"""Feedback and judging operators.

``TextGenerationOp`` writes the text that gets judged. ``UltraFeedbackOp``
scores one generation on helpfulness, honesty, instruction-following, and
truthfulness. Pass an object with ``inputs``, ``outputs``, ``format_input``,
and ``format_output`` and the same client will drive it.

TODO: score a list of generations in one prompt. That shape reads
``instruction`` and ``generations`` (a list of strings) and writes
``ratings``, ``rationales``, and ``model_name``. Aspects are
``helpfulness``, ``honesty``, ``instruction-following``, ``truthfulness``,
and the summary aspect ``overall-rating``. v0 scores a single
``generation`` column through ``LLMClient`` and flattens the aspects into
dataframe columns.

TODO: export instruction, generations, and optional ratings and rationales.
See ``Pipeline.export_argilla``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns
from DataGovernance.llm import LLMClient, ensure_llm

ULTRAFEEDBACK_ASPECTS: tuple[str, ...] = (
    "helpfulness",
    "honesty",
    "instruction_following",
    "truthfulness",
)

_ASPECT_ALIASES = {
    "helpfulness": "helpfulness",
    "honesty": "honesty",
    "instruction_following": "instruction_following",
    "truthfulness": "truthfulness",
}
_OVERALL_NAMES = {"overall", "overall_rating", "overall_score"}

DEFAULT_TEXT_SYSTEM = (
    "Answer the user directly. Put the final answer where a checker can read it."
)
DEFAULT_JUDGE_SYSTEM = "You are a careful preference judge. Return JSON only."

RECIPES = frozenset({"text_generation", "ultrafeedback"})


class ParsedJudge:
    """One UltraFeedback-like judgement."""

    def __init__(self, ratings: dict[str, float], rationales: dict[str, str], overall: float) -> None:
        self.ratings = ratings
        self.rationales = rationales
        self.overall = overall


class TaskOp(OperatorABC):
    """Run a built-in recipe or a duck-typed task.

    Recipes (``text_generation``, ``ultrafeedback``) take ``input_*`` and
    ``output_*`` arguments on ``run``, like other operators. A ``task``
    object supplies its own column names via ``inputs`` and ``outputs``;
    in that mode ``run`` takes only ``storage``.
    """

    def __init__(
        self,
        llm: LLMClient,
        recipe: str | None = None,
        task: Any = None,
        system_prompt: str | None = None,
    ) -> None:
        if task is None and recipe not in RECIPES:
            raise ValueError(
                "Pass recipe='text_generation' or recipe='ultrafeedback', "
                "or pass a task with inputs, outputs, "
                "format_input, and format_output."
            )
        if task is not None and recipe is not None:
            raise ValueError("Pass either recipe or task, not both")
        self.llm = ensure_llm(llm)
        self.recipe = recipe
        self.task = task
        self.system_prompt = system_prompt
        if task is not None:
            missing = [
                name
                for name in ("inputs", "outputs", "format_input", "format_output")
                if not hasattr(task, name)
            ]
            if missing:
                raise TypeError(f"task is missing {missing}")
            _task_list(task, "inputs")
            _task_list(task, "outputs")

    def declared_inputs(self) -> list[str] | None:
        if self.task is None:
            return None
        return _task_list(self.task, "inputs")

    def declared_outputs(self) -> list[str] | None:
        if self.task is None:
            return None
        return _task_list(self.task, "outputs")

    def run(self, storage: Any, **kwargs: Any) -> None:
        if self.task is not None:
            if kwargs:
                raise TypeError("A wrapped task takes only storage; columns come from the task")
            self._run_task(storage)
            return
        if self.recipe == "text_generation":
            self._run_text_generation(storage, **kwargs)
            return
        if self.recipe == "ultrafeedback":
            self._run_ultrafeedback(storage, **kwargs)
            return
        raise RuntimeError(f"Unknown recipe {self.recipe!r}")

    def _run_text_generation(
        self,
        storage: Any,
        input_key: str = "instruction",
        output_key: str = "generation",
    ) -> None:
        frame = _frame(storage)
        if frame.empty:
            frame[output_key] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, [input_key], self.name)
        system = self.system_prompt or DEFAULT_TEXT_SYSTEM
        chats = [
            build_text_generation_chat(str(text), system) for text in frame[input_key].tolist()
        ]
        frame[output_key] = _generate(self.llm, chats)
        storage.write(frame)

    def _run_ultrafeedback(
        self,
        storage: Any,
        input_key: str = "instruction",
        input_generation_key: str = "generation",
        output_ratings: str = "ratings",
        output_rationales: str = "rationales",
        output_overall: str = "overall",
        output_model: str | None = None,
    ) -> None:
        frame = _frame(storage)
        new_columns = [output_ratings, output_rationales, output_overall]
        if output_model:
            new_columns.append(output_model)
        if frame.empty:
            for column in new_columns:
                frame[column] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, [input_key, input_generation_key], self.name)
        system = self.system_prompt or DEFAULT_JUDGE_SYSTEM
        chats = [
            build_ultrafeedback_chat(str(instruction), str(generation), system)
            for instruction, generation in zip(
                frame[input_key].tolist(),
                frame[input_generation_key].tolist(),
                strict=True,
            )
        ]
        completions = _generate(self.llm, chats)
        parsed: list[ParsedJudge] = []
        for index, text in enumerate(completions):
            try:
                parsed.append(parse_ultrafeedback(text))
            except ValueError as exc:
                raise ValueError(f"{self.name} row {index} judge output was not usable") from exc
        frame[output_ratings] = [item.ratings for item in parsed]
        frame[output_rationales] = [item.rationales for item in parsed]
        frame[output_overall] = [item.overall for item in parsed]
        if output_model:
            frame[output_model] = getattr(self.llm, "model", "unknown")
        storage.write(frame)

    def _run_task(self, storage: Any) -> None:
        assert self.task is not None
        inputs = _task_list(self.task, "inputs")
        outputs = _task_list(self.task, "outputs")
        frame = _frame(storage)
        if frame.empty:
            for column in outputs:
                if column not in frame.columns:
                    frame[column] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, inputs, self.name)
        records = frame.to_dict(orient="records")
        chats = [_coerce_chat(self.task.format_input(record)) for record in records]
        completions = _generate(self.llm, chats)
        extras: list[dict[str, Any]] = []
        for record, text in zip(records, completions, strict=True):
            formatted = self.task.format_output(text, record)
            if not isinstance(formatted, dict):
                raise TypeError("task.format_output must return a dict")
            extras.append(formatted)
        for column in outputs:
            frame[column] = [row.get(column) for row in extras]
        storage.write(frame)


class TextGenerationOp(TaskOp):
    """Text generation operator.

    The built-in recipe reads ``input_key`` (default ``instruction``) and
    writes ``output_key`` (default ``generation``). Pass ``task`` to use an
    object with ``inputs``, ``outputs``, ``format_input``, and
    ``format_output`` instead of the built-in prompt.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        task: Any = None,
    ) -> None:
        super().__init__(
            llm,
            recipe=None if task is not None else "text_generation",
            task=task,
            system_prompt=system_prompt,
        )

    def run(
        self,
        storage: Any,
        input_key: str = "instruction",
        output_key: str = "generation",
    ) -> None:
        if self.task is not None:
            self._run_task(storage)
            return
        self._run_text_generation(storage, input_key=input_key, output_key=output_key)


class UltraFeedbackOp(TaskOp):
    """Single-response UltraFeedback-style judge.

    Reads an instruction and one generation. Writes ``ratings`` (aspect →
    score), ``rationales`` (aspect → text), and ``overall``. Aspect names
    follow the UltraFeedback paper, with ``instruction-following`` stored
    as ``instruction_following``.

    A list of generations can be judged in one prompt and returned as
    parallel ``ratings`` / ``rationales`` lists. Passing a ``task`` uses
    that object's own columns. See the module TODO.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        task: Any = None,
    ) -> None:
        super().__init__(
            llm,
            recipe=None if task is not None else "ultrafeedback",
            task=task,
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_JUDGE_SYSTEM,
        )

    def run(
        self,
        storage: Any,
        input_key: str = "instruction",
        input_generation_key: str = "generation",
        output_ratings: str = "ratings",
        output_rationales: str = "rationales",
        output_overall: str = "overall",
        output_model: str | None = None,
    ) -> None:
        if self.task is not None:
            self._run_task(storage)
            return
        self._run_ultrafeedback(
            storage,
            input_key=input_key,
            input_generation_key=input_generation_key,
            output_ratings=output_ratings,
            output_rationales=output_rationales,
            output_overall=output_overall,
            output_model=output_model,
        )


def build_text_generation_chat(instruction: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]


def build_ultrafeedback_chat(
    instruction: str,
    generation: str,
    system_prompt: str = DEFAULT_JUDGE_SYSTEM,
) -> list[dict[str, str]]:
    user = (
        "TASK: ultrafeedback\n"
        "Rate the assistant response on a 1-5 scale for each aspect: "
        "helpfulness, honesty, instruction_following, truthfulness.\n"
        "Return a JSON object only, with this shape:\n"
        '{"ratings": {"helpfulness": 1, "honesty": 1, "instruction_following": 1, '
        '"truthfulness": 1}, "rationales": {"helpfulness": "", "honesty": "", '
        '"instruction_following": "", "truthfulness": ""}, "overall": 1}\n\n'
        f"Instruction:\n{instruction}\n\n"
        f"Response:\n{generation}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def parse_ultrafeedback(text: str) -> ParsedJudge:
    """Parse a judge completion into aspect scores.

    Accepts the built-in JSON object (aspect → score), a flat aspect map,
    and aspect objects with ``rating`` / ``rationale``. A ``ratings`` list
    of floats is one score per generation, not one score
    per aspect; those become ``overall`` (the mean) and do not invent
    aspect names.
    """
    data = _extract_json_object(text)
    ratings: dict[str, float] = {}
    rationales: dict[str, str] = {}
    per_generation: list[float] = []
    hinted_overall: list[float] = []
    _consume_ratings(data.get("ratings"), ratings, rationales, per_generation, hinted_overall)
    _consume_rationales(data.get("rationales"), rationales)
    for aspect in ULTRAFEEDBACK_ASPECTS:
        if aspect in ratings or aspect not in data:
            continue
        value = data[aspect]
        ratings[aspect] = _as_score(value)
        if isinstance(value, dict) and value.get("rationale"):
            rationales[aspect] = str(value["rationale"])
    if not ratings and not per_generation and not hinted_overall and not _has_overall(data):
        raise ValueError("no UltraFeedback ratings found")
    overall = _overall_score(data, ratings, per_generation, hinted_overall)
    return ParsedJudge(ratings, rationales, overall)


def _generate(llm: LLMClient, chats: list[list[dict[str, str]]]) -> list[str]:
    outputs = list(llm.generate(chats))
    if len(outputs) != len(chats):
        raise RuntimeError(
            f"LLM returned {len(outputs)} completions for {len(chats)} chats"
        )
    for text in outputs:
        if not isinstance(text, str):
            raise TypeError("LLM generate() must return strings")
    return outputs


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("task operators expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _task_list(task: Any, name: str) -> list[str]:
    value = getattr(task, name)
    if callable(value):
        value = value()
    return [str(item) for item in value]


def _coerce_chat(chat: Any) -> list[dict[str, str]]:
    if isinstance(chat, str):
        return [{"role": "user", "content": chat}]
    if not isinstance(chat, (list, tuple)):
        raise TypeError("format_input must return a chat: a string or a list of messages")
    messages: list[dict[str, str]] = []
    for message in chat:
        if not isinstance(message, dict) or "content" not in message:
            raise TypeError("format_input messages must be dicts with content")
        messages.append(
            {"role": str(message.get("role", "user")), "content": str(message["content"])}
        )
    return messages


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, count=1).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"UltraFeedback response was not a JSON object: {text[:200]}") from None
        try:
            obj = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"UltraFeedback response was not a JSON object: {text[:200]}") from exc
    if not isinstance(obj, dict):
        raise ValueError("UltraFeedback response JSON must be an object")
    return obj


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _canon_aspect(name: str) -> str | None:
    return _ASPECT_ALIASES.get(_normalize_name(name))


def _as_score(value: Any) -> float:
    if isinstance(value, dict):
        if "rating" in value:
            value = value["rating"]
        elif "score" in value:
            value = value["score"]
        else:
            raise ValueError(f"rating object has no rating/score: {value}")
    if isinstance(value, bool) or value is None:
        raise ValueError(f"rating is not a number: {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rating is not a number: {value!r}") from exc


def _consume_ratings(
    block: Any,
    ratings: dict[str, float],
    rationales: dict[str, str],
    per_generation: list[float],
    hinted_overall: list[float],
) -> None:
    if isinstance(block, dict):
        for name, value in block.items():
            _take_rating(str(name), value, ratings, rationales, hinted_overall)
        return
    if not isinstance(block, list):
        return
    for item in block:
        if isinstance(item, dict) and "aspect" in item:
            _take_rating(
                str(item.get("aspect", "")),
                item.get("rating", item.get("score")),
                ratings,
                rationales,
                hinted_overall,
            )
            if item.get("rationale"):
                aspect = _canon_aspect(str(item.get("aspect", "")))
                if aspect:
                    rationales[aspect] = str(item["rationale"])
        else:
            # One float per generation, for one aspect.
            per_generation.append(_as_score(item))


def _take_rating(
    name: str,
    value: Any,
    ratings: dict[str, float],
    rationales: dict[str, str],
    hinted_overall: list[float],
) -> None:
    normalized = _normalize_name(name)
    if normalized in _OVERALL_NAMES:
        hinted_overall.append(_as_score(value))
        return
    aspect = _canon_aspect(name)
    if aspect is None:
        return
    ratings[aspect] = _as_score(value)
    if isinstance(value, dict) and value.get("rationale"):
        rationales[aspect] = str(value["rationale"])


def _consume_rationales(block: Any, rationales: dict[str, str]) -> None:
    if isinstance(block, dict):
        items = list(block.items())
    elif isinstance(block, list):
        items = []
        for index, item in enumerate(block):
            if isinstance(item, dict):
                aspect_name = str(item.get("aspect", ""))
                text = item.get("rationale", item.get("text"))
                if aspect_name and text:
                    items.append((aspect_name, text))
            else:
                # One rationale per generation, not per aspect.
                items.append((f"generation_{index}", item))
    else:
        return
    for name, value in items:
        if not name:
            continue
        if str(name).startswith("generation_"):
            rationales[str(name)] = str(value)
            continue
        aspect = _canon_aspect(str(name))
        if aspect is not None:
            rationales[aspect] = str(value)


def _has_overall(data: dict[str, Any]) -> bool:
    for name in ("overall", "overall_score", "overall-rating", "overall_rating"):
        value = data.get(name)
        if value is not None and not isinstance(value, dict):
            return True
    return False


def _overall_score(
    data: dict[str, Any],
    ratings: dict[str, float],
    per_generation: list[float],
    hinted_overall: list[float],
) -> float:
    for name in ("overall", "overall_score", "overall-rating", "overall_rating"):
        value = data.get(name)
        if value is not None and not isinstance(value, dict):
            return float(value)
    if hinted_overall:
        return hinted_overall[-1]
    if ratings:
        return sum(ratings.values()) / len(ratings)
    if per_generation:
        return sum(per_generation) / len(per_generation)
    raise ValueError("no UltraFeedback overall score")
