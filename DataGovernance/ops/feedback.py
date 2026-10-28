"""Feedback and judging operators.

``TextGenerationOp`` writes the text that gets judged. ``UltraFeedbackOp``
scores one response on a caller-chosen set of aspects. The default set is
helpfulness, honesty, instruction-following, and truthfulness. Pass
``aspects`` and ``input_context_keys`` for another task, such as math.
Pass an object with ``inputs``, ``outputs``, ``format_input``, and
``format_output`` and the same client will drive it.

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

from DataGovernance.core.checkpoint import map_completions
from DataGovernance.core.operator import OperatorABC, require_columns
from DataGovernance.core.prompt import Prompt
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

    This operator calls a model. Finished rows are checkpointed.

    Recipes (``text_generation``, ``ultrafeedback``) take ``input_*`` and
    ``output_*`` arguments on ``run``, like other operators. A ``task``
    object supplies its own column names via ``inputs`` and ``outputs``;
    in that mode ``run`` takes only ``storage``.
    """

    consumes_tokens = True

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

    def checkpoint_signature(self) -> str:
        if self.task is not None:
            template = type(self.task).__name__
        elif self.recipe == "ultrafeedback":
            aspects = getattr(self, "aspects", ULTRAFEEDBACK_ASPECTS)
            template = build_ultrafeedback_chat(
                "",
                "",
                self.system_prompt or DEFAULT_JUDGE_SYSTEM,
                aspects,
            )[1]["content"]
        else:
            template = self.system_prompt or DEFAULT_TEXT_SYSTEM
        return json.dumps(
            {
                "model": str(getattr(self.llm, "model", "")),
                "system": self.system_prompt or "",
                "template": template,
            },
            sort_keys=True,
        )

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
        frame[output_key] = map_completions(
            self,
            storage,
            {"input_key": input_key, "output_key": output_key},
            self.llm,
            chats,
            lambda _index, text: text,
        )
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
        input_context_keys: list[str] | None = None,
    ) -> None:
        frame = _frame(storage)
        new_columns = [output_ratings, output_rationales, output_overall]
        if output_model:
            new_columns.append(output_model)
        context_keys = list(input_context_keys or [])
        if frame.empty:
            for column in new_columns:
                frame[column] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, [input_key, input_generation_key, *context_keys], self.name)
        system = self.system_prompt or DEFAULT_JUDGE_SYSTEM
        aspects = getattr(self, "aspects", ULTRAFEEDBACK_ASPECTS)
        chats = []
        for record in frame.to_dict(orient="records"):
            context = {
                key: "" if record.get(key) is None else str(record[key]) for key in context_keys
            }
            chats.append(
                build_ultrafeedback_chat(
                    str(record[input_key]),
                    str(record[input_generation_key]),
                    system,
                    aspects,
                    context,
                )
            )
        def accept(index: int, text: str) -> ParsedJudge:
            try:
                return parse_ultrafeedback(text)
            except ValueError as exc:
                raise ValueError(f"{self.name} row {index} judge output was not usable") from exc

        parsed = map_completions(
            self,
            storage,
            {
                "input_key": input_key,
                "input_generation_key": input_generation_key,
                "output_ratings": output_ratings,
                "output_rationales": output_rationales,
                "output_overall": output_overall,
                "output_model": output_model,
                "input_context_keys": input_context_keys,
            },
            self.llm,
            chats,
            accept,
        )
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

        def accept(index: int, text: str) -> dict[str, Any]:
            formatted = self.task.format_output(text, records[index])
            if not isinstance(formatted, dict):
                raise TypeError("task.format_output must return a dict")
            return formatted

        extras = map_completions(self, storage, {}, self.llm, chats, accept)
        for column in outputs:
            frame[column] = [row.get(column) for row in extras]
        storage.write(frame)


class TextGenerationOp(TaskOp):
    """Text generation operator.

    Pass ``prompt`` and the operator owns that text. ``{column}``
    placeholders are the columns it reads, and ``output_key`` is the
    column it writes. Without a prompt, the built-in recipe reads
    ``input_key`` (default ``instruction``) and writes ``output_key``
    (default ``generation``). Pass ``task`` to use an object with
    ``inputs``, ``outputs``, ``format_input``, and ``format_output``.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        task: Any = None,
        prompt: Prompt | str | None = None,
        output_key: str = "generation",
    ) -> None:
        if prompt is not None and task is not None:
            raise ValueError("Pass prompt or task, not both")
        if prompt is not None and system_prompt is not None:
            raise ValueError("Put the system text on the Prompt")
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("output_key must be a non-empty string")
        super().__init__(
            llm,
            recipe=None if task is not None else "text_generation",
            task=task,
            system_prompt=system_prompt,
        )
        self.prompt_template = None if prompt is None else prompt if isinstance(prompt, Prompt) else Prompt(prompt)
        self.output_key = output_key

    def declared_inputs(self) -> list[str] | None:
        if self.prompt_template is not None:
            return self.prompt_template.input_columns()
        return super().declared_inputs()

    def declared_outputs(self) -> list[str] | None:
        if self.prompt_template is not None:
            return [self.output_key]
        return super().declared_outputs()

    def checkpoint_signature(self) -> str:
        if self.prompt_template is None:
            return super().checkpoint_signature()
        return json.dumps(
            {
                "model": str(getattr(self.llm, "model", "")),
                "system": self.prompt_template.system,
                "user": self.prompt_template.user,
                "output_key": self.output_key,
            },
            sort_keys=True,
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
        if self.prompt_template is not None:
            self._run_prompt(storage)
            return
        self._run_text_generation(storage, input_key=input_key, output_key=output_key)

    def _run_prompt(self, storage: Any) -> None:
        assert self.prompt_template is not None
        frame = _frame(storage)
        if frame.empty:
            frame[self.output_key] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, self.prompt_template.input_columns(), self.name)
        records = frame.to_dict(orient="records")
        chats = [self.prompt_template.render(record) for record in records]
        frame[self.output_key] = map_completions(
            self,
            storage,
            {},
            self.llm,
            chats,
            lambda _index, text: text.strip(),
        )
        storage.write(frame)


class UltraFeedbackOp(TaskOp):
    """Judge one response on a caller-chosen set of aspects.

    The default aspects are the UltraFeedback paper: helpfulness, honesty,
    instruction-following, and truthfulness. Pass ``aspects`` for another
    task, such as math correctness. ``input_key`` is the task text and
    ``input_generation_key`` is the response. ``input_context_keys`` adds
    other columns, such as a gold answer or a schema, to the same prompt.

    Writes ``ratings`` (aspect → score), ``rationales`` (aspect → text),
    and ``overall``. ``instruction-following`` is stored as
    ``instruction_following``. Passing a ``task`` uses that object's own
    columns.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        task: Any = None,
        aspects: list[str] | tuple[str, ...] | None = None,
        prompt: Prompt | None = None,
        output_model: str | None = None,
    ) -> None:
        if prompt is not None and task is not None:
            raise ValueError("Pass prompt or task, not both")
        super().__init__(
            llm,
            recipe=None if task is not None else "ultrafeedback",
            task=task,
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_JUDGE_SYSTEM,
        )
        self.prompt_template = prompt
        self.configured_model_column = output_model
        self.aspects = (
            ()
            if prompt is not None
            else normalize_aspects(ULTRAFEEDBACK_ASPECTS if aspects is None else aspects)
        )

    def declared_inputs(self) -> list[str] | None:
        if self.prompt_template is not None:
            return self.prompt_template.input_columns()
        return super().declared_inputs()

    def declared_outputs(self) -> list[str] | None:
        if self.prompt_template is None:
            return super().declared_outputs()
        columns = ["ratings", "rationales", "overall"]
        if self.configured_model_column:
            columns.append(self.configured_model_column)
        return columns

    def checkpoint_signature(self) -> str:
        if self.prompt_template is None:
            return super().checkpoint_signature()
        return json.dumps(
            {
                "model": str(getattr(self.llm, "model", "")),
                "system": self.prompt_template.system,
                "user": self.prompt_template.user,
                "output_model": self.configured_model_column or "",
            },
            sort_keys=True,
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
        input_context_keys: list[str] | None = None,
    ) -> None:
        if self.task is not None:
            self._run_task(storage)
            return
        if self.prompt_template is not None:
            self._run_configured_prompt(storage)
            return
        self._run_ultrafeedback(
            storage,
            input_key=input_key,
            input_generation_key=input_generation_key,
            output_ratings=output_ratings,
            output_rationales=output_rationales,
            output_overall=output_overall,
            output_model=output_model,
            input_context_keys=input_context_keys,
        )

    def _run_configured_prompt(self, storage: Any) -> None:
        assert self.prompt_template is not None
        frame = _frame(storage)
        columns = ["ratings", "rationales", "overall"]
        if self.configured_model_column:
            columns.append(self.configured_model_column)
        if frame.empty:
            for column in columns:
                frame[column] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, self.prompt_template.input_columns(), self.name)
        records = frame.to_dict(orient="records")
        chats = [self.prompt_template.render(record) for record in records]

        def accept(index: int, text: str) -> ParsedJudge:
            try:
                return parse_ultrafeedback(text)
            except ValueError as exc:
                raise ValueError(f"{self.name} row {index} judge output was not usable") from exc

        parsed = map_completions(self, storage, {}, self.llm, chats, accept)
        frame["ratings"] = [item.ratings for item in parsed]
        frame["rationales"] = [item.rationales for item in parsed]
        frame["overall"] = [item.overall for item in parsed]
        if self.configured_model_column:
            frame[self.configured_model_column] = getattr(self.llm, "model", "unknown")
        storage.write(frame)


def build_text_generation_chat(instruction: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]


def normalize_aspects(aspects: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize aspect names. ``overall`` is reserved for the summary score."""
    names: list[str] = []
    for aspect in aspects:
        if not isinstance(aspect, str) or not aspect.strip():
            raise ValueError(f"aspect must be a non-empty string, got {aspect!r}")
        normalized = _normalize_name(aspect)
        if normalized in _OVERALL_NAMES:
            raise ValueError("overall is computed from the aspect ratings")
        canon = _ASPECT_ALIASES.get(normalized, normalized)
        if canon not in names:
            names.append(canon)
    if not names:
        raise ValueError("UltraFeedback needs at least one aspect")
    return tuple(names)


def build_ultrafeedback_chat(
    instruction: str,
    generation: str,
    system_prompt: str = DEFAULT_JUDGE_SYSTEM,
    aspects: list[str] | tuple[str, ...] | None = None,
    context: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a judge chat for any task.

    ``aspects`` defaults to the UltraFeedback paper. ``context`` is extra
    labeled fields, in insertion order, such as a gold answer.
    """
    chosen = ULTRAFEEDBACK_ASPECTS if aspects is None else normalize_aspects(aspects)
    rating_shape = ", ".join(f"{json.dumps(name)}: 1" for name in chosen)
    rationale_shape = ", ".join(f'{json.dumps(name)}: ""' for name in chosen)
    lines = [
        "TASK: ultrafeedback",
        f"Rate the response on a 1-5 scale for each aspect: {', '.join(chosen)}.",
        "Return a JSON object only, with this shape:",
        '{"ratings": {' + rating_shape + '}, "rationales": {' + rationale_shape + '}, "overall": 1}',
        "",
        f"Instruction:\n{instruction}",
        "",
        f"Response:\n{generation}",
    ]
    for key, value in (context or {}).items():
        lines.extend(["", f"{key}:\n{value}"])
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(lines)},
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
    reserved = {"ratings", "rationales", *_OVERALL_NAMES}
    for name, value in data.items():
        if _normalize_name(str(name)) in reserved:
            continue
        aspect = _canon_aspect(str(name))
        if aspect is None or aspect in ratings:
            continue
        ratings[aspect] = _as_score(value)
        if isinstance(value, dict) and value.get("rationale"):
            rationales[aspect] = str(value["rationale"])
    if not ratings and not per_generation and not hinted_overall and not _has_overall(data):
        raise ValueError("no UltraFeedback ratings found")
    overall = _overall_score(data, ratings, per_generation, hinted_overall)
    return ParsedJudge(ratings, rationales, overall)


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
    normalized = _normalize_name(name)
    if not normalized or normalized in _OVERALL_NAMES:
        return None
    return _ASPECT_ALIASES.get(normalized, normalized)


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
