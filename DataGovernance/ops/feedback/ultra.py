"""``UltraFeedbackOp`` and the judge output it parses.

``UltraFeedbackOp`` scores one response on a caller-chosen set of
aspects. The default set is the UltraFeedback paper: helpfulness,
honesty, instruction-following, and truthfulness. Pass ``aspects``
and ``input_context_keys`` for another task, such as math.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from DataGovernance.core.checkpoint import map_completions
from DataGovernance.core.operator import model_checkpoint_fields, require_columns
from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import LLMClient
from DataGovernance.ops.feedback.task import DEFAULT_JUDGE_SYSTEM, TaskOp, _frame

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


class ParsedJudge:
    """One UltraFeedback-like judgement."""

    def __init__(self, ratings: dict[str, float], rationales: dict[str, str], overall: float) -> None:
        self.ratings = ratings
        self.rationales = rationales
        self.overall = overall


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
                **model_checkpoint_fields(self.llm),
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