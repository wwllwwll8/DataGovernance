"""``TaskOp``: the recipe dispatcher every operator in this package extends.

Recipes (``text_generation``, ``ultrafeedback``) take ``input_*`` and
``output_*`` arguments on ``run``, like other operators. A ``task``
object supplies its own column names via ``inputs`` and ``outputs``;
in that mode ``run`` takes only ``storage``.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from DataGovernance.core.checkpoint import map_completions
from DataGovernance.core.operator import OperatorABC, model_checkpoint_fields, require_columns
from DataGovernance.llm import LLMClient, ensure_llm

DEFAULT_TEXT_SYSTEM = (
    "Answer the user directly. Put the final answer where a checker can read it."
)
DEFAULT_JUDGE_SYSTEM = "You are a careful preference judge. Return JSON only."

RECIPES = frozenset({"text_generation", "ultrafeedback"})


class TaskOp(OperatorABC):
    """Run a built-in recipe or a duck-typed task.

    This operator calls a model. Finished rows are checkpointed.
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
            from DataGovernance.ops.feedback.ultra import (
                ULTRAFEEDBACK_ASPECTS,
                build_ultrafeedback_chat,
            )

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
                **model_checkpoint_fields(self.llm),
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
        from DataGovernance.ops.feedback.ultra import (
            ULTRAFEEDBACK_ASPECTS,
            ParsedJudge,
            build_ultrafeedback_chat,
            parse_ultrafeedback,
        )

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


def build_text_generation_chat(instruction: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]


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