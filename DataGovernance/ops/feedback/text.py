"""``TextGenerationOp``: write the text that gets judged.

Pass ``prompt`` and the operator owns that text. ``{column}``
placeholders are the columns it reads, and ``output_key`` is the
column it writes. Without a prompt, the built-in recipe reads
``input_key`` (default ``instruction``) and writes ``output_key``
(default ``generation``). Pass ``task`` to use an object with
``inputs``, ``outputs``, ``format_input``, and ``format_output``.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from DataGovernance.core.checkpoint import map_completions
from DataGovernance.core.operator import model_checkpoint_fields, require_columns
from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import LLMClient
from DataGovernance.ops.feedback.task import TaskOp, _frame


class TextGenerationOp(TaskOp):
    """Text generation operator."""

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
                **model_checkpoint_fields(self.llm),
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