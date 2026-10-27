"""General pipeline stages: generate, rank, and score.

Each stage renders a ``Prompt`` per row and writes one column. The prompt's
placeholders are the stage inputs, so column checks happen before the model
runs. Load and dedupe stay in ``ops``; ``Pipeline`` wires them in.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns
from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import LLMClient, ensure_llm


class _PromptStage(OperatorABC):
    """One prompted model call per row."""

    def __init__(self, llm: LLMClient, prompt: Prompt | str, output_key: str) -> None:
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("output_key must be a non-empty string")
        self.llm = ensure_llm(llm)
        self.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
        self.output_key = output_key

    def declared_inputs(self) -> list[str]:
        return self.prompt.input_columns()

    def declared_outputs(self) -> list[str]:
        return [self.output_key]

    def run(self, storage: Any) -> None:
        frame = _frame(storage)
        if frame.empty:
            frame[self.output_key] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, self.prompt.input_columns(), self.name)
        chats = [self.prompt.render(record) for record in frame.to_dict(orient="records")]
        completions = list(self.llm.generate(chats))
        if len(completions) != len(chats):
            raise RuntimeError(
                f"{self.name} expected {len(chats)} completions, got {len(completions)}"
            )
        frame[self.output_key] = [self.interpret(text) for text in completions]
        storage.write(frame)

    def interpret(self, text: str) -> Any:
        return text.strip()


class GenerateOp(_PromptStage):
    """Write a model completion from a prompt. The text is kept as returned."""


class RankOp(_PromptStage):
    """Ask the model to rank a row.

    A JSON object with ``rank`` stores that value. Any other completion is
    stored as text, so the prompt can ask for a number or an order.
    """

    def interpret(self, text: str) -> Any:
        data = _json_object(text)
        if isinstance(data, dict) and "rank" in data:
            return data["rank"]
        return text.strip()


class ScoreOp(_PromptStage):
    """Ask the model to score a row.

    A JSON object with ``score`` or ``overall`` stores that value. Any other
    completion is stored as text.
    """

    def interpret(self, text: str) -> Any:
        data = _json_object(text)
        if isinstance(data, dict):
            for key in ("score", "overall"):
                if key in data:
                    return data[key]
        return text.strip()


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("pipeline stages expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[: -3].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
