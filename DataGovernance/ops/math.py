"""Math answer check and its UltraFeedback judge.

``math_feedback`` scores a math answer with ``UltraFeedbackOp``.
The caller passes the prompt, usually ``math.judge``.
``MathVerifyFilter`` is the exact check: it
compares a prediction column with a gold column using ``math_verify``
when that package is installed, then normalized exact text, then the
last number on each side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns
from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import LLMClient
from DataGovernance.ops.feedback import UltraFeedbackOp

MATH_ASPECTS = ("correctness", "reasoning")


def math_feedback(llm: LLMClient, prompt: Prompt) -> UltraFeedbackOp:
    """UltraFeedback judge for a math answer.

    The caller passes the prompt, usually ``load_prompt("math.judge")``.
    ``MathVerifyFilter`` still decides whether the number matches.
    """
    return UltraFeedbackOp(llm, prompt=prompt)

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class AnswerCheck:
    """Result of comparing a prediction with a reference answer."""

    ok: bool
    method: str


def check_answer(prediction: Any, gold: Any) -> AnswerCheck:
    """Compare ``prediction`` to ``gold``.

    Order: optional ``math_verify``, then normalized exact text, then the
    last number in each side (including ``\\boxed{...}``).
    """
    pred = "" if prediction is None else str(prediction)
    reference = "" if gold is None else str(gold)
    verified = _math_verify(pred, reference)
    if verified is True:
        return AnswerCheck(True, "math_verify")
    if _normalize_text(pred) == _normalize_text(reference):
        return AnswerCheck(True, "exact")
    pred_number = _extract_last_number(pred)
    gold_number = _extract_last_number(reference)
    if pred_number is not None and gold_number is not None and abs(pred_number - gold_number) <= 1e-6:
        return AnswerCheck(True, "numeric")
    return AnswerCheck(False, "mismatch")


class MathVerifyFilter(OperatorABC):
    """Compare ``input_key`` to ``input_gold_key`` and write ``output_key``.

    Rows that do not match are dropped when ``drop_failures`` is true.
    The boolean column is written before the drop, so survivors are marked
    true and an empty result still carries the column.
    """

    def run(
        self,
        storage: Any,
        input_key: str = "generation",
        input_gold_key: str = "answer",
        output_key: str = "math_correct",
        drop_failures: bool = True,
    ) -> None:
        frame = _frame(storage)
        if frame.empty:
            frame[output_key] = pd.Series(dtype="boolean")
            storage.write(frame)
            return
        require_columns(frame, [input_key, input_gold_key], self.name)
        flags = [
            check_answer(pred, gold).ok
            for pred, gold in zip(
                frame[input_key].tolist(),
                frame[input_gold_key].tolist(),
                strict=True,
            )
        ]
        frame[output_key] = flags
        if drop_failures:
            frame = frame.loc[pd.Series(flags, index=frame.index)].reset_index(drop=True)
        storage.write(frame)


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("math operators expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _extract_last_number(text: str) -> float | None:
    boxed = _BOXED.findall(text)
    target = boxed[-1] if boxed else text
    numbers = _NUMBER.findall(target.replace(",", ""))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _math_verify(prediction: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        return bool(verify(parse(gold), parse(prediction)))
    except Exception:
        return None
