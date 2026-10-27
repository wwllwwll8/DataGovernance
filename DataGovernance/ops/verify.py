"""Objective verify operators: hash dedupe, math/exact check, SQL stub.

These gates do not call an LLM. ``MathVerifyFilter`` uses ``math_verify``
when that package is installed and always has a numeric/exact fallback.
``SQLExecFilter`` is a SELECT-only sqlite stub.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
_SQL_COMMENT = re.compile(r"/\*.*?\*/|--.*?$", re.DOTALL | re.MULTILINE)
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
SQL_RESULT_ROW_CAP = 50


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


class HashDeduplicateFilter(OperatorABC):
    """Drop duplicate rows. The hash covers ``input_key`` or ``input_keys``.

    Strings are stripped and internal whitespace is collapsed before
    hashing. The first row in each group is kept.
    """

    def validate(self, arguments: Mapping[str, Any]) -> None:
        has_one = bool(arguments.get("input_key"))
        has_many = bool(arguments.get("input_keys"))
        if has_one and has_many:
            raise TypeError("Pass input_key or input_keys, not both")
        if not has_one and not has_many:
            raise TypeError("HashDeduplicateFilter requires input_key or input_keys")

    def run(
        self,
        storage: Any,
        input_key: str | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        keys = list(input_keys) if input_keys else [input_key or ""]
        frame = _frame(storage)
        if frame.empty:
            storage.write(frame)
            return
        require_columns(frame, keys, self.name)
        seen: set[str] = set()
        keep: list[int] = []
        records = frame[keys].to_dict(orient="records")
        for index, record in enumerate(records):
            digest = _row_hash([record[key] for key in keys])
            if digest in seen:
                continue
            seen.add(digest)
            keep.append(index)
        storage.write(frame.iloc[keep].reset_index(drop=True))


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


class SQLExecFilter(OperatorABC):
    """SELECT-only sqlite check. This is a stub, not a general SQL sandbox.

    ``setup_sql`` is trusted and runs with ``executescript`` on a fresh
    in-memory database for every row. ``input_setup_key`` uses that column
    as the per-row schema instead. SQL from ``input_key`` must be a single
    SELECT or WITH ... SELECT. Writes, multiple statements, and execution
    errors mark the row failed. ``drop_failures`` removes those rows. At
    most ``SQL_RESULT_ROW_CAP`` result rows are stored.

    TODO: a fuller execution verifier would compare result sets under a
    schema, a timeout, and a read-only authorizer. v0 stops at this stub.
    """

    def __init__(self, setup_sql: str | None = None) -> None:
        self.setup_sql = setup_sql or ""

    def run(
        self,
        storage: Any,
        input_key: str = "sql",
        input_gold_key: str | None = None,
        input_setup_key: str | None = None,
        output_key: str = "sql_ok",
        output_result: str = "sql_result",
        output_error: str = "sql_error",
        drop_failures: bool = True,
    ) -> None:
        frame = _frame(storage)
        columns = [input_key]
        if input_gold_key:
            columns.append(input_gold_key)
        if input_setup_key:
            columns.append(input_setup_key)
        if frame.empty:
            frame[output_key] = pd.Series(dtype="boolean")
            frame[output_result] = pd.Series(dtype="object")
            frame[output_error] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, columns, self.name)
        oks: list[bool] = []
        results: list[str] = []
        errors: list[str] = []
        golds = (
            frame[input_gold_key].tolist()
            if input_gold_key
            else [None] * len(frame)
        )
        setups = (
            frame[input_setup_key].tolist()
            if input_setup_key
            else [self.setup_sql] * len(frame)
        )
        for sql, gold, setup in zip(frame[input_key].tolist(), golds, setups, strict=True):
            ok, payload, error = _execute_select(str(sql), _setup_text(setup), gold)
            oks.append(ok)
            results.append(payload)
            errors.append(error)
        frame[output_key] = oks
        frame[output_result] = results
        frame[output_error] = errors
        if drop_failures:
            frame = frame.loc[pd.Series(oks, index=frame.index)].reset_index(drop=True)
        storage.write(frame)


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("verify operators expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _row_hash(values: list[Any]) -> str:
    payload = json.dumps([_norm_cell(value) for value in values], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_cell(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    return value


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


def _setup_text(setup: Any) -> str:
    if setup is None:
        return ""
    try:
        if bool(pd.isna(setup)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(setup)


def _execute_select(sql: str, setup_sql: str, gold: Any) -> tuple[bool, str, str]:
    if not _is_select_only(sql):
        return False, "", "not a single SELECT statement"
    connection = sqlite3.connect(":memory:")
    try:
        if setup_sql.strip():
            connection.executescript(setup_sql)
        try:
            cursor = connection.execute(sql)
        except sqlite3.Error as exc:
            return False, "", str(exc)
        if cursor.description is None:
            return False, "", "statement did not return rows"
        rows = [list(row) for row in cursor.fetchmany(SQL_RESULT_ROW_CAP)]
        payload = json.dumps(rows, ensure_ascii=False)
        if gold is None or (isinstance(gold, float) and pd.isna(gold)):
            return True, payload, ""
        if _json_equal(payload, str(gold)):
            return True, payload, ""
        return False, payload, "result did not match gold"
    finally:
        connection.close()


def _is_select_only(sql: str) -> bool:
    stripped = _SQL_COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        return False
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        return False
    if _SQL_FORBIDDEN.search(stripped):
        return False
    return True


def _json_equal(left: str, right: str) -> bool:
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return left.strip() == right.strip()
