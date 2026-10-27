"""Hash deduplication.

``HashDeduplicateFilter`` keeps the first row in each group. Strings are
stripped and internal whitespace is collapsed before hashing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns


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


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("dedupe operators expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _row_hash(values: list[Any]) -> str:
    payload = json.dumps([_norm_cell(value) for value in values], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_cell(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    return value
