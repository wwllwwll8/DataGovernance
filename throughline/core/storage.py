"""JSONL step storage.

``operator_step`` starts at -1. ``step()`` advances it and returns a view
for one operator. ``read()`` loads the current step. Step 0 is the entry
file (``first_entry_file_name``). ``write()`` saves step ``operator_step + 1``
as ``{prefix}{n}.jsonl`` inside ``cache_path`` (default ``step1.jsonl``,
``step2.jsonl``, ...). The next operator's ``step()`` then reads that file.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FileStorage:
    """Disk-backed dataframe storage with a JSONL step chain."""

    def __init__(
        self,
        first_entry_file_name: str = "",
        cache_path: str | Path = "./cache",
        file_name_prefix: str = "step",
        cache_type: str = "jsonl",
    ) -> None:
        if cache_type != "jsonl":
            raise ValueError("v0 FileStorage writes JSONL only")
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = Path(cache_path)
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def step(self) -> FileStorage:
        """Advance the shared counter and return this operator's view."""
        self.operator_step += 1
        return self.view()

    def view(self) -> FileStorage:
        """Copy the counters into an independent storage view."""
        twin = FileStorage(
            first_entry_file_name=self.first_entry_file_name,
            cache_path=self.cache_path,
            file_name_prefix=self.file_name_prefix,
            cache_type=self.cache_type,
        )
        twin.operator_step = self.operator_step
        return twin

    def reset(self) -> FileStorage:
        """Set the counter back to -1. Does not delete step files."""
        self.operator_step = -1
        return self

    def get_keys_from_dataframe(self) -> list[str]:
        """Column names at the current step. Does not call an LLM."""
        return list(self.read().columns)

    def read(self, output_type: str = "dataframe") -> pd.DataFrame | list[dict[str, Any]]:
        """Load the current step as a DataFrame or a list of records."""
        if output_type not in {"dataframe", "dict"}:
            raise ValueError("output_type must be 'dataframe' or 'dict'")
        frame = self._read_frame()
        if output_type == "dict":
            return dataframe_to_records(frame)
        return frame

    def write(self, data: pd.DataFrame | list[dict[str, Any]]) -> Path:
        """Write this operator's output to the next step file."""
        if self.operator_step < 0:
            raise ValueError("Call storage.step() before writing data")
        frame = _as_frame(data)
        path = self.cache_file(self.operator_step + 1)
        write_jsonl(path, frame)
        return path

    def read_output(self) -> pd.DataFrame:
        """Load the file this view would write (step ``operator_step + 1``)."""
        if self.operator_step < 0:
            raise ValueError("Call storage.step() before reading output")
        path = self.cache_file(self.operator_step + 1)
        if not path.exists():
            raise FileNotFoundError(
                f"No output at {path}. The operator at step {self.operator_step} has not written yet."
            )
        return read_jsonl(path)

    def cache_file(self, step: int) -> Path:
        if step <= 0:
            raise ValueError(f"Cache files start at step 1, got {step}")
        return self.cache_path / f"{self.file_name_prefix}{step}.jsonl"

    def step_files(self) -> list[Path]:
        """Existing cache step files, in step order."""
        pattern = re.compile(rf"^{re.escape(self.file_name_prefix)}(\d+)\.jsonl$")
        found: list[tuple[int, Path]] = []
        for path in self.cache_path.iterdir():
            match = pattern.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [path for _, path in sorted(found)]

    def _read_frame(self) -> pd.DataFrame:
        if self.operator_step < 0:
            raise ValueError("Call storage.step() before reading data")
        if self.operator_step == 0:
            if not self.first_entry_file_name:
                return pd.DataFrame()
            path = Path(self.first_entry_file_name)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Entry file not found: {path}. "
                    "Set first_entry_file_name or start the pipeline with LoadJSONL."
                )
            return read_jsonl(path)
        path = self.cache_file(self.operator_step)
        if not path.exists():
            raise FileNotFoundError(
                f"Step file not found: {path}. The previous operator did not write it."
            )
        return read_jsonl(path)


def schema_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".schema.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(obj)
    return records


def jsonl_columns(path: Path) -> list[str]:
    """Union of object keys in first-seen order. Reads the file, not an LLM."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL source not found: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    found = False
    for record in iter_jsonl(path):
        found = True
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    if not found:
        raise ValueError(f"{path} has no records")
    return columns


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = iter_jsonl(path)
    if not records:
        schema = schema_path(path)
        if schema.is_file():
            columns = json.loads(schema.read_text(encoding="utf-8"))
            return pd.DataFrame(columns=list(columns))
        return pd.DataFrame()
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return pd.DataFrame.from_records(records, columns=columns)


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = dataframe_to_records(frame)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    schema_path(path).write_text(
        json.dumps(list(frame.columns)),
        encoding="utf-8",
    )


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _cell(value) for key, value in row.items()})
    return records


def _as_frame(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if isinstance(data[0], dict):
            return pd.DataFrame.from_records(data)
    raise TypeError("FileStorage.write expects a DataFrame or a list of objects")


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, dict, list)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    try:
        if value is pd.NA or value is pd.NaT or bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value
