"""Load a JSONL file as the first dataset step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC
from DataGovernance.core.storage import jsonl_columns, read_jsonl


class LoadJSONL(OperatorABC):
    """Source operator. Replaces the working dataset with a JSONL file.

    Column names are read from the file during compile (``declared_outputs``).
    That read is local I/O, not an LLM call. ``run`` takes only ``storage``;
    do not pass ``input_key`` / ``output_key``.
    """

    replaces_dataset = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def declared_outputs(self) -> list[str]:
        return jsonl_columns(self.path)

    def run(self, storage: Any) -> None:
        frame: pd.DataFrame = read_jsonl(self.path)
        frame = frame[self.declared_outputs()]
        storage.write(frame)
