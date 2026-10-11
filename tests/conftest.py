"""Shared JSONL helper for pipeline tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def write_jsonl():
    def _write(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    return _write
