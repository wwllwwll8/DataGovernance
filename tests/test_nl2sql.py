"""NL2SQL synthesis: SQL list parsing, per-row schema execution, full example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from DataGovernance import SQLExecFilter
from DataGovernance.ops.nl2sql import parse_sql_list, sql_complexity

ROOT = Path(__file__).resolve().parents[1]


def test_parse_sql_list_accepts_fenced_objects():
    text = """```json
    {"sql": [{"query": "SELECT 1"}, "SELECT 2"]}
    ```"""
    assert parse_sql_list(text) == ["SELECT 1", "SELECT 2"]
    assert sql_complexity("SELECT a FROM t JOIN u ON t.id = u.id") == "medium"
    assert sql_complexity("SELECT a FROM t WHERE a > 1") == "easy"
    with pytest.raises(ValueError, match="no statements"):
        parse_sql_list('{"queries": []}')


def test_sql_filter_uses_each_rows_schema(tmp_path, write_jsonl):
    from DataGovernance import FileStorage

    seeds = write_jsonl(
        tmp_path / "in.jsonl",
        [
            {
                "sql": "SELECT name FROM t",
                "schema": "CREATE TABLE t(name TEXT); INSERT INTO t VALUES ('ada');",
            },
            {
                "sql": "SELECT name FROM t",
                "schema": "CREATE TABLE other(name TEXT);",
            },
        ],
    )
    storage = FileStorage(first_entry_file_name=str(seeds), cache_path=tmp_path / "cache")
    SQLExecFilter().run(
        storage.step(),
        input_key="sql",
        input_setup_key="schema",
        output_key="sql_ok",
    )
    frame = storage.read_output()
    assert len(frame) == 1
    assert frame.iloc[0]["sql_result"] == '[["ada"]]'
    assert json.loads(frame.iloc[0]["sql_result"]) == [["ada"]]


def test_nl2sql_example_script():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "nl2sql_synthesize.py"), "--mock"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rows out: 6" in result.stdout
    assert "step rows: 3, 15, 9, 9, 9, 6" in result.stdout
    assert "dead columns:" in result.stdout
