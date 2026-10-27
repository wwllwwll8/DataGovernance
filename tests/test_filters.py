"""Hash dedupe, math checks, and the SELECT-only SQL filter."""

from __future__ import annotations

import pandas as pd
import pytest

from DataGovernance import (
    FileStorage,
    HashDeduplicateFilter,
    MathVerifyFilter,
    Pipeline,
    SQLExecFilter,
    check_answer,
)


def _run(tmp_path, rows, op, **kwargs):
    entry = tmp_path / "in.jsonl"
    entry.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(entry, orient="records", lines=True, force_ascii=False)
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    view = storage.step()
    op.run(view, **kwargs)
    return view.read_output()


def test_check_answer_exact_numeric_and_boxed():
    assert check_answer("same", "same").method == "exact"
    assert check_answer("Foo", "foo").ok
    assert check_answer("4 ", "4").method == "exact"
    numeric = check_answer("The answer is 42.", "42")
    assert numeric.ok and numeric.method == "numeric"
    assert check_answer("4.0", "4").method == "numeric"
    assert check_answer("final \\boxed{7}", "7").ok
    assert check_answer("nope", "4").ok is False


def test_hash_keeps_first_and_normalizes_whitespace(tmp_path):
    frame = _run(
        tmp_path,
        [
            {"instruction": "a", "generation": "4", "tag": "1"},
            {"instruction": "a", "generation": "  4\n", "tag": "2"},
            {"instruction": "b", "generation": "4", "tag": "3"},
        ],
        HashDeduplicateFilter(),
        input_keys=["instruction", "generation"],
    )
    assert frame["tag"].tolist() == ["1", "3"]


def test_hash_requires_a_column_at_compile(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "in.jsonl", [{"instruction": "a"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")

    class Bare(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = storage
            self.dedup = HashDeduplicateFilter()

        def forward(self) -> None:
            self.dedup.run(self.storage.step())

    with pytest.raises(TypeError, match="input_key or input_keys"):
        Bare().compile()


def test_math_filter_drops_failures_and_can_keep_them(tmp_path):
    rows = [
        {"generation": "4", "answer": "4"},
        {"generation": r"\boxed{42}", "answer": "42"},
        {"generation": "0", "answer": "10"},
    ]
    dropped = _run(
        tmp_path,
        rows,
        MathVerifyFilter(),
        input_key="generation",
        input_gold_key="answer",
        output_key="math_correct",
    )
    assert dropped["generation"].tolist() == ["4", r"\boxed{42}"]
    assert list(dropped["math_correct"]) == [True, True]

    kept = _run(
        tmp_path / "keep",
        rows,
        MathVerifyFilter(),
        input_key="generation",
        input_gold_key="answer",
        output_key="math_correct",
        drop_failures=False,
    )
    assert list(kept["math_correct"]) == [True, True, False]


def test_math_filter_can_drop_every_row(tmp_path):
    frame = _run(
        tmp_path,
        [{"generation": "1", "answer": "2"}],
        MathVerifyFilter(),
        input_key="generation",
        input_gold_key="answer",
    )
    assert frame.empty
    assert "math_correct" in frame.columns


def test_sql_select_only_and_gold(tmp_path):
    setup = "CREATE TABLE t(id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a');"
    op = SQLExecFilter(setup_sql=setup)
    frame = _run(
        tmp_path,
        [
            {"sql": "SELECT name FROM t WHERE id = 1", "gold": '[["a"]]'},
            {"sql": "DELETE FROM t", "gold": ""},
            {"sql": "SELECT 1; DROP TABLE t", "gold": ""},
            {"sql": "WITH c AS (SELECT 1 AS n) SELECT n FROM c", "gold": "[[1]]"},
        ],
        op,
        input_key="sql",
        input_gold_key="gold",
        drop_failures=False,
    )
    assert list(frame["sql_ok"]) == [True, False, False, True]
    assert frame.iloc[0]["sql_result"] == '[["a"]]'
    assert "not a single SELECT" in frame.iloc[1]["sql_error"]
    assert frame.iloc[3]["sql_result"] == "[[1]]"

    kept = _run(
        tmp_path / "drop",
        [
            {"sql": "SELECT 1", "gold": "[[1]]"},
            {"sql": "DROP TABLE t", "gold": ""},
        ],
        SQLExecFilter(),
        input_key="sql",
        input_gold_key="gold",
    )
    assert len(kept) == 1
    assert bool(kept.iloc[0]["sql_ok"])
