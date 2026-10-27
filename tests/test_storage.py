"""Step-file storage: entry file is step 0, writes land on the next step."""

from __future__ import annotations

from DataGovernance import FileStorage


def test_step_chain_reads_entry_then_cache(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "hi", "answer": "4"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    assert storage.operator_step == -1

    first = storage.step()
    assert first.operator_step == 0
    frame = first.read()
    assert frame.iloc[0]["answer"] == "4"
    assert isinstance(frame.iloc[0]["answer"], str)
    frame = frame.copy()
    frame["generation"] = ["4"]
    written = first.write(frame)
    assert written.name == "step1.jsonl"

    second = storage.step()
    assert second.operator_step == 1
    loaded = second.read()
    assert loaded.iloc[0]["generation"] == "4"
    assert list(loaded.columns) == ["instruction", "answer", "generation"]
    assert storage.step_files() == [written]


def test_empty_write_keeps_columns(tmp_path):
    storage = FileStorage(cache_path=tmp_path / "cache")
    view = storage.step()
    import pandas as pd

    view.write(pd.DataFrame(columns=["instruction", "math_correct"]))
    loaded = view.read_output()
    assert loaded.empty
    assert list(loaded.columns) == ["instruction", "math_correct"]


def test_read_before_step_raises(tmp_path):
    storage = FileStorage(cache_path=tmp_path / "cache")
    try:
        storage.read()
    except ValueError as exc:
        assert "step()" in str(exc)
    else:
        raise AssertionError("read() before step() should fail")
