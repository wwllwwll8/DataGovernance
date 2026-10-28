"""Producer → consumer text dump, including dead columns."""

from __future__ import annotations

import pytest

from DataGovernance import FileStorage, HashDeduplicateFilter, MockLLM, Pipeline, TextGenerationOp
from DataGovernance.core.operator import OperatorABC


class NoteOp(OperatorABC):
    def run(self, storage, input_key: str = "instruction", output_key: str = "scratch_note"):
        frame = storage.read().copy()
        frame[output_key] = ["note"] * len(frame)
        storage.write(frame)


class TinyPipe(Pipeline):
    def __init__(self, storage: FileStorage, llm: MockLLM) -> None:
        super().__init__()
        self.storage = storage
        self.gen = TextGenerationOp(llm)
        self.dedup = HashDeduplicateFilter()

    def forward(self) -> None:
        self.gen.run(self.storage.step(), input_key="instruction", output_key="generation")
        self.dedup.run(self.storage.step(), input_key="generation")


class NotePipe(Pipeline):
    def __init__(self, storage: FileStorage, llm: MockLLM) -> None:
        super().__init__()
        self.storage = storage
        self.gen = TextGenerationOp(llm)
        self.note = NoteOp()
        self.dedup = HashDeduplicateFilter()

    def forward(self) -> None:
        self.gen.run(self.storage.step(), input_key="instruction", output_key="generation")
        self.note.run(self.storage.step(), input_key="instruction", output_key="scratch_note")
        self.dedup.run(self.storage.step(), input_key="generation")


def test_draw_lists_producer_consumer_and_dead_seed(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "hi", "answer": "4"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    pipe = TinyPipe(storage, MockLLM(responses=["4"]))
    with pytest.raises(RuntimeError, match="compile\\(\\) before draw"):
        pipe.draw()
    pipe.compile()
    text = pipe.draw()
    assert text == (
        "pipeline: gen:TextGenerationOp -> dedup:HashDeduplicateFilter\n"
        "column lineage\n"
        "instruction: DATASET-INPUT -> gen:TextGenerationOp\n"
        "answer: DATASET-INPUT -> (none)\n"
        "generation: gen:TextGenerationOp -> dedup:HashDeduplicateFilter\n"
        "dead columns: answer\n"
    )
    assert pipe.draw() == text
    assert pipe.lineage is not None
    assert pipe.lineage.dead_columns() == ["answer"]
    assert pipe.lineage.producers["generation"].readers[0].owner.endswith("HashDeduplicateFilter")


def test_lineage_lists_unread_middle_column(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "hi", "answer": "4"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    pipe = NotePipe(storage, MockLLM(responses=["4"]))
    pipe.compile()
    assert pipe.lineage is not None
    dead = pipe.lineage.dead_columns()
    assert "scratch_note" in dead
    assert "answer" in dead
    assert "generation" not in dead
    assert "scratch_note: note:NoteOp -> (none)" in pipe.draw()
