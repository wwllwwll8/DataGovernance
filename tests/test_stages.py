"""A prompt belongs to the operator. Compile checks its columns first."""

from __future__ import annotations

import pytest

from DataGovernance import (
    FileStorage,
    HashDeduplicateFilter,
    LoadJSONL,
    MockLLM,
    Pipeline,
    Prompt,
    TextGenerationOp,
)


def _respond(chat):
    text = "\n".join(message["content"] for message in chat)
    return text.split("Item: ", 1)[1].strip()


def test_prompt_lists_columns_in_order():
    prompt = Prompt("Answer {instruction} about {topic}", system="Stay on {topic}")
    assert prompt.input_columns() == ["topic", "instruction"]
    rendered = prompt.render({"instruction": "add", "topic": "math"})
    assert rendered[0]["content"] == "Stay on math"
    assert rendered[1]["content"] == "Answer add about math"


def test_prompt_on_the_operator_dedupes(tmp_path, write_jsonl):
    seeds = write_jsonl(
        tmp_path / "seeds.jsonl",
        [
            {"instruction": "alpha"},
            {"instruction": "alpha"},
            {"instruction": "beta"},
        ],
    )
    llm = MockLLM(responder=_respond)

    class Draft(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = FileStorage(cache_path=tmp_path / "cache")
            self.load = LoadJSONL(seeds)
            self.gen = TextGenerationOp(
                llm,
                prompt=Prompt("Item: {instruction}", system="Copy the item."),
            )
            self.dedup = HashDeduplicateFilter()

        def forward(self) -> None:
            self.load.run(self.storage.step())
            self.gen.run(self.storage.step())
            self.dedup.run(self.storage.step(), input_key="instruction")

    pipe = Draft()
    pipe.compile()
    assert llm.calls == []
    frame = pipe.run()
    assert frame["instruction"].tolist() == ["alpha", "beta"]
    assert frame["generation"].tolist() == ["alpha", "beta"]
    assert len(llm.calls) == 3
    assert all(len(batch) == 1 for batch in llm.calls)


def test_missing_prompt_column_fails_before_the_model(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}])
    llm = MockLLM(responder=_respond)

    class Draft(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = FileStorage(cache_path=tmp_path / "cache")
            self.load = LoadJSONL(seeds)
            self.gen = TextGenerationOp(llm, prompt=Prompt("Use {missing}"))

        def forward(self) -> None:
            self.load.run(self.storage.step())
            self.gen.run(self.storage.step())

    with pytest.raises(KeyError, match="missing"):
        Draft().compile()
    assert llm.calls == []
