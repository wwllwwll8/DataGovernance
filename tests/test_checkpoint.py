"""Token-using steps resume from the cache instead of calling the model again."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DataGovernance import FileStorage, LoadJSONL, MockLLM, Pipeline, Prompt, TextGenerationOp


def _copy(chat):
    text = "\n".join(message["content"] for message in chat)
    return json.dumps({"rank": 1}) if "Rank" in text else "ok"


def _pipe(tmp_path, llm, seeds, prompt="Item: {instruction}"):
    class Draft(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = FileStorage(cache_path=tmp_path / "cache")
            self.load = LoadJSONL(seeds)
            self.gen = TextGenerationOp(llm, prompt=Prompt(prompt))
            self.rank = TextGenerationOp(llm, prompt=Prompt("Rank {generation}"), output_key="rank")

        def forward(self) -> None:
            self.load.run(self.storage.step())
            self.gen.run(self.storage.step())
            self.rank.run(self.storage.step())

    return Draft()


def test_finished_token_steps_are_reused(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}, {"instruction": "beta"}])
    llm = MockLLM(responder=lambda chat: _copy(chat))
    pipe = _pipe(tmp_path, llm, seeds)
    pipe.compile()
    assert list((tmp_path / "cache").glob("*.checkpoint.json")) == []
    frame = pipe.run()
    assert frame["generation"].tolist() == ["ok", "ok"]
    spent = len(llm.calls)
    assert spent == 4
    again = pipe.run()
    assert again["generation"].tolist() == ["ok", "ok"]
    assert len(llm.calls) == spent
    assert pipe.reused == ["gen:TextGenerationOp", "rank:TextGenerationOp"]


def test_partial_batch_resumes_after_the_failed_row(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}, {"instruction": "beta"}])

    class Flaky:
        model = "mock"

        def __init__(self) -> None:
            self.calls: list[list] = []
            self.fail_at: int | None = 2

        def generate(self, chats):
            self.calls.append(list(chats))
            if self.fail_at is not None and len(self.calls) == self.fail_at:
                raise RuntimeError("stopped")
            return ["kept"]

    llm = Flaky()

    class Draft(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = FileStorage(cache_path=tmp_path / "cache")
            self.load = LoadJSONL(seeds)
            self.gen = TextGenerationOp(llm, prompt=Prompt("Item: {instruction}"))

        def forward(self) -> None:
            self.load.run(self.storage.step())
            self.gen.run(self.storage.step())

    pipe = Draft()
    pipe.compile()
    with pytest.raises(RuntimeError, match="stopped"):
        pipe.run()
    assert len(llm.calls) == 2
    assert list((tmp_path / "cache").glob("*.progress.jsonl"))
    llm.fail_at = None
    frame = pipe.run()
    assert frame["generation"].tolist() == ["kept", "kept"]
    assert len(llm.calls) == 3
    assert pipe.reused == []
    assert list((tmp_path / "cache").glob("*.progress.jsonl")) == []


def test_changed_prompt_does_not_reuse(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}])

    def respond(chat):
        text = "\n".join(message["content"] for message in chat)
        if "Rank" in text:
            return json.dumps({"rank": 1})
        if text.startswith("Rewrite:"):
            return "two"
        return "one"

    llm = MockLLM(responder=respond)
    first = _pipe(tmp_path, llm, seeds, prompt="Item: {instruction}")
    first.compile()
    assert first.run().iloc[0]["generation"] == "one"
    second = _pipe(Path(tmp_path), llm, seeds, prompt="Rewrite: {instruction}")
    second.compile()
    frame = second.run()
    assert frame.iloc[0]["generation"] == "two"
    assert second.reused == []


def test_later_token_step_can_fail_without_redoing_the_earlier_one(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}])
    state = {"fail_rank": True}

    def respond(chat):
        text = "\n".join(message["content"] for message in chat)
        if "Rank" in text:
            if state["fail_rank"]:
                raise RuntimeError("rank down")
            return json.dumps({"rank": 1})
        return "ok"

    llm = MockLLM(responder=respond)
    pipe = _pipe(tmp_path, llm, seeds)
    pipe.compile()
    with pytest.raises(RuntimeError, match="rank down"):
        pipe.run()
    spent = len(llm.calls)
    state["fail_rank"] = False
    frame = pipe.run()
    assert frame.iloc[0]["generation"] == "ok"
    assert json.loads(frame.iloc[0]["rank"])["rank"] == 1
    assert len(llm.calls) == spent + 1
    assert pipe.reused == ["gen:TextGenerationOp"]
