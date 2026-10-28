"""General pipeline stages: load, generate, rank, score, and dedupe."""

from __future__ import annotations

import json

import pytest

from DataGovernance import FileStorage, MockLLM, Pipeline, Prompt


def _respond(chat):
    text = "\n".join(message["content"] for message in chat)
    if "Rank" in text:
        return json.dumps({"rank": 2 if "beta" in text else 1})
    if "Score" in text:
        return json.dumps({"score": 1 if "beta" in text else 5})
    return text.split("Item: ", 1)[1]


def test_prompt_lists_columns_in_order():
    prompt = Prompt("Answer {instruction} about {topic}", system="Stay on {topic}")
    assert prompt.input_columns() == ["topic", "instruction"]
    rendered = prompt.render({"instruction": "add", "topic": "math"})
    assert rendered[0]["content"] == "Stay on math"
    assert rendered[1]["content"] == "Answer add about math"


def test_pipeline_generate_rank_score_and_dedupe(tmp_path, write_jsonl):
    seeds = write_jsonl(
        tmp_path / "seeds.jsonl",
        [
            {"instruction": "alpha"},
            {"instruction": "alpha"},
            {"instruction": "beta"},
        ],
    )
    llm = MockLLM(responder=_respond)
    pipe = Pipeline()
    pipe.storage = FileStorage(cache_path=tmp_path / "cache")
    pipe.load(seeds).generate(
        llm,
        Prompt("Item: {instruction}", system="Copy the item."),
    ).rank(
        llm,
        Prompt("Rank {generation}. Item was {instruction}."),
    ).score(
        llm,
        Prompt("Score {generation}."),
    ).dedupe(input_key="instruction")

    pipe.compile()
    assert llm.calls == []
    frame = pipe.run()
    assert frame["instruction"].tolist() == ["alpha", "beta"]
    assert frame["generation"].tolist() == ["alpha", "beta"]
    assert frame["rank"].tolist() == [1, 2]
    assert frame["score"].tolist() == [5, 1]
    assert len(llm.calls) == 9
    assert all(len(batch) == 1 for batch in llm.calls)


def test_missing_prompt_column_fails_before_the_model(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "alpha"}])
    llm = MockLLM(responder=_respond)
    pipe = Pipeline()
    pipe.storage = FileStorage(cache_path=tmp_path / "cache")
    pipe.load(seeds).generate(llm, Prompt("Use {missing}"))
    with pytest.raises(KeyError, match="missing"):
        pipe.compile()
    assert llm.calls == []
