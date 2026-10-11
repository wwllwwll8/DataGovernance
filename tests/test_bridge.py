"""Built-in judge parser, optional distilabel import, and task wrapping."""

from __future__ import annotations

import json
import sys

from df_distilabel import (
    FileStorage,
    LoadJSONL,
    MockLLM,
    Pipeline,
    TextGenerationOp,
    parse_ultrafeedback,
)


def test_bridge_does_not_import_distilabel():
    assert "distilabel" not in sys.modules
    import df_distilabel.ops.distilabel_bridge as bridge

    assert bridge.distilabel_available() is False
    assert "distilabel" not in sys.modules


def test_parse_ultrafeedback_shapes():
    fenced = """```json
    {"ratings": {"helpfulness": 4, "honesty": "5", "instruction-following": {"rating": 3, "rationale": "partial"}, "truthfulness": 4},
     "rationales": {"helpfulness": "useful"}}
    ```"""
    parsed = parse_ultrafeedback(fenced)
    assert parsed.ratings["helpfulness"] == 4
    assert parsed.ratings["honesty"] == 5
    assert parsed.ratings["instruction_following"] == 3
    assert parsed.rationales["instruction_following"] == "partial"
    assert parsed.rationales["helpfulness"] == "useful"
    assert parsed.overall == 4

    listed = parse_ultrafeedback(
        json.dumps(
            {
                "ratings": [
                    {"aspect": "helpfulness", "rating": 2, "rationale": "thin"},
                    {"aspect": "overall-rating", "rating": 2},
                ]
            }
        )
    )
    assert listed.ratings == {"helpfulness": 2}
    assert listed.overall == 2
    assert listed.rationales["helpfulness"] == "thin"

    distilabel_like = parse_ultrafeedback(
        json.dumps({"ratings": [5, 1], "rationales": ["good", "wrong city"]})
    )
    assert distilabel_like.ratings == {}
    assert distilabel_like.overall == 3
    assert distilabel_like.rationales["generation_0"] == "good"
    assert distilabel_like.rationales["generation_1"] == "wrong city"


class _FakeTask:
    inputs = ["instruction"]
    outputs = ["generation"]

    def format_input(self, record):
        return [{"role": "user", "content": record["instruction"]}]

    def format_output(self, output, record):
        return {"generation": output.upper()}


class TaskPipe(Pipeline):
    def __init__(self, cache, seeds, llm):
        super().__init__()
        self.storage = FileStorage(cache_path=cache)
        self.load = LoadJSONL(seeds)
        self.gen = TextGenerationOp(llm, task=_FakeTask())

    def forward(self):
        self.load.run(self.storage.step())
        self.gen.run(self.storage.step())


def test_duck_typed_task_formats_without_distilabel(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "Hello"}])
    llm = MockLLM(responses=["world"])
    frame = TaskPipe(tmp_path / "cache", seeds, llm).run()
    assert frame.iloc[0]["generation"] == "WORLD"
    assert llm.calls[0][0][0]["content"] == "Hello"
