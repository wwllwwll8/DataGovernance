"""Built-in judge parser and task wrapping."""

from __future__ import annotations

import json

from DataGovernance import (
    FileStorage,
    LoadJSONL,
    MockLLM,
    Pipeline,
    TaskOp,
    TextGenerationOp,
    parse_ultrafeedback,
)


def test_task_module_exports_base_operator():
    import DataGovernance.ops.tasks as tasks

    assert tasks.TaskOp is TaskOp
    assert issubclass(TextGenerationOp, TaskOp)


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

    per_generation = parse_ultrafeedback(
        json.dumps({"ratings": [5, 1], "rationales": ["good", "wrong city"]})
    )
    assert per_generation.ratings == {}
    assert per_generation.overall == 3
    assert per_generation.rationales["generation_0"] == "good"
    assert per_generation.rationales["generation_1"] == "wrong city"


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


def test_duck_typed_task_formats_output(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "Hello"}])
    llm = MockLLM(responses=["world"])
    frame = TaskPipe(tmp_path / "cache", seeds, llm).run()
    assert frame.iloc[0]["generation"] == "WORLD"
    assert llm.calls[0][0][0]["content"] == "Hello"
