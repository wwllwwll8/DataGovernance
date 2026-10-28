"""UltraFeedback parsing and a duck-typed feedback task."""

from __future__ import annotations

import json

from DataGovernance import (
    FileStorage,
    LoadJSONL,
    MockLLM,
    Pipeline,
    TaskOp,
    TextGenerationOp,
    math_feedback,
    parse_ultrafeedback,
)
from DataGovernance.recipes import load_prompt


def test_feedback_module_exports_task_op():
    import DataGovernance.ops.feedback as feedback

    assert feedback.TaskOp is TaskOp
    assert issubclass(TextGenerationOp, TaskOp)


def test_prompt_file_columns_are_not_hardcoded_parameters():
    from DataGovernance import MockLLM
    from DataGovernance.ops.nl2sql import SQLQuestionTask, SQLSynthesisOp
    from DataGovernance.recipes import load_prompt, load_section

    synthesize = load_section("nl2sql.synthesize")
    sql = SQLSynthesisOp(
        MockLLM(),
        prompt=synthesize.prompt,
        generate_num=synthesize.values["generate_num"],
        copy=synthesize.values["copy"],
    )
    assert sql.prompt.input_columns() == ["schema"]
    assert sql.generate_num == 5
    assert sql.declared_inputs() == ["schema", "db_id"]
    assert SQLQuestionTask(load_prompt("nl2sql.question")).inputs == ["schema", "sql"]
    assert load_prompt("nl2sql.judge").input_columns() == ["question", "sql"]
    assert load_prompt("math.judge").input_columns() == ["instruction", "generation", "answer"]
    assert load_prompt("math.generate").input_columns() == ["instruction"]


def test_parse_ultrafeedback_keeps_a_task_aspect():
    parsed = parse_ultrafeedback(
        json.dumps(
            {
                "ratings": {"correctness": 4, "reasoning": "3"},
                "rationales": {"correctness": "the total matches"},
                "overall": 4,
            }
        )
    )
    assert parsed.ratings == {"correctness": 4.0, "reasoning": 3.0}
    assert parsed.rationales["correctness"] == "the total matches"
    assert parsed.overall == 4


def test_math_feedback_shows_the_gold_answer(tmp_path, write_jsonl):
    def respond(chat):
        text = "\n".join(message["content"] for message in chat)
        assert "correctness, reasoning" in text
        assert "answer:\n10" in text
        return json.dumps(
            {
                "ratings": {"correctness": 2, "reasoning": 3},
                "rationales": {"correctness": "wrong total"},
                "overall": 2,
            }
        )

    entry = write_jsonl(
        tmp_path / "seeds.jsonl",
        [{"instruction": "What is 9 + 1?", "generation": "0", "answer": "10"}],
    )
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    view = storage.step()
    math_feedback(MockLLM(responder=respond), prompt=load_prompt("math.judge")).run(view)
    frame = view.read_output()
    assert frame.iloc[0]["ratings"]["correctness"] == 2
    assert frame.iloc[0]["overall"] == 2


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
    pipe = TaskPipe(tmp_path / "cache", seeds, llm)
    pipe.compile()
    frame = pipe.run()
    assert frame.iloc[0]["generation"] == "WORLD"
    assert llm.calls[0][0][0]["content"] == "Hello"
