"""Mock preference pipeline: generate, judge, dedupe, math-check, export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DataGovernance import (
    FileStorage,
    HashDeduplicateFilter,
    LoadJSONL,
    MathVerifyFilter,
    MockLLM,
    Pipeline,
    TextGenerationOp,
    UltraFeedbackOp,
    get_operator,
)


def _judge() -> str:
    return json.dumps(
        {
            "ratings": {
                "helpfulness": 5,
                "honesty": 5,
                "instruction_following": 4,
                "truthfulness": 5,
            },
            "rationales": {"helpfulness": "direct"},
            "overall": 4.5,
        }
    )


def _respond(chat):
    text = "\n".join(message["content"] for message in chat)
    if "TASK: ultrafeedback" in text:
        return _judge()
    if "9 + 1" in text:
        return "0"
    if "6 * 7" in text:
        return "42"
    if "10 - 3" in text:
        return "7"
    if "2 + 2" in text:
        return "4"
    return "0"


class PrefPipe(Pipeline):
    def __init__(self, cache: Path, seeds: Path, llm: MockLLM) -> None:
        super().__init__()
        self.storage = FileStorage(cache_path=cache)
        self.load = LoadJSONL(seeds)
        self.gen = TextGenerationOp(llm)
        self.judge = UltraFeedbackOp(llm)
        self.dedup = HashDeduplicateFilter()
        self.math = MathVerifyFilter()
        self.llm = llm

    def forward(self) -> None:
        self.load.run(self.storage.step())
        self.gen.run(self.storage.step(), input_key="instruction", output_key="generation")
        self.judge.run(
            self.storage.step(),
            input_key="instruction",
            input_generation_key="generation",
        )
        self.dedup.run(self.storage.step(), input_keys=["instruction", "generation"])
        self.math.run(
            self.storage.step(),
            input_key="generation",
            input_gold_key="answer",
            output_key="math_correct",
        )


def test_mock_pref_pipeline_drops_wrong_and_duplicate_rows(tmp_path, write_jsonl):
    seeds = write_jsonl(
        tmp_path / "seeds.jsonl",
        [
            {"instruction": "What is 2 + 2?", "answer": "4"},
            {"instruction": "What is 2 + 2?", "answer": "4"},
            {"instruction": "What is 6 * 7?", "answer": "42"},
            {"instruction": "Compute 10 - 3.", "answer": "7"},
            {"instruction": "What is 9 + 1?", "answer": "10"},
        ],
    )
    llm = MockLLM(responder=_respond)
    pipe = PrefPipe(tmp_path / "cache", seeds, llm)
    pipe.compile()
    assert llm.calls == []
    frame = pipe.run()
    assert frame["instruction"].tolist() == [
        "What is 2 + 2?",
        "What is 6 * 7?",
        "Compute 10 - 3.",
    ]
    assert frame["generation"].tolist() == ["4", "42", "7"]
    assert list(frame["math_correct"]) == [True, True, True]
    assert frame.iloc[0]["ratings"]["helpfulness"] == 5
    assert frame.iloc[0]["overall"] == pytest.approx(4.5)
    assert sum(len(batch) for batch in llm.calls) == 10
    assert all(len(batch) == 1 for batch in llm.calls)
    counts = [
        sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        for path in pipe.storage.step_files()
    ]
    assert counts == [5, 5, 5, 4, 3]

    exported = pipe.export_jsonl(tmp_path / "result.jsonl")
    manifest = json.loads(exported.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "DataGovernance-export"
    assert manifest["rows"] == 3
    assert "LoadJSONL" in manifest["operators"][0]
    with pytest.raises(NotImplementedError, match="not implemented"):
        pipe.export_argilla()


def test_entry_file_without_load_operator(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "in.jsonl", [{"instruction": "hi", "answer": "ok"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    llm = MockLLM(responses=["yo"])

    class OneStep(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = storage
            self.gen = TextGenerationOp(llm)

        def forward(self) -> None:
            self.gen.run(self.storage.step(), input_key="instruction", output_key="generation")

    pipe = OneStep()
    pipe.compile()
    frame = pipe.run()
    assert frame.iloc[0]["generation"] == "yo"
    assert (tmp_path / "cache" / "step1.jsonl").is_file()
    assert entry.read_text(encoding="utf-8").count("hi") == 1


def test_get_operator_builds_registered_class():
    op = get_operator("HashDeduplicateFilter")
    assert isinstance(op, HashDeduplicateFilter)
