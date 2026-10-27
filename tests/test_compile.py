"""compile() checks columns and does not call operator.run or the LLM."""

from __future__ import annotations

import pytest

from DataGovernance import FileStorage, MockLLM, Pipeline, TextGenerationOp
from DataGovernance.core.operator import OperatorABC


class FlagOp(OperatorABC):
    def __init__(self) -> None:
        self.ran = False

    def run(self, storage, input_key: str = "instruction", output_key: str = "generation"):
        self.ran = True
        frame = storage.read().copy()
        frame[output_key] = ["ok"] * len(frame)
        storage.write(frame)


class FlagPipe(Pipeline):
    def __init__(self, storage: FileStorage, op: FlagOp) -> None:
        super().__init__()
        self.storage = storage
        self.flag = op

    def forward(self) -> None:
        self.flag.run(self.storage.step(), input_key="instruction", output_key="generation")


class TypoPipe(Pipeline):
    def __init__(self, storage: FileStorage, llm: MockLLM) -> None:
        super().__init__()
        self.storage = storage
        self.gen = TextGenerationOp(llm)

    def forward(self) -> None:
        self.gen.run(self.storage.step(), input_key="instructon", output_key="generation")


class BadArgPipe(Pipeline):
    def __init__(self, storage: FileStorage, llm: MockLLM) -> None:
        super().__init__()
        self.storage = storage
        self.gen = TextGenerationOp(llm)

    def forward(self) -> None:
        self.gen.run(self.storage.step(), input_ky="instruction", output_key="generation")


def _entry(tmp_path, write_jsonl):
    entry = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "hi", "answer": "4"}])
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    return storage


def test_compile_typo_raises_before_llm(tmp_path, write_jsonl):
    storage = _entry(tmp_path, write_jsonl)
    llm = MockLLM(responses=["hello"])
    pipe = TypoPipe(storage, llm)
    with pytest.raises(KeyError, match="instructon") as caught:
        pipe.compile()
    message = str(caught.value)
    assert "input_key" in message
    assert "LLM" in message
    assert llm.calls == []
    assert list((tmp_path / "cache").glob("step*.jsonl")) == []
    assert pipe.compiled is False


def test_compile_success_does_not_run_operators(tmp_path, write_jsonl):
    storage = _entry(tmp_path, write_jsonl)
    flag = FlagOp()
    pipe = FlagPipe(storage, flag)
    pipe.compile()
    assert flag.ran is False
    assert list((tmp_path / "cache").glob("step*.jsonl")) == []
    frame = pipe.run()
    assert flag.ran is True
    assert frame.iloc[0]["generation"] == "ok"
    assert (tmp_path / "cache" / "step1.jsonl").is_file()


def test_unknown_run_argument_fails_at_compile(tmp_path, write_jsonl):
    storage = _entry(tmp_path, write_jsonl)
    llm = MockLLM()
    pipe = BadArgPipe(storage, llm)
    with pytest.raises(TypeError, match="input_ky"):
        pipe.compile()
    assert llm.calls == []
