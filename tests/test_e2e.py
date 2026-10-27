"""Live path: config.toml -> ChatCompletionsClient -> operator -> step file.

Skipped when api_key is empty, so the rest of the suite stays offline.
"""

from __future__ import annotations

import pytest

from DataGovernance import (
    ChatCompletionsClient,
    FileStorage,
    LLMConfig,
    LoadJSONL,
    Pipeline,
    TextGenerationOp,
)


def test_config_file_generates_and_writes_a_step(tmp_path, write_jsonl):
    config = LLMConfig.load()
    if not config.api_key:
        pytest.skip("Set api_key in DataGovernance/llm/config.toml to run the live end-to-end test.")

    seeds = write_jsonl(
        tmp_path / "seeds.jsonl",
        [{"instruction": "Reply with exactly the single character 4."}],
    )
    llm = ChatCompletionsClient(config)

    class GenerateOne(Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.storage = FileStorage(cache_path=tmp_path / "cache")
            self.load = LoadJSONL(seeds)
            self.gen = TextGenerationOp(llm, system_prompt="Reply with exactly the characters you are asked for.")

        def forward(self) -> None:
            self.load.run(self.storage.step())
            self.gen.run(self.storage.step(), input_key="instruction", output_key="generation")

    pipe = GenerateOne()
    pipe.compile()
    frame = pipe.run()
    text = str(frame.iloc[0]["generation"]).strip()
    assert "4" in text
    written = (tmp_path / "cache" / "step2.jsonl").read_text(encoding="utf-8")
    assert "4" in written
