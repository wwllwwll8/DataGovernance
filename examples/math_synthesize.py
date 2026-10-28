"""Generate, judge, then verify a tiny math set with the configured model.

Five math seeds go through LoadJSONL, text generation, an UltraFeedback
judge on correctness and reasoning, hash dedupe, and a numeric check
against ``answer``. The judge sees the gold answer. Rows whose generation
does not match are dropped. Finished model steps are reused from the
cache. Delete that cache to start over.

    python3 examples/math_synthesize.py

Results land in ``examples/output/math_synthesize/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataGovernance import (  # noqa: E402
    ChatCompletionsClient,
    FileStorage,
    HashDeduplicateFilter,
    LLMConfig,
    LoadJSONL,
    MathVerifyFilter,
    Pipeline,
    TextGenerationOp,
    math_feedback,
)

SEEDS = Path(__file__).resolve().parent / "data" / "math_seeds.jsonl"
OUTPUT = Path(__file__).resolve().parent / "output" / "math_synthesize"

class MathSynthesize(Pipeline):
    """Load seeds, draft, judge, dedupe, then check the math."""

    def __init__(self, cache_path: Path, llm: ChatCompletionsClient) -> None:
        super().__init__()
        self.llm = llm
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL(SEEDS)
        self.gen = TextGenerationOp(llm)
        self.judge = math_feedback(llm)
        self.dedup = HashDeduplicateFilter()
        self.math = MathVerifyFilter()

    def forward(self) -> None:
        self.load.run(self.storage.step())
        self.gen.run(
            self.storage.step(),
            input_key="instruction",
            output_key="generation",
        )
        self.judge.run(
            self.storage.step(),
            input_key="instruction",
            input_generation_key="generation",
            input_context_keys=["answer"],
            output_ratings="ratings",
            output_rationales="rationales",
            output_overall="overall",
        )
        self.dedup.run(
            self.storage.step(),
            input_keys=["instruction", "generation"],
        )
        self.math.run(
            self.storage.step(),
            input_key="generation",
            input_gold_key="answer",
            output_key="math_correct",
        )


def main() -> None:
    config = LLMConfig.load()
    if not config.api_key:
        raise SystemExit("Set api_key in DataGovernance/llm/config.toml")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipe = MathSynthesize(OUTPUT / "cache", ChatCompletionsClient(config))
    print(pipe.draw(), end="")
    frame = pipe.run()
    if pipe.reused:
        print("reused: " + ", ".join(pipe.reused))
    counts = [_line_count(path) for path in pipe.storage.step_files()]
    print("step rows: " + ", ".join(str(count) for count in counts))
    print(f"rows out: {len(frame)}")
    columns = ["instruction", "generation", "answer", "math_correct", "overall"]
    print(frame[columns].to_string(index=False))
    exported = pipe.export_jsonl(OUTPUT / "result.jsonl")
    print(f"wrote {exported}")
    if frame.empty:
        raise SystemExit("the model produced no surviving rows")
    if not all(frame["math_correct"]):
        raise SystemExit("math_correct should be true on every surviving row")
    if frame["overall"].isna().any():
        raise SystemExit("a surviving row has no score")


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
