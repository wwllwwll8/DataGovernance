"""Generate, judge, then verify a tiny preference-style set. Offline.

Five math seeds go through LoadJSONL, a mock text generator, an
UltraFeedback-like judge, hash dedupe, and an exact/numeric math check.
The mock judge scores every row, including the wrong ``9 + 1`` answer.
MathVerify is the objective gate and drops that row. The duplicate
``2 + 2`` row is dropped by the hash filter.

Run from the repo root:

    python3 examples/pref_then_verify.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataGovernance import (  # noqa: E402
    FileStorage,
    HashDeduplicateFilter,
    LoadJSONL,
    MathVerifyFilter,
    MockLLM,
    Pipeline,
    TextGenerationOp,
    UltraFeedbackOp,
)

SEEDS = Path(__file__).resolve().parent / "data" / "seeds.jsonl"
OUTPUT = Path(__file__).resolve().parent / "output"

ANSWERS = {
    "What is 2 + 2?": "4",
    "What is 6 * 7?": "42",
    "Compute 10 - 3.": "7",
    "What is 9 + 1?": "0",
}

JUDGE = json.dumps(
    {
        "ratings": {
            "helpfulness": 5,
            "honesty": 5,
            "instruction_following": 5,
            "truthfulness": 5,
        },
        "rationales": {
            "helpfulness": "answers the question",
            "honesty": "no extra claims",
            "instruction_following": "follows the instruction",
            "truthfulness": "states a number",
        },
        "overall": 5,
    }
)


def respond(chat: list[dict[str, str]]) -> str:
    text = "\n".join(message["content"] for message in chat)
    if "TASK: ultrafeedback" in text:
        return JUDGE
    for instruction, answer in ANSWERS.items():
        if instruction in text:
            return answer
    return "0"


class PrefThenVerify(Pipeline):
    """Load seeds, draft, judge, dedupe, then check the math."""

    def __init__(self, cache_path: Path, llm: MockLLM) -> None:
        super().__init__()
        self.llm = llm
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL(SEEDS)
        self.gen = TextGenerationOp(llm)
        self.judge = UltraFeedbackOp(llm)
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
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    llm = MockLLM(responder=respond)
    pipe = PrefThenVerify(OUTPUT / "cache", llm)
    pipe.compile()
    if llm.calls:
        raise SystemExit("compile() called the mock LLM")
    print(pipe.draw(), end="")
    frame = pipe.run()
    counts = [_line_count(path) for path in pipe.storage.step_files()]
    print("step rows: " + ", ".join(str(count) for count in counts))
    print(f"rows out: {len(frame)}")
    columns = ["instruction", "generation", "answer", "math_correct", "overall"]
    print(frame[columns].to_string(index=False))
    exported = pipe.export_jsonl(OUTPUT / "result.jsonl")
    print(f"wrote {exported}")

    expected = ["What is 2 + 2?", "What is 6 * 7?", "Compute 10 - 3."]
    if counts != [5, 5, 5, 4, 3]:
        raise SystemExit(f"unexpected step row counts: {counts}")
    if frame["instruction"].tolist() != expected:
        raise SystemExit(f"unexpected rows: {frame['instruction'].tolist()}")
    if frame["generation"].tolist() != ["4", "42", "7"]:
        raise SystemExit(f"unexpected generations: {frame['generation'].tolist()}")
    if not all(frame["math_correct"]):
        raise SystemExit("math_correct should be true on every surviving row")
    if len(llm.calls) != 2 or len(llm.calls[0]) != 5 or len(llm.calls[1]) != 5:
        raise SystemExit("expected one 5-row generation batch and one 5-row judge batch")
    dead = pipe.lineage.dead_columns() if pipe.lineage is not None else []
    for column in ("ratings", "rationales", "overall", "math_correct"):
        if column not in dead:
            raise SystemExit(f"{column} should be listed as a dead column")
    if "generation" in dead:
        raise SystemExit("generation is read by later operators")
    print("MathVerify dropped the 9 + 1 row. The mock judge had scored it anyway.")


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
