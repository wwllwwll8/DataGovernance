"""Load, generate, rank, score, then dedupe.

The pipeline reads ``instruction``, writes ``generation`` from a prompt,
then asks the model in ``config.toml`` to rank and score that answer.
Duplicate instructions are dropped after scoring. ``compile()`` checks the
prompt columns before any model call. A finished generate, rank, or score
step is reused from the cache. Delete that cache to start over.

    python3 examples/generate_rank_score.py

Results land in ``examples/output/generate_rank_score/``.
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
    LLMConfig,
    Pipeline,
    Prompt,
)

SEEDS = Path(__file__).resolve().parent / "data" / "generate_rank_score.jsonl"
OUTPUT = Path(__file__).resolve().parent / "output" / "generate_rank_score"

GENERATE = Prompt(
    "Answer in one or two words.\nQuestion: {instruction}",
    system="Answer the question directly.",
)
RANK = Prompt(
    'Rank this answer from 1 (best) to 3. Return JSON only: {{"rank": 1}}\n'
    "Question: {instruction}\nAnswer: {generation}"
)
SCORE = Prompt(
    'Score this answer from 1 to 5. Return JSON only: {{"score": 5}}\n'
    "Question: {instruction}\nAnswer: {generation}"
)


def main() -> None:
    config = LLMConfig.load()
    if not config.api_key:
        raise SystemExit("Set api_key in DataGovernance/llm/config.toml")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline()
    pipe.storage = FileStorage(cache_path=OUTPUT / "cache")
    (
        pipe.load(SEEDS)
        .generate(ChatCompletionsClient(config), GENERATE)
        .rank(ChatCompletionsClient(config), RANK)
        .score(ChatCompletionsClient(config), SCORE)
        .dedupe(input_key="instruction")
    )
    frame = pipe.run()
    print(pipe.draw(), end="")
    if pipe.reused:
        print("reused: " + ", ".join(pipe.reused))
    print(f"rows out: {len(frame)}")
    print(frame[["instruction", "generation", "rank", "score"]].to_string(index=False))
    exported = pipe.export_jsonl(OUTPUT / "result.jsonl")
    print(f"wrote {exported}")
    if frame.empty:
        raise SystemExit("the model produced no rows")
    if not all(str(value).strip() for value in frame["generation"]):
        raise SystemExit("a row has an empty generation")
    if frame["rank"].isna().any() or frame["score"].isna().any():
        raise SystemExit("a row is missing rank or score")


if __name__ == "__main__":
    main()
