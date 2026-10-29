"""One offline pass over the features that are not the NL2SQL story.

It selects a registered prompt version, writes two generations, scores
that list in one judge call, runs the seed SQL on a sqlite file, then
writes the JSONL, the Argilla records, and a local stand-in for the Hub
upload. No API key.

    python3 examples/feature_tour.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataGovernance import (  # noqa: E402
    DatabaseManager,
    FileStorage,
    HashDeduplicateFilter,
    LoadJSONL,
    MockLLM,
    Pipeline,
    SQLExecFilter,
    TextGenerationOp,
    UltraFeedbackOp,
)
from DataGovernance.core.prompt import Prompt  # noqa: E402
from DataGovernance.recipes import load_prompt, register_prompt, use_prompt  # noqa: E402

SEEDS = Path(__file__).resolve().parent / "data" / "feature_tour.jsonl"
OUTPUT = Path(__file__).resolve().parent / "output" / "feature_tour"


def _llm() -> MockLLM:
    drafts = iter(["4", "four"])

    def respond(chat):
        text = "\n".join(message["content"] for message in chat)
        if "Response 1" in text and "Response 2" in text:
            return json.dumps(
                {"ratings": [5, 2], "rationales": ["the number", "the word"], "overall": 3}
            )
        return next(drafts)

    return MockLLM(responder=respond)


def _library(path: Path) -> DatabaseManager:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT);"
        "INSERT INTO books (id, title) VALUES (1, 'Dune');"
    )
    connection.close()
    manager = DatabaseManager()
    manager.register("library", path=path)
    return manager


class FeatureTour(Pipeline):
    """Draft twice, score the list, then execute the seed SQL."""

    def __init__(self, cache_path: Path, llm: MockLLM, database: DatabaseManager) -> None:
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL(SEEDS)
        self.gen = TextGenerationOp(
            llm,
            prompt=load_prompt("tour.generate"),
            num_generations=2,
        )
        self.judge = UltraFeedbackOp(llm)
        self.exec = SQLExecFilter(database=database)
        self.dedup = HashDeduplicateFilter()

    def forward(self) -> None:
        self.load.run(self.storage.step())
        self.gen.run(self.storage.step())
        self.judge.run(
            self.storage.step(),
            input_generation_key=None,
            input_generations_key="generation",
            output_model="model_name",
        )
        self.exec.run(self.storage.step(), input_db_id_key="db_id", input_setup_key=None)
        self.dedup.run(self.storage.step(), input_key="sql")


def main() -> None:
    register_prompt(
        "tour.generate",
        Prompt("{instruction}", system="Answer with one token."),
        version="plain",
    )
    register_prompt(
        "tour.generate",
        Prompt("Put only the number on the last line.\n{instruction}", system="Answer with one token."),
        version="strict",
    )
    use_prompt("tour.generate", "strict")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipe = FeatureTour(OUTPUT / "cache", _llm(), _library(OUTPUT / "library.sqlite"))
    pipe.compile()
    print(pipe.draw(), end="")
    frame = pipe.run()
    print(frame[["instruction", "generation", "ratings", "overall", "sql_result"]].to_string(index=False))
    print(f"wrote {pipe.export_jsonl(OUTPUT / 'result.jsonl')}")
    print(f"wrote {pipe.export_argilla(OUTPUT / 'review.jsonl')}")

    def save_locally(repo_id, filename, content, token):
        target = OUTPUT / "hub" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    print(f"wrote {pipe.push_to_hub('local/feature-tour', uploader=save_locally)}")
    row = frame.iloc[0]
    if row["generation"] != ["4", "four"]:
        raise SystemExit("expected two generations")
    if row["ratings"] != [5.0, 2.0]:
        raise SystemExit("expected one score per generation")
    if json.loads(row["sql_result"]) != [["Dune"]]:
        raise SystemExit("expected the sqlite file to answer the seed SQL")


if __name__ == "__main__":
    main()
