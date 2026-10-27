"""Synthesize NL2SQL rows, then score them.

Three database schemas (library, school, shop) go through the Text-to-SQL
synthesis shape and the Distilabel scoring shape:

1. ``SQLSynthesisOp`` asks for several SELECT statements.
2. ``SQLExecFilter`` runs each statement on that row's schema and drops
   anything that is not a single executable SELECT.
3. ``SQLQuestionTask`` (a Distilabel task on ``TextGenerationOp``) writes
   the natural-language question.
4. ``UltraFeedbackOp`` scores the question and the SQL.
5. ``HashDeduplicateFilter`` keeps the first copy of each SQL.

The run reads ``DataGovernance/llm/config.toml`` and calls that model
for SQL synthesis, question writing, and UltraFeedback scoring.

    python3 examples/nl2sql_synthesize.py
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
    ChatCompletionsClient,
    FileStorage,
    HashDeduplicateFilter,
    LLMConfig,
    LLMClient,
    LoadJSONL,
    Pipeline,
    SQLExecFilter,
    SQLQuestionTask,
    SQLSynthesisOp,
    TextGenerationOp,
    UltraFeedbackOp,
)
from DataGovernance.llm.client import _urllib_post  # noqa: E402

SEEDS = Path(__file__).resolve().parent / "data" / "nl2sql_seeds.jsonl"
OUTPUT = Path(__file__).resolve().parent / "output" / "nl2sql"

class NL2SQLSynthesize(Pipeline):
    """Schema to SQL, execution filter, question, UltraFeedback, dedupe."""

    def __init__(self, cache_path: Path, llm: LLMClient) -> None:
        super().__init__()
        self.llm = llm
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL(SEEDS)
        self.sql = SQLSynthesisOp(llm)
        self.exec = SQLExecFilter()
        self.question = TextGenerationOp(llm, task=SQLQuestionTask())
        self.judge = UltraFeedbackOp(llm)
        self.dedup = HashDeduplicateFilter()

    def forward(self) -> None:
        self.load.run(self.storage.step())
        self.sql.run(
            self.storage.step(),
            input_schema_key="schema",
            input_db_id_key="db_id",
            output_sql_key="sql",
            output_schema_key="schema",
            output_db_id_key="db_id",
            output_complexity_key="sql_complexity",
            generate_num=5,
        )
        self.exec.run(
            self.storage.step(),
            input_key="sql",
            input_setup_key="schema",
            output_key="sql_ok",
        )
        self.question.run(self.storage.step())
        self.judge.run(
            self.storage.step(),
            input_key="question",
            input_generation_key="sql",
            output_model="model_name",
        )
        self.dedup.run(self.storage.step(), input_key="sql")


def main() -> None:
    config = LLMConfig.load()
    if not config.api_key:
        raise SystemExit("Set api_key in DataGovernance/llm/config.toml")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    secret = config.api_key
    log_lines: list[str] = []

    def logging_transport(url, headers, body, timeout):
        safe_headers = {
            key: ("Bearer [REDACTED]" if key.lower() == "authorization" else value)
            for key, value in headers.items()
        }
        log_lines.append(f"\n----- request -----\nPOST {url}\ntimeout: {timeout}")
        log_lines.append(json.dumps(safe_headers, ensure_ascii=False, indent=2))
        log_lines.append(json.dumps(json.loads(body.decode("utf-8")), ensure_ascii=False, indent=2))
        status, raw = _urllib_post(url, headers, body, timeout)
        text = raw.decode("utf-8", errors="replace").replace(secret, "[REDACTED]")
        log_lines.append(f"\n----- response -----\nstatus: {status}\nbytes: {len(raw)}")
        try:
            log_lines.append(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            log_lines.append(text)
        return status, raw

    llm = ChatCompletionsClient(config, transport=logging_transport)
    pipe = NL2SQLSynthesize(OUTPUT / "cache", llm)
    print(pipe.draw(), end="")
    log_path = OUTPUT / "live.log"
    try:
        frame = pipe.run()
    finally:
        log_path.write_text(
            f"model: {config.model}\nbase_url: {config.base_url}\n" + "\n".join(log_lines) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {log_path}")
    counts = [_line_count(path) for path in pipe.storage.step_files()]
    print("step rows: " + ", ".join(str(count) for count in counts))
    print(f"rows out: {len(frame)}")
    columns = ["db_id", "sql_complexity", "question", "sql", "overall", "model_name"]
    print(frame[columns].to_string(index=False))
    exported = pipe.export_jsonl(OUTPUT / "result.jsonl")
    print(f"wrote {exported}")
    if frame.empty:
        raise SystemExit("the model produced no surviving rows")
    if not all(str(question).strip() for question in frame["question"]):
        raise SystemExit("a surviving row has an empty question")
    if not all(frame["sql_ok"]):
        raise SystemExit("a surviving row did not execute")
    if frame["overall"].isna().any():
        raise SystemExit("a surviving row has no score")


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
