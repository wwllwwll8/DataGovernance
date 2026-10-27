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

The default run reads ``DataGovernance/llm/config.toml`` and calls that
model for SQL synthesis, question writing, and UltraFeedback scoring.
``--mock`` replays a fixed script instead: five statements per schema, of
which execution and dedup leave two scored rows each.

Run from the repo root:

    python3 examples/nl2sql_synthesize.py
    python3 examples/nl2sql_synthesize.py --mock
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
    MockLLM,
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

LIBRARY_SQL = "SELECT title FROM books WHERE year >= 1984"
LIBRARY_JOIN = (
    "SELECT books.title, loans.borrower FROM books "
    "JOIN loans ON books.id = loans.book_id"
)
SCHOOL_SQL = "SELECT name FROM students WHERE grade >= 10"
SCHOOL_JOIN = (
    "SELECT students.name, enrollments.course FROM students "
    "JOIN enrollments ON students.id = enrollments.student_id"
)
SHOP_SQL = "SELECT name FROM products WHERE price < 20"
SHOP_JOIN = (
    "SELECT products.name, orders.buyer FROM products "
    "JOIN orders ON products.id = orders.product_id"
)


def _candidates(keep: str, joined: str, table: str) -> list[str]:
    return [
        keep,
        keep,
        joined,
        f"SELECT missing_column FROM {table}",
        f"INSERT INTO {table} (id) VALUES (9)",
    ]


LIBRARY_SQLS = _candidates(LIBRARY_SQL, LIBRARY_JOIN, "books")
SCHOOL_SQLS = _candidates(SCHOOL_SQL, SCHOOL_JOIN, "students")
SHOP_SQLS = _candidates(SHOP_SQL, SHOP_JOIN, "products")

JUDGE = json.dumps(
    {
        "ratings": {
            "helpfulness": 5,
            "honesty": 5,
            "instruction_following": 4,
            "truthfulness": 5,
        },
        "rationales": {"helpfulness": "the SQL matches the question"},
        "overall": 4.75,
    }
)


def respond(chat: list[dict[str, str]]) -> str:
    text = "\n".join(message["content"] for message in chat)
    if "TASK: ultrafeedback" in text:
        return JUDGE
    if "natural-language question" in text:
        if "JOIN loans" in text:
            return "Which books are on loan, and who borrowed them?"
        if "JOIN enrollments" in text:
            return "Which course is each student enrolled in?"
        if "JOIN orders" in text:
            return "Who bought each product?"
        if "grade >= 10" in text:
            return "Which students are in grade 10 or above?"
        if "price < 20" in text:
            return "Which products cost less than 20?"
        return "Which books were published in 1984 or later?"
    if "SQLite SELECT" in text:
        if "CREATE TABLE students" in text:
            return json.dumps({"sql": SCHOOL_SQLS})
        if "CREATE TABLE products" in text:
            return json.dumps({"sql": SHOP_SQLS})
        return json.dumps({"sql": LIBRARY_SQLS})
    raise RuntimeError("mock LLM saw an unexpected prompt")


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
            generate_num=len(LIBRARY_SQLS),
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


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--mock"]:
        _run_mock()
        return
    if args:
        raise SystemExit("usage: python3 examples/nl2sql_synthesize.py [--mock]")
    _run_live()


def _run_live() -> None:
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


def _run_mock() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    llm = MockLLM(responder=respond)
    pipe = NL2SQLSynthesize(OUTPUT / "cache", llm)
    pipe.compile()
    if llm.calls:
        raise SystemExit("compile() called the mock LLM")
    print(pipe.draw(), end="")
    frame = pipe.run()
    counts = [_line_count(path) for path in pipe.storage.step_files()]
    print("step rows: " + ", ".join(str(count) for count in counts))
    print(f"rows out: {len(frame)}")
    columns = ["db_id", "sql_complexity", "question", "sql", "overall", "model_name"]
    print(frame[columns].to_string(index=False))
    exported = pipe.export_jsonl(OUTPUT / "result.jsonl")
    print(f"wrote {exported}")

    if counts != [3, 15, 9, 9, 9, 6]:
        raise SystemExit(f"unexpected step row counts: {counts}")
    if frame["db_id"].tolist() != ["library", "library", "school", "school", "shop", "shop"]:
        raise SystemExit(f"unexpected databases: {frame['db_id'].tolist()}")
    if frame["sql"].tolist() != [
        LIBRARY_SQL,
        LIBRARY_JOIN,
        SCHOOL_SQL,
        SCHOOL_JOIN,
        SHOP_SQL,
        SHOP_JOIN,
    ]:
        raise SystemExit(f"unexpected SQL rows: {frame['sql'].tolist()}")
    if frame["sql_complexity"].tolist() != ["easy", "medium"] * 3:
        raise SystemExit(f"unexpected complexity: {frame['sql_complexity'].tolist()}")
    if frame["question"].tolist() != [
        "Which books were published in 1984 or later?",
        "Which books are on loan, and who borrowed them?",
        "Which students are in grade 10 or above?",
        "Which course is each student enrolled in?",
        "Which products cost less than 20?",
        "Who bought each product?",
    ]:
        raise SystemExit(f"unexpected questions: {frame['question'].tolist()}")
    if not all(frame["sql_ok"]):
        raise SystemExit("surviving rows should be executable")
    if set(frame["overall"]) != {4.75}:
        raise SystemExit(f"unexpected scores: {frame['overall'].tolist()}")
    if set(frame["model_name"]) != {"mock"}:
        raise SystemExit(f"unexpected model names: {frame['model_name'].tolist()}")
    if len(llm.calls) != 5:
        raise SystemExit("expected one SQL call per schema, then question and judge batches")
    print("Three schemas in. Execution and dedup leave two scored rows for each.")


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
