"""NL2SQL operators: synthesize SQL, execute it, then write a question.

``SQLSynthesisOp`` turns one schema into several SELECT statements.
``SQLExecFilter`` keeps the statements that run on that schema.
``SQLQuestionTask`` is the Distilabel task that writes the natural-language
question for ``TextGenerationOp``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

import pandas as pd

from DataGovernance.core.checkpoint import map_completions
from DataGovernance.core.operator import OperatorABC, model_checkpoint_fields, require_columns
from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import LLMClient, ensure_llm
from DataGovernance.ops.database import DatabaseManager

_FENCE = re.compile(r"^```(?:json|sql)?\s*|\s*```$", re.IGNORECASE)
_SQL_COMMENT = re.compile(r"/\*.*?\*/|--.*?$", re.DOTALL | re.MULTILINE)
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
SQL_RESULT_ROW_CAP = 50


class SQLSynthesisOp(OperatorABC):
    """Expand each schema row into several candidate SQL statements.

    One model call per input row. A row that already returned is kept if a
    later schema fails. The completion must be a JSON object
    ``{"sql": ["SELECT ...", ...]}``. Each statement becomes its own row
    and keeps ``db_id`` and the schema. A local label is written to
    ``output_complexity_key`` (``medium`` when the statement joins or
    aggregates, otherwise ``easy``).
    """

    consumes_tokens = True

    def __init__(
        self,
        llm: LLMClient,
        prompt: Prompt,
        generate_num: int = 5,
        copy: list[str] | None = None,
    ) -> None:
        self.llm = ensure_llm(llm)
        if not isinstance(prompt, Prompt):
            raise TypeError("SQLSynthesisOp expects a Prompt")
        self.prompt = prompt
        if isinstance(generate_num, bool) or not isinstance(generate_num, int) or generate_num < 1:
            raise ValueError("generate_num must be an integer >= 1")
        self.generate_num = generate_num
        columns = ["db_id"] if copy is None else copy
        if not isinstance(columns, list) or not all(isinstance(item, str) and item for item in columns):
            raise ValueError("copy must be a list of column names")
        self.copy_columns = list(columns)

    def declared_inputs(self) -> list[str]:
        columns = self.prompt.input_columns()
        return columns + [name for name in self.copy_columns if name not in columns]

    def declared_outputs(self) -> list[str]:
        return ["db_id", "schema", "sql", "sql_complexity"]

    def checkpoint_signature(self) -> str:
        return json.dumps(
            {
                **model_checkpoint_fields(self.llm),
                "system": self.prompt.system,
                "user": self.prompt.user,
                "generate_num": self.generate_num,
            },
            sort_keys=True,
        )

    def run(self, storage: Any) -> None:
        frame = storage.read()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("SQLSynthesisOp expects storage.read() to return a DataFrame")
        frame = frame.copy().reset_index(drop=True)
        columns = ["db_id", "schema", "sql", "sql_complexity"]
        if frame.empty:
            storage.write(pd.DataFrame(columns=list(dict.fromkeys([*frame.columns, *columns]))))
            return
        require_columns(frame, self.declared_inputs(), self.name)
        records = frame.to_dict(orient="records")
        chats = [
            self.prompt.render(record, generate_num=self.generate_num) for record in records
        ]
        parsed = map_completions(
            self,
            storage,
            {},
            self.llm,
            chats,
            lambda _index, text: parse_sql_list(text),
        )
        rows: list[dict[str, Any]] = []
        for record, statements in zip(records, parsed, strict=True):
            schema = str(record["schema"])
            db_id = record["db_id"]
            for statement in statements:
                rows.append(
                    {
                        **record,
                        "db_id": db_id,
                        "schema": schema,
                        "sql": statement,
                        "sql_complexity": sql_complexity(statement),
                    }
                )
        if not rows:
            storage.write(pd.DataFrame(columns=list(dict.fromkeys([*frame.columns, *columns]))))
            return
        storage.write(pd.DataFrame.from_records(rows))


class SQLQuestionTask:
    """Distilabel task: one natural-language question for an executable SQL.

    The caller passes the prompt, usually ``load_prompt("nl2sql.question")``.
    ``TextGenerationOp`` reads the prompt columns and writes ``question``.
    """

    outputs = ["question"]

    def __init__(self, prompt: Prompt) -> None:
        if not isinstance(prompt, Prompt):
            raise TypeError("SQLQuestionTask expects a Prompt")
        self.prompt = prompt
        self.inputs = self.prompt.input_columns()

    def format_input(self, record: dict[str, Any]) -> list[dict[str, str]]:
        return self.prompt.render(record)

    def format_output(self, output: str, record: dict[str, Any]) -> dict[str, str]:
        question = str(output).strip()
        if not question:
            raise ValueError("question synthesis returned an empty question")
        return {"question": question}


def parse_sql_list(text: str) -> list[str]:
    """Read a JSON sql list. Fenced completions and a bare list are accepted."""
    payload = _load_json(text)
    if isinstance(payload, dict):
        payload = payload.get("sql", [])
    if not isinstance(payload, list):
        raise ValueError("SQL synthesis response has no sql list")
    statements: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            item = item.get("query", item.get("sql", ""))
        if not isinstance(item, str):
            continue
        statement = item.strip()
        if statement:
            statements.append(statement)
    if not statements:
        raise ValueError("SQL synthesis response listed no statements")
    return statements


def sql_complexity(statement: str) -> str:
    """Label a statement easy or medium from its clauses. No model call."""
    folded = f" {statement.casefold()} "
    markers = (" join ", " group by ", " union ", " intersect ", " except ")
    if any(marker in folded for marker in markers):
        return "medium"
    return "easy"


class SQLExecFilter(OperatorABC):
    """SELECT-only sqlite check for a synthesized statement.

    ``setup_sql`` is trusted and runs with ``executescript`` on a fresh
    in-memory database for every row. ``input_setup_key`` uses that column
    as the per-row schema instead. SQL from ``input_key`` must be a single
    SELECT or WITH ... SELECT. Writes, multiple statements, and execution
    errors mark the row failed. ``drop_failures`` removes those rows. At
    most ``SQL_RESULT_ROW_CAP`` result rows are stored.
    """

    def __init__(
        self,
        setup_sql: str | None = None,
        database: DatabaseManager | None = None,
    ) -> None:
        self.setup_sql = setup_sql or ""
        if database is not None and not isinstance(database, DatabaseManager):
            raise TypeError("database must be a DatabaseManager")
        self.database = database

    def run(
        self,
        storage: Any,
        input_key: str = "sql",
        input_gold_key: str | None = None,
        input_setup_key: str | None = None,
        output_key: str = "sql_ok",
        output_result: str = "sql_result",
        output_error: str = "sql_error",
        drop_failures: bool = True,
        input_db_id_key: str | None = None,
    ) -> None:
        frame = _frame(storage)
        columns = [input_key]
        if input_gold_key:
            columns.append(input_gold_key)
        if input_setup_key:
            columns.append(input_setup_key)
        if input_db_id_key:
            columns.append(input_db_id_key)
        if frame.empty:
            frame[output_key] = pd.Series(dtype="boolean")
            frame[output_result] = pd.Series(dtype="object")
            frame[output_error] = pd.Series(dtype="object")
            storage.write(frame)
            return
        require_columns(frame, columns, self.name)
        oks: list[bool] = []
        results: list[str] = []
        errors: list[str] = []
        golds = (
            frame[input_gold_key].tolist()
            if input_gold_key
            else [None] * len(frame)
        )
        setups = (
            frame[input_setup_key].tolist()
            if input_setup_key
            else [self.setup_sql] * len(frame)
        )
        db_ids = frame[input_db_id_key].tolist() if input_db_id_key else [None] * len(frame)
        for sql, gold, setup, db_id in zip(
            frame[input_key].tolist(), golds, setups, db_ids, strict=True
        ):
            if self.database is not None and db_id is not None and str(db_id).strip():
                ok, payload, error = _execute_managed(self.database, str(db_id), str(sql), gold)
            else:
                ok, payload, error = _execute_select(str(sql), _setup_text(setup), gold)
            oks.append(ok)
            results.append(payload)
            errors.append(error)
        frame[output_key] = oks
        frame[output_result] = results
        frame[output_error] = errors
        if drop_failures:
            frame = frame.loc[pd.Series(oks, index=frame.index)].reset_index(drop=True)
        storage.write(frame)


def _frame(storage: Any) -> pd.DataFrame:
    frame = storage.read()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("NL2SQL operators expect storage.read() to return a DataFrame")
    return frame.copy().reset_index(drop=True)


def _setup_text(setup: Any) -> str:
    if setup is None:
        return ""
    try:
        if bool(pd.isna(setup)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(setup)


def _execute_select(sql: str, setup_sql: str, gold: Any) -> tuple[bool, str, str]:
    if not _is_select_only(sql):
        return False, "", "not a single SELECT statement"
    connection = sqlite3.connect(":memory:")
    try:
        if setup_sql.strip():
            connection.executescript(setup_sql)
        return _query(connection, sql, gold)
    finally:
        connection.close()


def _execute_managed(database: DatabaseManager, db_id: str, sql: str, gold: Any) -> tuple[bool, str, str]:
    if not _is_select_only(sql):
        return False, "", "not a single SELECT statement"
    try:
        connection = database.connect(db_id)
    except KeyError as exc:
        return False, "", str(exc)
    try:
        return _query(connection, sql, gold)
    finally:
        connection.close()


def _query(connection: sqlite3.Connection, sql: str, gold: Any) -> tuple[bool, str, str]:
    try:
        cursor = connection.execute(sql)
    except sqlite3.Error as exc:
        return False, "", str(exc)
    if cursor.description is None:
        return False, "", "statement did not return rows"
    rows = [list(row) for row in cursor.fetchmany(SQL_RESULT_ROW_CAP)]
    payload = json.dumps(rows, ensure_ascii=False)
    if gold is None or (isinstance(gold, float) and pd.isna(gold)):
        return True, payload, ""
    if _json_equal(payload, str(gold)):
        return True, payload, ""
    return False, payload, "result did not match gold"


def _is_select_only(sql: str) -> bool:
    stripped = _SQL_COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        return False
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        return False
    if _SQL_FORBIDDEN.search(stripped):
        return False
    return True


def _json_equal(left: str, right: str) -> bool:
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return left.strip() == right.strip()


def _load_json(text: str) -> Any:
    stripped = text.strip()
    stripped = _FENCE.sub("", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_obj = stripped.find("{")
        start_list = stripped.find("[")
        starts = [index for index in (start_obj, start_list) if index >= 0]
        if not starts:
            raise ValueError("SQL synthesis response was not JSON") from None
        start = min(starts)
        end = stripped.rfind("}" if stripped[start] == "{" else "]")
        if end <= start:
            raise ValueError("SQL synthesis response was not JSON") from None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("SQL synthesis response was not JSON") from exc
