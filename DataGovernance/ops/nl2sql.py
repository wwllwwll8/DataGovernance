"""NL2SQL synthesis: schema to SQL, then SQL to a natural-language question.

The SQL step follows the Text-to-SQL synthesis shape: one schema yields
several SELECT statements, which an execution filter can drop. The
question step is a Distilabel task (``inputs``, ``outputs``,
``format_input``, ``format_output``) driven by ``TextGenerationOp``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from DataGovernance.core.operator import OperatorABC, require_columns
from DataGovernance.llm import LLMClient, ensure_llm

_FENCE = re.compile(r"^```(?:json|sql)?\s*|\s*```$", re.IGNORECASE)


class SQLSynthesisOp(OperatorABC):
    """Expand each schema row into several candidate SQL statements.

    One model call per input row. The completion must be a JSON object
    ``{"sql": ["SELECT ...", ...]}``. Each statement becomes its own row
    and keeps ``db_id`` and the schema. A local label is written to
    ``output_complexity_key`` (``medium`` when the statement joins or
    aggregates, otherwise ``easy``).
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = ensure_llm(llm)

    def run(
        self,
        storage: Any,
        input_schema_key: str = "schema",
        input_db_id_key: str = "db_id",
        output_sql_key: str = "sql",
        output_schema_key: str = "schema",
        output_db_id_key: str = "db_id",
        output_complexity_key: str = "sql_complexity",
        generate_num: int = 4,
    ) -> None:
        if generate_num < 1:
            raise ValueError("generate_num must be at least 1")
        frame = storage.read()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("SQLSynthesisOp expects storage.read() to return a DataFrame")
        frame = frame.copy().reset_index(drop=True)
        columns = [output_db_id_key, output_schema_key, output_sql_key, output_complexity_key]
        if frame.empty:
            storage.write(pd.DataFrame(columns=list(dict.fromkeys([*frame.columns, *columns]))))
            return
        require_columns(frame, [input_schema_key, input_db_id_key], self.name)
        rows: list[dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            schema = str(record[input_schema_key])
            db_id = record[input_db_id_key]
            chat = build_sql_synthesis_chat(schema, generate_num)
            completion = _one(self.llm, chat)
            for statement in parse_sql_list(completion):
                rows.append(
                    {
                        **record,
                        output_db_id_key: db_id,
                        output_schema_key: schema,
                        output_sql_key: statement,
                        output_complexity_key: sql_complexity(statement),
                    }
                )
        if not rows:
            storage.write(pd.DataFrame(columns=list(dict.fromkeys([*frame.columns, *columns]))))
            return
        storage.write(pd.DataFrame.from_records(rows))


class SQLQuestionTask:
    """Distilabel task: one natural-language question for an executable SQL.

    ``TextGenerationOp`` reads ``schema`` and ``sql`` and writes ``question``.
    """

    inputs = ["schema", "sql"]
    outputs = ["question"]

    def format_input(self, record: dict[str, Any]) -> list[dict[str, str]]:
        user = (
            "Write one natural-language question that this SQL answers. "
            "Return only the question.\n\n"
            f"Schema:\n{record['schema']}\n\n"
            f"SQL:\n{record['sql']}"
        )
        return [
            {"role": "system", "content": "You write database questions. Return only the question."},
            {"role": "user", "content": user},
        ]

    def format_output(self, output: str, record: dict[str, Any]) -> dict[str, str]:
        question = str(output).strip()
        if not question:
            raise ValueError("question synthesis returned an empty question")
        return {"question": question}


def build_sql_synthesis_chat(schema: str, generate_num: int) -> list[dict[str, str]]:
    user = (
        f"Generate {generate_num} different SQLite SELECT statements for this schema.\n"
        "Cover filters and, when the schema allows it, one join.\n"
        'Return a JSON object only, with this shape: {"sql": ["SELECT ..."]}\n\n'
        f"Schema:\n{schema}"
    )
    return [
        {"role": "system", "content": "You write SQLite SELECT statements. Return JSON only."},
        {"role": "user", "content": user},
    ]


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


def _one(llm: LLMClient, chat: list[dict[str, str]]) -> str:
    outputs = list(llm.generate([chat]))
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise RuntimeError("SQL synthesis expected one string completion")
    return outputs[0]


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
