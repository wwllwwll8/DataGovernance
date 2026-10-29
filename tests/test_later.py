"""List scoring, a live database, and concurrent chat calls."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from DataGovernance import (
    DatabaseManager,
    FileStorage,
    MockLLM,
    SQLExecFilter,
    TextGenerationOp,
    UltraFeedbackOp,
)
from DataGovernance.llm import ChatCompletionsClient, LLMConfig


def test_num_generations_are_scored_together(tmp_path, write_jsonl):
    seeds = write_jsonl(tmp_path / "seeds.jsonl", [{"instruction": "2+2"}])
    storage = FileStorage(first_entry_file_name=str(seeds), cache_path=tmp_path / "cache")
    drafts = iter(["4", "four"])

    def respond(chat):
        text = "\n".join(message["content"] for message in chat)
        if "Response 1" in text and "Response 2" in text:
            return json.dumps({"ratings": [5, 2], "rationales": ["right", "wordy"], "overall": 3})
        return next(drafts)

    llm = MockLLM(responder=respond)
    view = storage.step()
    TextGenerationOp(llm, num_generations=2).run(view)
    drafted = view.read_output()
    assert drafted.iloc[0]["generation"] == ["4", "four"]
    judge = storage.step()
    UltraFeedbackOp(llm).run(
        judge,
        input_generation_key=None,
        input_generations_key="generation",
        output_model="model_name",
    )
    scored = judge.read_output()
    assert scored.iloc[0]["ratings"] == [5.0, 2.0]
    assert scored.iloc[0]["overall"] == 3.0
    assert scored.iloc[0]["model_name"] == "mock"


def test_database_manager_runs_sql_on_a_file(tmp_path):
    path = tmp_path / "library.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT);"
        "INSERT INTO books (id, title) VALUES (1, 'Dune');"
    )
    connection.close()
    manager = DatabaseManager()
    manager.register("library", path=path)
    entry = tmp_path / "rows.jsonl"
    entry.write_text(
        json.dumps({"db_id": "library", "sql": "SELECT title FROM books"}) + "\n",
        encoding="utf-8",
    )
    storage = FileStorage(first_entry_file_name=str(entry), cache_path=tmp_path / "cache")
    view = storage.step()
    SQLExecFilter(database=manager).run(view, input_db_id_key="db_id", input_setup_key=None)
    frame = view.read_output()
    assert list(frame["sql_ok"]) == [True]
    assert json.loads(frame.iloc[0]["sql_result"]) == [["Dune"]]


def test_concurrent_generate_keeps_order():
    state = {"inflight": 0, "max": 0}
    lock = threading.Lock()

    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        text = payload["messages"][-1]["content"]
        with lock:
            state["inflight"] += 1
            state["max"] = max(state["max"], state["inflight"])
        time.sleep(0.05)
        with lock:
            state["inflight"] -= 1
        raw = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        return 200, raw

    client = ChatCompletionsClient(
        LLMConfig(model="demo", base_url="https://example.test/v1", api_key="k", concurrency=2),
        transport=transport,
    )
    chats = [[{"role": "user", "content": item}] for item in ("a", "b")]
    assert client.generate(chats) == ["a", "b"]
    assert state["max"] == 2
