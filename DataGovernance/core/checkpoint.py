"""Checkpoints for operators that spend tokens.

A finished token-using step stores a fingerprint next to its step file.
The next ``Pipeline.run`` reuses that file when the input rows, the prompt,
and the model are unchanged. A step that dies mid-batch keeps each accepted
completion in a progress file, so the retry calls the model only for the
rows that did not finish.

Operators that do not spend tokens always run. They are cheap, and their
output is what the next token step fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from DataGovernance.core.operator import OperatorABC
from DataGovernance.core.storage import FileStorage, dataframe_to_records
from DataGovernance.llm import LLMClient

Accept = Callable[[int, str], Any]


def fingerprint(storage: FileStorage, operator: OperatorABC, arguments: dict[str, Any]) -> str:
    """Hash the input frame, the operator signature, and the run arguments."""
    frame = storage.read()
    payload = {
        "signature": operator.checkpoint_signature(),
        "arguments": arguments,
        "columns": list(frame.columns),
        "rows": dataframe_to_records(frame),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_stamp(storage: FileStorage) -> str | None:
    """Return the fingerprint of a finished step, if one was stored."""
    path = _stamp_path(storage)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    stamp = data.get("fingerprint") if isinstance(data, dict) else None
    if not isinstance(stamp, str) or not stamp:
        return None
    return stamp


def write_stamp(storage: FileStorage, digest: str) -> None:
    """Record that this step's output matches ``digest``."""
    path = _stamp_path(storage)
    path.write_text(json.dumps({"fingerprint": digest}) + "\n", encoding="utf-8")


def clear_progress(storage: FileStorage) -> None:
    """Drop the partial-row file after the step file is complete."""
    path = _progress_path(storage)
    if path.is_file():
        path.unlink()


def map_completions(
    operator: OperatorABC,
    storage: FileStorage,
    arguments: dict[str, Any],
    llm: LLMClient,
    chats: list[list[dict[str, str]]],
    accept: Accept,
) -> list[Any]:
    """Call the model once per missing row.

    ``accept`` turns one completion into the row value. A row is stored
    only after ``accept`` returns. A cached completion whose ``accept``
    now fails is dropped, so the next run asks the model again.
    """
    digest = fingerprint(storage, operator, arguments)
    done = _load_progress(storage, digest)
    results: list[Any] = []
    for index, chat in enumerate(chats):
        cached = index in done
        text = done[index] if cached else _one(llm, chat)
        try:
            result = accept(index, text)
        except Exception:
            if cached:
                done.pop(index, None)
                _write_progress(storage, digest, done)
            raise
        if not cached:
            done[index] = text
            _append_progress(storage, digest, index, text)
        results.append(result)
    return results


def _one(llm: LLMClient, chat: list[dict[str, str]]) -> str:
    outputs = list(llm.generate([chat]))
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise RuntimeError(
            f"LLM returned {len(outputs)} completions for 1 chat"
        )
    return outputs[0]


def _stamp_path(storage: FileStorage) -> Path:
    return _output_file(storage).with_suffix(".checkpoint.json")


def _progress_path(storage: FileStorage) -> Path:
    return _output_file(storage).with_suffix(".progress.jsonl")


def _output_file(storage: FileStorage) -> Path:
    if storage.operator_step < 0:
        raise ValueError("Call storage.step() before checkpointing")
    return storage.cache_file(storage.operator_step + 1)


def _load_progress(storage: FileStorage, digest: str) -> dict[int, str]:
    path = _progress_path(storage)
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        path.unlink()
        return {}
    if not isinstance(header, dict) or header.get("fingerprint") != digest:
        path.unlink()
        return {}
    done: dict[int, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        completion = item.get("completion")
        if isinstance(index, int) and isinstance(completion, str):
            done[index] = completion
    return done


def _append_progress(storage: FileStorage, digest: str, index: int, completion: str) -> None:
    path = _progress_path(storage)
    if not path.is_file():
        path.write_text(json.dumps({"fingerprint": digest}) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"index": index, "completion": completion}, ensure_ascii=False) + "\n"
        )
        handle.flush()


def _write_progress(storage: FileStorage, digest: str, done: dict[int, str]) -> None:
    path = _progress_path(storage)
    lines = [json.dumps({"fingerprint": digest})]
    for index in sorted(done):
        lines.append(
            json.dumps({"index": index, "completion": done[index]}, ensure_ascii=False)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
