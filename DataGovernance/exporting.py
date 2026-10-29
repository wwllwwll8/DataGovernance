"""Dataset artifacts for review and publishing.

``to_argilla_records`` is the preference shape Distilabel sends to Argilla:
an instruction, the candidate generations, and the optional scores.
``push_dataset`` uploads the JSONL. The default uploader is Hugging Face
Hub and is imported only when a publish actually runs.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

Uploader = Callable[[str, str, bytes, str | None], str]


def to_argilla_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One Argilla-style record per row."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        instruction = row.get("instruction")
        if instruction is None or (isinstance(instruction, float) and pd.isna(instruction)):
            instruction = row.get("question", "")
        generations = _generations(row)
        records.append(
            {
                "fields": {
                    "instruction": "" if instruction is None else str(instruction),
                    "generations": generations,
                },
                "metadata": {
                    "ratings": _jsonable(row.get("ratings")),
                    "rationales": _jsonable(row.get("rationales")),
                    "overall": _jsonable(row.get("overall")),
                },
            }
        )
    return records


def write_argilla(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write Argilla records as JSONL. Returns the file path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in to_argilla_records(frame)]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def push_dataset(
    repo_id: str,
    frame: pd.DataFrame,
    *,
    token: str | None = None,
    uploader: Uploader | None = None,
) -> str:
    """Upload ``result.jsonl`` to ``repo_id`` and return the dataset URL."""
    if not isinstance(repo_id, str) or not repo_id.strip() or "/" not in repo_id:
        raise ValueError("repo_id must look like 'owner/name'")
    body = json.dumps(frame.to_dict(orient="records"), ensure_ascii=False, default=str).encode("utf-8")
    send = uploader or huggingface_upload
    return send(repo_id.strip(), "result.jsonl", body, token)


def huggingface_upload(repo_id: str, filename: str, content: bytes, token: str | None) -> str:
    """Upload one file to a dataset repo. Requires the ``hub`` extra."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install the hub extra: pip install 'DataGovernance[hub]'") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(content),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _generations(row: dict[str, Any]) -> list[str]:
    generations = row.get("generations")
    if isinstance(generations, list) and generations:
        return [str(item) for item in generations]
    for key in ("generation", "sql"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value
