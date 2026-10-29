"""Named SQLite databases for NL2SQL execution.

A row can name a ``db_id`` instead of carrying its schema. Register either
a schema script or a sqlite file. ``SQLExecFilter`` opens a connection
from this registry when ``input_db_id_key`` is set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class DatabaseManager:
    """db_id to a schema script or a sqlite file."""

    def __init__(self) -> None:
        self._schemas: dict[str, str] = {}
        self._paths: dict[str, Path] = {}

    def register(self, db_id: str, *, schema: str | None = None, path: str | Path | None = None) -> None:
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError("db_id must be a non-empty string")
        if (schema is None) == (path is None):
            raise ValueError("register a schema script or a sqlite file")
        key = db_id.strip()
        self._schemas.pop(key, None)
        self._paths.pop(key, None)
        if schema is not None:
            if not isinstance(schema, str) or not schema.strip():
                raise ValueError("schema must be a non-empty SQL script")
            self._schemas[key] = schema
            return
        file = Path(path)  # type: ignore[arg-type]
        if not file.is_file():
            raise FileNotFoundError(f"sqlite database not found: {file}")
        self._paths[key] = file

    def connect(self, db_id: str) -> sqlite3.Connection:
        """Open a connection. The caller closes it.

        A schema registration returns a fresh in-memory database with the
        script applied. A path registration opens that file.
        """
        key = db_id.strip()
        if key in self._paths:
            return sqlite3.connect(self._paths[key])
        if key in self._schemas:
            connection = sqlite3.connect(":memory:")
            connection.executescript(self._schemas[key])
            return connection
        known = ", ".join(sorted([*self._schemas, *self._paths])) or "(none)"
        raise KeyError(f"Unknown database {db_id!r}. Known: {known}")
