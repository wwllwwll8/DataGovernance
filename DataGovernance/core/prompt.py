"""Prompt templates for generate, rank, and score.

A prompt is text plus the column names it reads. ``{column}`` placeholders
become inputs, so ``compile()`` can check them before any model call.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter


class Prompt:
    """System and user text rendered once per row.

    ``user`` and ``system`` may contain ``{column}`` placeholders. Those
    names are the prompt's input columns, in first-seen order.
    """

    def __init__(self, user: str, system: str = "") -> None:
        if not isinstance(user, str) or not user.strip():
            raise ValueError("prompt user text must be a non-empty string")
        if not isinstance(system, str):
            raise ValueError("prompt system text must be a string")
        self.user = user
        self.system = system
        self._columns = _placeholders(system) + [
            name for name in _placeholders(user) if name not in _placeholders(system)
        ]

    def input_columns(self) -> list[str]:
        return list(self._columns)

    def render(self, record: Mapping[str, object]) -> list[dict[str, str]]:
        values = {name: "" if record.get(name) is None else str(record[name]) for name in self._columns}
        messages: list[dict[str, str]] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system.format(**values)})
        messages.append({"role": "user", "content": self.user.format(**values)})
        return messages


def _placeholders(template: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for _, name, _, _ in Formatter().parse(template):
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names
