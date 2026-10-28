"""Prompt text owned by an operator.

A prompt is system and user text plus the column names it reads.
``{column}`` placeholders are that operator's inputs, so ``compile()``
can check them before any model call.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter


class Prompt:
    """System and user text rendered once per row.

    ``user`` and ``system`` may contain ``{column}`` placeholders. Those
    names are the prompt's input columns, in first-seen order.
    """

    def __init__(
        self,
        user: str,
        system: str = "",
        parameters: list[str] | tuple[str, ...] = (),
    ) -> None:
        if not isinstance(user, str) or not user.strip():
            raise ValueError("prompt user text must be a non-empty string")
        if not isinstance(system, str):
            raise ValueError("prompt system text must be a string")
        self.user = user
        self.system = system
        self._parameters = tuple(parameters)
        names = _placeholders(system) + [
            name for name in _placeholders(user) if name not in _placeholders(system)
        ]
        missing = [name for name in self._parameters if name not in names]
        if missing:
            raise ValueError(f"prompt parameters are not in the text: {', '.join(missing)}")
        self._columns = [name for name in names if name not in self._parameters]

    def input_columns(self) -> list[str]:
        return list(self._columns)

    def render(self, record: Mapping[str, object], **parameters: object) -> list[dict[str, str]]:
        """Fill column placeholders from ``record`` and parameters from kwargs."""
        missing = [name for name in self._parameters if name not in parameters]
        if missing:
            raise ValueError(f"prompt is missing parameters: {', '.join(missing)}")
        extra = [name for name in parameters if name not in self._parameters]
        if extra:
            raise ValueError(f"prompt received unknown parameters: {', '.join(extra)}")
        values = {name: "" if record.get(name) is None else str(record[name]) for name in self._columns}
        values.update({name: str(parameters[name]) for name in self._parameters})
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
