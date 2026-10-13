"""DataFlow-like operator contract and name registry.

An operator implements ``run(storage, input_key=..., output_key=...)``.
Column parameters use an ``input_`` or ``output_`` prefix so
``Pipeline.compile()`` can see them without calling ``run``. Other
parameters (booleans, limits) are runtime options and are not columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pandas as pd

OPERATOR_REGISTRY: dict[str, type["OperatorABC"]] = {}


class OperatorABC(ABC):
    """Base class for a single pipeline step.

    ``replaces_dataset`` is true for source operators that discard the
    current frame and write a new one (``LoadJSONL``).

    ``declared_inputs`` / ``declared_outputs`` let a source or a wrapped
    Distilabel task name columns that do not appear as ``input_*`` /
    ``output_*`` arguments. Both must be free of LLM calls. Return ``None``
    when the ``run`` signature carries the columns instead.
    """

    replaces_dataset: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        OPERATOR_REGISTRY[cls.__name__] = cls

    @property
    def name(self) -> str:
        return type(self).__name__

    def declared_inputs(self) -> list[str] | None:
        return None

    def declared_outputs(self) -> list[str] | None:
        return None

    def validate(self, arguments: Mapping[str, Any]) -> None:
        """Reject a bad argument combination during compile.

        ``arguments`` includes signature defaults. This method must not
        read the dataset or call an LLM.
        """

    @abstractmethod
    def run(self, storage: Any, **kwargs: Any) -> None:
        """Read the current step, write the next one."""


def get_operator(name: str, **kwargs: Any) -> OperatorABC:
    """Build a registered operator by class name.

    Importing ``throughline`` registers the built-in operators. A
    subclass of ``OperatorABC`` registers itself when the class statement
    runs.
    """
    if name not in OPERATOR_REGISTRY:
        import throughline.ops  # noqa: F401

    try:
        cls = OPERATOR_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(OPERATOR_REGISTRY))
        raise KeyError(f"Unknown operator {name!r}. Known: {known}") from None
    return cls(**kwargs)


def require_columns(frame: pd.DataFrame, columns: list[str], operator: str) -> None:
    """Raise KeyError if ``frame`` is missing any of ``columns``."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{operator} missing input column(s) {missing}. "
            f"Available columns: {list(frame.columns)}."
        )
