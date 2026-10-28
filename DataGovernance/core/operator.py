"""Operator contract and name registry.

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
    task declare columns that do not appear as ``input_*`` /
    ``output_*`` arguments. Both must be free of LLM calls. Return ``None``
    when the ``run`` signature carries the columns instead.

    ``consumes_tokens`` is true when ``run`` calls a model. Those steps
    are checkpointed. A later failure reuses the finished step and any
    rows that already came back.
    """

    replaces_dataset: bool = False
    consumes_tokens: bool = False

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

    def checkpoint_signature(self) -> str:
        """Prompt and model identity for a token checkpoint.

        The pipeline mixes this with the input rows. Change the text the
        model sees, the model name, or the temperature, and the stored
        step is not reused.
        """
        return ""

    def validate(self, arguments: Mapping[str, Any]) -> None:
        """Reject a bad argument combination during compile.

        ``arguments`` includes signature defaults. This method must not
        read the dataset or call an LLM.
        """

    @abstractmethod
    def run(self, storage: Any, **kwargs: Any) -> None:
        """Read the current step, write the next one."""


def model_checkpoint_fields(llm: object) -> dict[str, object]:
    """Model name and sampling settings that change a stored completion."""
    fields: dict[str, object] = {"model": str(getattr(llm, "model", ""))}
    temperature = getattr(llm, "temperature", None)
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        fields["temperature"] = float(temperature)
    return fields


def get_operator(name: str, **kwargs: Any) -> OperatorABC:
    """Build a registered operator by class name.

    Importing ``DataGovernance`` registers the built-in operators. A
    subclass of ``OperatorABC`` registers itself when the class statement
    runs.
    """
    if name not in OPERATOR_REGISTRY:
        import DataGovernance.ops  # noqa: F401

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
