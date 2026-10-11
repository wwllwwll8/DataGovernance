"""Compiling pipeline: AutoOP record, column checks, KeyNode lineage.

``compile()`` wraps each operator attribute in ``AutoOP`` and calls
``forward()``. ``AutoOP.run`` records the ``input_*`` / ``output_*``
arguments and does not call ``Operator.run``, so no LLM client runs.
The recorded calls are checked against the entry file (or a source
operator's declared columns). ``forward()`` is then replaced with the
real execution of those recorded calls.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from df_distilabel.core.operator import OperatorABC
from df_distilabel.core.storage import FileStorage, write_jsonl


@dataclass
class KeyNode:
    """One column endpoint in the compiled graph.

    An output (or seed) node is the producer of ``key``. Its ``readers``
    are the input nodes that consume that value. That list is the
    producer → consumer edge for the column.
    """

    key: str
    parameter: str
    owner: str
    kind: str
    index: int
    readers: list[KeyNode] = field(default_factory=list)


@dataclass(frozen=True)
class ColumnBinding:
    """A column name taken from one ``run`` parameter."""

    parameter: str
    key: str


@dataclass
class OPRuntime:
    """One recorded ``operator.run`` call captured during compile."""

    op: OperatorABC
    op_name: str
    storage: FileStorage
    kwargs: dict[str, Any]
    inputs: list[ColumnBinding]
    outputs: list[ColumnBinding]

    @property
    def label(self) -> str:
        return f"{self.op_name}:{type(self.op).__name__}"


class Lineage:
    """Column producers and consumers collected from KeyNodes."""

    def __init__(self) -> None:
        self.nodes: list[KeyNode] = []
        self.producers: dict[str, KeyNode] = {}
        self.order: list[str] = []

    def seed(self, key: str) -> None:
        self._add_producer(KeyNode(key, "dataset", "DATASET-INPUT", "seed", -1))

    def read(self, key: str, parameter: str, owner: str, index: int) -> KeyNode:
        reader = KeyNode(key, parameter, owner, "input", index)
        self.producers[key].readers.append(reader)
        self.nodes.append(reader)
        return reader

    def write(self, key: str, parameter: str, owner: str, index: int) -> None:
        self._add_producer(KeyNode(key, parameter, owner, "output", index))

    def dead_columns(self) -> list[str]:
        """Columns no later operator reads.

        Terminal products (a score column nobody thresholds, a verify
        flag) are included. They are exported on the row and unused by
        the rest of the graph.
        """
        return [key for key in self.order if not self.producers[key].readers]

    def draw(self, labels: list[str]) -> str:
        lines = ["pipeline: " + " -> ".join(labels)]
        lines.append("column lineage")
        for key in self.order:
            node = self.producers[key]
            if node.readers:
                dest = ", ".join(reader.owner for reader in node.readers)
            else:
                dest = "(none)"
            lines.append(f"{key}: {node.owner} -> {dest}")
        dead = self.dead_columns()
        lines.append("dead columns: " + (", ".join(dead) if dead else "(none)"))
        return "\n".join(lines) + "\n"

    def _add_producer(self, node: KeyNode) -> None:
        if node.key not in self.producers:
            self.order.append(node.key)
        self.producers[node.key] = node
        self.nodes.append(node)


class AutoOP:
    """Record-only view of an operator used by ``Pipeline.compile``.

    ``run`` stores the storage view and the column arguments. It does not
    call the wrapped operator, and it does not call an LLM.
    """

    def __init__(self, operator: OperatorABC, op_name: str, pipeline: Pipeline) -> None:
        self.operator = operator
        self.op_name = op_name
        self.pipeline = pipeline

    def run(self, storage: FileStorage | None = None, **kwargs: Any) -> None:
        if not isinstance(storage, FileStorage):
            raise TypeError(
                f"{self.op_name}.run() expects a FileStorage from storage.step(), "
                f"got {type(storage).__name__}"
            )
        declared_inputs = _declared(self.operator, "declared_inputs")
        declared_outputs = _declared(self.operator, "declared_outputs")
        if declared_inputs or declared_outputs:
            if kwargs:
                raise TypeError(
                    f"{self.op_name} declares its own columns and takes no "
                    f"input_/output_ arguments. Got {sorted(kwargs)}."
                )
            inputs = [ColumnBinding("declared_inputs", key) for key in declared_inputs]
            outputs = [ColumnBinding("declared_outputs", key) for key in declared_outputs]
            arguments: dict[str, Any] = {"storage": storage}
        else:
            signature = inspect.signature(self.operator.run)
            try:
                bound = signature.bind(storage, **kwargs)
            except TypeError as exc:
                raise TypeError(f"{self.op_name}.run() {exc}") from exc
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            inputs, outputs = _bindings_from_arguments(arguments)
        self.operator.validate(arguments)
        self.pipeline.op_runtimes.append(
            OPRuntime(
                op=self.operator,
                op_name=self.op_name,
                storage=storage,
                kwargs=dict(kwargs),
                inputs=inputs,
                outputs=outputs,
            )
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.operator, name)


class Pipeline:
    """Subclass this, assign operators to ``self``, and implement ``forward``.

    ``forward`` should be a straight sequence of
    ``self.op.run(self.storage.step(), input_key=..., output_key=...)``.
    Operators must be attributes of the pipeline so compile can wrap them.
    Call ``run()`` (or ``compile()`` then ``forward()``) to execute.
    """

    def __init__(self) -> None:
        self.op_runtimes: list[OPRuntime] = []
        self.compiled = False
        self.lineage: Lineage | None = None
        self._executed = False

    def forward(self) -> Any:
        """Override with the operator calls. Compile records this method."""
        raise NotImplementedError(
            "Implement forward() with operator.run(self.storage.step(), input_key=..., output_key=...)"
        )

    def compile(self) -> Lineage:
        """Record ``forward`` and validate columns. Does not call ``Operator.run``."""
        if self.compiled:
            if self.lineage is None:
                raise RuntimeError("Pipeline is marked compiled without a lineage")
            return self.lineage
        storage = self._storage()
        originals: dict[str, OperatorABC] = {}
        try:
            storage.reset()
            self.op_runtimes = []
            for key, value in list(vars(self).items()):
                if isinstance(value, OperatorABC):
                    originals[key] = value
                    setattr(self, key, AutoOP(value, key, self))
            self.forward()
            if not self.op_runtimes:
                raise ValueError("forward() did not call any operators")
            lineage = self._build_lineage()
        except Exception:
            self.op_runtimes = []
            raise
        finally:
            for key, value in originals.items():
                setattr(self, key, value)
            storage.reset()
        self.lineage = lineage
        self.compiled = True
        self.forward = self._compiled_forward.__get__(self, type(self))  # type: ignore[method-assign]
        return lineage

    def run(self) -> Any:
        """Compile if needed, then execute the recorded operators."""
        if not self.compiled:
            self.compile()
        return self.forward()

    def draw(self) -> str:
        """Text dump of pipeline order and column lineage."""
        lineage = self.compile()
        return lineage.draw([runtime.label for runtime in self.op_runtimes])

    def latest_frame(self) -> Any:
        """DataFrame written by the last operator. Requires ``run()`` first."""
        if not self._executed or not self.op_runtimes:
            raise RuntimeError("Pipeline has not been run. Call run() before reading output.")
        return self.op_runtimes[-1].storage.read_output()

    def export_jsonl(self, path: str | Path) -> Path:
        """Write the latest step as JSONL plus a Distiset-like manifest.

        The manifest records operators and lineage. It is the v0 stand-in
        for a Distilabel ``Distiset``.
        """
        if self.lineage is None or not self.op_runtimes:
            raise RuntimeError("Compile the pipeline before export_jsonl()")
        frame = self.latest_frame()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, frame)
        manifest = {
            "kind": "distiset-like",
            "version": "0",
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "operators": [runtime.label for runtime in self.op_runtimes],
            "lineage": self.lineage.draw([runtime.label for runtime in self.op_runtimes]),
            "notes": {
                "ultrafeedback": (
                    "v0 scores one generation column. "
                    "distilabel.steps.tasks.UltraFeedback scores a list of generations."
                ),
                "argilla": (
                    "Call export_argilla() once PreferenceToArgilla is wired. "
                    "Distiset.push_to_hub is the Hub path."
                ),
            },
        }
        sidecar = target.with_suffix(".manifest.json")
        sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return target

    def export_argilla(self, **kwargs: Any) -> None:
        """Future hook for Distilabel's Argilla exporter.

        TODO(argilla): push the JSONL from ``export_jsonl`` with
        ``distilabel.steps.PreferenceToArgilla``, which reads
        ``instruction``, ``generations``, and optional ``ratings`` and
        ``rationales``. ``Distiset.push_to_hub`` publishes the same
        artifact to the Hugging Face Hub.
        """
        raise NotImplementedError(
            "Argilla export is not implemented in v0. "
            "export_jsonl() writes the Distiset-like artifact. "
            "A later adapter should call distilabel.steps.PreferenceToArgilla "
            "or Distiset.push_to_hub."
        )

    def _compiled_forward(self) -> Any:
        last: OPRuntime | None = None
        for runtime in self.op_runtimes:
            runtime.op.run(runtime.storage, **runtime.kwargs)
            frame = runtime.storage.read_output()
            missing = [spec.key for spec in runtime.outputs if spec.key not in frame.columns]
            if missing:
                raise RuntimeError(
                    f"{runtime.label} declared output column(s) {missing} "
                    "but did not write them"
                )
            last = runtime
        self._executed = True
        if last is None:
            raise RuntimeError("Pipeline has no operators")
        return last.storage.read_output()

    def _build_lineage(self) -> Lineage:
        lineage = Lineage()
        first = self.op_runtimes[0]
        if not first.op.replaces_dataset:
            for key in first.storage.get_keys_from_dataframe():
                lineage.seed(key)

        errors: list[str] = []
        for index, runtime in enumerate(self.op_runtimes):
            if runtime.op.replaces_dataset:
                if index != 0:
                    raise ValueError(
                        f"{runtime.label} replaces the working dataset and must be the first step"
                    )
                lineage = Lineage()
            for spec in runtime.inputs:
                if spec.key not in lineage.producers:
                    errors.append(
                        f"- Input key {spec.key!r} in `{runtime.op_name}` "
                        f"(class <{type(runtime.op).__name__}>) does not match any output key "
                        f"from previous operators or the dataset. "
                        f"Check parameter {spec.parameter!r} in `{runtime.op_name}.run()`. "
                        f"Available columns: {list(lineage.producers)}."
                    )
                else:
                    lineage.read(spec.key, spec.parameter, runtime.label, index)
            for spec in runtime.outputs:
                lineage.write(spec.key, spec.parameter, runtime.label, index)
        if errors:
            details = "\n".join(errors)
            raise KeyError(
                "Key Matching Error in following Operators during pipeline.compile():\n"
                f"{details}\n"
                "Compile stopped before any operator.run() or LLM call."
            )
        return lineage

    def _storage(self) -> FileStorage:
        found = [value for value in vars(self).values() if isinstance(value, FileStorage)]
        if len(found) != 1:
            raise RuntimeError(
                "Pipeline needs exactly one FileStorage attribute "
                f"(found {len(found)})."
            )
        return found[0]

    def __repr__(self) -> str:
        if self.op_runtimes:
            labels = ", ".join(runtime.op_name for runtime in self.op_runtimes)
        else:
            names = [key for key, value in vars(self).items() if isinstance(value, OperatorABC)]
            labels = ", ".join(names) or "empty"
        return f"Pipeline({labels})"


def _declared(operator: OperatorABC, method_name: str) -> list[str]:
    method = getattr(operator, method_name, None)
    if method is None:
        return []
    value = method()
    if not value:
        return []
    return [str(item) for item in value]


def _bindings_from_arguments(arguments: Mapping[str, Any]) -> tuple[list[ColumnBinding], list[ColumnBinding]]:
    inputs: list[ColumnBinding] = []
    outputs: list[ColumnBinding] = []
    for name, value in arguments.items():
        if name == "storage":
            continue
        if name.startswith("input_"):
            inputs.extend(ColumnBinding(name, key) for key in _as_keys(value))
        elif name.startswith("output_"):
            outputs.extend(ColumnBinding(name, key) for key in _as_keys(value))
    return inputs, outputs


def _as_keys(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        keys: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise TypeError(f"Column lists must contain non-empty strings, got {item!r}")
            keys.append(item)
        return keys
    raise TypeError(
        "input_* and output_* values must be a column name or a list of column names, "
        f"got {type(value).__name__}"
    )
