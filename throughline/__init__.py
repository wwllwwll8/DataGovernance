"""Throughline: column-lineage pipelines with LLM task recipes.

Operators read and write a JSONL step chain. ``Pipeline.compile()`` checks
column dependencies and records producer-to-consumer lineage before any
LLM call. Task recipes (text generation, UltraFeedback-like scoring) talk
to a pluggable ``LLMClient``. Tests use ``MockLLM``.
"""

from throughline.core.operator import OperatorABC, get_operator
from throughline.core.pipeline import KeyNode, Lineage, Pipeline
from throughline.core.storage import FileStorage
from throughline.llm import LLMClient, MockLLM
from throughline.ops.tasks import (
    ULTRAFEEDBACK_ASPECTS,
    TaskOp,
    TextGenerationOp,
    UltraFeedbackOp,
    parse_ultrafeedback,
)
from throughline.ops.load import LoadJSONL
from throughline.ops.verify import (
    HashDeduplicateFilter,
    MathVerifyFilter,
    SQLExecFilter,
    check_answer,
)

__version__ = "0.1.0"

__all__ = [
    "TaskOp",
    "FileStorage",
    "HashDeduplicateFilter",
    "KeyNode",
    "LLMClient",
    "Lineage",
    "LoadJSONL",
    "MathVerifyFilter",
    "MockLLM",
    "OperatorABC",
    "Pipeline",
    "SQLExecFilter",
    "TextGenerationOp",
    "ULTRAFEEDBACK_ASPECTS",
    "UltraFeedbackOp",
    "__version__",
    "check_answer",
    "get_operator",
    "parse_ultrafeedback",
]
