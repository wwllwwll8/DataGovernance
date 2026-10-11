"""DataFlow-style pipelines with Distilabel-style LLM task recipes.

Operators read and write a JSONL step chain. ``Pipeline.compile()`` checks
column dependencies and records producer-to-consumer lineage before any
LLM call. Task recipes (text generation, UltraFeedback-like scoring) talk
to a pluggable ``LLMClient``. Tests use ``MockLLM``.
"""

from df_distilabel.core.operator import OperatorABC, get_operator
from df_distilabel.core.pipeline import KeyNode, Lineage, Pipeline
from df_distilabel.core.storage import FileStorage
from df_distilabel.llm import LLMClient, MockLLM
from df_distilabel.ops.distilabel_bridge import (
    ULTRAFEEDBACK_ASPECTS,
    DistilabelTaskOp,
    TextGenerationOp,
    UltraFeedbackOp,
    distilabel_available,
    parse_ultrafeedback,
)
from df_distilabel.ops.load import LoadJSONL
from df_distilabel.ops.verify import (
    HashDeduplicateFilter,
    MathVerifyFilter,
    SQLExecFilter,
    check_answer,
)

__version__ = "0.1.0"

__all__ = [
    "DistilabelTaskOp",
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
    "distilabel_available",
    "get_operator",
    "parse_ultrafeedback",
]
