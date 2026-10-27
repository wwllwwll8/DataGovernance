"""DataGovernance: column-lineage pipelines with LLM task recipes.

Operators read and write a JSONL step chain. ``Pipeline.compile()`` checks
column dependencies and records producer-to-consumer lineage before any
LLM call. Task recipes (text generation, UltraFeedback-like scoring) talk
to a pluggable ``LLMClient``. ``LLMConfig`` holds the endpoint URL and
API key. ``ChatCompletionsClient`` calls an OpenAI-compatible HTTP API
with that config. Tests use ``MockLLM``.
"""

from DataGovernance.core.operator import OperatorABC, get_operator
from DataGovernance.core.pipeline import KeyNode, Lineage, Pipeline
from DataGovernance.core.storage import FileStorage
from DataGovernance.llm import ChatCompletionsClient, LLMClient, LLMConfig, MockLLM
from DataGovernance.ops.tasks import (
    ULTRAFEEDBACK_ASPECTS,
    TaskOp,
    TextGenerationOp,
    UltraFeedbackOp,
    parse_ultrafeedback,
)
from DataGovernance.ops.load import LoadJSONL
from DataGovernance.ops.nl2sql import SQLQuestionTask, SQLSynthesisOp
from DataGovernance.ops.verify import (
    HashDeduplicateFilter,
    MathVerifyFilter,
    SQLExecFilter,
    check_answer,
)

__version__ = "0.1.0"

__all__ = [
    "TaskOp",
    "ChatCompletionsClient",
    "FileStorage",
    "HashDeduplicateFilter",
    "KeyNode",
    "LLMClient",
    "LLMConfig",
    "Lineage",
    "LoadJSONL",
    "MathVerifyFilter",
    "MockLLM",
    "OperatorABC",
    "Pipeline",
    "SQLExecFilter",
    "SQLQuestionTask",
    "SQLSynthesisOp",
    "TextGenerationOp",
    "ULTRAFEEDBACK_ASPECTS",
    "UltraFeedbackOp",
    "__version__",
    "check_answer",
    "get_operator",
    "parse_ultrafeedback",
]
