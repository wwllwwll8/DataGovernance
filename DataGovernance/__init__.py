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
from DataGovernance.core.prompt import Prompt
from DataGovernance.core.stages import GenerateOp, RankOp, ScoreOp
from DataGovernance.core.storage import FileStorage
from DataGovernance.llm import ChatCompletionsClient, LLMClient, LLMConfig, MockLLM
from DataGovernance.ops.feedback import (
    ULTRAFEEDBACK_ASPECTS,
    TaskOp,
    TextGenerationOp,
    UltraFeedbackOp,
    parse_ultrafeedback,
)
from DataGovernance.ops.load import LoadJSONL
from DataGovernance.ops.dedupe import HashDeduplicateFilter
from DataGovernance.ops.math import MATH_ASPECTS, MathVerifyFilter, check_answer, math_feedback
from DataGovernance.ops.nl2sql import SQLExecFilter, SQLQuestionTask, SQLSynthesisOp

__version__ = "0.1.0"

__all__ = [
    "TaskOp",
    "ChatCompletionsClient",
    "FileStorage",
    "GenerateOp",
    "HashDeduplicateFilter",
    "KeyNode",
    "LLMClient",
    "LLMConfig",
    "Lineage",
    "LoadJSONL",
    "MATH_ASPECTS",
    "MathVerifyFilter",
    "MockLLM",
    "OperatorABC",
    "Pipeline",
    "Prompt",
    "RankOp",
    "ScoreOp",
    "SQLExecFilter",
    "SQLQuestionTask",
    "SQLSynthesisOp",
    "TextGenerationOp",
    "ULTRAFEEDBACK_ASPECTS",
    "UltraFeedbackOp",
    "__version__",
    "check_answer",
    "get_operator",
    "math_feedback",
    "parse_ultrafeedback",
]
