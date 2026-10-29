"""Feedback and judging operators.

``TaskOp`` runs a built-in recipe or a duck-typed task, ``TextGenerationOp``
writes the text that gets judged, and ``UltraFeedbackOp`` scores one
response on a caller-chosen set of aspects. The default set is
helpfulness, honesty, instruction-following, and truthfulness. Pass
``aspects`` and ``input_context_keys`` for another task, such as math.
Pass ``input_generations_key`` to score a list of responses in one prompt.
``TextGenerationOp(num_generations=N)`` writes that list.
"""

from DataGovernance.ops.feedback.task import (
    DEFAULT_JUDGE_SYSTEM,
    DEFAULT_TEXT_SYSTEM,
    RECIPES,
    TaskOp,
    build_text_generation_chat,
)
from DataGovernance.ops.feedback.text import TextGenerationOp
from DataGovernance.ops.feedback.ultra import (
    ULTRAFEEDBACK_ASPECTS,
    ParsedJudge,
    UltraFeedbackOp,
    build_ultrafeedback_chat,
    normalize_aspects,
    parse_ultrafeedback,
)

__all__ = [
    "DEFAULT_JUDGE_SYSTEM",
    "DEFAULT_TEXT_SYSTEM",
    "ParsedJudge",
    "RECIPES",
    "TaskOp",
    "TextGenerationOp",
    "ULTRAFEEDBACK_ASPECTS",
    "UltraFeedbackOp",
    "build_text_generation_chat",
    "build_ultrafeedback_chat",
    "normalize_aspects",
    "parse_ultrafeedback",
]