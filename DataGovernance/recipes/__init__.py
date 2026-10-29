"""Prompt content and the binding from a section to its LLM endpoint."""

from DataGovernance.recipes.loader import (
    CURRENT_VERSION,
    DEFAULT_RECIPES_DIR,
    PromptSection,
    client_for,
    load_prompt,
    load_section,
    register_prompt,
    use_prompt,
    versions,
)

__all__ = [
    "CURRENT_VERSION",
    "DEFAULT_RECIPES_DIR",
    "PromptSection",
    "client_for",
    "load_prompt",
    "load_section",
    "register_prompt",
    "use_prompt",
    "versions",
]