"""Registered prompt versions keep one name and several wordings."""

from __future__ import annotations

import pytest

from DataGovernance.core.prompt import Prompt
from DataGovernance.recipes import (
    CURRENT_VERSION,
    load_prompt,
    register_prompt,
    use_prompt,
    versions,
)


def test_shipped_prompt_is_the_current_version():
    prompt = load_prompt("math.judge")
    assert CURRENT_VERSION in versions("math.judge")
    assert "answer" in prompt.input_columns()
    assert load_prompt("math.judge", version=CURRENT_VERSION).user == prompt.user


def test_registered_version_is_opt_in():
    strict = Prompt("Score only the final number.\n{generation}\n{answer}", system="Strict.")
    register_prompt("math.judge", strict, version="strict", llm="judge")
    try:
        assert "strict" in versions("math.judge")
        assert load_prompt("math.judge").user != strict.user
        assert load_prompt("math.judge", version="strict").user == strict.user
        use_prompt("math.judge", "strict")
        assert load_prompt("math.judge").user == strict.user
    finally:
        use_prompt("math.judge", CURRENT_VERSION)


def test_unknown_version_lists_the_known_ones():
    with pytest.raises(KeyError, match="missing"):
        load_prompt("math.judge", version="missing")
