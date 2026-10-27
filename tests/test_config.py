"""LLMConfig.load reads the endpoint URL and API key from a TOML file."""

from __future__ import annotations

from pathlib import Path

import pytest

from DataGovernance.llm.config import DEFAULT_CONFIG_PATH, LLMConfig


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_reads_toml_file(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = " demo "
        base_url = "https://example.test/v1/"
        api_key = " secret "
        timeout = 15
        """,
    )

    config = LLMConfig.load(path)

    assert config.model == "demo"
    assert config.api_key == "secret"
    assert config.base_url == "https://example.test/v1"
    assert config.timeout == 15
    assert config.chat_completions_url == "https://example.test/v1/chat/completions"
    assert config.require_api_key() == "secret"


def test_example_config_ships_without_a_key():
    example = Path(__file__).resolve().parents[1] / "DataGovernance" / "llm" / "config.example.toml"
    config = LLMConfig.load(example)

    assert DEFAULT_CONFIG_PATH.name == "config.toml"
    assert config.api_key is None
    assert config.model
    assert config.base_url.startswith("https://")
    assert config.timeout > 0


def test_blank_api_key_stays_unset(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = "demo"
        base_url = "https://example.test/v1"
        api_key = "  "
        """,
    )

    config = LLMConfig.load(path)

    assert config.api_key is None
    with pytest.raises(RuntimeError, match="config.toml"):
        config.require_api_key()


def test_completions_suffix_in_the_file_is_stripped_once(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = "demo"
        base_url = "https://example.test/v1/chat/completions"
        api_key = "k"
        """,
    )

    config = LLMConfig.load(path)

    assert config.base_url == "https://example.test/v1"
    assert config.chat_completions_url == "https://example.test/v1/chat/completions"


def test_rejects_a_bad_config_file(tmp_path):
    missing = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError, match="not found"):
        LLMConfig.load(missing)

    broken = _write(tmp_path / "broken.toml", "model = [\n")
    with pytest.raises(ValueError, match="not valid TOML"):
        LLMConfig.load(broken)

    no_url = _write(tmp_path / "no-url.toml", 'model = "demo"\n')
    with pytest.raises(ValueError, match="base_url"):
        LLMConfig.load(no_url)

    blank_model = _write(
        tmp_path / "blank.toml",
        'model = "  "\nbase_url = "https://example.test/v1"\n',
    )
    with pytest.raises(ValueError, match="model"):
        LLMConfig.load(blank_model)

    bad_scheme = _write(
        tmp_path / "ftp.toml",
        'model = "demo"\nbase_url = "ftp://example.test"\n',
    )
    with pytest.raises(ValueError, match="base_url"):
        LLMConfig.load(bad_scheme)

    bad_timeout = _write(
        tmp_path / "timeout.toml",
        'model = "demo"\nbase_url = "https://example.test/v1"\ntimeout = 0\n',
    )
    with pytest.raises(ValueError, match="timeout"):
        LLMConfig.load(bad_timeout)
