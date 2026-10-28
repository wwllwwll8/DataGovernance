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


def test_named_models_keep_their_own_base_url(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = "default-model"
        base_url = "https://example.test/v1"
        api_key = "root-key"
        timeout = 15

        [llm.draft]
        model = "draft-model"
        base_url = "https://draft.example/v1"
        timeout = 20

        [llm.judge]
        model = "judge-model"
        base_url = "https://judge.example/v1/"
        timeout = 30
        """,
    )

    catalog = LLMConfig.load_catalog(path)

    assert catalog.get().model == "default-model"
    assert catalog.get().base_url == "https://example.test/v1"
    assert catalog.get().timeout == 15
    assert catalog.get("draft").model == "draft-model"
    assert catalog.get("draft").base_url == "https://draft.example/v1"
    assert catalog.get("draft").api_key == "root-key"
    assert catalog.get("draft").timeout == 20
    assert catalog.get("judge").model == "judge-model"
    assert catalog.get("judge").base_url == "https://judge.example/v1"
    assert catalog.get("judge").timeout == 30
    with pytest.raises(KeyError, match="missing"):
        catalog.get("missing")


def test_named_model_without_base_url_is_rejected(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = "default-model"
        base_url = "https://example.test/v1"
        api_key = "root-key"

        [llm.draft]
        model = "draft-model"
        timeout = 20
        """,
    )

    with pytest.raises(ValueError, match=r"llm\.draft.*base_url"):
        LLMConfig.load_catalog(path)


def test_named_model_without_timeout_is_rejected(tmp_path):
    path = _write(
        tmp_path / "config.toml",
        """
        model = "default-model"
        base_url = "https://example.test/v1"
        api_key = "root-key"
        timeout = 15

        [llm.draft]
        model = "draft-model"
        base_url = "https://draft.example/v1"
        """,
    )

    with pytest.raises(ValueError, match=r"llm\.draft.*timeout"):
        LLMConfig.load_catalog(path)


def test_example_config_ships_without_a_key():
    example = Path(__file__).resolve().parents[1] / "DataGovernance" / "llm" / "config.example.toml"
    config = LLMConfig.load(example)

    assert DEFAULT_CONFIG_PATH.name == "config.toml"
    assert config.api_key is None
    assert config.model
    assert config.base_url.startswith("https://")
    assert config.timeout > 0
    catalog = LLMConfig.load_catalog(example)
    for name in ("draft", "judge"):
        assert catalog.get(name).base_url.startswith("https://")
        assert catalog.get(name).timeout > 0
    assert catalog.get("judge").model != catalog.get("draft").model


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
