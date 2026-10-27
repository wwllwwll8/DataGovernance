"""Load the online model endpoint from ``config.toml``.

Edit ``DataGovernance/llm/config.toml``. ``LLMConfig.load`` reads that file.
``ChatCompletionsClient`` receives the loaded config and does not read the
environment or another copy of these settings.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class LLMConfig:
    """Chat-completions endpoint settings loaded from a TOML file.

    ``base_url`` is the origin plus optional path, without a trailing
    ``/chat/completions``. ``api_key`` may be missing until ``generate``;
    compile does not need a key.
    """

    model: str
    base_url: str
    api_key: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        model = self.model.strip() if isinstance(self.model, str) else ""
        if not model:
            raise ValueError("model must be a non-empty string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be a positive number")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(self, "api_key", _blank_to_none(self.api_key))
        object.__setattr__(self, "timeout", float(self.timeout))

    @classmethod
    def load(cls, path: str | Path | None = None) -> LLMConfig:
        """Read model, base_url, api_key, and timeout from a TOML file.

        ``path`` defaults to ``DataGovernance/llm/config.toml``.
        """
        file = DEFAULT_CONFIG_PATH if path is None else Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"LLM config file not found: {file}")
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{file} is not valid TOML") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{file} must be a TOML table")
        missing = [key for key in ("model", "base_url") if key not in data]
        if missing:
            raise ValueError(f"{file} is missing {', '.join(missing)}")
        timeout = data.get("timeout", 60)
        return cls(
            model=data["model"],
            base_url=data["base_url"],
            api_key=data.get("api_key"),
            timeout=timeout,
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def require_api_key(self) -> str:
        """Return the key, or raise before any request is sent."""
        if not self.api_key:
            raise RuntimeError(
                "LLMConfig has no API key. Set api_key in DataGovernance/llm/config.toml."
            )
        return self.api_key


def normalize_base_url(base_url: str) -> str:
    """Return an http(s) base URL without a trailing slash or completions suffix."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be an http(s) URL")
    text = base_url.strip().rstrip("/")
    scheme, separator, rest = text.partition("://")
    if separator != "://" or scheme not in {"http", "https"} or not rest or rest.startswith("/"):
        raise ValueError("base_url must be an http(s) URL")
    if text.endswith("/chat/completions"):
        text = text.removesuffix("/chat/completions")
    return text


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("api_key must be a string or None")
    text = value.strip()
    return text or None
