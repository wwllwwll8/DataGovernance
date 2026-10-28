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
class LLMCatalog:
    """The default endpoint plus named endpoints from ``[llm.<name>]`` tables.

    A named table sets its own ``model``, ``base_url``, ``timeout``, and
    ``temperature``. It inherits ``api_key`` from the top of the file
    unless it sets it.
    """

    default: LLMConfig
    models: dict[str, LLMConfig]

    def get(self, name: str | None = None) -> LLMConfig:
        """Return the named endpoint, or the default when ``name`` is empty."""
        if name is None or name == "":
            return self.default
        try:
            return self.models[name]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise KeyError(f"Unknown LLM config {name!r}. Known: {known}") from None


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
    temperature: float = 1.0

    def __post_init__(self) -> None:
        model = self.model.strip() if isinstance(self.model, str) else ""
        if not model:
            raise ValueError("model must be a non-empty string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be a positive number")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number from 0 to 2")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be a number from 0 to 2")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(self, "api_key", _blank_to_none(self.api_key))
        object.__setattr__(self, "timeout", float(self.timeout))
        object.__setattr__(self, "temperature", float(self.temperature))

    @classmethod
    def load(cls, path: str | Path | None = None) -> LLMConfig:
        """Read model, base_url, api_key, and timeout from a TOML file.

        ``path`` defaults to ``DataGovernance/llm/config.toml``.
        """
        file = DEFAULT_CONFIG_PATH if path is None else Path(path)
        return cls._from_mapping(_read_toml(file), file)

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

    @classmethod
    def load_catalog(cls, path: str | Path | None = None) -> LLMCatalog:
        """Read the default endpoint and every ``[llm.<name>]`` table."""
        file = DEFAULT_CONFIG_PATH if path is None else Path(path)
        data = _read_toml(file)
        default = cls._from_mapping(data, file)
        tables = data.get("llm", {})
        if tables is None:
            tables = {}
        if not isinstance(tables, dict):
            raise ValueError(f"{file} section llm must be a table of named models")
        models: dict[str, LLMConfig] = {}
        for name, table in tables.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{file} has an empty LLM config name")
            if not isinstance(table, dict):
                raise ValueError(f"{file} llm.{name} must be a table")
            merged = {
                "model": table.get("model", data.get("model")),
                "base_url": table.get("base_url"),
                "api_key": table.get("api_key", data.get("api_key")),
                "timeout": table.get("timeout"),
                "temperature": table.get("temperature"),
            }
            models[name] = cls._from_mapping(
                merged,
                file,
                label=f"llm.{name}",
                required=("model", "base_url", "timeout", "temperature"),
            )
        return LLMCatalog(default, models)

    @classmethod
    def _from_mapping(
        cls,
        data: dict,
        file: Path,
        label: str = "",
        required: tuple[str, ...] = ("model", "base_url"),
    ) -> LLMConfig:
        missing = [key for key in required if key not in data or data[key] in (None, "")]
        if missing:
            where = f"{file} {label}".strip()
            raise ValueError(f"{where} is missing {', '.join(missing)}")
        return cls(
            model=data["model"],
            base_url=data["base_url"],
            api_key=data.get("api_key"),
            timeout=data.get("timeout", 60),
            temperature=data.get("temperature", 1),
        )


def _read_toml(file: Path) -> dict:
    if not file.is_file():
        raise FileNotFoundError(f"LLM config file not found: {file}")
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{file} is not valid TOML") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{file} must be a TOML table")
    return data


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
