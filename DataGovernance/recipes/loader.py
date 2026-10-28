"""Prompt recipes loaded from per-domain TOML files.

Edit the TOML files in this directory: ``math.toml``, ``nl2sql.toml``.
A dotted section name such as ``math.judge`` resolves to the file named
after its first part. ``{column}`` names are columns. Names listed under
``parameters`` are filled by the operator. ``llm`` names an
``[llm.<name>]`` table in ``llm/config.toml``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from DataGovernance.core.prompt import Prompt

DEFAULT_RECIPES_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptSection:
    """One ``[section]`` table and the prompt parsed from it."""

    name: str
    prompt: Prompt
    values: dict[str, Any]


def client_for(section: str, *, transport=None, catalog=None):
    """Build the chat client named by ``llm`` in that prompt section.

    An empty ``llm`` value uses the top-level model in ``config.toml``.
    """
    from DataGovernance.llm import ChatCompletionsClient
    from DataGovernance.llm.config import LLMConfig

    spec = load_section(section)
    name = spec.values.get("llm")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{section} llm must be a config name")
    chosen = None if name is None else name.strip()
    configs = catalog if catalog is not None else LLMConfig.load_catalog()
    return ChatCompletionsClient(configs.get(chosen), transport=transport)


def load_prompt(name: str, path: str | Path | None = None) -> Prompt:
    """Return the prompt for ``name``, such as ``math.judge``."""
    return load_section(name, path).prompt


def load_section(name: str, path: str | Path | None = None) -> PromptSection:
    """Read one dotted section from a recipe file.

    ``path`` is the recipe directory (default: this package). The first
    part of ``name`` picks the file: ``math.judge`` reads ``math.toml``
    and walks the full dotted path inside it.
    """
    parts = name.split(".")
    if not all(part.strip() for part in parts):
        raise ValueError(f"section name must be dot-separated, got {name!r}")
    file = _recipe_file(parts[0], path)
    data = _read_toml(file)
    section = _walk(data, parts, file)
    if not isinstance(section, dict):
        raise ValueError(f"{file} section {name} must be a table")
    user = section.get("user")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"{file} section {name} needs a user prompt")
    system = section.get("system", "")
    if not isinstance(system, str):
        raise ValueError(f"{file} section {name} system must be a string")
    parameters = section.get("parameters", [])
    if not isinstance(parameters, list) or not all(isinstance(item, str) and item for item in parameters):
        raise ValueError(f"{file} section {name} parameters must be a list of names")
    values = {
        key: value
        for key, value in section.items()
        if key not in {"user", "system", "parameters"}
    }
    return PromptSection(
        name,
        Prompt(user.strip(), system, parameters=parameters),
        values,
    )


def _recipe_file(domain: str, path: str | Path | None) -> Path:
    base = DEFAULT_RECIPES_DIR if path is None else Path(path)
    if base.is_file():
        return base
    file = base / f"{domain}.toml"
    if not file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {file}")
    return file


def _read_toml(file: Path) -> dict:
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{file} is not valid TOML") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{file} must be a TOML table")
    return data


def _walk(data: object, keys: list[str], file: Path) -> object:
    current = data
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"{file} is missing section {'.'.join(walked)}")
        current = current[key]
    return current