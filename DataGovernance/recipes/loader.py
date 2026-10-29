"""Prompt recipes loaded from per-domain TOML files.

Edit the TOML files in this directory: ``math.toml``, ``nl2sql.toml``.
A dotted section name such as ``math.judge`` resolves to the file named
after its first part. That shipped text is version ``current``.
``register_prompt`` adds another wording of the same name, and
``use_prompt`` or ``load_prompt(..., version=)`` selects it.
``{column}`` names are columns. Names listed under ``parameters`` are
filled by the operator. ``llm`` names an ``[llm.<name>]`` table in
``llm/config.toml``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from DataGovernance.core.prompt import Prompt
from DataGovernance.llm import ChatCompletionsClient
from DataGovernance.llm.config import LLMConfig

DEFAULT_RECIPES_DIR = Path(__file__).parent
CURRENT_VERSION = "current"

_PROMPT_FIELDS = {"user", "system", "parameters"}
_versions: dict[str, dict[str, PromptSection]] = {}
_active: dict[str, str] = {}
_builtins_loaded = False


@dataclass(frozen=True)
class PromptSection:
    """One ``[section]`` table and the prompt parsed from it."""

    name: str
    prompt: Prompt
    values: dict[str, Any]


def register_prompt(
    name: str,
    prompt: Prompt,
    *,
    version: str,
    llm: str | None = None,
    **values: Any,
) -> PromptSection:
    """Store another wording of ``name`` without changing the active one.

    The text shipped in ``recipes/*.toml`` is version ``current``. A later
    ``register_prompt(..., version="strict")`` keeps ``current`` active
    until ``use_prompt`` selects ``strict``.
    """
    _ensure_builtins()
    _check_name(name)
    _check_version(version)
    if not isinstance(prompt, Prompt):
        raise TypeError("register_prompt expects a Prompt")
    stored = dict(values)
    if llm is not None:
        if not isinstance(llm, str) or not llm.strip():
            raise ValueError("llm must be a config name")
        stored["llm"] = llm.strip()
    section = PromptSection(name, prompt, stored)
    _put(name, version, section)
    return section


def use_prompt(name: str, version: str) -> PromptSection:
    """Make ``version`` the wording ``load_prompt(name)`` returns."""
    _ensure_builtins()
    bucket = _bucket(name)
    _check_version(version)
    if version not in bucket:
        known = ", ".join(sorted(bucket))
        raise KeyError(f"Unknown prompt version {name!r} {version!r}. Known: {known}")
    _active[name] = version
    return bucket[version]


def versions(name: str) -> tuple[str, ...]:
    """Return the registered versions of ``name``, in sorted order."""
    _ensure_builtins()
    return tuple(sorted(_bucket(name)))


def client_for(section: str, *, transport=None, catalog=None, version: str | None = None):
    """Build the chat client named by ``llm`` in that prompt section.

    An empty ``llm`` value uses the top-level model in ``config.toml``.
    ``version`` selects a registered wording. The default is the active one.
    """
    spec = load_section(section, version=version)
    name = spec.values.get("llm")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{section} llm must be a config name")
    chosen = None if name is None else name.strip()
    configs = catalog if catalog is not None else LLMConfig.load_catalog()
    return ChatCompletionsClient(configs.get(chosen), transport=transport)


def load_prompt(name: str, path: str | Path | None = None, *, version: str | None = None) -> Prompt:
    """Return the prompt for ``name``, such as ``math.judge``.

    With no ``version``, this is the active wording. ``current`` is the
    text in the recipe file.
    """
    return load_section(name, path, version=version).prompt


def load_section(
    name: str,
    path: str | Path | None = None,
    *,
    version: str | None = None,
) -> PromptSection:
    """Return one prompt section.

    ``path`` reads that file directly and does not use the registry.
    ``version`` selects a registered wording of the shipped name.
    """
    if path is not None and version is not None:
        raise ValueError("Pass path or version, not both")
    if path is not None:
        return _load_from_file(name, path)
    _ensure_builtins()
    bucket = _bucket(name)
    chosen = _active[name] if version is None else version
    if chosen not in bucket:
        known = ", ".join(sorted(bucket))
        raise KeyError(f"Unknown prompt version {name!r} {chosen!r}. Known: {known}")
    return bucket[chosen]


def _ensure_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for file in sorted(DEFAULT_RECIPES_DIR.glob("*.toml")):
        data = _read_toml(file)
        for name in _prompt_names(data):
            table = _walk(data, name.split("."), file)
            _put(name, CURRENT_VERSION, _section_from_table(name, table, file))


def _put(name: str, version: str, section: PromptSection) -> None:
    bucket = _versions.setdefault(name, {})
    bucket[version] = section
    _active.setdefault(name, version)


def _bucket(name: str) -> dict[str, PromptSection]:
    _check_name(name)
    try:
        return _versions[name]
    except KeyError:
        known = ", ".join(sorted(_versions)) or "(none)"
        raise KeyError(f"Unknown prompt {name!r}. Known: {known}") from None


def _check_name(name: str) -> None:
    parts = name.split(".")
    if not parts or not all(part.strip() for part in parts):
        raise ValueError(f"section name must be dot-separated, got {name!r}")


def _check_version(version: str) -> None:
    if not isinstance(version, str) or not version.strip():
        raise ValueError("prompt version must be a non-empty string")


def _load_from_file(name: str, path: str | Path) -> PromptSection:
    parts = name.split(".")
    _check_name(name)
    file = _recipe_file(parts[0], path)
    data = _read_toml(file)
    return _section_from_table(name, _walk(data, parts, file), file)


def _section_from_table(name: str, section: object, file: Path) -> PromptSection:
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
    values = {key: value for key, value in section.items() if key not in _PROMPT_FIELDS}
    return PromptSection(name, Prompt(user.strip(), system, parameters=parameters), values)


def _prompt_names(data: dict, prefix: str = "") -> list[str]:
    names: list[str] = []
    if isinstance(data.get("user"), str):
        names.append(prefix)
    for key, value in data.items():
        if key in _PROMPT_FIELDS or not isinstance(value, dict):
            continue
        child = f"{prefix}.{key}" if prefix else key
        names.extend(_prompt_names(value, child))
    return names


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