"""Pluggable chat LLM client used by task operators.

A client accepts a batch of chats and returns one completion per chat, in
order. ``MockLLM`` is deterministic and does not open a network connection.
``ChatCompletionsClient`` posts each chat to an OpenAI-compatible
``/chat/completions`` endpoint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from DataGovernance.llm.config import LLMConfig


Chat = Sequence[Mapping[str, str]]
Responder = Callable[[Sequence[Mapping[str, str]]], str]
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


class LLMClient(Protocol):
    """Minimal generate() contract shared by mock and real backends.

    ``chats`` is a batch. Each chat is a list of ``{"role", "content"}``
    messages. The method returns one string per chat, in the same order.
    """

    def generate(self, chats: Sequence[Chat]) -> Sequence[str]:
        """Return one completion for each chat."""


class MockLLM:
    """Offline stand-in for an LLM client.

    Pass ``responses`` to replay a script (one string per chat, across
    batches), or ``responder`` to compute a completion from the chat.
    With neither, each completion is ``mock-response-{n}``.
    """

    model = "mock"

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Responder | None = None,
    ) -> None:
        if responses is not None and responder is not None:
            raise ValueError("Pass responses or responder, not both")
        self._responses = list(responses) if responses is not None else None
        self._responder = responder
        self._cursor = 0
        self.calls: list[list[list[dict[str, str]]]] = []

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        batch = [_copy_chat(chat) for chat in chats]
        self.calls.append(batch)
        if self._responder is not None:
            outputs = [self._responder(chat) for chat in batch]
        elif self._responses is None:
            outputs = [f"mock-response-{index}" for index in range(len(batch))]
        else:
            outputs = []
            for _ in batch:
                if self._cursor >= len(self._responses):
                    raise RuntimeError("MockLLM ran out of scripted responses")
                outputs.append(self._responses[self._cursor])
                self._cursor += 1
        for text in outputs:
            if not isinstance(text, str):
                raise TypeError("LLM generate() must return strings")
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"LLM returned {len(outputs)} completions for {len(batch)} chats"
            )
        return outputs


class ChatCompletionsClient:
    """Online client for an OpenAI-compatible chat completions API.

    Pass an ``LLMConfig``. This client posts to
    ``config.chat_completions_url`` and does not read environment
    variables. Nothing is sent until ``generate`` runs.

    Pass ``transport`` in tests. The default transport is ``urllib``.
    """

    def __init__(self, config: LLMConfig, *, transport: Transport | None = None) -> None:
        if not isinstance(config, LLMConfig):
            raise TypeError("ChatCompletionsClient expects an LLMConfig")
        self.config = config
        self.model = config.model
        self.base_url = config.base_url
        self.timeout = config.timeout
        self._transport = transport or _urllib_post

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        batch = [_copy_chat(chat) for chat in chats]
        if not batch:
            return []
        for chat in batch:
            if not chat:
                raise ValueError("Each chat needs at least one message")
        key = self.config.require_api_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        outputs: list[str] = []
        for chat in batch:
            body = json.dumps(
                {"model": self.model, "messages": chat},
                ensure_ascii=False,
            ).encode("utf-8")
            status, raw = self._transport(
                self.config.chat_completions_url,
                headers,
                body,
                self.timeout,
            )
            outputs.append(_completion_text(status, raw))
        return outputs


def ensure_llm(llm: LLMClient) -> LLMClient:
    """Reject objects that do not implement the client protocol."""
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        raise TypeError("LLM client must provide generate(chats) -> list[str]")
    return llm


def _completion_text(status: int, raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if status < 200 or status >= 300:
        raise RuntimeError(f"Chat completions HTTP {status}: {text[:500]}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Chat completions response was not JSON") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Chat completions response has no message content") from exc
    if not isinstance(content, str):
        raise TypeError("LLM generate() must return strings")
    return content


def _urllib_post(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            return int(status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat completions request failed: {exc.reason}") from exc


def _copy_chat(chat: Chat) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for message in chat:
        if "content" not in message:
            raise TypeError("Each chat message needs a content field")
        copied.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message["content"]),
            }
        )
    return copied
