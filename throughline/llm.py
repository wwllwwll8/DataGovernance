"""Pluggable chat LLM client used by Distilabel-style task operators.

A client accepts a batch of chats and returns one completion per chat, in
order. ``MockLLM`` is deterministic and does not open a network connection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol


Chat = Sequence[Mapping[str, str]]
Responder = Callable[[Sequence[Mapping[str, str]]], str]


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


def ensure_llm(llm: LLMClient) -> LLMClient:
    """Reject objects that do not implement the client protocol."""
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        raise TypeError("LLM client must provide generate(chats) -> list[str]")
    return llm


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
