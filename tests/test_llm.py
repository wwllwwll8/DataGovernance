"""ChatCompletionsClient sends the URL and key already resolved by LLMConfig."""

from __future__ import annotations

import json

import pytest

from DataGovernance.llm import ChatCompletionsClient, LLMConfig, ensure_llm


def _ok(content: str) -> tuple[int, bytes]:
    payload = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return 200, json.dumps(payload).encode()


def _config(**overrides: object) -> LLMConfig:
    values: dict[str, object] = {
        "model": "demo",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


def test_generate_posts_each_chat_in_order():
    seen: list[dict] = []

    def transport(url, headers, body, timeout):
        assert url == "https://example.test/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert timeout == 60.0
        payload = json.loads(body)
        seen.append(payload)
        text = payload["messages"][-1]["content"]
        return _ok(text.upper())

    client = ChatCompletionsClient(_config(), transport=transport)
    chats = [
        [{"role": "user", "content": "a"}],
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "b"}],
    ]
    assert client.generate(chats) == ["A", "B"]
    assert [item["messages"][-1]["content"] for item in seen] == ["a", "b"]
    assert seen[0]["model"] == "demo"
    assert ensure_llm(client) is client


def test_empty_batch_does_not_call_transport():
    def transport(url, headers, body, timeout):
        raise AssertionError("empty generate must not send")

    client = ChatCompletionsClient(_config(), transport=transport)
    assert client.generate([]) == []


def test_missing_api_key_does_not_send(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    def transport(url, headers, body, timeout):
        raise AssertionError("missing key must not send")

    client = ChatCompletionsClient(_config(api_key=None), transport=transport)
    with pytest.raises(RuntimeError, match="API key"):
        client.generate([[{"role": "user", "content": "hi"}]])


def test_http_error_and_bad_payload():
    def http_error(url, headers, body, timeout):
        return 401, b'{"error":"nope"}'

    client = ChatCompletionsClient(_config(), transport=http_error)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.generate([[{"role": "user", "content": "hi"}]])

    def not_json(url, headers, body, timeout):
        return 200, b"not-json"

    client = ChatCompletionsClient(_config(), transport=not_json)
    with pytest.raises(RuntimeError, match="not JSON"):
        client.generate([[{"role": "user", "content": "hi"}]])

    def missing(url, headers, body, timeout):
        return 200, b'{"choices":[]}'

    client = ChatCompletionsClient(_config(), transport=missing)
    with pytest.raises(RuntimeError, match="no message content"):
        client.generate([[{"role": "user", "content": "hi"}]])

    def non_string(url, headers, body, timeout):
        return 200, json.dumps({"choices": [{"message": {"content": 1}}]}).encode()

    client = ChatCompletionsClient(_config(), transport=non_string)
    with pytest.raises(TypeError, match="strings"):
        client.generate([[{"role": "user", "content": "hi"}]])


def test_rejects_client_without_llm_config_and_empty_chat():
    with pytest.raises(TypeError, match="LLMConfig"):
        ChatCompletionsClient("demo")  # type: ignore[arg-type]

    def transport(url, headers, body, timeout):
        raise AssertionError("invalid chat must not send")

    client = ChatCompletionsClient(_config(), transport=transport)
    with pytest.raises(ValueError, match="at least one message"):
        client.generate([[]])
    with pytest.raises(TypeError, match="content"):
        client.generate([[{"role": "user"}]])
