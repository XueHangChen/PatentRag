import json

import pytest

from patent_rag.llm.client import (
    DashScopeChatClient,
    LLMProviderError,
    _parse_openai_compatible_chat_completion,
    create_chat_client,
)


def test_parse_openai_compatible_chat_completion() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "这是一个基于证据的回答。",
                    }
                }
            ]
        },
        ensure_ascii=False,
    )

    answer = _parse_openai_compatible_chat_completion(body)

    assert answer == "这是一个基于证据的回答。"


def test_parse_openai_compatible_chat_completion_rejects_invalid_response() -> None:
    with pytest.raises(LLMProviderError):
        _parse_openai_compatible_chat_completion("{}")


def test_create_dashscope_chat_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATENT_RAG_DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")

    with pytest.raises(ValueError, match="DashScope API key is required"):
        create_chat_client(provider="dashscope")


def test_create_dashscope_chat_client_with_explicit_key() -> None:
    client = create_chat_client(
        provider="dashscope",
        model_name="qwen-plus",
        api_key="fake-key",
    )

    assert isinstance(client, DashScopeChatClient)
    assert client.model_name == "qwen-plus"
