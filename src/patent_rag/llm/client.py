"""LLM clients used by RAG services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from patent_rag.config import get_settings
from patent_rag.retrieval.embedding import DEFAULT_DASHSCOPE_BASE_URL


class LLMProviderError(RuntimeError):
    """Raised when an external LLM provider request fails."""


class ChatMessage(BaseModel):
    """A chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatClient(Protocol):
    """Minimal interface expected by RAG services."""

    model_name: str

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate an answer from chat messages."""


@dataclass(frozen=True)
class DashScopeChatClient:
    """Qwen/DashScope chat client using an OpenAI-compatible endpoint."""

    api_key: str
    model_name: str = "qwen-plus"
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_seconds: int = 90

    def generate(self, messages: list[ChatMessage]) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError(
                f"DashScope chat request failed: HTTP {exc.code} {error_body}"
            ) from exc
        except URLError as exc:
            raise LLMProviderError(f"DashScope chat request failed: {exc.reason}") from exc

        return _parse_openai_compatible_chat_completion(body)


def create_chat_client(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatClient:
    """Create a chat client by provider name."""

    settings = get_settings()
    resolved_provider = provider or settings.llm_provider
    normalized_provider = resolved_provider.strip().lower()

    if normalized_provider in {"dashscope", "qwen"}:
        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        resolved_api_key = resolved_api_key or settings.dashscope_api_key
        if not resolved_api_key:
            raise ValueError(
                "DashScope API key is required. Set PATENT_RAG_DASHSCOPE_API_KEY "
                "or DASHSCOPE_API_KEY."
            )
        return DashScopeChatClient(
            api_key=resolved_api_key,
            model_name=model_name or settings.llm_model,
            base_url=base_url or settings.dashscope_base_url,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def _parse_openai_compatible_chat_completion(body: str) -> str:
    payload = json.loads(body)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderError("Chat response is missing a valid `choices` list.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMProviderError("Chat response contains an invalid choice.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("Chat response choice is missing `message`.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("Chat response message is missing text content.")
    return content.strip()
