"""Embedding models used by vector retrieval.

The default implementation is intentionally local and deterministic so tests
and demos do not depend on external model downloads. It can be replaced by a
real semantic embedding provider without changing the vector store layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from patent_rag.config import get_settings
from patent_rag.retrieval.keyword import tokenize

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"


class EmbeddingProviderError(RuntimeError):
    """Raised when an external embedding provider request fails."""


class EmbeddingModel(Protocol):
    """Minimal interface expected by vector stores."""

    model_name: str
    dimension: int | None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


@dataclass(frozen=True)
class HashingEmbeddingModel:
    """A deterministic local embedding model based on token hashing.

    This is a development baseline, not a true semantic model. It gives us a
    reproducible Chroma pipeline now, while keeping the provider boundary ready
    for BGE/OpenAI embeddings later.
    """

    dimension: int = 384
    model_name: str = "local-hashing-v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.25 if len(token) >= 2 else 1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


@dataclass(frozen=True)
class DashScopeEmbeddingModel:
    """Qwen/DashScope embedding client using an OpenAI-compatible endpoint."""

    api_key: str
    model_name: str = DEFAULT_DASHSCOPE_EMBEDDING_MODEL
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    dimension: int | None = None
    batch_size: int = 10
    timeout_seconds: int = 60

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), max(self.batch_size, 1)):
            batch = texts[start : start + max(self.batch_size, 1)]
            embeddings.extend(self._request_embeddings(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._request_embeddings([text])[0]

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.dimension:
            payload["dimensions"] = self.dimension

        request = Request(
            url=f"{self.base_url.rstrip('/')}/embeddings",
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
            raise EmbeddingProviderError(
                f"DashScope embedding request failed: HTTP {exc.code} {error_body}"
            ) from exc
        except URLError as exc:
            raise EmbeddingProviderError(
                f"DashScope embedding request failed: {exc.reason}"
            ) from exc

        return _parse_openai_compatible_embeddings(body)


def create_embedding_model(
    provider: str | None = None,
    dimension: int | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingModel:
    """Create an embedding model by provider name."""

    settings = get_settings()
    resolved_provider = provider or os.getenv("DASHSCOPE_EMBEDDING_PROVIDER")
    resolved_provider = resolved_provider or settings.embedding_provider
    normalized_provider = resolved_provider.strip().lower()

    if normalized_provider == "hashing":
        return HashingEmbeddingModel(dimension=dimension or settings.embedding_dimension or 384)

    if normalized_provider in {"dashscope", "qwen"}:
        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        resolved_api_key = resolved_api_key or settings.dashscope_api_key
        if not resolved_api_key:
            raise ValueError(
                "DashScope API key is required. Set PATENT_RAG_DASHSCOPE_API_KEY "
                "or DASHSCOPE_API_KEY."
            )
        return DashScopeEmbeddingModel(
            api_key=resolved_api_key,
            model_name=model_name or settings.embedding_model or DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
            base_url=base_url or settings.dashscope_base_url,
            dimension=dimension or settings.embedding_dimension,
        )

    raise ValueError(f"Unsupported embedding provider: {resolved_provider}")


def _parse_openai_compatible_embeddings(body: str) -> list[list[float]]:
    payload = json.loads(body)
    data = payload.get("data")
    if not isinstance(data, list):
        raise EmbeddingProviderError("Embedding response is missing a valid `data` list.")

    sorted_items = sorted(
        data,
        key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
    )
    embeddings: list[list[float]] = []
    for item in sorted_items:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise EmbeddingProviderError("Embedding response contains an invalid item.")
        embeddings.append([float(value) for value in item["embedding"]])
    return embeddings
