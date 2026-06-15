import json

import pytest

from patent_rag.retrieval.embedding import (
    DashScopeEmbeddingModel,
    EmbeddingProviderError,
    _parse_openai_compatible_embeddings,
    create_embedding_model,
)


def test_parse_openai_compatible_embeddings_keeps_input_order() -> None:
    body = json.dumps(
        {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
    )

    embeddings = _parse_openai_compatible_embeddings(body)

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_parse_openai_compatible_embeddings_rejects_invalid_response() -> None:
    with pytest.raises(EmbeddingProviderError):
        _parse_openai_compatible_embeddings("{}")


def test_create_dashscope_embedding_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATENT_RAG_DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")

    with pytest.raises(ValueError, match="DashScope API key is required"):
        create_embedding_model(provider="dashscope")


def test_create_dashscope_embedding_model_with_explicit_key() -> None:
    model = create_embedding_model(
        provider="dashscope",
        model_name="text-embedding-v4",
        api_key="fake-key",
    )

    assert isinstance(model, DashScopeEmbeddingModel)
    assert model.model_name == "text-embedding-v4"
