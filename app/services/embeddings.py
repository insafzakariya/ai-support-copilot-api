from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import Settings
from ..exceptions import EmbeddingProviderError


def create_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider.lower() != "huggingface":
        raise EmbeddingProviderError("Unsupported embedding provider")
    try:
        print(
            f"Initializing embedding provider: {settings.embedding_provider} with model: {settings.embedding_model}"
        )
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    except Exception as exc:
        raise EmbeddingProviderError(
            "Embedding provider initialization failed"
        ) from exc
