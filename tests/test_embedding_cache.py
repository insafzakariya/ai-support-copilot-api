import pytest
from app.services.embedding_cache import CachedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


@pytest.fixture
def base_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def cached_embeddings(base_embeddings):
    return CachedEmbeddings(base_embeddings, cache_size=10)


def test_embedding_cache_hit(cached_embeddings):
    """Test that same question returns cached embedding."""
    question = "What is machine learning?"

    # First call - cache miss
    embedding1 = cached_embeddings.embed_query(question)
    stats_before = cached_embeddings.get_cache_stats()
    assert stats_before["hits"] == 0
    assert stats_before["misses"] == 1

    # Second call - cache hit
    embedding2 = cached_embeddings.embed_query(question)
    stats_after = cached_embeddings.get_cache_stats()
    assert stats_after["hits"] == 1
    assert stats_after["misses"] == 1

    # Embeddings should be identical
    assert embedding1 == embedding2


def test_embedding_cache_miss_different_questions(cached_embeddings):
    """Test that different questions generate different embeddings."""
    q1 = "What is machine learning?"
    q2 = "What is deep learning?"

    emb1 = cached_embeddings.embed_query(q1)
    emb2 = cached_embeddings.embed_query(q2)

    stats = cached_embeddings.get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2

    # Embeddings should be different
    assert emb1 != emb2


def test_cache_hit_rate(cached_embeddings):
    """Test cache hit rate calculation."""
    # 3 unique questions
    cached_embeddings.embed_query("Question 1")
    cached_embeddings.embed_query("Question 2")
    cached_embeddings.embed_query("Question 3")

    # Ask same questions again (3 hits)
    cached_embeddings.embed_query("Question 1")
    cached_embeddings.embed_query("Question 2")
    cached_embeddings.embed_query("Question 3")

    stats = cached_embeddings.get_cache_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 3
    assert stats["total"] == 6
    assert stats["hit_rate_percent"] == 50.0


def test_clear_cache(cached_embeddings):
    """Test clearing the cache."""
    cached_embeddings.embed_query("Test question")
    stats = cached_embeddings.get_cache_stats()
    assert stats["hits"] + stats["misses"] > 0

    cached_embeddings.clear_cache()
    stats_after = cached_embeddings.get_cache_stats()
    assert stats_after["hits"] == 0
    assert stats_after["misses"] == 0
