import hashlib
import logging
from collections import OrderedDict

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class CachedEmbeddings(Embeddings):
    """Wrapper around embeddings that caches results in memory."""

    def __init__(self, base_embeddings: Embeddings, cache_size: int = 1000):
        """
        Initialize cached embeddings.

        Args:
            base_embeddings: The underlying embedding model
            cache_size: Max number of embeddings to cache (LRU)
        """
        self.base_embeddings = base_embeddings
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query with caching.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check if in cache
        if text in self._cache:
            self.cache_hits += 1
            logger.debug(
                "embedding_cache_hit",
                extra={"text_hash": text_hash, "hits": self.cache_hits},
            )
            # Move to end (LRU)
            self._cache.move_to_end(text)
            return self._cache[text]

        # Cache miss - generate embedding
        try:
            embedding = self.base_embeddings.embed_query(text)
            self.cache_misses += 1

            # Add to cache
            self._cache[text] = embedding

            # Evict oldest if cache is full
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

            logger.debug(
                "embedding_cache_miss",
                extra={"text_hash": text_hash, "misses": self.cache_misses},
            )
            return embedding
        except Exception:
            logger.debug("embedding_generation_failed", extra={"text_hash": text_hash})
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents (usually not cached as documents are unique).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # For bulk document embeddings, bypass cache to avoid memory bloat
        return self.base_embeddings.embed_documents(texts)

    def get_cache_stats(self) -> dict:
        """Get cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate_percent": round(hit_rate, 2),
        }

    def clear_cache(self) -> None:
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("embedding_cache_cleared")
