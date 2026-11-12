from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from ..config import Settings
from ..exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantRepository:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )

    def index(
        self,
        document_id: str,
        document_hash: str,
        filename: str,
        extension: str,
        chunks: list[Document],
        embeddings: Embeddings,
    ) -> tuple[int, bool]:
        try:
            vectors = embeddings.embed_documents(
                [chunk.page_content for chunk in chunks]
            )
            if not vectors:
                raise VectorStoreError("No embeddings were generated")
            self._ensure_collection(len(vectors[0]))
            point_ids = [
                self._point_id(document_hash, index) for index in range(len(chunks))
            ]
            existing = self.client.retrieve(
                collection_name=self.settings.qdrant_collection,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            if len(existing) == len(point_ids):
                return len(chunks), True

            points = []
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                payload: dict[str, Any] = {
                    "document_id": document_id,
                    "document_hash": document_hash,
                    "source_filename": filename,
                    "source_extension": extension,
                    "text": chunk.page_content,
                    "chunk_index": index,
                    "ingestion_version": 1,
                    **chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=point_ids[index],
                        vector={self.settings.qdrant_vector_name: vector},
                        payload=payload,
                    )
                )
            self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
                wait=True,
            )
            return len(chunks), False
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Vector indexing failed") from exc

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            collections = self.client.get_collections().collections
            names = {collection.name for collection in collections}
            print(f"Existing collections: {names}")
            if self.settings.qdrant_collection not in names:
                self.client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config={
                        self.settings.qdrant_vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
                return
            info = self.client.get_collection(self.settings.qdrant_collection)
            configured = info.config.params.vectors
            if isinstance(configured, dict):
                configured = configured.get(self.settings.qdrant_vector_name)
            size = configured.size if hasattr(configured, "size") else None
            if size != vector_size:
                raise VectorStoreError(
                    "Qdrant vector dimension does not match embeddings"
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant collection is unavailable") from exc

    def truncate(self) -> None:
        try:
            collections = self.client.get_collections().collections
            names = {c.name for c in collections}
            if self.settings.qdrant_collection in names:
                self.client.delete(
                    collection_name=self.settings.qdrant_collection,
                    points_selector=models.FilterSelector(filter=models.Filter()),
                )
        except Exception as exc:
            raise VectorStoreError("Failed to truncate vector store") from exc

    def search_semantic(
        self,
        embedding: list[float],
        top_k: int,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Vector similarity search using cosine distance.
        Returns LangChain Documents with preserved metadata.
        """
        try:
            query_filter = None
            if filter_dict:
                query_filter = self._build_filter(filter_dict)

            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=embedding,
                using=self.settings.qdrant_vector_name,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            documents = []
            for point in response.points:
                payload = point.payload
                doc = Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        "document_id": payload.get("document_id", ""),
                        "source_filename": payload.get("source_filename", ""),
                        "semantic_score": float(point.score) if point.score else 0.0,
                        "chunk_index": payload.get("chunk_index", 0),
                        **{
                            k: v
                            for k, v in payload.items()
                            if k
                            not in {
                                "text",
                                "document_id",
                                "source_filename",
                                "chunk_index",
                            }
                        },
                    },
                )
                documents.append(doc)

            logger.debug(
                "semantic_search_completed",
                extra={"top_k": top_k, "results_count": len(documents)},
            )
            return documents
        except Exception as exc:
            logger.exception("semantic_search_failed")
            raise VectorStoreError("Semantic search failed") from exc

    def search_keyword(
        self,
        keywords: list[str],
        top_k: int,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Keyword-based search using TF scoring.
        Returns LangChain Documents with keyword scores.
        """
        try:
            query_filter = None
            if filter_dict:
                query_filter = self._build_filter(filter_dict)

            scored_docs: list[tuple[Document, float]] = []
            offset = None

            while True:
                points, offset = self.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    scroll_filter=query_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload
                    text = payload.get("text", "").lower()

                    keyword_score = sum(
                        1 for keyword in keywords if keyword.lower() in text
                    )

                    if keyword_score > 0:
                        doc = Document(
                            page_content=payload.get("text", ""),
                            metadata={
                                "document_id": payload.get("document_id", ""),
                                "source_filename": payload.get(
                                    "source_filename", ""
                                ),
                                "keyword_score": float(keyword_score),
                                "chunk_index": payload.get("chunk_index", 0),
                                **{
                                    k: v
                                    for k, v in payload.items()
                                    if k
                                    not in {
                                        "text",
                                        "document_id",
                                        "source_filename",
                                        "chunk_index",
                                    }
                                },
                            },
                        )
                        scored_docs.append((doc, keyword_score))

                if offset is None:
                    break

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            documents = [doc for doc, _ in scored_docs[:top_k]]

            logger.debug(
                "keyword_search_completed",
                extra={"keywords": keywords, "results_count": len(documents)},
            )
            return documents
        except Exception as exc:
            logger.exception("keyword_search_failed")
            raise VectorStoreError("Keyword search failed") from exc

    @staticmethod
    def _build_filter(filter_dict: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                models.HasPayloadCondition(key=key, has_payload=True)
            )
        return models.Filter(must=conditions) if conditions else models.Filter()

    @staticmethod
    def _point_id(document_hash: str, index: int) -> int:
        hash_bytes = hashlib.sha256(f"{document_hash}:{index}".encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")
