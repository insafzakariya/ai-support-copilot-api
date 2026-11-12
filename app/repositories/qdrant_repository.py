from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from ..config import Settings
from ..exceptions import VectorStoreError


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

    @staticmethod
    def _point_id(document_hash: str, index: int) -> int:
        hash_bytes = hashlib.sha256(f"{document_hash}:{index}".encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")
