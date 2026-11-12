from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..repositories.qdrant_repository import QdrantRepository
from .document_parsing import load_documents, normalize_documents
from .document_validation import read_upload
from .embeddings import create_embeddings
from .text_processing import split_documents


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        repository: QdrantRepository,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.embeddings = embeddings

    async def ingest(self, upload) -> dict[str, object]:
        filename, extension, content = await read_upload(upload, self.settings)
        document_hash = hashlib.sha256(content).hexdigest()
        document_id = f"doc_{document_hash[:24]}"
        documents = normalize_documents(
            load_documents(content, extension, self.settings)
        )
        chunks = split_documents(documents, self.settings)
        print(f"Document {filename} split into {len(chunks)} chunks")
        embeddings = self.embeddings or create_embeddings(self.settings)
        print(f"Using embedding model: {self.settings.embedding_model}")
        total_chunks, already_indexed = self.repository.index(
            document_id, document_hash, filename, extension, chunks, embeddings
        )
        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "total_chunks": total_chunks,
            "status": "already_indexed" if already_indexed else "indexed",
            "message": (
                "Document was already indexed"
                if already_indexed
                else "Document successfully uploaded and indexed"
            ),
        }
