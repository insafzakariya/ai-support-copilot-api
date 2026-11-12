class DocumentIngestionError(Exception):
    """Base exception for safe, client-facing ingestion failures."""


class InvalidDocumentError(DocumentIngestionError):
    pass


class DocumentTooLargeError(DocumentIngestionError):
    pass


class EmbeddingProviderError(DocumentIngestionError):
    pass


class VectorStoreError(DocumentIngestionError):
    pass
