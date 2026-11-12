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


class RetrievalError(Exception):
    """Base exception for retrieval failures."""


class NoRelevantDocumentsError(RetrievalError):
    """Raised when no relevant documents found."""


class RerankingError(RetrievalError):
    """Raised when reranking fails."""


class AnswerGenerationError(Exception):
    """Base exception for answer generation failures."""


class CitationValidationError(AnswerGenerationError):
    """Raised when citation validation fails."""
