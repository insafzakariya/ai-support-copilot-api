from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    frontend_orgin: str = "http://localhost:3000"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "support-documents"
    qdrant_vector_name: str = "text"
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    max_extracted_text_chars: int = 2_000_000
    max_pdf_pages: int = 200
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # Retrieval configuration
    reranker_enabled: bool = True
    top_k_default: int = 5
    top_k_max: int = 20
    confidence_threshold_default: float = 0.3

    # Question constraints
    question_min_length: int = 3
    question_max_length: int = 2000

    # RRF configuration
    rrf_k: int = 60

    # Reranker model
    reranker_model: str = "llama-3.3-70b-versatile"
    reranker_temperature: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
