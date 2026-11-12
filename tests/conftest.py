import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        max_upload_size_bytes=1024,
        chunk_size=20,
        chunk_overlap=5,
    )
