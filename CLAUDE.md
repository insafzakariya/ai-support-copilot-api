# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup & Dependencies**
```bash
# Install with uv (project uses uv for dependency management)
uv sync

# Activate venv
source .venv/bin/activate
```

**Running the Server**
```bash
# Start FastAPI dev server (auto-reloads on file changes)
fastapi dev app/main.py

# Server runs on http://localhost:8000
# Health check: GET /health
# API docs: http://localhost:8000/docs
```

**Testing**
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a single test file
pytest tests/test_document_validation.py

# Run a single test function
pytest tests/test_document_validation.py::test_name

# Run with coverage (if pytest-cov installed)
pytest --cov=app

# Watch mode (re-run on file changes)
pytest-watch

# Run async tests with output
pytest -s -v
```

**Code Quality**
```bash
# Format code with Ruff
ruff format app tests

# Lint with Ruff
ruff check app tests --fix

# Type check with mypy (if installed)
mypy app
```

## Architecture Overview

This is a **Production RAG (Retrieval-Augmented Generation)** application that provides AI-powered customer support ticket analysis. The system ingests documents, indexes them in a vector database, and uses an LLM to generate support responses grounded in company knowledge.

### Core Flow

1. **Document Ingestion Pipeline** → **Vector Indexing** → **Retrieval & Synthesis**
   - User uploads document (PDF/DOCX) via `/api/v1/documents/upload`
   - Document is validated, parsed, normalized, chunked, and embedded
   - Embeddings are indexed in Qdrant
   - When analyzing a support ticket, relevant documents are retrieved and used to generate a response

### Layered Architecture

**API Layer** (`app/api/`)
- `documents.py`: FastAPI router handling document uploads with dependency injection
- Uses `DocumentIngestionService` to orchestrate the pipeline

**Application/Service Layer** (`app/services/`)
- `document_ingestion.py`: Main orchestrator for the ingestion pipeline (reads → validates → parses → chunks → embeds → indexes)
- `document_validation.py`: File size, type, and content validation
- `document_parsing.py`: Extracts text from PDF and DOCX files
- `text_processing.py`: Uses `RecursiveCharacterTextSplitter` for semantic chunking
- `embeddings.py`: Creates embeddings using HuggingFace sentence transformers

**Infrastructure Layer** (`app/repositories/`)
- `qdrant_repository.py`: Encapsulates all Qdrant vector store operations (collection management, indexing, retrieval)
- Handles idempotent ingestion via document hashing

**Domain/Config** (`app/`)
- `config.py`: Settings management with Pydantic (loaded from `.env`)
- `schemas.py`: Pydantic models for request/response validation
- `exceptions.py`: Custom exception hierarchy
- `ai_pipeline.py`: LLM chain orchestration (uses LangChain)
- `knowledge_base.py`: Document retrieval (currently a simple in-memory implementation for testing)

### Key Design Patterns

**Dependency Injection**: Services are instantiated in API endpoints via `Depends()` with settings injected
```python
def get_ingestion_service(settings: Settings = Depends(get_settings)) -> DocumentIngestionService:
    return DocumentIngestionService(settings, QdrantRepository(settings))
```

**Idempotent Ingestion**: Documents are deduplicated using SHA256 hashing on the content
- Same document uploaded twice = skipped (already indexed)
- Allows safe retries without duplication

**Structured LLM Output**: Uses LangChain's `with_structured_output()` to ensure responses match `TicketAnalysis` schema

**Repository Pattern**: `QdrantRepository` encapsulates all vector store operations, making it replaceable

### Configuration

Settings in `app/config.py` are loaded from `.env`:
- `groq_api_key`: LLM API key
- `groq_model`: LLM model name (default: `llama-3.3-70b-versatile`)
- `qdrant_url`: Vector database endpoint (default: `http://localhost:6333`)
- `qdrant_collection`: Collection name for documents (default: `support-documents`)
- `embedding_model`: HuggingFace model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `chunk_size`: Text chunk size for splitting (default: 1200)
- `chunk_overlap`: Overlap between chunks (default: 150)
- `max_upload_size_bytes`: Max file size (default: 10MB)
- `max_pdf_pages`: Max pages in a PDF (default: 200)

## Testing

Tests use `pytest` with async fixtures via `pytest-asyncio`.

**Test Organization**
- `tests/conftest.py`: Shared fixtures (settings factory)
- `tests/test_document_validation.py`: File validation and parsing
- `tests/test_document_upload.py`: Full ingestion flow
- `tests/test_qdrant_repository.py`: Vector store operations
- `tests/test_text_processing.py`: Text chunking

**Running Tests**: When adding features, run tests locally first:
```bash
pytest -v              # All tests
pytest tests/test_document_validation.py -v  # Specific file
```

**Test Fixtures**: The `settings` fixture in `conftest.py` provides minimal overrides for testing (1KB max upload, 20-char chunks). Use it or extend it.

## Important Constraints & Patterns (from SKILL.md)

- **No silent failures**: All exceptions are logged and propagated
- **Type hints required**: All functions must have type annotations
- **Pydantic validation**: Use schemas for all I/O
- **Async where applicable**: Document ingestion is async; ensure tests handle async properly
- **Security**: Uploaded files are treated as untrusted; validate type, size, and content
- **Prompt injection defense**: Knowledge base context is assembled, never user-controlled
- **No fabricated citations**: Citations must map to actual source documents

## Common Modifications

**Adding a New Document Type** (e.g., `.txt`)
1. Update `document_validation.py` to allow the extension
2. Add a parser in `document_parsing.py` (implement `load_documents` case)
3. Test with `tests/test_document_validation.py`

**Changing Chunk Strategy**
- Modify `chunk_size` and `chunk_overlap` in `.env` or `Settings`
- Chunk strategy is in `text_processing.py` (currently `RecursiveCharacterTextSplitter`)

**Switching Vector Database** (away from Qdrant)
- Replace `QdrantRepository` with a new implementation (e.g., `PineconeRepository`)
- Update the repository in `get_ingestion_service()` dependency

**Changing Embedding Model**
- Update `embedding_model` in `.env` or `Settings`
- Qdrant will auto-detect vector dimensions on first upsert

**Using a Different LLM** (away from Groq)
- Replace `ChatGroq` in `ai_pipeline.py` with another LangChain LLM (e.g., `ChatOpenAI`)
- Adjust the prompt template if needed

## Debugging

**Enable Structured Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Qdrant Collection Status**
```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
print(client.get_collection("support-documents"))
```

**Test a Single Ingestion**
```bash
pytest tests/test_document_upload.py::test_upload_simple_pdf -v -s
```

**Trace LangChain Execution**
```python
import langchain
langchain.debug = True
```
