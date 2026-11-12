# Ask Endpoint Implementation Plan

## Overview

This document provides a comprehensive implementation plan for building a high-quality **Ask endpoint** with advanced retrieval strategies, security measures, and accurate source citations.

The endpoint will accept user questions, retrieve the most relevant information from the Qdrant vector database using hybrid search with Reciprocal Rank Fusion (RRF), rerank results for improved relevance, and generate well-grounded answers using an LLM.

---

## 1. Architecture Overview

### 1.1 Request Flow

```
User Question
    ↓
[Input Validation & Sanitization]
    ↓
[Query Embedding Generation]
    ↓
[Hybrid Retrieval: Semantic + Keyword Search with RRF]
    ↓
[Reranking for Relevance]
    ↓
[Context Assembly & Citation Extraction]
    ↓
[LLM Answer Generation with Safeguards]
    ↓
[Output Validation & Citation Verification]
    ↓
Response with Answer + Citations + Confidence
```

### 1.2 Layered Architecture

**API Layer** (`app/api/ask.py`)
- FastAPI endpoint handling requests with dependency injection
- Input validation and error handling

**Application/Service Layer** (`app/services/`)
- `retrieval.py`: Orchestrates semantic search, keyword search, hybrid fusion, reranking
- `answer.py`: LLM-based answer generation and citation extraction

**Infrastructure Layer** (`app/repositories/`)
- `qdrant_repository.py`: Extends existing repo with retrieval methods

**Domain/Config** (`app/`)
- `schemas.py`: Pydantic models for requests/responses
- `exceptions.py`: Custom exceptions for retrieval/answer failures
- `config.py`: Settings for retrieval configuration

---

## 2. Retrieval Strategy

### 2.1 Semantic Search

**Method**: Vector similarity search in Qdrant
- **Embedding Model**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Query Process**:
  1. Generate embedding for user question using same model
  2. Vector search in Qdrant using cosine distance
  3. Return top-k documents ranked by similarity score (0-1)

**Implementation**: `RetrievalService.semantic_search(question, top_k)`

### 2.2 Keyword Search

**Method**: Term matching across Qdrant payload text
- **Keyword Extraction**:
  1. Split question into tokens
  2. Filter tokens by length (min 3 characters)
  3. Convert to lowercase

- **Scoring**:
  1. Search Qdrant payloads for keyword presence
  2. Calculate TF-IDF-like score per document
  3. Normalize scores to 0-1 range

**Implementation**: `RetrievalService.keyword_search(question, top_k)`

### 2.3 Hybrid Search with Reciprocal Rank Fusion (RRF)

**Method**: Combine semantic and keyword rankings
- **RRF Formula**: `score = 1/(k + rank)` where k=60
- **Process**:
  1. Run semantic_search and keyword_search in parallel
  2. For each result, calculate RRF score from both methods
  3. Merge results, deduplicating by document_id
  4. Re-rank by combined score
  5. Return top-k merged results

**Benefit**: Captures both semantic meaning (vector similarity) and literal matches (keyword relevance)

**Implementation**: `RetrievalService.hybrid_search(question, top_k, include_reranking)`

### 2.4 Reranking

**Method**: LLM-based relevance scoring
- **Process**:
  1. Pass question + candidate documents to LLM
  2. Ask LLM to score relevance (1-10 scale)
  3. Extract scores from structured LLM output
  4. Re-order documents by LLM score
  5. Return reranked results

- **Benefit**: More intelligent relevance assessment than pure vector/keyword matching
- **Cost**: One additional LLM call; uses existing Groq infrastructure

**Implementation**: `RetrievalService.rerank(question, documents, top_k)`

### 2.5 Metadata Filtering (Future-Ready)

**Architecture**: Support filter queries for document-level access control
- **Current State**: Demo app; no access control enforced
- **Future**: Can add user context filtering (e.g., only docs user can access)
- **Implementation**: Pass optional `filter_dict` to retrieval methods
  - Example: `{"source_filename": "public-policies.pdf"}`
  - Passed to Qdrant filter queries

**Implementation**: Add `filter_dict: dict | None` parameter to all retrieval methods

---

## 3. Security Measures

### 3.1 Input Security

✅ **Question Validation**
- Min length: 3 characters
- Max length: 2000 characters
- Reject if empty or whitespace-only
- Reject if contains suspicious patterns:
  - `"system:"`, `"ignore:"`, `"override:"`, `"forget:"`
  - `"<system>"`, `"{{`" (template injection markers)

✅ **Input Sanitization**
- Trim leading/trailing whitespace
- Normalize internal whitespace (collapse multiple spaces)
- No special character escaping needed (passed to embedding, not shell)

✅ **Rate Limiting** (placeholder for production)
- Recommend: Implement in production with `slowapi` or similar
- Structure code to support adding rate limit middleware

### 3.2 Prompt Injection Defense

✅ **Strict System Prompt** (cannot be overridden by retrieved content)

✅ **Structured Output Validation**
- Use Pydantic's `with_structured_output()` on LLM
- Validates LLM response matches expected schema
- Rejects malformed responses

✅ **No User Input in Prompt Template**
- Question is embedded via variable substitution `{question}`
- Not concatenated into prompt string
- Cannot break out of template format

✅ **Retrieved Content as Data, Not Code**
- Context formatted as plaintext blocks
- Not included in prompt template itself
- Passed as separate context variable

✅ **System Instruction Lock**
- System prompt explicitly forbids changing behavior based on retrieved content
- No conditional logic that acts on content keywords

### 3.3 Output Security

✅ **Citation Validation**
- Every citation must refer to actual retrieved chunk text
- Verify `citation.excerpt` exists in original chunk (substring match)
- Reject fabricated citations
- Use exact matching algorithm to prevent false positives

✅ **Citation Format**
- Only expose: `document_id`, `filename`, `excerpt`
- Do NOT expose: `document_hash`, `ingestion_version`, `source_extension`
- Excerpt limited to 200 characters to prevent large data leaks

✅ **Graceful Handling of Low Confidence**
- If no relevant documents found: return `found_relevant_docs=false`
- Answer: "I don't have enough information to answer that question."
- Confidence: 0.0
- Citations: empty list

✅ **Confidence Thresholding**
- Configurable threshold (default 0.3 / 30%)
- If confidence < threshold: return result but mark as low-confidence
- Never suppress answers; return with appropriate confidence score

### 3.4 Data Leakage Prevention

✅ **Metadata Exposure**
- Internal metadata never exposed
- No chunk_index, document_hash, ingestion_version in responses

✅ **Logging Strategy**
- Log: Sanitized question, document_ids retrieved, confidence score
- Never log: Full question text (if contains PII), full document content

✅ **Error Messages**
- Never expose internal LLM prompts
- Never expose Qdrant errors
- Return safe messages: "The AI service is temporarily unavailable"

✅ **Sensitive Data Handling**
- Current: No PII detection; assumes admin-curated documents
- Future: Could add regex markers and filter chunks

### 3.5 Hallucination Prevention

✅ **Context Grounding**
- System prompt forbids inventing information
- Answer only from retrieved context

✅ **Confidence Scoring**
- Return explicit confidence 0-1
- Low confidence when no relevant documents retrieved

✅ **No Fabricated Citations**
- Citations extracted from actual retrieved chunks
- Never cite documents that weren't retrieved

✅ **Fallback Responses**
- When uncertain: "I don't have enough information"
- When no docs match: Explicit `found_relevant_docs=false`

---

## 4. API Design

### 4.1 Endpoint Specification

```
POST /api/v1/ask
Content-Type: application/json
```

### 4.2 Request Schema: `AskRequest`

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_reranking: bool = Field(default=True)
```

### 4.3 Response Schema: `AskResponse`

```python
class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1)
    found_relevant_docs: bool
    retrieval_method: str
```

### 4.4 Citation Schema (existing, reuse)

```python
class Citation(BaseModel):
    document_id: str
    filename: str
    excerpt: str
```

---

## 5. Implementation Components

### 5.1 Files to Create/Modify

| File | Type | Purpose |
|------|------|---------|
| `app/services/retrieval.py` | NEW | Semantic + keyword search, hybrid fusion, reranking |
| `app/services/answer.py` | NEW | LLM answer generation, citation extraction |
| `app/repositories/qdrant_repository.py` | MODIFY | Add retrieval methods |
| `app/api/ask.py` | NEW | API endpoint and dependencies |
| `app/schemas.py` | MODIFY | Add AskRequest, AskResponse |
| `app/exceptions.py` | MODIFY | Add RetrievalError, NoRelevantDocumentsError |
| `app/main.py` | MODIFY | Register ask router |
| `app/config.py` | MODIFY | Add retrieval settings |
| `tests/test_retrieval.py` | NEW | Retrieval service unit tests |
| `tests/test_answer.py` | NEW | Answer generation unit tests |
| `tests/test_ask_api.py` | NEW | API endpoint integration tests |

### 5.2 RetrievalService Key Methods

```python
class RetrievalService:
    async def semantic_search(question: str, top_k: int) -> list[RetrievalResult]
    async def keyword_search(question: str, top_k: int) -> list[RetrievalResult]
    async def hybrid_search(question: str, top_k: int, include_reranking: bool) -> list[RetrievalResult]
    async def rerank(question: str, documents: list[Document], top_k: int) -> list[RetrievalResult]
```

### 5.3 AnswerService Key Methods

```python
class AnswerService:
    async def generate_answer(question: str, documents: list[Document]) -> Answer
    def extract_citations(answer: str, documents: list[Document]) -> list[Citation]
    def validate_answer(answer: str, question: str, documents: list[Document]) -> float
```

### 5.4 Updated QdrantRepository Methods

```python
async def search_semantic(embedding: list[float], top_k: int, filter_dict: dict | None = None) -> list[Document]
async def search_keyword(keywords: list[str], top_k: int, filter_dict: dict | None = None) -> list[Document]
```

---

## 6. Testing Strategy

### 6.1 Test Coverage Areas

- **Semantic Search**: Returns relevant documents, respects top_k, normalized scores
- **Keyword Search**: Finds matching terms, ignores short words, case-insensitive
- **Hybrid Search**: Combines results, correct RRF ordering, proper deduplication
- **Reranking**: Improves relevance, respects top_k
- **Answer Generation**: Uses context, validates schema, confidence scoring
- **Citation Extraction**: Valid citations, no fabrication, excerpt validation
- **API Endpoint**: Happy path, no relevant docs, invalid input, respects top_k
- **Security**: Prompt injection defense, no sensitive data in errors

### 6.2 Test Files

- `tests/test_retrieval.py`: Retrieval service tests
- `tests/test_answer.py`: Answer generation tests
- `tests/test_ask_api.py`: API endpoint tests
- `tests/test_security.py`: Security-specific tests

---

## 7. Key Implementation Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Reranking** | LLM-based (Groq) | Uses existing infra, intelligent scoring |
| **Hybrid Fusion** | RRF (Reciprocal Rank Fusion) | Proven, scalable, no tuning needed |
| **Keyword Search** | Simple term matching (v1) | Fast to implement, works for MVP |
| **Confidence** | Composite (retrieval + LLM) | Combines both signal sources |
| **Citation** | Direct matching from docs | Prevents fabrication |
| **Access Control** | Infrastructure ready, not enforced | Demo app; ready for future |

---

## 8. Configuration

### 8.1 New Settings in `.env`

```env
RERANKER_ENABLED=true
TOP_K_DEFAULT=5
TOP_K_MAX=20
CONFIDENCE_THRESHOLD_DEFAULT=0.3
QUESTION_MIN_LENGTH=3
QUESTION_MAX_LENGTH=2000
RRF_K=60
RERANKER_MODEL=llama-3.3-70b-versatile
RERANKER_TEMPERATURE=0.3
```

### 8.2 Updated config.py

```python
class Settings(BaseSettings):
    # Retrieval
    reranker_enabled: bool = True
    top_k_default: int = 5
    top_k_max: int = 20
    confidence_threshold_default: float = 0.3
    
    # Question constraints
    question_min_length: int = 3
    question_max_length: int = 2000
    
    # RRF
    rrf_k: int = 60
    
    # Reranker
    reranker_model: str = "llama-3.3-70b-versatile"
    reranker_temperature: float = 0.3
```

---

## 9. Phased Implementation

### Phase 1: MVP (Semantic Search + Answer Generation)
- RetrievalService.semantic_search()
- AnswerService.generate_answer()
- API endpoint `/api/v1/ask`
- Basic testing
- **Timeline**: ~2-3 days

### Phase 2: Hybrid Retrieval & Reranking
- RetrievalService.keyword_search()
- RetrievalService.hybrid_search()
- RetrievalService.rerank()
- Update tests
- **Timeline**: ~1-2 days

### Phase 3: Security & Polish
- Input validation & sanitization
- Prompt injection defense testing
- Citation validation hardening
- Security test suite
- **Timeline**: ~1-2 days

---

## 10. Qdrant Payload Structure

**Existing Payload** (from document ingestion):

```json
{
  "document_id": "doc_abc123...",
  "document_hash": "sha256...",
  "source_filename": "policies.pdf",
  "source_extension": ".pdf",
  "text": "chunk content here",
  "chunk_index": 0,
  "ingestion_version": 1
}
```

**Retrieval will leverage**:
- `text`: Keyword search + LLM context
- `document_id`: Citation tracking
- `source_filename`: User-facing citations
- `chunk_index`: Deduplication

---

## 11. Success Criteria

### Functional
✅ Ask endpoint returns accurate answers from knowledge base
✅ Citations are traceable to source documents (no fabrication)
✅ Hybrid search improves retrieval quality vs. semantic-only
✅ Graceful handling when no relevant documents found
✅ Confidence scoring reflects answer quality

### Security
✅ No successful prompt injection attempts
✅ No citation fabrication
✅ Input properly validated and sanitized
✅ Output safely formatted (no leakage)
✅ Error messages don't expose internals

### Quality
✅ Unit tests: >85% code coverage
✅ Integration tests: full flow tested
✅ Security tests: injection + edge cases
✅ All tests pass locally before merge

### Performance
✅ Ask endpoint responds < 5 seconds (semantic only)
✅ With reranking: < 8 seconds (acceptable for demo)

---

**End of Implementation Plan**
