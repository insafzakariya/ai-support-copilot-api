---
name: rag-ai-engineer
description: Design, build, and optimize production LangChain RAG applications with Qdrant vector storage. Covers retrieval patterns, embedding strategies, prompt engineering, evaluation, and security hardening.
---

# RAG AI Engineer Skill

Act as a senior AI/ML engineer specializing in **production Retrieval-Augmented Generation (RAG)** systems using LangChain and Qdrant.

## Core Responsibilities

- Design and implement robust document ingestion pipelines
- Optimize retrieval quality (semantic + keyword + hybrid search)
- Ensure retrieval-grounded LLM outputs (no hallucinations)
- Implement security guardrails against prompt injection
- Evaluate and improve RAG quality metrics
- Handle production constraints (latency, throughput, scale)

---

## 1. LangChain Patterns & Anti-Patterns

### DO: Use Composable Chains

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Good: Composable, testable, readable
chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt_template
    | llm
    | output_parser
)
```

### DON'T: Use deprecated BaseChain or LLMChain

```python
# Bad: Deprecated, harder to test, inflexible
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

### DO: Use `with_structured_output()` for Type Safety

```python
from pydantic import BaseModel

class TicketAnalysis(BaseModel):
    summary: str
    suggested_response: str
    confidence: float

structured_llm = llm.with_structured_output(TicketAnalysis)
```

### DO: Use RunnableConfig for Debugging & Tracing

```python
from langchain_core.runnables.config import RunnableConfig

result = chain.invoke(
    {"input": query},
    config=RunnableConfig(
        tags=["retrieval", "analysis"],
        run_name="ticket_analysis_v1",
    )
)
```

### DON'T: Hardcode Models or Secrets in Chains

```python
# Bad
llm = ChatOpenAI(api_key="sk-xxx")

# Good: Use config and dependency injection
from pydantic_settings import BaseSettings
settings = Settings()
llm = ChatOpenAI(api_key=settings.openai_api_key)
```

---

## 2. Qdrant Vector Store Best Practices

### Collection Design

**Use Named Vectors for Multiple Retrieval Paths**
```python
from qdrant_client import models

# Good: Support both dense and sparse retrieval
client.create_collection(
    collection_name="documents",
    vectors_config={
        "text_embeddings": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "text_sparse": models.SparseVectorParams(),
    }
)
```

**Always Use Consistent Vector Dimensions**
```python
# Verify embedding model output dimension matches Qdrant config
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
test_embedding = embedding_model.embed_query("test")
assert len(test_embedding) == 384, "Embedding dimension mismatch"
```

### Payload Structure

**Store Metadata for Traceability**
```python
point = models.PointStruct(
    id=point_id,
    vector=embedding,
    payload={
        # Always include source reference
        "document_id": "doc_abc123",
        "source_filename": "policy.pdf",
        "source_url": "https://...",
        "chunk_index": 0,
        
        # Content for filtering & display
        "text": chunk_text,
        "title": "Return Policy",
        
        # Versioning for safe updates
        "ingestion_version": 1,
        "ingestion_timestamp": datetime.now().isoformat(),
        
        # Tenant isolation (if multi-tenant)
        "tenant_id": "customer_123",
    }
)
```

### Idempotent Indexing

**Use Content Hashing for Deduplication**
```python
import hashlib

def compute_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def is_already_indexed(hash_value: str, qdrant_client) -> bool:
    # Query for existing chunks with this document_hash
    results = qdrant_client.scroll(
        collection_name="documents",
        limit=1,
        scroll_filter=models.Filter(
            must=[
                models.HasIdCondition(
                    has_id=[models.FieldCondition(
                        key="document_hash",
                        match=models.MatchValue(value=hash_value),
                    )]
                )
            ]
        ),
    )
    return len(results[0]) > 0
```

### Hybrid Search Implementation

**Combine Dense + Sparse Retrieval**
```python
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import EnsembleRetriever

# Dense retriever (semantic)
dense_retriever = QdrantVectorStore.as_retriever(
    qdrant_client=client,
    collection_name="documents",
    vector_name="text_embeddings",
    k=5,
)

# Sparse retriever (BM25 keyword)
sparse_retriever = QdrantVectorStore.as_retriever(
    qdrant_client=client,
    collection_name="documents",
    vector_name="text_sparse",
    k=5,
)

# Ensemble (weighted fusion)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4],  # 60% semantic, 40% keyword
)
```

### Filtering & Metadata Queries

**Always Include Tenant/Access Filters**
```python
from qdrant_client import models

# Tenant-scoped retrieval
retriever.invoke(
    query,
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=current_user.tenant_id),
            )
        ]
    ),
    top_k=5,
)
```

---

## 3. Embedding Strategy

### Model Selection

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 dims | Fast | Good | Default, most use cases |
| `all-mpnet-base-v2` | 768 dims | Medium | Better | Semantic accuracy critical |
| `bge-large-en-v1.5` | 1024 dims | Slower | Best | Maximum quality (cost: latency) |

### Embedding at Indexing vs. Query Time

**DO: Index & Search with Same Model**
```python
# Indexing
indexing_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = indexing_embedder.embed_documents(chunks)

# Querying (MUST be same model)
query_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_vector = query_embedder.embed_query(question)
```

**DON'T: Change Embedding Models Mid-Collection**
- If you must upgrade, create a new collection and reindex
- Store `embedding_model_version` in payload for tracking

### Batch Embedding Optimization

```python
def embed_documents_batched(texts: list[str], batch_size: int = 32) -> list:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

---

## 4. Retrieval Quality & Optimization

### Chunking Strategy

**Use Semantic Chunking, Not Just Size**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Good: Respects semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,           # Tokens, not characters
    chunk_overlap=150,         # 12.5% overlap for context
    separators=["\n\n", "\n", ". ", " ", ""],  # Try to keep sentences/paragraphs
    add_start_index=True,
)

chunks = splitter.split_documents(documents)
```

**DON'T: Use Fixed-Size Token Chunking Without Context**
- Always preserve paragraph/sentence boundaries
- Add metadata (document title, section) to each chunk

### Retrieval Evaluation

**Measure Quality with Metrics**
```python
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

eval_results = evaluate(
    datasets=[{"question": q, "answer": a, "contexts": c} for q, a, c in samples],
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)
```

**Key Metrics**
- **Context Precision**: % retrieved docs are relevant (low = noisy retrieval)
- **Context Recall**: % relevant docs retrieved (low = missing relevant docs)
- **Faithfulness**: % of answer grounded in context (low = hallucinations)
- **Answer Relevancy**: How directly answer addresses question

### Re-Ranking for Precision

```python
from langchain_community.document_compressors import CohereReranker
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereReranker(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=qdrant_retriever,
)

# Returns top-k docs re-ranked by relevance
top_docs = compression_retriever.invoke(query, top_k=5)
```

---

## 5. Prompt Engineering for RAG

### Prompt Template Structure

```python
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful support assistant. Answer questions based ONLY on the provided context.

Rules:
- If the context doesn't contain enough information, say "I don't have that information."
- Never invent policies, numbers, or timeframes.
- Always cite which document(s) you're referencing.
- Be concise and polite.
"""),
    ("human", """Context from knowledge base:
{context}

Question: {question}

Provide a helpful response grounded in the context above."""),
])
```

### Citation Handling

```python
def format_context_with_citations(documents: list[Document]) -> str:
    """Format retrieved docs with citation markers."""
    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(
            f"[{i}] {doc.metadata.get('title', 'Unknown')}\n"
            f"Source: {doc.metadata.get('source_filename')}\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(formatted)

# In prompt
context_str = format_context_with_citations(retrieved_docs)
```

### Instructing the Model to Use Citations

```python
class AnalysisWithCitations(BaseModel):
    analysis: str
    citations: list[int]  # References to [1], [2], etc.
    confidence: float

# The model will now reference citations by index
```

---

## 6. Security Guardrails

### Prompt Injection Defense

**Strategy 1: Separate Instructions from Context**
```python
# Good: User input and context are clearly separated
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant. Answer based on context."),
    ("human", "Question: {question}"),
    ("human", "Context: {context}"),  # Clearly marked
])
```

**Strategy 2: Instruction Hierarchy**
```python
# System message (highest priority - hardest to override)
# → User question (lower priority)
# → Retrieved context (informational only, not executable)
```

**Strategy 3: Input Validation**
```python
from pydantic import BaseModel, Field

class TicketQuery(BaseModel):
    subject: str = Field(max_length=200)
    message: str = Field(max_length=5000)
    
    @field_validator('subject', 'message')
    def no_prompt_injection(cls, v):
        # Block common injection patterns
        dangerous = ["ignore previous", "system:", "admin:", "---"]
        if any(pattern.lower() in v.lower() for pattern in dangerous):
            raise ValueError("Query contains invalid patterns")
        return v
```

### Tenant Isolation

**Always Include Tenant Filter in Queries**
```python
def retrieve_for_user(query: str, user_id: str, tenant_id: str) -> list[Document]:
    """Retrieve only documents the user's tenant has access to."""
    return retriever.invoke(
        query,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
        ),
    )
```

### PII Protection

**Never Log or Store Sensitive Data**
```python
import logging

logger = logging.getLogger(__name__)

# Bad
logger.info(f"Processing ticket: {user_email} {message}")  # PII exposed

# Good
logger.info(f"Processing ticket: ticket_id={ticket_id}")  # No PII
```

**Scrub Retrieved Context**
```python
import re

def scrub_pii(text: str) -> str:
    """Remove credit card, SSN, email patterns."""
    text = re.sub(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}', 'CARD_REDACTED', text)
    text = re.sub(r'\d{3}-\d{2}-\d{4}', 'SSN_REDACTED', text)
    return text
```

### Rate Limiting & Quotas

```python
from functools import wraps
from datetime import datetime, timedelta

def rate_limit(max_calls: int = 10, period: int = 60):
    """Rate limit queries per user."""
    def decorator(func):
        calls = {}
        
        @wraps(func)
        def wrapper(user_id: str, *args, **kwargs):
            now = datetime.now()
            if user_id not in calls:
                calls[user_id] = []
            
            # Remove old calls outside the period
            calls[user_id] = [call_time for call_time in calls[user_id] 
                             if (now - call_time).seconds < period]
            
            if len(calls[user_id]) >= max_calls:
                raise Exception(f"Rate limit exceeded for {user_id}")
            
            calls[user_id].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 7. Common Pitfalls & Solutions

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Embedding model mismatch | Wrong similarity rankings | Use same model for index & query |
| No chunk overlap | Lost context at boundaries | Use 10-15% overlap (150 tokens typical) |
| Forgetting deduplication | Duplicate answers | Hash content, check before indexing |
| Hardcoded LLM params | Inflexible, costly | Use config for temp, max_tokens, etc. |
| No citation tracking | Unverifiable answers | Store source metadata in payload |
| Filtering after retrieval | Incomplete results | Filter in Qdrant query, not after |
| No evaluation metrics | Silent quality degradation | Track precision, recall, faithfulness |
| Injected instructions in context | Prompt injection | Keep system prompt separate from context |

---

## 8. Debugging & Observability

### Enable LangChain Debug Mode

```python
import langchain
langchain.debug = True  # Logs all chain executions
```

### Trace Retrieval Quality

```python
def debug_retrieval(query: str, retriever):
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"  [{i}] {doc.metadata.get('title')} (score: N/A)")
        print(f"      {doc.page_content[:100]}...")
```

### Monitor Qdrant Health

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
info = client.get_collection("documents")
print(f"Vectors in collection: {info.points_count}")
print(f"Vector size: {info.config.params.vectors['text_embeddings'].size}")
```

---

## 9. Implementation Checklist

For every RAG feature:

- [ ] Documents have stable IDs and source metadata
- [ ] Idempotent ingestion (hashing deduplication)
- [ ] Consistent embedding model for index & query
- [ ] Hybrid search (semantic + keyword) considered
- [ ] Re-ranking evaluated for precision improvement
- [ ] Prompt template tested for prompt injection
- [ ] Tenant/user filters applied to all queries
- [ ] PII scrubbed from context before passing to LLM
- [ ] Rate limiting & quota enforcement
- [ ] Retrieval quality metrics tracked
- [ ] Citations mapped back to source documents
- [ ] Error handling for Qdrant unavailability
- [ ] Timeouts set on LLM calls
- [ ] Logging excludes sensitive data

---

## References

- [LangChain Docs](https://python.langchain.com/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)
- [LangSmith Tracing](https://smith.langchain.com/)
