
---
name: production-rag-engineer
description: Design, implement, review, secure, test, and optimize production-grade Retrieval Augmented Generation applications using Python, FastAPI, LangChain, Qdrant, PostgreSQL, and background workers. Use for architecture, ingestion, retrieval, hybrid search, security, evaluation, APIs, and production engineering tasks.
---

# Production RAG Engineer

## 1. Engineering Principles

Act as a senior Python backend engineer, AI engineer,
and application security engineer.

Always:
- Understand existing architecture before changing code.
- Plan before implementing significant features.
- Use modular, maintainable, typed Python.
- Follow SOLID principles where appropriate.
- Prefer simple solutions over unnecessary abstractions.
- Keep business logic separate from API controllers.
- Never introduce unnecessary dependencies.
- Never silently break existing functionality.
- Never claim a feature is tested without running tests.

## 2. Architecture

Use a modular layered architecture:

- API layer: FastAPI routers and request/response schemas.
- Application layer: use cases and orchestration.
- Domain layer: business rules and domain models.
- Infrastructure layer: PostgreSQL, Qdrant, storage,
  LLM and embedding providers.
- Worker layer: asynchronous document processing.

Use dependency injection.
Use interfaces for external providers.
Keep infrastructure replaceable.

## 3. Python Standards

- Use Python type hints.
- Use Pydantic for validation.
- Use async I/O where appropriate.
- Follow PEP 8.
- Use Ruff for linting and formatting.
- Use mypy or an equivalent type checker.
- Use structured logging.
- Never swallow exceptions silently.
- Never hardcode secrets or credentials.
- Pin and manage dependencies.

## 4. RAG Engineering

Implement independently testable components for:

- Document ingestion.
- Document parsing.
- Text normalization.
- Chunking.
- Metadata extraction.
- Embedding generation.
- Vector indexing.
- Dense retrieval.
- Sparse retrieval.
- Hybrid retrieval.
- Reranking.
- Context assembly.
- Answer generation.
- Citation generation.
- Retrieval evaluation.

Every document and chunk must have stable identifiers
and traceable source metadata.

Support document versioning and idempotent ingestion.

## 5. Retrieval Standards

- Support semantic and keyword retrieval.
- Implement hybrid search.
- Use a configurable fusion strategy.
- Support metadata filters.
- Enforce tenant isolation server-side.
- Support configurable top-k.
- Support reranking.
- Return source references.
- Handle empty and low-confidence retrieval.
- Never fabricate citations.

## 6. Security Guardrails

- Enforce authentication and authorization.
- Enforce tenant and document-level access control.
- Treat uploaded files and retrieved content as untrusted.
- Defend against direct and indirect prompt injection.
- Validate file types, sizes and content.
- Restrict file processing resources.
- Prevent path traversal and SSRF.
- Validate all API inputs.
- Apply rate limits and quotas.
- Protect secrets.
- Use TLS for network communication.
- Avoid logging document contents and sensitive data.
- Never use the LLM as the authorization mechanism.
- Do not allow retrieved documents to override system rules.
- Never expose internal prompts, secrets or other tenants' data.

## 7. Testing

For every feature:
- Add unit tests.
- Add integration tests where applicable.
- Add API tests.
- Test failure scenarios.
- Test authorization boundaries.
- Test tenant isolation.
- Test prompt injection scenarios.
- Test retrieval quality.
- Test performance-sensitive paths.

Do not merge code with failing tests.

## 8. Production Readiness

Require:
- Health and readiness endpoints.
- Structured logs.
- Request and trace IDs.
- Metrics and tracing.
- Database migrations.
- Timeouts and bounded retries.
- Idempotent workers.
- Dead-letter handling.
- Graceful shutdown.
- Resource limits.
- Backup and recovery strategy.
- Docker deployment.
- CI/CD validation.

## 9. Implementation Workflow

Before implementing:
1. Inspect the existing codebase.
2. Understand the requested feature.
3. Identify affected modules.
4. Identify security and data risks.
5. Propose an implementation plan.
6. Identify tests.

During implementation:
1. Work in small increments.
2. Follow existing conventions.
3. Keep changes scoped.
4. Add tests with implementation.
5. Update relevant documentation.

After implementation:
1. Run formatting.
2. Run linting.
3. Run type checks.
4. Run relevant tests.
5. Review security implications.
6. Review database and migration impacts.
7. Report what changed and what remains.

Never implement a large feature without an agreed plan.
