# Multi-Agent Support Ticket Analysis Pipeline — Implementation Plan

## What We're Building

A **4-agent pipeline** that takes a support ticket and returns a fully analyzed, grounded, cited response. Reactivates the commented-out `/api/tickets/analyze` endpoint with a proper multi-agent architecture.

The key selling point for the tech lead interview:
- Typed Pydantic handoffs between agents (no string parsing)
- `TriageAgent` generates an **optimized search query** (not raw customer text) → better retrieval
- `ResearchAgent` has no LLM call — it wraps the existing `RetrievalService` directly (shows you know not every agent needs an LLM)
- Two LLM temperatures: `0.2` for classification/review, `0.4` for drafting
- `pipeline_stages` in the response body — full trace visible in a single API call
- Zero new dependencies

---

## Pipeline Flow

```
POST /api/tickets/analyze
         │
         ▼
   AnalyzeTicketRequest
         │
         ▼
  [1] TriageAgent   ──► TriageOutput        classify + produce optimized search query
         │
         ▼
  [2] ResearchAgent ──► ResearchOutput      hybrid search via existing RetrievalService (no LLM)
         │
         ▼
  [3] DrafterAgent  ──► DraftOutput         write grounded, cited customer reply
         │
         ▼
  [4] ReviewerAgent ──► ReviewOutput        QA gate: hallucination + tone check
         │
         ▼
   TicketAnalysisResponse (HTTP 200)
```

---

## Files to Create

| File | Purpose |
|---|---|
| `app/agents/__init__.py` | Re-export all agents |
| `app/agents/base.py` | `BaseAgent(ABC)` — `llm: ChatGroq`, `async run()` |
| `app/agents/triage_agent.py` | Classify ticket + generate search query |
| `app/agents/research_agent.py` | Wrap `RetrievalService.hybrid_search` (no LLM) |
| `app/agents/drafter_agent.py` | Write grounded customer reply |
| `app/agents/reviewer_agent.py` | QA gate: hallucination + tone |
| `app/services/agent_pipeline.py` | Orchestrator — calls all 4 agents sequentially |
| `app/api/tickets.py` | FastAPI router for the endpoint |

## Files to Modify

| File | Change |
|---|---|
| `app/schemas.py` | Add 5 new Pydantic models (see below) |
| `app/exceptions.py` | Add `AgentPipelineError`, `AgentExecutionError` |
| `app/main.py` | Register `tickets_router`; remove old commented-out code |

---

## New Pydantic Models (add to `app/schemas.py`)

```python
from typing import Literal
from pydantic import BaseModel, Field

class TriageOutput(BaseModel):
    category: Literal["billing", "technical", "account", "shipping", "general"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    urgency_score: float = Field(ge=0.0, le=1.0)
    search_query: str  # optimized KB search query — the key insight
    reasoning: str

class ResearchOutput(BaseModel):
    retrieved_chunks: list[str]
    source_citations: list[Citation]
    retrieval_method: str
    found_relevant_docs: bool

class DraftOutput(BaseModel):
    draft_response: str
    confidence: float = Field(ge=0.0, le=1.0)
    cited_document_ids: list[str]

class ReviewOutput(BaseModel):
    approved: bool
    revised_response: str
    issues_found: list[str] = Field(default_factory=list)
    final_confidence: float = Field(ge=0.0, le=1.0)

class TicketAnalysisResponse(BaseModel):
    ticket_id: str
    category: str
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    urgency_score: float
    draft_response: str
    citations: list[Citation]
    confidence: float
    approved: bool
    issues_found: list[str]
    retrieval_method: str
    found_relevant_docs: bool
    pipeline_stages: list[str]  # e.g. ["triage", "research", "draft", "review"]
```

---

## Agent System Prompts

**TriageAgent** (temp 0.2):
> "You are a support triage specialist. Classify the ticket into the correct category and priority, assess customer sentiment, and produce a targeted knowledge-base search query. Do not infer information not stated in the ticket."

**ResearchAgent** — No system prompt, no LLM. Calls:
```python
await self.retrieval_service.hybrid_search(
    question=triage.search_query, top_k=5, include_reranking=True
)
```
Catches `NoRelevantDocumentsError` and returns `found_relevant_docs=False` gracefully.

**DrafterAgent** (temp 0.4):
> "You are a customer support writer. Draft a helpful, empathetic reply using ONLY the supplied knowledge-base context. Cite document IDs inline. If the context is insufficient, tell the customer a human agent will follow up. Never invent policies, refunds, timelines, or company procedures."

**ReviewerAgent** (temp 0.2):
> "You are a QA reviewer for customer support. Check the draft for: (1) hallucinations — claims not supported by the KB context, (2) inappropriate tone, (3) missing critical information. Rewrite if needed. Set approved=true only when the response is factually grounded and professionally toned."

---

## `app/agents/base.py`

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_groq import ChatGroq

class BaseAgent(ABC):
    name: str

    def __init__(self, llm: ChatGroq) -> None:
        self.llm = llm

    @abstractmethod
    async def run(self, **kwargs) -> BaseModel:
        ...
```

---

## `app/services/agent_pipeline.py` (Orchestrator)

```python
class AgentPipeline:
    def __init__(self, settings: Settings, retrieval_service: RetrievalService) -> None:
        self._llm_precise  = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.2, max_tokens=800)
        self._llm_creative = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.4, max_tokens=1200)
        self.triage_agent   = TriageAgent(llm=self._llm_precise)
        self.research_agent = ResearchAgent(llm=self._llm_precise, retrieval_service=retrieval_service)
        self.drafter_agent  = DrafterAgent(llm=self._llm_creative)
        self.reviewer_agent = ReviewerAgent(llm=self._llm_precise)

    async def run(self, request: AnalyzeTicketRequest) -> TicketAnalysisResponse:
        stages = []
        triage   = await self.triage_agent.run(subject=request.subject, message=request.message);   stages.append("triage")
        research = await self.research_agent.run(triage=triage);                                     stages.append("research")
        draft    = await self.drafter_agent.run(subject=request.subject, message=request.message, triage=triage, research=research); stages.append("draft")
        review   = await self.reviewer_agent.run(subject=request.subject, message=request.message, draft=draft, research=research);  stages.append("review")

        return TicketAnalysisResponse(
            ticket_id=request.ticket_id,
            category=triage.category, priority=triage.priority, sentiment=triage.sentiment,
            urgency_score=triage.urgency_score, draft_response=review.revised_response,
            citations=research.source_citations, confidence=review.final_confidence,
            approved=review.approved, issues_found=review.issues_found,
            retrieval_method=research.retrieval_method,
            found_relevant_docs=research.found_relevant_docs,
            pipeline_stages=stages,
        )
```

---

## `app/api/tickets.py` (Endpoint)

Same DI pattern as `app/api/ask.py`:

```python
router = APIRouter(prefix="/api/tickets", tags=["tickets"])

@router.post("/analyze", response_model=TicketAnalysisResponse, status_code=200)
async def analyze_ticket(
    request: AnalyzeTicketRequest,
    pipeline: Annotated[AgentPipeline, Depends(get_agent_pipeline)],
) -> TicketAnalysisResponse:
    return await pipeline.run(request)
```

---

## `app/main.py` change

```python
# Add after existing imports:
from .api.tickets import router as tickets_router

# Add after existing include_router calls:
app.include_router(tickets_router)

# Delete the old commented-out block at the bottom of the file
```

---

## Implementation Order

1. `app/schemas.py` — add 5 models (everything imports from here)
2. `app/exceptions.py` — add `AgentPipelineError`, `AgentExecutionError`
3. `app/agents/base.py`
4. `app/agents/triage_agent.py`
5. `app/agents/research_agent.py`
6. `app/agents/drafter_agent.py`
7. `app/agents/reviewer_agent.py`
8. `app/agents/__init__.py`
9. `app/services/agent_pipeline.py`
10. `app/api/tickets.py`
11. `app/main.py`

---

## Reused Existing Assets

| Existing code | Reused in |
|---|---|
| `RetrievalService.hybrid_search()` | `ResearchAgent.run()` |
| `NoRelevantDocumentsError` | Caught in `ResearchAgent` |
| `llm.with_structured_output(Model)` pattern | All 3 LLM agents (same as `AnswerService`) |
| `Citation` schema | Populated in `ResearchOutput` |
| DI pattern from `app/api/ask.py` | Replicated in `app/api/tickets.py` |

---

## Verification

```bash
# 1. Start server
fastapi dev app/main.py

# 2. Upload a test document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@some_support_doc.pdf"

# 3. Analyze a ticket
curl -X POST http://localhost:8000/api/tickets/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT-001",
    "subject": "Payment went through but subscription still inactive",
    "message": "I was charged $29 three days ago but my account still shows free tier."
  }'

# 4. Expected response includes:
#    pipeline_stages: ["triage", "research", "draft", "review"]
#    citations pointing to real uploaded docs
#    approved: true/false from reviewer
```

---

## Expected API Response Shape

```json
{
  "ticket_id": "TKT-001",
  "category": "billing",
  "priority": "high",
  "sentiment": "frustrated",
  "urgency_score": 0.82,
  "draft_response": "Thank you for reaching out. We can confirm your payment was received...",
  "citations": [
    {
      "document_id": "billing-001",
      "filename": "support-handbook.pdf",
      "excerpt": "If a payment succeeds but the subscription remains inactive..."
    }
  ],
  "confidence": 0.87,
  "approved": true,
  "issues_found": [],
  "retrieval_method": "hybrid+rerank",
  "found_relevant_docs": true,
  "pipeline_stages": ["triage", "research", "draft", "review"]
}
```
