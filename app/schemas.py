from typing import Literal
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    id: str = Field(..., description="The unique identifier for the ticket.")
    customer_name: str
    subject: str
    message: str
    status: Literal["open", "in_review", "resolved"] = "open"


class AnalyzeTicketRequest(BaseModel):
    ticket_id: str = Field(min_length=1)
    subject: str = Field(min_length=3, max_length=300)
    message: str = Field(min_length=5, max_length=10_000)


class Citation(BaseModel):
    document_id: str
    filename: str
    excerpt: str = Field(..., max_length=200)


class TicketAnalysis(BaseModel):
    category: str
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    confidence: float = Field(ge=0, le=1)
    summary: str
    recommended_action: str
    # draft_response: str
    # citations: list[Citation]


class FeedbackRequest(BaseModel):
    ticket_id: str = Field(min_length=1)
    approved: bool
    final_response: str = Field(min_length=1, max_length=10_000)


class MessageResponse(BaseModel):
    message: str


class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    total_chunks: int
    status: Literal["indexed", "already_indexed"]
    message: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_reranking: bool = Field(default=True)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    found_relevant_docs: bool
    retrieval_method: str
