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
    title: str
    excerpt: str


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
