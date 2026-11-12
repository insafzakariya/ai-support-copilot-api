import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ai_pipeline import analyze_ticket
from .api.documents import router as documents_router

from .config import get_settings
from .schemas import (
    AnalyzeTicketRequest,
    FeedbackRequest,
    MessageResponse,
    Ticket,
    TicketAnalysis,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="AI Support Copilot API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[settings.frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(documents_router)


@app.get("/health", response_model=MessageResponse)
def health_check() -> MessageResponse:
    return MessageResponse(message="ok")


@app.post("/api/tickets/analyze", response_model=TicketAnalysis)
def analyze_ticket_endpoint(payload: AnalyzeTicketRequest) -> TicketAnalysis:
    logger.info("Analyzing ticket_id=%s", payload.ticket_id)

    try:
        return analyze_ticket(payload.subject, payload.message)
    except Exception as exc:
        logger.exception("Ticket analysis failed")
        raise HTTPException(
            status_code=502,
            detail="The AI analysis service is temporarily unavailable.",
        )
