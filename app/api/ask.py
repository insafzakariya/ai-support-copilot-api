import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Settings, get_settings
from ..exceptions import (
    AnswerGenerationError,
    NoRelevantDocumentsError,
    RetrievalError,
)
from ..repositories.qdrant_repository import QdrantRepository
from ..schemas import AskRequest, AskResponse
from ..services.answer import AnswerService
from ..services.embeddings import create_embeddings
from ..services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ask"])


def get_embeddings(settings: Settings = Depends(get_settings)):
    return create_embeddings(settings)


def get_qdrant_repo(settings: Settings = Depends(get_settings)):
    return QdrantRepository(settings)


def get_retrieval_service(
    settings: Settings = Depends(get_settings),
    qdrant_repo: QdrantRepository = Depends(get_qdrant_repo),
    embeddings=Depends(get_embeddings),
) -> RetrievalService:
    return RetrievalService(settings, qdrant_repo, embeddings)


def get_answer_service(
    settings: Settings = Depends(get_settings),
) -> AnswerService:
    return AnswerService(settings)


def _validate_question(question: str, settings: Settings) -> str:
    """Validate and sanitize question input."""
    if len(question) < settings.question_min_length:
        raise ValueError(f"Question too short (minimum {settings.question_min_length} characters)")
    if len(question) > settings.question_max_length:
        raise ValueError(f"Question too long (maximum {settings.question_max_length} characters)")

    injection_patterns = [
        "system:",
        "ignore:",
        "override:",
        "forget:",
        "<system>",
        "{{",
    ]
    question_lower = question.lower()
    for pattern in injection_patterns:
        if pattern in question_lower:
            raise ValueError("Question contains suspicious patterns")

    normalized = " ".join(question.split())
    return normalized


@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: AskRequest,
    retrieval_service: Annotated[
        RetrievalService, Depends(get_retrieval_service)
    ],
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    """
    Ask a question to the knowledge base.

    Returns an answer with citations, confidence score, and metadata about the retrieval.
    """
    try:
        sanitized_question = _validate_question(request.question, settings)

        logger.info(
            "ask_request_received",
            extra={
                "question_length": len(sanitized_question),
                "top_k": request.top_k,
                "include_reranking": request.include_reranking,
            },
        )

        try:
            if request.include_reranking:
                results = await retrieval_service.hybrid_search(
                    question=sanitized_question,
                    top_k=request.top_k,
                    include_reranking=True,
                )
                retrieval_method = "hybrid"
            else:
                results = await retrieval_service.semantic_search(
                    question=sanitized_question,
                    top_k=request.top_k,
                )
                retrieval_method = "semantic"

            documents = [r.document for r in results]

        except NoRelevantDocumentsError:
            logger.info("ask_no_relevant_documents")
            return AskResponse(
                answer="I don't have enough information to answer that question.",
                citations=[],
                confidence=0.0,
                found_relevant_docs=False,
                retrieval_method="semantic",
            )

        if not documents:
            return AskResponse(
                answer="I don't have enough information to answer that question.",
                citations=[],
                confidence=0.0,
                found_relevant_docs=False,
                retrieval_method=retrieval_method,
            )

        answer = await answer_service.generate_answer(
            question=sanitized_question, documents=documents
        )

        citations = answer_service.extract_citations(
            answer=answer.text, documents=documents
        )

        confidence = (
            answer.confidence
            if answer.confidence >= request.confidence_threshold
            else answer.confidence
        )

        response = AskResponse(
            answer=answer.text,
            citations=citations,
            confidence=confidence,
            found_relevant_docs=True,
            retrieval_method=retrieval_method,
        )

        logger.info(
            "ask_request_completed",
            extra={
                "documents_retrieved": len(documents),
                "citations_returned": len(citations),
                "confidence": confidence,
                "retrieval_method": retrieval_method,
            },
        )

        return response

    except ValueError as exc:
        logger.warning("ask_request_validation_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RetrievalError as exc:
        logger.exception("ask_retrieval_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The retrieval service is temporarily unavailable.",
        )
    except AnswerGenerationError as exc:
        logger.exception("ask_answer_generation_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI service is temporarily unavailable.",
        )
    except Exception as exc:
        logger.exception("ask_request_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )
