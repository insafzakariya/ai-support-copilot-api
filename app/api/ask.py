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

from llm_guard.input_scanners import PromptInjection, Toxicity
from llm_guard.vault import Vault
from ..services.embedding_cache import CachedEmbeddings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ask"])
_cached_embeddings: CachedEmbeddings | None = None


def get_embeddings(settings: Settings = Depends(get_settings)):
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = create_embeddings(settings)
    return _cached_embeddings


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
    question = question.strip()
    if len(question) < settings.question_min_length:
        raise ValueError(
            f"Question too short (minimum {settings.question_min_length} characters)"
        )
    if len(question) > settings.question_max_length:
        raise ValueError(
            f"Question too long (maximum {settings.question_max_length} characters)"
        )

    injection_patterns = [
        "system:",
        "ignore previous",
        "ignore all",
        "override:",
        "forget:",
        "<system>",
        "{{",
        "[inst]",
        "###",
        "human:",
        "assistant:",
        "<|system|>",
        "<|user|>",
        "developer mode",
        "jailbreak",
        "new instruction",
        "act as if",
        "pretend you",
        "you are now",
        "i am an admin",
        "clearance level",
        "bypass",
        "base64",
        "decode this",
        "reveal the",
        "system instructions",
        "system prompt",
        "tell me your system",
        "tell me the system",
        "help me hack",
        "how to hack",
        "exploit",
        "breach",
        "unauthorized access",
    ]
    question_lower = question.lower()
    for pattern in injection_patterns:
        if pattern in question_lower:
            raise ValueError("Question contains suspicious patterns")

    normalized = " ".join(question.split())
    return normalized


async def _scan_input_with_llm_guard(question: str) -> None:
    """Scan question for prompt injection and toxicity using llm-guard."""
    try:

        vault = Vault()

        prompt_injection_scanner = PromptInjection(vault=vault)
        toxicity_scanner = Toxicity(vault=vault)

        is_injection_safe, _, reason = prompt_injection_scanner.scan(question)
        if not is_injection_safe:
            logger.warning(
                "ask_request_blocked_prompt_injection", extra={"reason": reason}
            )
            raise ValueError("Question contains potential prompt injection attack")

        is_toxicity_safe, _, reason = toxicity_scanner.scan(question)
        if not is_toxicity_safe:
            logger.warning("ask_request_blocked_toxicity", extra={"reason": reason})
            raise ValueError("Question contains toxic content")

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        logger.debug("llm_guard_scan_error", extra={"error": str(exc)})
        pass


@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: AskRequest,
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    """
    Ask a question to the knowledge base.

    Returns an answer with citations, confidence score, and metadata about the retrieval.
    """
    guardrails_triggered: list[str] = []
    try:
        sanitized_question = _validate_question(request.question, settings)
        await _scan_input_with_llm_guard(sanitized_question)

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

        if answer.raw_response.get("output_safety_flagged", False):
            guardrails_triggered.append("sensitive_data_detection")

        confidence_below_threshold = answer.confidence < request.confidence_threshold
        if confidence_below_threshold:
            guardrails_triggered.append("confidence_threshold")

        confidence = answer.confidence

        response = AskResponse(
            answer=answer.text,
            citations=citations,
            confidence=confidence,
            found_relevant_docs=not confidence_below_threshold,
            retrieval_method=retrieval_method,
            guardrails_triggered=guardrails_triggered,
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
        error_msg = str(exc)
        logger.warning("ask_request_validation_failed", extra={"error": error_msg})

        if "too short" in error_msg.lower():
            guardrails_triggered.append("input_length_min")
        elif "too long" in error_msg.lower():
            guardrails_triggered.append("input_length_max")
        elif "suspicious patterns" in error_msg.lower():
            guardrails_triggered.append("prompt_injection_pattern")
        elif "prompt injection" in error_msg.lower():
            guardrails_triggered.append("prompt_injection_ml")
        elif "toxic" in error_msg.lower():
            guardrails_triggered.append("toxicity_detection")

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


@router.get("/cache-stats", status_code=status.HTTP_200_OK)
async def get_cache_stats():
    """Get embedding cache statistics."""
    if _cached_embeddings is None:
        return {"error": "Cache not initialized"}

    stats = _cached_embeddings.get_cache_stats()
    return {
        "embedding_cache": stats,
        "message": "Cache hit rate shows percentage of questions answered from cache"
    }


@router.post("/cache-clear", status_code=status.HTTP_200_OK)
async def clear_cache():
    """Clear the embedding cache."""
    if _cached_embeddings is None:
        return {"error": "Cache not initialized"}

    _cached_embeddings.clear_cache()
    return {"message": "Embedding cache cleared"}
