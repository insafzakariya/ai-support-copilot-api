import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ..config import Settings, get_settings
from ..exceptions import DocumentIngestionError, DocumentTooLargeError, InvalidDocumentError, VectorStoreError
from ..repositories.qdrant_repository import QdrantRepository
from ..schemas import DocumentUploadResponse, MessageResponse
from ..services.document_ingestion import DocumentIngestionService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> DocumentIngestionService:
    return DocumentIngestionService(settings, QdrantRepository(settings))


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    try:
        result = await service.ingest(file)
        return DocumentUploadResponse.model_validate(result)
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentIngestionError as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Document indexing is temporarily unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected document ingestion failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document ingestion failed") from exc


@router.delete("", response_model=MessageResponse, status_code=status.HTTP_200_OK)
def truncate_documents(
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> MessageResponse:
    try:
        service.truncate()
        return MessageResponse(message="Vector store truncated successfully")
    except VectorStoreError as exc:
        logger.exception("Vector store truncation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store truncation is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected truncation failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Truncation failed",
        ) from exc
