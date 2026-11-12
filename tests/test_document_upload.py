import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.documents import router


class FakeService:
    async def ingest(self, upload):
        assert upload.filename == "guide.txt"
        return {
            "success": True,
            "document_id": "doc_test",
            "filename": "guide.txt",
            "total_chunks": 2,
            "status": "indexed",
            "message": "Document successfully uploaded and indexed",
        }


def test_successful_upload_response():
    app = FastAPI()
    app.include_router(router)
    from app.api.documents import get_ingestion_service

    app.dependency_overrides[get_ingestion_service] = lambda: FakeService()
    response = TestClient(app).post(
        "/api/v1/documents/upload",
        files={"file": ("guide.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["total_chunks"] == 2
