import io
import zipfile

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.exceptions import DocumentTooLargeError, InvalidDocumentError
from app.services.document_validation import read_upload


def make_upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=__import__("io").BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.asyncio
async def test_txt_validation(settings):
    filename, extension, content = await read_upload(
        make_upload("notes.txt", b"hello", "text/plain"), settings
    )
    assert (filename, extension, content) == ("notes.txt", ".txt", b"hello")


@pytest.mark.asyncio
async def test_pdf_validation(settings):
    result = await read_upload(make_upload("guide.pdf", b"%PDF-1.7", "application/pdf"), settings)
    assert result[1] == ".pdf"


@pytest.mark.asyncio
async def test_docx_validation(settings):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document />")
    result = await read_upload(
        make_upload(
            "guide.docx",
            buffer.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        settings,
    )
    assert result[1] == ".docx"


@pytest.mark.asyncio
async def test_invalid_signature_is_rejected(settings):
    with pytest.raises(InvalidDocumentError):
        await read_upload(make_upload("file.pdf", b"not a pdf", "application/pdf"), settings)


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected(settings):
    with pytest.raises(DocumentTooLargeError):
        await read_upload(make_upload("file.txt", b"x" * 1025, "text/plain"), settings)


@pytest.mark.asyncio
async def test_unsupported_file_is_rejected(settings):
    with pytest.raises(InvalidDocumentError):
        await read_upload(make_upload("file.exe", b"MZ", "application/octet-stream"), settings)
