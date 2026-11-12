from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from fastapi import UploadFile

from ..config import Settings
from ..exceptions import DocumentTooLargeError, InvalidDocumentError

ALLOWED_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
}
SAFE_FILENAME = re.compile(r"^[^/\\\x00]+$")


async def read_upload(upload: UploadFile, settings: Settings) -> tuple[str, str, bytes]:
    filename = Path(upload.filename or "").name
    if not filename or not SAFE_FILENAME.fullmatch(filename):
        raise InvalidDocumentError("A safe filename is required")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_TYPES:
        raise InvalidDocumentError("Unsupported document type")
    if upload.content_type not in ALLOWED_TYPES[extension]:
        raise InvalidDocumentError("Invalid document MIME type")

    chunks: list[bytes] = []
    total = 0
    while data := await upload.read(1024 * 1024):
        total += len(data)
        if total > settings.max_upload_size_bytes:
            raise DocumentTooLargeError("The document exceeds the maximum size")
        chunks.append(data)

    content = b"".join(chunks)
    if not content:
        raise InvalidDocumentError("The document is empty")
    _validate_signature(extension, content)
    return filename, extension, content


def _validate_signature(extension: str, content: bytes) -> None:
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise InvalidDocumentError("Invalid PDF signature")
    if extension == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as exc:
            raise InvalidDocumentError("Invalid DOCX signature") from exc
        if not {"[Content_Types].xml", "word/document.xml"}.issubset(names):
            raise InvalidDocumentError("Invalid DOCX signature")
    if extension == ".txt":
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("TXT files must be valid UTF-8") from exc
        if "\x00" in decoded:
            raise InvalidDocumentError("Binary content is not allowed in TXT files")
