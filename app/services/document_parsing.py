from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..config import Settings
from ..exceptions import InvalidDocumentError


def load_documents(content: bytes, extension: str, settings: Settings) -> list[Document]:
    with tempfile.NamedTemporaryFile(suffix=extension) as temporary:
        temporary.write(content)
        temporary.flush()
        path = Path(temporary.name)
        if extension == ".pdf":
            documents = PyPDFLoader(str(path)).load()
            if len(documents) > settings.max_pdf_pages:
                raise InvalidDocumentError("The PDF exceeds the page limit")
        elif extension == ".docx":
            documents = Docx2txtLoader(str(path)).load()
        else:
            documents = TextLoader(str(path), encoding="utf-8").load()

    text_length = sum(len(document.page_content) for document in documents)
    if text_length == 0 or text_length > settings.max_extracted_text_chars:
        raise InvalidDocumentError("The document has no usable text or is too large")
    return documents


def normalize_documents(documents: list[Document]) -> list[Document]:
    normalized: list[Document] = []
    for document in documents:
        text = document.page_content.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = "\n\n".join(part.strip() for part in text.split("\n\n") if part.strip())
        if text:
            normalized.append(Document(page_content=text, metadata=dict(document.metadata)))
    if not normalized:
        raise InvalidDocumentError("The document has no usable text")
    return normalized
