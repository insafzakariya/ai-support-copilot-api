from langchain_core.documents import Document

from app.services.document_parsing import normalize_documents
from app.services.text_processing import split_documents


def test_normalization_and_chunk_generation(settings):
    documents = normalize_documents([Document(page_content=" one\r\n\r\n two ")])
    chunks = split_documents(documents, settings)
    assert documents[0].page_content == "one\n\ntwo"
    assert chunks
    assert all(chunk.page_content for chunk in chunks)
