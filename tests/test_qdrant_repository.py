import pytest
from langchain_core.embeddings import Embeddings

from app.exceptions import VectorStoreError
from app.repositories.qdrant_repository import QdrantRepository


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        raise RuntimeError("provider failed")


class FakeQdrant:
    def __init__(self, size=2):
        self.size = size
        self.ids = set()
        self.upserts = 0

    def get_collections(self):
        collections = [type("Collection", (), {"name": "support-documents"})()] if self.size == 3 else []
        return type("Collections", (), {"collections": collections})()

    def create_collection(self, **kwargs):
        vectors_config = kwargs["vectors_config"]
        if isinstance(vectors_config, dict):
            vectors_config = next(iter(vectors_config.values()))
        self.size = vectors_config.size

    def get_collection(self, name):
        params = type("Params", (), {"vectors": type("Vectors", (), {"size": self.size})()})()
        return type("Info", (), {"config": type("Config", (), {"params": params})()})()

    def retrieve(self, **kwargs):
        return [object() for point_id in kwargs["ids"] if point_id in self.ids]

    def upsert(self, **kwargs):
        self.upserts += 1
        self.ids.update(point.id for point in kwargs["points"])


def test_duplicate_indexing_is_idempotent(settings):
    client = FakeQdrant()
    repository = QdrantRepository(settings, client=client)
    from langchain_core.documents import Document

    chunks = [Document(page_content="hello")]
    first = repository.index("doc_1", "hash_1", "a.txt", ".txt", chunks, FakeEmbeddings())
    second = repository.index("doc_1", "hash_1", "a.txt", ".txt", chunks, FakeEmbeddings())
    assert first == (1, False)
    assert second == (1, True)
    assert client.upserts == 1


def test_embedding_failure_is_wrapped(settings):
    repository = QdrantRepository(settings, client=FakeQdrant())
    from langchain_core.documents import Document

    with pytest.raises(VectorStoreError, match="Vector indexing failed"):
        repository.index("doc_1", "hash_1", "a.txt", ".txt", [Document(page_content="x")], FailingEmbeddings())


def test_dimension_mismatch_is_rejected(settings):
    repository = QdrantRepository(settings, client=FakeQdrant(size=3))
    from langchain_core.documents import Document

    with pytest.raises(VectorStoreError, match="dimension"):
        repository.index("doc_1", "hash_1", "a.txt", ".txt", [Document(page_content="x")], FakeEmbeddings())
