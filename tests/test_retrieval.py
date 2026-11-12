import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from app.config import Settings
from app.exceptions import NoRelevantDocumentsError, RetrievalError
from app.services.retrieval import RetrievalService, RetrievalResult


@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1] * 384
    return embeddings


@pytest.fixture
def mock_qdrant_repo():
    repo = MagicMock()
    return repo


@pytest.fixture
def retrieval_service(settings, mock_qdrant_repo, mock_embeddings):
    return RetrievalService(settings, mock_qdrant_repo, mock_embeddings)


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_semantic.return_value = [
            Document(
                page_content="Payment handling policy",
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                    "semantic_score": 0.95,
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="Refund policy",
                metadata={
                    "document_id": "doc2",
                    "source_filename": "policies.pdf",
                    "semantic_score": 0.85,
                    "chunk_index": 1,
                },
            ),
        ]

        results = await retrieval_service.semantic_search(
            "How to handle payments?", top_k=5
        )

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].semantic_score == 0.95
        assert results[1].semantic_score == 0.85
        assert results[0].combined_score == 0.95

    @pytest.mark.asyncio
    async def test_semantic_search_respects_top_k(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_semantic.return_value = [
            Document(
                page_content=f"Document {i}",
                metadata={
                    "document_id": f"doc{i}",
                    "source_filename": "test.pdf",
                    "semantic_score": 0.9 - i * 0.05,
                    "chunk_index": i,
                },
            )
            for i in range(5)
        ]

        results = await retrieval_service.semantic_search(
            "test query", top_k=5
        )

        mock_qdrant_repo.search_semantic.assert_called_once()
        call_args = mock_qdrant_repo.search_semantic.call_args
        assert call_args.kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_semantic_search_no_documents_raises_error(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_semantic.return_value = []

        with pytest.raises(NoRelevantDocumentsError):
            await retrieval_service.semantic_search("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_semantic_search_embed_query_called(
        self, retrieval_service, mock_qdrant_repo, mock_embeddings
    ):
        mock_qdrant_repo.search_semantic.return_value = [
            Document(
                page_content="test",
                metadata={
                    "document_id": "doc1",
                    "semantic_score": 0.95,
                    "chunk_index": 0,
                },
            )
        ]

        await retrieval_service.semantic_search("test query", top_k=5)

        mock_embeddings.embed_query.assert_called_once_with("test query")


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_keyword_search_returns_results(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_keyword.return_value = [
            Document(
                page_content="Payment handling policy",
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                    "keyword_score": 2.0,
                    "chunk_index": 0,
                },
            ),
        ]

        results = await retrieval_service.keyword_search(
            "How to handle payments?", top_k=5
        )

        assert len(results) == 1
        assert results[0].keyword_score == 2.0

    @pytest.mark.asyncio
    async def test_keyword_search_extracts_keywords(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_keyword.return_value = [
            Document(
                page_content="test",
                metadata={
                    "document_id": "doc1",
                    "keyword_score": 1.0,
                    "chunk_index": 0,
                },
            )
        ]

        await retrieval_service.keyword_search("payment handling", top_k=5)

        mock_qdrant_repo.search_keyword.assert_called_once()
        call_args = mock_qdrant_repo.search_keyword.call_args
        keywords = call_args.kwargs["keywords"]
        assert "payment" in keywords
        assert "handling" in keywords

    @pytest.mark.asyncio
    async def test_keyword_search_filters_short_words(self, retrieval_service):
        keywords = RetrievalService._extract_keywords(
            "a the to payment handling is very"
        )

        assert "a" not in keywords
        assert "to" not in keywords
        assert "is" not in keywords
        assert "the" in keywords
        assert "payment" in keywords
        assert "handling" in keywords
        assert "very" in keywords

    @pytest.mark.asyncio
    async def test_keyword_search_no_documents_raises_error(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_keyword.return_value = []

        with pytest.raises(NoRelevantDocumentsError):
            await retrieval_service.keyword_search("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_keyword_search_no_keywords_raises_error(
        self, retrieval_service
    ):
        with pytest.raises(NoRelevantDocumentsError):
            await retrieval_service.keyword_search("a b c", top_k=5)


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(
        self, retrieval_service, mock_qdrant_repo
    ):
        semantic_docs = [
            Document(
                page_content="Payment policy",
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                    "semantic_score": 0.95,
                    "chunk_index": 0,
                },
            ),
        ]
        keyword_docs = [
            Document(
                page_content="Payment handling",
                metadata={
                    "document_id": "doc2",
                    "source_filename": "policies.pdf",
                    "keyword_score": 2.0,
                    "chunk_index": 1,
                },
            ),
        ]

        mock_qdrant_repo.search_semantic.return_value = semantic_docs
        mock_qdrant_repo.search_keyword.return_value = keyword_docs

        results = await retrieval_service.hybrid_search(
            "How to handle payments?", top_k=5, include_reranking=False
        )

        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_handles_missing_semantic(
        self, retrieval_service, mock_qdrant_repo
    ):
        keyword_docs = [
            Document(
                page_content="Payment handling",
                metadata={
                    "document_id": "doc2",
                    "source_filename": "policies.pdf",
                    "keyword_score": 2.0,
                    "chunk_index": 1,
                },
            ),
        ]

        mock_qdrant_repo.search_semantic.side_effect = NoRelevantDocumentsError()
        mock_qdrant_repo.search_keyword.return_value = keyword_docs

        results = await retrieval_service.hybrid_search(
            "payment handling", top_k=5, include_reranking=False
        )

        assert len(results) > 0
        assert results[0].document.metadata["document_id"] == "doc2"

    @pytest.mark.asyncio
    async def test_hybrid_search_no_results_raises_error(
        self, retrieval_service, mock_qdrant_repo
    ):
        mock_qdrant_repo.search_semantic.side_effect = NoRelevantDocumentsError()
        mock_qdrant_repo.search_keyword.side_effect = NoRelevantDocumentsError()

        with pytest.raises(NoRelevantDocumentsError):
            await retrieval_service.hybrid_search(
                "test query", top_k=5, include_reranking=False
            )


class TestRRFCalculation:
    def test_rrf_scores_calculation(self, retrieval_service):
        semantic_results = [
            RetrievalResult(
                document=Document(
                    page_content="test",
                    metadata={"document_id": "doc1"},
                ),
                semantic_score=0.95,
                keyword_score=0.0,
                combined_score=0.95,
                rank=0,
            ),
            RetrievalResult(
                document=Document(
                    page_content="test2",
                    metadata={"document_id": "doc2"},
                ),
                semantic_score=0.85,
                keyword_score=0.0,
                combined_score=0.85,
                rank=1,
            ),
        ]

        keyword_results = [
            RetrievalResult(
                document=Document(
                    page_content="test3",
                    metadata={"document_id": "doc1"},
                ),
                semantic_score=0.0,
                keyword_score=2.0,
                combined_score=2.0,
                rank=0,
            ),
        ]

        rrf_scores = retrieval_service._calculate_rrf_scores(
            semantic_results, keyword_results
        )

        assert "doc1" in rrf_scores
        assert "doc2" in rrf_scores
        assert rrf_scores["doc1"] > 0
        assert rrf_scores["doc2"] > 0

    def test_rrf_formula(self, retrieval_service, settings):
        semantic_results = [
            RetrievalResult(
                document=Document(
                    page_content="test",
                    metadata={"document_id": "doc1"},
                ),
                semantic_score=0.95,
                keyword_score=0.0,
                combined_score=0.95,
                rank=0,
            ),
        ]

        rrf_scores = retrieval_service._calculate_rrf_scores(
            semantic_results, []
        )

        expected_score = 1.0 / (settings.rrf_k + 1)
        assert abs(rrf_scores["doc1"] - expected_score) < 0.0001

    def test_merge_deduplicates(self, retrieval_service):
        semantic_results = [
            RetrievalResult(
                document=Document(
                    page_content="test",
                    metadata={"document_id": "doc1"},
                ),
                semantic_score=0.95,
                keyword_score=0.0,
                combined_score=0.95,
                rank=0,
            ),
        ]

        keyword_results = [
            RetrievalResult(
                document=Document(
                    page_content="test (duplicate)",
                    metadata={"document_id": "doc1"},
                ),
                semantic_score=0.0,
                keyword_score=2.0,
                combined_score=2.0,
                rank=0,
            ),
        ]

        rrf_scores = {"doc1": 0.05}
        merged = retrieval_service._merge_results(
            semantic_results, keyword_results, rrf_scores
        )

        assert len(merged) == 1
        assert merged[0].document.metadata["document_id"] == "doc1"
