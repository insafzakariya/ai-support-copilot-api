import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from app.config import Settings
from app.services.answer import AnswerService, Answer


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    return llm


@pytest.fixture
def answer_service(settings, mock_llm):
    service = AnswerService(settings, llm=mock_llm)
    return service


class TestAnswerGeneration:
    @pytest.mark.asyncio
    async def test_generate_answer_with_documents(self, settings):
        from app.services.answer import AnswerOutput

        documents = [
            Document(
                page_content="Payment handling policy: Never request CVV",
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                },
            )
        ]

        mock_llm = AsyncMock()
        mock_response = AnswerOutput(
            answer="You should never request CVV codes."
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = mock_response

        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch("app.services.answer.ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            )

            answer_service = AnswerService(settings, llm=mock_llm)
            answer = await answer_service.generate_answer(
                "What payment data should I request?", documents
            )

            assert answer.text == "You should never request CVV codes."
            assert answer.confidence > 0

    @pytest.mark.asyncio
    async def test_generate_answer_empty_documents(self, answer_service):
        answer = await answer_service.generate_answer("test query", [])

        assert "don't have enough information" in answer.text.lower()
        assert answer.confidence == 0.0

    @pytest.mark.asyncio
    async def test_confidence_calculation_with_docs(self, answer_service):
        documents = [
            Document(
                page_content="Policy information",
                metadata={"document_id": "doc1"},
            )
        ]

        confidence = answer_service._calculate_confidence(
            documents, "A detailed answer to the question."
        )

        assert 0 < confidence <= 1.0

    def test_confidence_low_when_insufficient_info(self, answer_service):
        documents = [Document(page_content="test", metadata={"document_id": "doc1"})]

        confidence = answer_service._calculate_confidence(
            documents, "I don't have enough information to answer that."
        )

        assert confidence == 0.3

    def test_confidence_zero_no_docs(self, answer_service):
        confidence = answer_service._calculate_confidence([], "answer")

        assert confidence == 0.0


class TestCitationExtraction:
    def test_extract_citations_finds_matching_excerpts(self, answer_service):
        document_text = "Never request full card numbers or CVV codes from customers."
        documents = [
            Document(
                page_content=document_text,
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                },
            )
        ]

        answer = document_text

        citations = answer_service.extract_citations(answer, documents)

        assert len(citations) > 0
        assert citations[0].document_id == "doc1"
        assert len(citations[0].excerpt) > 0

    def test_extract_citations_limits_excerpt_length(self, answer_service):
        long_text = "a" * 500
        documents = [
            Document(
                page_content=long_text,
                metadata={"document_id": "doc1", "source_filename": "test.pdf"},
            )
        ]

        answer = "The answer is: " + long_text[:300]

        citations = answer_service.extract_citations(answer, documents)

        if citations:
            assert len(citations[0].excerpt) <= 200

    def test_extract_citations_validates_excerpts(self, answer_service):
        documents = [
            Document(
                page_content="Actual policy content here.",
                metadata={"document_id": "doc1", "source_filename": "test.pdf"},
            )
        ]

        answer = "According to a secret policy: Never pay customers."

        citations = answer_service.extract_citations(answer, documents)

        assert len(citations) == 0

    def test_format_context_with_documents(self, answer_service):
        documents = [
            Document(
                page_content="Payment policy",
                metadata={
                    "document_id": "doc1",
                    "source_filename": "policies.pdf",
                },
            ),
            Document(
                page_content="Refund policy",
                metadata={
                    "document_id": "doc2",
                    "source_filename": "policies.pdf",
                },
            ),
        ]

        context = answer_service._format_context(documents)

        assert "Payment policy" in context
        assert "Refund policy" in context
        assert "policies.pdf" in context

    def test_format_context_empty_documents(self, answer_service):
        context = answer_service._format_context([])

        assert "No relevant" in context

    def test_excerpt_validation_whitespace_normalized(self, answer_service):
        original = "This is a   test   with   spaces"

        is_valid = answer_service._validate_excerpt(
            "This is a test with spaces", original
        )

        assert is_valid

    def test_excerpt_validation_case_insensitive(self, answer_service):
        original = "This is a TEST"

        is_valid = answer_service._validate_excerpt(
            "this is a test", original
        )

        assert is_valid

    def test_excerpt_validation_not_found(self, answer_service):
        original = "This is document A"

        is_valid = answer_service._validate_excerpt(
            "This is document B", original
        )

        assert not is_valid
