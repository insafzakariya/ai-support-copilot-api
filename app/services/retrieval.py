import logging
from typing import NamedTuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..exceptions import NoRelevantDocumentsError, RetrievalError
from ..repositories.qdrant_repository import QdrantRepository

logger = logging.getLogger(__name__)


class RetrievalResult(NamedTuple):
    document: Document
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        qdrant_repo: QdrantRepository,
        embeddings: Embeddings,
    ) -> None:
        self.settings = settings
        self.qdrant_repo = qdrant_repo
        self.embeddings = embeddings

    async def semantic_search(
        self, question: str, top_k: int, filter_dict: dict | None = None
    ) -> list[RetrievalResult]:
        """
        Semantic similarity search using embeddings.
        Returns documents ranked by vector similarity score.
        """
        try:
            embedding = self.embeddings.embed_query(question)

            documents = self.qdrant_repo.search_semantic(
                embedding=embedding, top_k=top_k, filter_dict=filter_dict
            )

            if not documents:
                logger.debug("semantic_search: no documents found")
                raise NoRelevantDocumentsError(
                    "No semantically similar documents found"
                )

            results = [
                RetrievalResult(
                    document=doc,
                    semantic_score=doc.metadata.get("semantic_score", 0.0),
                    keyword_score=0.0,
                    combined_score=doc.metadata.get("semantic_score", 0.0),
                    rank=idx,
                )
                for idx, doc in enumerate(documents)
            ]

            logger.info(
                "semantic_search_success",
                extra={"question_length": len(question), "results_count": len(results)},
            )
            return results
        except NoRelevantDocumentsError:
            raise
        except Exception as exc:
            logger.exception("semantic_search_failed")
            raise RetrievalError("Semantic search failed") from exc

    async def keyword_search(
        self, question: str, top_k: int, filter_dict: dict | None = None
    ) -> list[RetrievalResult]:
        """
        Keyword-based search using term matching.
        Useful for capturing exact matches semantic search might miss.
        """
        try:
            keywords = self._extract_keywords(question)

            if not keywords:
                logger.debug("keyword_search: no keywords extracted")
                raise NoRelevantDocumentsError(
                    "No valid keywords extracted from question"
                )

            documents = self.qdrant_repo.search_keyword(
                keywords=keywords, top_k=top_k, filter_dict=filter_dict
            )

            if not documents:
                logger.debug("keyword_search: no documents found")
                raise NoRelevantDocumentsError("No keyword matches found")

            results = [
                RetrievalResult(
                    document=doc,
                    semantic_score=0.0,
                    keyword_score=doc.metadata.get("keyword_score", 0.0),
                    combined_score=doc.metadata.get("keyword_score", 0.0),
                    rank=idx,
                )
                for idx, doc in enumerate(documents)
            ]

            logger.info(
                "keyword_search_success",
                extra={"keywords": keywords, "results_count": len(results)},
            )
            return results
        except NoRelevantDocumentsError:
            raise
        except Exception as exc:
            logger.exception("keyword_search_failed")
            raise RetrievalError("Keyword search failed") from exc

    async def hybrid_search(
        self,
        question: str,
        top_k: int,
        include_reranking: bool = True,
        filter_dict: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid search combining semantic and keyword retrieval with RRF fusion.
        Falls back to semantic-only if keyword search fails.
        """
        try:
            semantic_results = await self.semantic_search(
                question=question, top_k=top_k * 2, filter_dict=filter_dict
            )
        except NoRelevantDocumentsError:
            logger.debug("hybrid_search: semantic search returned no results")
            semantic_results = []

        keyword_results = []
        try:
            keyword_results = await self.keyword_search(
                question=question, top_k=top_k * 2, filter_dict=filter_dict
            )
        except NoRelevantDocumentsError:
            logger.debug("hybrid_search: keyword search returned no results")

        if not semantic_results and not keyword_results:
            raise NoRelevantDocumentsError(
                "No relevant documents found from hybrid search"
            )

        rrf_scores = self._calculate_rrf_scores(semantic_results, keyword_results)
        merged = self._merge_results(semantic_results, keyword_results, rrf_scores)
        final_results = sorted(merged, key=lambda x: x.combined_score, reverse=True)[
            :top_k
        ]

        if include_reranking:
            final_results = await self.rerank(
                question=question,
                documents=[r.document for r in final_results],
                top_k=top_k,
            )

        logger.info(
            "hybrid_search_success",
            extra={
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results),
                "final_count": len(final_results),
                "reranked": include_reranking,
            },
        )
        return final_results

    async def rerank(
        self, question: str, documents: list[Document], top_k: int
    ) -> list[RetrievalResult]:
        """
        LLM-based reranking using Groq for improved relevance.
        Scores documents 1-10 and re-orders them.
        """
        if not self.settings.reranker_enabled or not documents:
            return [
                RetrievalResult(
                    document=doc,
                    semantic_score=0.0,
                    keyword_score=0.0,
                    combined_score=1.0 / (self.settings.rrf_k + idx + 1),
                    rank=idx,
                )
                for idx, doc in enumerate(documents[:top_k])
            ]

        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate
            import json
            import re

            reranker = ChatGroq(
                model=self.settings.reranker_model,
                api_key=self.settings.groq_api_key,
                temperature=self.settings.reranker_temperature,
            )

            rerank_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a relevance expert. Rate each document's relevance to the query on a scale of 1-10.",
                    ),
                    (
                        "human",
                        """Query: {question}

Documents:
{documents}

Return a JSON array where each object has "index" and "score" (1-10) properties, sorted by score descending.""",
                    ),
                ]
            )

            docs_text = "\n\n".join(
                f"[{idx}] {doc.page_content[:500]}"
                for idx, doc in enumerate(documents[:top_k])
            )

            chain = rerank_prompt | reranker
            response = await chain.ainvoke(
                {"question": question, "documents": docs_text}
            )

            json_match = re.search(r"\[[\s\S]*\]", response.content)
            if json_match:
                scores = json.loads(json_match.group())
                score_map = {item.get("index"): item.get("score", 0) for item in scores}

                sorted_indices = sorted(
                    range(len(documents[:top_k])),
                    key=lambda x: score_map.get(x, 0.0),
                    reverse=True,
                )

                results = [
                    RetrievalResult(
                        document=documents[idx],
                        semantic_score=0.0,
                        keyword_score=0.0,
                        combined_score=score_map.get(idx, 0.0) / 10.0,
                        rank=rank,
                    )
                    for rank, idx in enumerate(sorted_indices)
                ]

                logger.info(
                    "rerank_success",
                    extra={
                        "documents_reranked": len(results),
                        "score_map": score_map,
                    },
                )
                return results
            else:
                logger.warning("rerank_invalid_response: no JSON found")
                raise ValueError("Reranker returned invalid JSON")

        except Exception as exc:
            logger.warning("rerank_failed, falling back to original order")
            return [
                RetrievalResult(
                    document=doc,
                    semantic_score=0.0,
                    keyword_score=0.0,
                    combined_score=1.0 / (self.settings.rrf_k + idx + 1),
                    rank=idx,
                )
                for idx, doc in enumerate(documents[:top_k])
            ]

    @staticmethod
    def _extract_keywords(question: str, min_length: int = 3) -> list[str]:
        """Extract keywords from question by tokenizing and filtering."""
        tokens = question.lower().split()
        keywords = [t for t in tokens if len(t) >= min_length]
        return keywords

    def _calculate_rrf_scores(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> dict[str, float]:
        """
        Calculate RRF (Reciprocal Rank Fusion) scores.
        Formula: score = 1 / (k + rank), where k=60
        """
        rrf_scores: dict[str, float] = {}

        for rank, result in enumerate(semantic_results):
            doc_id = result.document.metadata.get("document_id", "")
            rrf_score = 1.0 / (self.settings.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        for rank, result in enumerate(keyword_results):
            doc_id = result.document.metadata.get("document_id", "")
            rrf_score = 1.0 / (self.settings.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        return rrf_scores

    def _merge_results(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        rrf_scores: dict[str, float],
    ) -> list[RetrievalResult]:
        """Merge and deduplicate results from semantic and keyword search."""
        seen: dict[str, RetrievalResult] = {}

        for result in semantic_results:
            doc_id = result.document.metadata.get("document_id", "")
            if doc_id not in seen:
                seen[doc_id] = result

        for result in keyword_results:
            doc_id = result.document.metadata.get("document_id", "")
            if doc_id not in seen:
                seen[doc_id] = result

        merged = []
        for doc_id, result in seen.items():
            combined_score = rrf_scores.get(doc_id, 0.0)
            merged.append(
                RetrievalResult(
                    document=result.document,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    combined_score=combined_score,
                    rank=len(merged),
                )
            )

        return merged
