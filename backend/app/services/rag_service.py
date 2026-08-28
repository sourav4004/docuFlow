"""RAG (Retrieval-Augmented Generation) orchestration service.

Connects the retrieval pipeline to the LLM service for grounded
question answering.

Pipeline:

User question
    ↓
Question validation
    ↓
Retrieval (embedding → vector search → filtering → context)
    ↓
[If no context → return "no information" response]
    ↓
RAG prompt construction
    ↓
LLM generation
    ↓
Response validation
    ↓
Structured RAG response with sources

No LLM call is made if retrieval produces zero acceptable chunks.
This prevents hallucination when documents don't contain the answer.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from .retrieval_service import (
    retrieve_context,
    RetrievalResult,
    RetrievalResponse,
    QueryValidationError,
    RetrievalError,
    DEFAULT_MIN_SIMILARITY,
)
from .rag_prompt import get_system_prompt, build_rag_user_prompt
from .llm.service import LLMService
from .llm.base import LLMProviderError, LLMTimeoutError, LLMConfigurationError

logger = logging.getLogger(__name__)

# Maximum question length
MAX_QUESTION_LENGTH = 2000


class RAGError(Exception):
    """Raised when RAG generation fails."""


@dataclass
class SourceReference:
    """A source chunk used to generate the answer."""
    document_id: int
    filename: Optional[str]
    chunk_id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    similarity_score: Optional[float] = None


@dataclass
class RAGResponse:
    """Structured RAG answer with sources and metadata."""
    answer: str
    sources: List[SourceReference]
    grounded: bool
    retrieval_count: int
    model: Optional[str] = None
    provider: Optional[str] = None


def validate_question(question: str) -> str:
    """Validate and clean a user question.

    Args:
        question: Raw user question.

    Returns:
        Cleaned question string.

    Raises:
        RAGError: If question is invalid.
    """
    if not isinstance(question, str):
        raise RAGError("Question must be a string")

    cleaned = question.strip()

    if not cleaned:
        raise RAGError("Question must not be empty")

    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise RAGError(
            f"Question must not exceed {MAX_QUESTION_LENGTH} characters"
        )

    return cleaned


def answer_question(
    db: Session,
    user_id: int,
    question: str,
    document_id: Optional[int] = None,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    embedding_provider=None,
    llm_service: Optional[LLMService] = None,
) -> RAGResponse:
    """Answer a question using RAG over the user's documents.

    Full pipeline: validate → retrieve → build prompt → LLM → response.

    If retrieval produces zero acceptable chunks, returns a controlled
    "no information" response WITHOUT calling the LLM.

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID.
        question: The user's question.
        document_id: Optional. Restrict to one document.
        top_k: Number of retrieval candidates.
        min_similarity: Minimum similarity threshold.
        embedding_provider: Optional override for testing.
        llm_service: Optional override for testing.

    Returns:
        RAGResponse with answer, sources, and metadata.

    Raises:
        RAGError: If validation or LLM generation fails.
    """
    start_time = time.monotonic()

    # 1. Validate question
    cleaned_question = validate_question(question)

    # 2. Retrieve relevant context
    if llm_service is None:
        llm_service = LLMService()

    try:
        retrieval = retrieve_context(
            db=db,
            user_id=user_id,
            query=cleaned_question,
            document_id=document_id,
            top_k=top_k,
            min_similarity=min_similarity,
            provider=embedding_provider,
        )
    except QueryValidationError as exc:
        raise RAGError(f"Invalid question: {exc}") from exc
    except RetrievalError as exc:
        raise RAGError(f"Retrieval failed: {exc}") from exc

    # 3. No context → no hallucination
    if retrieval.total_results == 0:
        elapsed = time.monotonic() - start_time
        logger.info(
            "RAG: no relevant context found (user=%s, elapsed=%.2fs)",
            user_id, elapsed,
        )
        return RAGResponse(
            answer=(
                "I don't have enough information in the provided documents "
                "to answer this question."
            ),
            sources=[],
            grounded=False,
            retrieval_count=0,
        )

    # 4. Build sources list (from retrieval results, before LLM)
    sources = [
        SourceReference(
            document_id=r.document_id,
            filename=r.original_filename,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            page_start=r.page_start,
            page_end=r.page_end,
            similarity_score=r.similarity_score,
        )
        for r in retrieval.results
    ]

    # 5. Build RAG prompt
    system_prompt = get_system_prompt()
    user_prompt = build_rag_user_prompt(cleaned_question, retrieval.context)

    # 6. Call LLM
    try:
        llm_response = llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except LLMTimeoutError as exc:
        raise RAGError(f"LLM request timed out: {exc}") from exc
    except LLMConfigurationError as exc:
        raise RAGError(f"LLM configuration error: {exc}") from exc
    except LLMProviderError as exc:
        raise RAGError(f"LLM generation failed: {exc}") from exc

    # 7. Validate LLM response
    answer_text = llm_response.text.strip()
    if not answer_text:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "RAG: LLM returned empty response (user=%s, elapsed=%.2fs)",
            user_id, elapsed,
        )
        return RAGResponse(
            answer=(
                "I don't have enough information in the provided documents "
                "to answer this question."
            ),
            sources=sources,
            grounded=False,
            retrieval_count=retrieval.total_results,
            model=llm_response.model,
            provider=llm_response.provider,
        )

    elapsed = time.monotonic() - start_time
    logger.info(
        "RAG: answer generated (user=%s, sources=%d, model=%s, elapsed=%.2fs)",
        user_id, len(sources), llm_response.model, elapsed,
    )

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        grounded=True,
        retrieval_count=retrieval.total_results,
        model=llm_response.model,
        provider=llm_response.provider,
    )
