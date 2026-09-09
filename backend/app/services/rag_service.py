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
import math
from dataclasses import dataclass
from typing import List, Optional, Iterator, AsyncIterator

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
from .rag_prompt import get_system_prompt, build_rag_user_prompt, build_conversation_aware_user_prompt
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
    # Enhanced citation fields
    keyword_score: Optional[float] = None
    final_score: Optional[float] = None
    match_type: str = "vector"  # "vector", "keyword", or "both"


@dataclass
class ConfidenceScore:
    """Confidence/grounding signals for a RAG response."""
    level: str = "LOW"  # "LOW", "MEDIUM", "HIGH"
    grounding_score: float = 0.0  # 0.0-1.0
    supporting_sources: int = 0
    avg_relevance: float = 0.0
    source_diversity: int = 0  # number of distinct documents
    min_relevance: float = 0.0


@dataclass
class RAGResponse:
    """Structured RAG answer with sources and metadata."""
    answer: str
    sources: List[SourceReference]
    grounded: bool
    retrieval_count: int
    model: Optional[str] = None
    provider: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    # Observability
    total_time_ms: float = 0.0  # Total RAG pipeline time in milliseconds
    retrieval_time_ms: float = 0.0  # Retrieval time in milliseconds
    llm_time_ms: float = 0.0  # LLM generation time in milliseconds


def _build_sources(retrieval) -> List[SourceReference]:
    """Build SourceReference list from RetrievalResponse with enhanced metadata."""
    return [
        SourceReference(
            document_id=r.document_id,
            filename=r.original_filename,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            page_start=r.page_start,
            page_end=r.page_end,
            similarity_score=r.similarity_score,
            keyword_score=r.keyword_score,
            final_score=r.final_score,
            match_type=r.match_type,
        )
        for r in retrieval.results
    ]


def _compute_confidence(retrieval, sources: List[SourceReference]) -> ConfidenceScore:
    """Compute confidence/grounding signals for a RAG response.

    Signals:
    - Number of supporting sources
    - Average relevance score
    - Minimum relevance score
    - Source diversity (distinct documents)
    - Overall grounding score (weighted combination)
    """
    if not sources:
        return ConfidenceScore(
            level="LOW", grounding_score=0.0, supporting_sources=0,
            avg_relevance=0.0, source_diversity=0, min_relevance=0.0,
        )

    # Collect scores
    scores = []
    for s in sources:
        score = s.final_score if s.final_score is not None else s.similarity_score
        if score is not None:
            scores.append(score)

    avg_rel = sum(scores) / len(scores) if scores else 0.0
    min_rel = min(scores) if scores else 0.0
    source_diversity = len(set(s.document_id for s in sources))
    supporting = len(sources)

    # Grounding score: weighted combination
    # - avg_relevance (40%)
    # - source_count factor (30%): more sources = higher confidence, capped
    # - diversity factor (20%): more documents = higher confidence
    # - min_relevance (10%): weakest link
    source_count_factor = min(supporting / 5.0, 1.0)  # 5+ sources = max
    diversity_factor = min(source_diversity / 3.0, 1.0)  # 3+ docs = max
    grounding = (
        0.4 * avg_rel
        + 0.3 * source_count_factor
        + 0.2 * diversity_factor
        + 0.1 * min_rel
    )
    grounding = max(0.0, min(1.0, grounding))

    # Level classification
    if grounding >= 0.7:
        level = "HIGH"
    elif grounding >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return ConfidenceScore(
        level=level,
        grounding_score=round(grounding, 4),
        supporting_sources=supporting,
        avg_relevance=round(avg_rel, 4),
        source_diversity=source_diversity,
        min_relevance=round(min_rel, 4),
    )


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

    retrieval_start = time.monotonic()
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
    retrieval_time = (time.monotonic() - retrieval_start) * 1000

    # 3. No context → no hallucination
    if retrieval.total_results == 0:
        total_time = (time.monotonic() - start_time) * 1000
        logger.info(
            "RAG: no relevant context found (elapsed=%.1fms)",
            total_time,
        )
        return RAGResponse(
            answer=(
                "I don't have enough information in the provided documents "
                "to answer this question."
            ),
            sources=[],
            grounded=False,
            retrieval_count=0,
            total_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
        )

    # 4. Build sources list with enhanced metadata
    sources = _build_sources(retrieval)

    # 5. Compute confidence score
    confidence = _compute_confidence(retrieval, sources)

    # 6. Build RAG prompt
    system_prompt = get_system_prompt()
    user_prompt = build_rag_user_prompt(cleaned_question, retrieval.context)

    # 7. Call LLM
    llm_start = time.monotonic()
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
    llm_time = (time.monotonic() - llm_start) * 1000

    # 8. Validate LLM response
    answer_text = llm_response.text.strip()
    if not answer_text:
        total_time = (time.monotonic() - start_time) * 1000
        logger.warning(
            "RAG: LLM returned empty response (elapsed=%.1fms)",
            total_time,
        )
        return RAGResponse(
            answer=(
                "I don't have enough information in the provided documents "
                "to answer this question."
            ),
            sources=sources,
            grounded=False,
            retrieval_count=retrieval.total_results,
            confidence=confidence,
            model=llm_response.model,
            provider=llm_response.provider,
            total_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            llm_time_ms=llm_time,
        )

    total_time = (time.monotonic() - start_time) * 1000
    logger.info(
        "RAG: answer generated (sources=%d, model=%s, retrieval=%.1fms, llm=%.1fms, total=%.1fms)",
        len(sources), llm_response.model, retrieval_time, llm_time, total_time,
    )

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        grounded=True,
        retrieval_count=retrieval.total_results,
        confidence=confidence,
        model=llm_response.model,
        provider=llm_response.provider,
        total_time_ms=total_time,
        retrieval_time_ms=retrieval_time,
        llm_time_ms=llm_time,
    )


def answer_question_with_history(
    db: Session,
    user_id: int,
    question: str,
    conversation_history: str = "",
    document_id: Optional[int] = None,
    collection_id: Optional[int] = None,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    embedding_provider=None,
    llm_service: Optional[LLMService] = None,
) -> RAGResponse:
    """Answer a question using RAG with conversation history context.

    Extends answer_question() by including conversation history in the
    prompt. History helps the model understand follow-up references
    (e.g., "that" referring to a previous topic), but the document
    context remains the authoritative source for factual answers.

    Pipeline: validate → retrieve → build prompt with history → LLM → response.

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID.
        question: The user's current question.
        conversation_history: Formatted conversation history string.
            Empty for first messages. Not treated as factual truth.
        document_id: Optional. Restrict to one document.
        collection_id: Optional. Restrict to documents in this collection.
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
            collection_id=collection_id,
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
            "RAG (history-aware): no relevant context found (user=%s, elapsed=%.2fs)",
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

    # 4. Build sources list with enhanced metadata
    sources = _build_sources(retrieval)

    # 5. Compute confidence score
    confidence = _compute_confidence(retrieval, sources)

    # 6. Build RAG prompt with conversation history
    system_prompt = get_system_prompt()
    user_prompt = build_conversation_aware_user_prompt(
        question=cleaned_question,
        context=retrieval.context,
        conversation_history=conversation_history,
    )

    # 7. Call LLM
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

    # 8. Validate LLM response
    answer_text = llm_response.text.strip()
    if not answer_text:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "RAG (history-aware): LLM returned empty response (user=%s, elapsed=%.2fs)",
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
            confidence=confidence,
            model=llm_response.model,
            provider=llm_response.provider,
        )

    elapsed = time.monotonic() - start_time
    logger.info(
        "RAG (history-aware): answer generated (user=%s, sources=%d, history=%s, model=%s, elapsed=%.2fs)",
        user_id, len(sources), "yes" if conversation_history.strip() else "no",
        llm_response.model, elapsed,
    )

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        grounded=True,
        retrieval_count=retrieval.total_results,
        confidence=confidence,
        model=llm_response.model,
        provider=llm_response.provider,
    )


def answer_question_with_history_streaming(
    db: Session,
    user_id: int,
    question: str,
    conversation_history: str = "",
    document_id: Optional[int] = None,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    embedding_provider=None,
    llm_service: Optional[LLMService] = None,
) -> Iterator[str]:
    """Stream a RAG answer incrementally using conversation history.

    Pipeline: validate → retrieve → build prompt → stream LLM → yield chunks.
    Sources and metadata are yielded at the end as a special event.

    This is a generator that yields text chunks. The caller is responsible
    for accumulating the complete response and persisting the assistant message.

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID.
        question: The user's current question.
        conversation_history: Formatted conversation history string.
        document_id: Optional. Restrict to one document.
        top_k: Number of retrieval candidates.
        min_similarity: Minimum similarity threshold.
        embedding_provider: Optional override for testing.
        llm_service: Optional override for testing.

    Yields:
        Text chunks from the LLM response.
        The last yield may be a JSON metadata event (not yielded but passed as return).

    Raises:
        RAGError: If validation or retrieval fails.
    """
    if llm_service is None:
        llm_service = LLMService()

    start_time = time.monotonic()

    # 1. Validate question
    cleaned_question = validate_question(question)

    # 2. Retrieve relevant context
    retrieval_start = time.monotonic()
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
    retrieval_time = (time.monotonic() - retrieval_start) * 1000

    # 3. No context → no hallucination
    if retrieval.total_results == 0:
        total_time = (time.monotonic() - start_time) * 1000
        logger.info(
            "RAG (streaming): no relevant context found (elapsed=%.1fms)",
            total_time,
        )
        no_answer = (
            "I don't have enough information in the provided documents "
            "to answer this question."
        )
        yield no_answer
        return

    # 4. Build sources and confidence
    sources = _build_sources(retrieval)
    confidence = _compute_confidence(retrieval, sources)

    # 5. Build RAG prompt with conversation history
    system_prompt = get_system_prompt()
    user_prompt = build_conversation_aware_user_prompt(
        question=cleaned_question,
        context=retrieval.context,
        conversation_history=conversation_history,
    )

    # 6. Stream LLM generation
    llm_start = time.monotonic()
    try:
        for chunk in llm_service.stream_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            if chunk:
                yield chunk
    except LLMTimeoutError as exc:
        raise RAGError(f"LLM request timed out: {exc}") from exc
    except LLMConfigurationError as exc:
        raise RAGError(f"LLM configuration error: {exc}") from exc
    except LLMProviderError as exc:
        raise RAGError(f"LLM generation failed: {exc}") from exc
    llm_time = (time.monotonic() - llm_start) * 1000

    total_time = (time.monotonic() - start_time) * 1000
    logger.info(
        "RAG (streaming): generation completed (sources=%d, retrieval=%.1fms, llm=%.1fms, total=%.1fms)",
        len(sources), retrieval_time, llm_time, total_time,
    )
