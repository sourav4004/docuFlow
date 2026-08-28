"""Retrieval pipeline service.

Connects user questions to vector search results with relevance filtering
and structured context construction for future LLM usage.

Pipeline:

User Question
    ↓
Query Validation
    ↓
Query Embedding (EmbeddingProvider)
    ↓
Vector Search (pgvector)
    ↓
Relevance Filtering
    ↓
Context Construction
    ↓
Structured RetrievalResult

No LLM calls are made here. Phase 4.7 will use this output.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from .embedding_service import get_embedding_provider
from .embeddings.base import EmbeddingError
from .vector_search import search_similar_chunks, SearchResult, MAX_TOP_K

logger = logging.getLogger(__name__)

# Retrieval configuration defaults
DEFAULT_RETRIEVAL_TOP_K = 5
MAX_RETRIEVAL_TOP_K = 20
DEFAULT_MIN_SIMILARITY = 0.3
DEFAULT_MAX_CONTEXT_CHARS = 8000
MAX_QUERY_LENGTH = 2000


class RetrievalError(Exception):
    """Raised when retrieval fails."""


class QueryValidationError(Exception):
    """Raised when the user query fails validation."""


@dataclass
class RetrievalResult:
    """A single retrieval result with source metadata."""
    document_id: int
    chunk_id: int
    chunk_index: int
    text: str
    similarity_score: float
    original_filename: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class RetrievalResponse:
    """Complete retrieval response with results and constructed context."""
    query: str
    results: List[RetrievalResult]
    context: str
    total_results: int
    document_id: Optional[int] = None


def validate_query(query: str) -> str:
    """Validate and clean a user query string.

    Args:
        query: Raw user query.

    Returns:
        Cleaned query string.

    Raises:
        QueryValidationError: If query is invalid.
    """
    if not isinstance(query, str):
        raise QueryValidationError("Query must be a string")

    cleaned = query.strip()

    if not cleaned:
        raise QueryValidationError("Query must not be empty")

    if len(cleaned) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query must not exceed {MAX_QUERY_LENGTH} characters"
        )

    return cleaned


def build_context(results: List[RetrievalResult], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    """Build a structured context string from retrieval results.

    Produces a deterministic, source-attributed context block suitable
    for inclusion in an LLM prompt.

    Args:
        results: Retrieval results ordered by relevance.
        max_chars: Maximum total context character count. Chunks are added
            in order until this limit is reached; partial chunks are not included.

    Returns:
        Formatted context string. Empty if no results.
    """
    if not results:
        return ""

    parts: List[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        # Source header
        header_parts = [f"[Source {i}]"]
        if result.original_filename:
            header_parts.append(f"Document: {result.original_filename}")
        header_parts.append(f"Chunk: {result.chunk_index}")
        if result.page_start is not None:
            page_str = f"Page: {result.page_start}"
            if result.page_end is not None and result.page_end != result.page_start:
                page_str += f"–{result.page_end}"
            header_parts.append(page_str)

        header = "\n".join(header_parts)
        block = f"{header}\n\n{result.text}"

        # Check context size limit — skip chunk if it would exceed limit
        # (except the first chunk, which is always included)
        if total_chars + len(block) > max_chars and i > 1:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(parts)


def retrieve_context(
    db: Session,
    user_id: int,
    query: str,
    document_id: Optional[int] = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    provider=None,
) -> RetrievalResponse:
    """Retrieve relevant document context for a user query.

    Full pipeline: validate → embed → search → filter → build context.

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID. All results scoped to this user.
        query: User's natural-language question.
        document_id: Optional. Restrict search to this specific document.
        top_k: Number of vector search candidates to retrieve.
        min_similarity: Minimum cosine similarity to include in results.
        max_context_chars: Maximum characters in constructed context.
        provider: Optional EmbeddingProvider override (for testing).

    Returns:
        RetrievalResponse with results, context, and metadata.

    Raises:
        QueryValidationError: If the query is invalid.
        RetrievalError: If embedding or search fails.
    """
    # 1. Validate query
    cleaned_query = validate_query(query)

    # 2. Generate query embedding
    if provider is None:
        provider = get_embedding_provider()

    try:
        query_embedding = provider.embed_text(cleaned_query)
    except EmbeddingError as exc:
        raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc
    except Exception as exc:
        raise RetrievalError(f"Embedding provider failed: {exc}") from exc

    # 3. Vector search
    try:
        search_results = search_similar_chunks(
            db=db,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )
    except ValueError as exc:
        raise RetrievalError(f"Vector search validation failed: {exc}") from exc
    except Exception as exc:
        raise RetrievalError(f"Vector search failed: {exc}") from exc

    # 4. Relevance filtering
    filtered = [
        RetrievalResult(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            text=r.text,
            similarity_score=r.similarity_score,
            original_filename=r.original_filename,
            page_start=r.page_start,
            page_end=r.page_end,
        )
        for r in search_results
        if r.similarity_score >= min_similarity
    ]

    logger.debug(
        "retrieve_context: %d search results, %d after filtering (min_sim=%.2f)",
        len(search_results), len(filtered), min_similarity,
    )

    # 5. Build context
    context = build_context(filtered, max_chars=max_context_chars)

    return RetrievalResponse(
        query=cleaned_query,
        results=filtered,
        context=context,
        total_results=len(filtered),
        document_id=document_id,
    )
