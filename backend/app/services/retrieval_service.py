"""Retrieval pipeline service — production quality.

Connects user questions to search results with relevance filtering
and structured context construction for LLM usage.

Supports two retrieval modes:

  Vector-only (hybrid_search_enabled=False):
    Query → Embedding → pgvector cosine similarity → context

  Hybrid (hybrid_search_enabled=True):
    Query → Embedding → pgvector cosine similarity (vector_score)
    Query → PostgreSQL tsvector/tsquery (lexical_score)
    → Weighted combination → dedup → filter → context

Fallback behavior:
  - Hybrid enabled, both succeed: combine both
  - Hybrid enabled, vector fails: keyword fallback (if enabled)
  - Hybrid enabled, keyword fails: vector fallback (if enabled)
  - Both fail: empty results with diagnostic
  - Hybrid disabled: vector-only (original behavior)

No LLM calls are made here.
"""

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from ..core.config import settings
from .embedding_service import get_embedding_provider
from .embeddings.base import EmbeddingError
from .vector_search import search_similar_chunks, SearchResult, MAX_TOP_K
from .keyword_search import search_by_keywords, KeywordSearchResult, MAX_KEYWORD_TOP_K

logger = logging.getLogger(__name__)

# Retrieval configuration defaults
DEFAULT_RETRIEVAL_TOP_K = 5
MAX_RETRIEVAL_TOP_K = 50
DEFAULT_MIN_SIMILARITY = 0.3
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CHUNKS_IN_CONTEXT = 20
MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 1


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
    # Hybrid search fields
    keyword_score: Optional[float] = None  # Normalized keyword score (0-1)
    final_score: Optional[float] = None  # Weighted combined score
    match_type: str = "vector"  # "vector", "keyword", or "both"

    @property
    def score(self) -> float:
        """Best available score: final_score if hybrid, else similarity_score."""
        return self.final_score if self.final_score is not None else self.similarity_score


@dataclass
class RetrievalMetadata:
    """Trace/explanation metadata for a retrieval operation."""
    retrieval_mode: str = "vector"  # "vector" or "hybrid"
    candidate_count: int = 0  # Total candidates before filtering
    vector_candidate_count: int = 0  # Candidates from vector search
    keyword_candidate_count: int = 0  # Candidates from keyword search
    returned_count: int = 0  # Results after filtering
    threshold: float = 0.0  # Minimum score threshold used
    vector_weight: float = 0.0  # Vector weight used
    keyword_weight: float = 0.0  # Keyword weight used
    # Timing (seconds, monotonic)
    embedding_time: float = 0.0
    vector_search_time: float = 0.0
    keyword_search_time: float = 0.0
    merge_time: float = 0.0
    context_time: float = 0.0
    total_time: float = 0.0
    # Fallback info
    vector_failed: bool = False
    keyword_failed: bool = False
    fallback_used: str = ""  # "keyword_fallback", "vector_fallback", ""


@dataclass
class RetrievalResponse:
    """Complete retrieval response with results, context, and metadata."""
    query: str
    results: List[RetrievalResult]
    context: str
    total_results: int
    document_id: Optional[int] = None
    collection_id: Optional[int] = None
    retrieval_mode: str = "vector"
    metadata: Optional[RetrievalMetadata] = None


def normalize_query(query: str) -> str:
    """Lightweight, safe query normalization.

    - Trims surrounding whitespace
    - Normalizes repeated internal whitespace to single spaces
    - Preserves Unicode, technical terms, and meaning
    - Does NOT stem, rewrite, or use LLM

    Args:
        query: Raw user query string (already validated as non-empty).

    Returns:
        Normalized query string.
    """
    # Collapse repeated whitespace (spaces, tabs, newlines) to single space
    normalized = re.sub(r'\s+', ' ', query).strip()
    return normalized if normalized else query.strip()


def validate_query(query: str) -> str:
    """Validate and clean a user query string.

    Args:
        query: Raw user query.

    Returns:
        Cleaned and normalized query string.

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

    # Apply light normalization
    cleaned = normalize_query(cleaned)

    # Re-check length after normalization (shouldn't grow, but be safe)
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query must not exceed {MAX_QUERY_LENGTH} characters"
        )

    return cleaned


def _validate_weights(vector_weight: float, keyword_weight: float) -> tuple:
    """Validate and normalize retrieval weights.

    If weights don't sum to 1.0, normalize them. Negative weights are
    clamped to 0.0. Both zero → equal split.

    Args:
        vector_weight: Weight for vector similarity score.
        keyword_weight: Weight for keyword/lexical score.

    Returns:
        Tuple of (normalized_vector_weight, normalized_keyword_weight).
    """
    vw = max(0.0, float(vector_weight))
    kw = max(0.0, float(keyword_weight))

    total = vw + kw
    if total <= 0.0:
        # Both zero or negative → equal split
        return 0.5, 0.5
    if abs(total - 1.0) > 1e-6:
        # Normalize to sum to 1.0
        vw = vw / total
        kw = kw / total

    return vw, kw


def _safe_normalize_score(score: float) -> float:
    """Safely normalize a score to [0, 1], protecting against NaN/infinity."""
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def build_context(
    results: List[RetrievalResult],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS_IN_CONTEXT,
    group_by_document: bool = False,
) -> str:
    """Build a structured context string from retrieval results.

    Produces a deterministic, source-attributed context block suitable
    for inclusion in an LLM prompt.

    When group_by_document=True, chunks are grouped by document with
    document-level headers, enabling better multi-document reasoning.

    Args:
        results: Retrieval results ordered by relevance.
        max_chars: Maximum total context character count. Chunks are added
            in order until this limit is reached; partial chunks are not included.
        max_chunks: Maximum number of chunks to include in context.
        group_by_document: If True, group chunks under document headers.

    Returns:
        Formatted context string. Empty if no results.
    """
    if not results:
        return ""

    # Filter empty chunks
    valid_results = [r for r in results if r.text and r.text.strip()]
    if not valid_results:
        return ""

    if group_by_document:
        return _build_grouped_context(valid_results, max_chars, max_chunks)
    else:
        return _build_flat_context(valid_results, max_chars, max_chunks)


def _build_flat_context(
    results: List[RetrievalResult],
    max_chars: int,
    max_chunks: int,
) -> str:
    """Build flat context — each chunk as a separate source block."""
    parts: List[str] = []
    total_chars = 0
    chunks_added = 0

    for i, result in enumerate(results, 1):
        if chunks_added >= max_chunks:
            break

        header_parts = [f"[Source {i}]"]
        if result.original_filename:
            header_parts.append(f"Document: {result.original_filename}")
        header_parts.append(f"Chunk: {result.chunk_index}")
        if result.page_start is not None:
            page_str = f"Page: {result.page_start}"
            if result.page_end is not None and result.page_end != result.page_start:
                page_str += f"\u2013{result.page_end}"
            header_parts.append(page_str)

        header = "\n".join(header_parts)
        block = f"{header}\n\n{result.text}"

        if total_chars + len(block) > max_chars and chunks_added > 0:
            break

        parts.append(block)
        total_chars += len(block)
        chunks_added += 1

    return "\n\n---\n\n".join(parts)


def _build_grouped_context(
    results: List[RetrievalResult],
    max_chars: int,
    max_chunks: int,
) -> str:
    """Build context grouped by document — chunks under document headers.

    Useful for multi-document reasoning where the LLM needs to distinguish
    facts by source document.
    """
    # Group by document_id, preserving result order within each group
    from collections import OrderedDict
    doc_groups: Dict[int, List[RetrievalResult]] = OrderedDict()
    for r in results:
        if r.document_id not in doc_groups:
            doc_groups[r.document_id] = []
        doc_groups[r.document_id].append(r)

    parts: List[str] = []
    total_chars = 0
    chunks_added = 0

    for doc_id, doc_results in doc_groups.items():
        if chunks_added >= max_chunks:
            break

        # Document header
        filename = doc_results[0].original_filename
        doc_header = f"DOCUMENT: {filename or f'Document #{doc_id}'}"

        chunk_lines: List[str] = []
        for r in doc_results:
            if chunks_added >= max_chunks:
                break

            # Source label with page info
            label_parts = [f"[Chunk {r.chunk_index}]"]
            if r.page_start is not None:
                page_str = f"Page {r.page_start}"
                if r.page_end is not None and r.page_end != r.page_start:
                    page_str += f"- {r.page_end}"
                label_parts.append(page_str)

            block = f"{chr(10).join(label_parts)}\n\n{r.text}"

            if total_chars + len(block) > max_chars and chunks_added > 0:
                break

            chunk_lines.append(block)
            total_chars += len(block)
            chunks_added += 1

        if chunk_lines:
            doc_block = f"{doc_header}\n\n" + "\n\n".join(chunk_lines)
            parts.append(doc_block)

    return "\n\n===\n\n".join(parts)


def _normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a score to [0, 1] range with NaN/infinity protection.

    If the range is zero (min == max):
      - score > 0: all results match equally, normalize to 1.0
      - score == 0: no signal, normalize to 0.0
    """
    if not math.isfinite(score):
        return 0.0
    if not math.isfinite(min_val):
        min_val = 0.0
    if not math.isfinite(max_val):
        max_val = 1.0
    if max_val <= min_val:
        return 1.0 if score > 0 else 0.0
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def _deduplicate_results(
    results: Dict[int, RetrievalResult],
) -> List[RetrievalResult]:
    """Deduplicate results by chunk_id with deterministic tie-breaking.

    Sorting order:
      1. final_score DESC (or similarity_score for vector-only)
      2. similarity_score DESC (secondary)
      3. keyword_score DESC (tertiary)
      4. chunk_id ASC (final tie-breaker for determinism)

    Args:
        results: Dict mapping chunk_id → RetrievalResult.

    Returns:
        Deduplicated list sorted by the criteria above.
    """
    deduped = list(results.values())
    deduped.sort(key=lambda r: (
        -(r.final_score if r.final_score is not None else r.similarity_score),
        -r.similarity_score,
        -(r.keyword_score if r.keyword_score is not None else 0.0),
        r.chunk_id,
    ))
    return deduped


def _hybrid_retrieve(
    db: Session,
    user_id: int,
    cleaned_query: str,
    query_embedding: List[float],
    document_id: Optional[int],
    vector_weight: float,
    keyword_weight: float,
    vector_top_k: int,
    keyword_top_k: int,
    enable_keyword_fallback: bool,
    enable_vector_fallback: bool,
    metadata: RetrievalMetadata,
) -> List[RetrievalResult]:
    """Perform hybrid retrieval with graceful fallback.

    Cases handled:
      A: Both succeed → combine
      B: Vector fails, keyword succeeds → keyword fallback
      C: Keyword fails, vector succeeds → vector fallback
      D: Both fail → empty results

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID.
        cleaned_query: Validated query string.
        query_embedding: Vector embedding of the query.
        document_id: Optional document restriction.
        vector_weight: Weight for vector similarity score.
        keyword_weight: Weight for keyword/lexical score.
        vector_top_k: Number of vector candidates to fetch.
        keyword_top_k: Number of keyword candidates to fetch.
        enable_keyword_fallback: Use keyword search when vector fails.
        enable_vector_fallback: Use vector search when keyword fails.
        metadata: Mutable metadata object to populate.

    Returns:
        List of RetrievalResult sorted by final_score DESC.
    """
    vector_results: List[SearchResult] = []
    keyword_results: List[KeywordSearchResult] = []

    # --- Case B/C/D: Try each strategy independently ---
    vector_failed = False
    keyword_failed = False

    v_start = time.monotonic()
    try:
        vector_results = search_similar_chunks(
            db=db, user_id=user_id, query_embedding=query_embedding,
            top_k=vector_top_k, document_id=document_id,
        )
    except Exception as exc:
        vector_failed = True
        logger.warning("Hybrid: vector search failed: %s", type(exc).__name__)
    metadata.vector_search_time = time.monotonic() - v_start
    metadata.vector_candidate_count = len(vector_results)

    k_start = time.monotonic()
    try:
        keyword_results = search_by_keywords(
            db=db, user_id=user_id, query=cleaned_query,
            top_k=keyword_top_k, document_id=document_id,
        )
    except Exception as exc:
        keyword_failed = True
        logger.warning("Hybrid: keyword search failed: %s", type(exc).__name__)
    metadata.keyword_search_time = time.monotonic() - k_start
    metadata.keyword_candidate_count = len(keyword_results)

    metadata.vector_failed = vector_failed
    metadata.keyword_failed = keyword_failed

    # --- Case D: Both failed ---
    if vector_failed and keyword_failed:
        metadata.fallback_used = "both_failed"
        return []

    # --- Determine effective fallback ---
    if vector_failed and not keyword_failed:
        if not enable_keyword_fallback:
            metadata.fallback_used = "keyword_disabled"
            return []
        metadata.fallback_used = "keyword_fallback"
        logger.info("Hybrid: using keyword fallback (vector failed)")
    elif keyword_failed and not vector_failed:
        if not enable_vector_fallback:
            metadata.fallback_used = "vector_disabled"
            return []
        metadata.fallback_used = "vector_fallback"
        logger.info("Hybrid: using vector fallback (keyword failed)")
    else:
        metadata.fallback_used = ""

    # --- Normalize scores ---
    # Vector scores: cosine similarity already in [0, 1] typically
    vector_scores = [r.similarity_score for r in vector_results]
    v_min = min(vector_scores) if vector_scores else 0.0
    v_max = max(vector_scores) if vector_scores else 1.0

    # Keyword scores: ts_rank returns [0, 1] but normalize for safety
    kw_scores = [r.lexical_score for r in keyword_results]
    k_min = min(kw_scores) if kw_scores else 0.0
    k_max = max(kw_scores) if kw_scores else 1.0

    m_start = time.monotonic()

    # Merge results into a dict keyed by chunk_id for deduplication
    merged: Dict[int, RetrievalResult] = {}

    # Add vector results
    for r in vector_results:
        norm_v = _normalize_score(r.similarity_score, v_min, v_max)
        norm_v = _safe_normalize_score(norm_v)
        norm_k = 0.0
        final = _safe_normalize_score(vector_weight * norm_v + keyword_weight * norm_k)
        merged[r.chunk_id] = RetrievalResult(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            text=r.text,
            similarity_score=r.similarity_score,
            original_filename=r.original_filename,
            page_start=r.page_start,
            page_end=r.page_end,
            keyword_score=0.0,
            final_score=final,
            match_type="vector",
        )

    # Add/update with keyword results
    for r in keyword_results:
        norm_k = _normalize_score(r.lexical_score, k_min, k_max)
        norm_k = _safe_normalize_score(norm_k)
        if r.chunk_id in merged:
            # Chunk found in both: combine scores, mark as "both"
            existing = merged[r.chunk_id]
            norm_v = _normalize_score(existing.similarity_score, v_min, v_max)
            norm_v = _safe_normalize_score(norm_v)
            final = _safe_normalize_score(vector_weight * norm_v + keyword_weight * norm_k)
            existing.keyword_score = norm_k
            existing.final_score = final
            existing.match_type = "both"
        else:
            # Chunk only in keyword results
            norm_v = 0.0
            final = _safe_normalize_score(vector_weight * norm_v + keyword_weight * norm_k)
            merged[r.chunk_id] = RetrievalResult(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                chunk_index=r.chunk_index,
                text=r.text,
                similarity_score=0.0,
                original_filename=r.original_filename,
                page_start=r.page_start,
                page_end=r.page_end,
                keyword_score=norm_k,
                final_score=final,
                match_type="keyword",
            )

    metadata.merge_time = time.monotonic() - m_start

    return _deduplicate_results(merged)


def retrieve_context(
    db: Session,
    user_id: int,
    query: str,
    document_id: Optional[int] = None,
    collection_id: Optional[int] = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    provider=None,
    hybrid: Optional[bool] = None,
    vector_weight: Optional[float] = None,
    keyword_weight: Optional[float] = None,
) -> RetrievalResponse:
    """Retrieve relevant document context for a user query.

    Supports both vector-only and hybrid (vector + keyword) retrieval.
    Supports collection filtering for collection-aware RAG.

    Pipeline (vector-only):
        validate → embed → vector search → filter → build context

    Pipeline (hybrid):
        validate → embed → [vector search + keyword search] →
        normalize scores → weighted combine → dedup → filter → build context

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID. All results scoped to this user.
        query: User's natural-language question.
        document_id: Optional. Restrict search to this specific document.
        collection_id: Optional. Restrict search to documents in this collection.
        top_k: Number of results to retrieve (final output).
        min_similarity: Minimum score to include in results.
        max_context_chars: Maximum characters in constructed context.
        provider: Optional EmbeddingProvider override (for testing).
        hybrid: Override hybrid_search_enabled setting. None = use config.
        vector_weight: Override vector_search_weight setting.
        keyword_weight: Override keyword_search_weight setting.

    Returns:
        RetrievalResponse with results, context, and metadata.

    Raises:
        QueryValidationError: If the query is invalid.
        RetrievalError: If embedding or search fails.
    """
    total_start = time.monotonic()

    # Resolve collection_id to document_ids if provided
    collection_doc_ids = None
    if collection_id is not None:
        from ..models.collection import Collection
        col = db.query(Collection).filter(
            Collection.id == collection_id,
            Collection.user_id == user_id,
        ).first()
        if not col:
            raise RetrievalError(f"Collection {collection_id} not found")
        collection_doc_ids = [d.id for d in col.documents]
        if not collection_doc_ids:
            # Empty collection - return no results
            return RetrievalResponse(
                query=query,
                results=[],
                context="",
                total_results=0,
                collection_id=collection_id,
            )

    # 1. Validate and normalize query
    cleaned_query = validate_query(query)

    # 2. Determine mode and validate weights
    use_hybrid = hybrid if hybrid is not None else settings.hybrid_search_enabled
    vw_raw = vector_weight if vector_weight is not None else settings.vector_search_weight
    kw_raw = keyword_weight if keyword_weight is not None else settings.keyword_search_weight
    vw, kw = _validate_weights(vw_raw, kw_raw)

    # Initialize metadata
    meta = RetrievalMetadata(
        retrieval_mode="hybrid" if use_hybrid else "vector",
        threshold=min_similarity,
        vector_weight=vw,
        keyword_weight=kw,
    )

    # 3. Generate query embedding
    if provider is None:
        provider = get_embedding_provider()

    emb_start = time.monotonic()
    try:
        query_embedding = provider.embed_text(cleaned_query)
    except EmbeddingError as exc:
        raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc
    except Exception as exc:
        raise RetrievalError(f"Embedding provider failed: {exc}") from exc
    meta.embedding_time = time.monotonic() - emb_start

    if use_hybrid:
        # 4a. Hybrid retrieval
        vector_top = min(settings.hybrid_top_k, MAX_TOP_K)
        keyword_top = min(settings.keyword_only_top_k, MAX_KEYWORD_TOP_K)

        # If collection filtering, we need to search each doc separately or filter after
        if collection_doc_ids and not document_id:
            # Collection-aware: search all docs, then filter to collection
            try:
                all_results = _hybrid_retrieve(
                    db=db, user_id=user_id, cleaned_query=cleaned_query,
                    query_embedding=query_embedding, document_id=None,
                    vector_weight=vw, keyword_weight=kw,
                    vector_top_k=vector_top, keyword_top_k=keyword_top,
                    enable_keyword_fallback=settings.enable_keyword_fallback,
                    enable_vector_fallback=settings.enable_vector_fallback,
                    metadata=meta,
                )
            except Exception as exc:
                raise RetrievalError(f"Hybrid search failed: {exc}") from exc
            # Filter to collection documents
            all_results = [r for r in all_results if r.document_id in collection_doc_ids]
        else:
            try:
                all_results = _hybrid_retrieve(
                    db=db, user_id=user_id, cleaned_query=cleaned_query,
                    query_embedding=query_embedding, document_id=document_id,
                    vector_weight=vw, keyword_weight=kw,
                    vector_top_k=vector_top, keyword_top_k=keyword_top,
                    enable_keyword_fallback=settings.enable_keyword_fallback,
                    enable_vector_fallback=settings.enable_vector_fallback,
                    metadata=meta,
                )
            except Exception as exc:
                raise RetrievalError(f"Hybrid search failed: {exc}") from exc

        meta.candidate_count = len(all_results)

        # 4b. Filter by minimum score
        filtered = [r for r in all_results if r.score >= min_similarity]

        # 4c. Limit to top_k
        filtered = filtered[:top_k]

        retrieval_mode = "hybrid"

        logger.debug(
            "retrieve_context (hybrid): %d candidates, %d after filter/slice "
            "(min=%.2f, vw=%.2f, kw=%.2f, fallback=%s)",
            meta.candidate_count, len(filtered), min_similarity, vw, kw,
            meta.fallback_used or "none",
        )
    else:
        # 4d. Vector-only retrieval (original behavior)
        try:
            search_results = search_similar_chunks(
                db=db, user_id=user_id, query_embedding=query_embedding,
                top_k=min(top_k, MAX_TOP_K), document_id=document_id,
            )
        except ValueError as exc:
            raise RetrievalError(f"Vector search validation failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        meta.vector_candidate_count = len(search_results)
        meta.candidate_count = len(search_results)

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

        retrieval_mode = "vector"

        logger.debug(
            "retrieve_context (vector): %d results, %d after filtering (min_sim=%.2f)",
            meta.vector_candidate_count, len(filtered), min_similarity,
        )

    meta.returned_count = len(filtered)

    # 5. Build context
    ctx_start = time.monotonic()
    max_chunks = getattr(settings, 'retrieval_max_chunks_in_context', DEFAULT_MAX_CHUNKS_IN_CONTEXT)
    group_by_doc = getattr(settings, 'group_context_by_document', False)
    context = build_context(
        filtered, max_chars=max_context_chars, max_chunks=max_chunks,
        group_by_document=group_by_doc,
    )
    meta.context_time = time.monotonic() - ctx_start

    # 6. Finalize timing
    meta.total_time = time.monotonic() - total_start

    logger.info(
        "retrieve_context: mode=%s, query_len=%d, candidates=%d, returned=%d, "
        "total=%.3fs",
        retrieval_mode, len(cleaned_query), meta.candidate_count,
        meta.returned_count, meta.total_time,
    )

    return RetrievalResponse(
        query=cleaned_query,
        results=filtered,
        context=context,
        total_results=len(filtered),
        document_id=document_id,
        collection_id=collection_id,
        retrieval_mode=retrieval_mode,
        metadata=meta,
    )
