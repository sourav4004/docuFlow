"""Vector similarity search service for document chunks.

Provides semantic search over document embeddings using the configured
vector backend (pgvector or JSON fallback).

Pipeline:

User question
    ↓
EmbeddingProvider
    ↓
query embedding
    ↓
VectorBackend (pgvector or JSON fallback)
    ↓
ranked DocumentChunks
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from ..models.document_chunk import DocumentChunk
from ..models.document import Document
from .vector_backend import get_vector_backend_singleton, VectorSearchResult

logger = logging.getLogger(__name__)

# Default and maximum top_k values
DEFAULT_TOP_K = 5
MAX_TOP_K = 50


@dataclass
class SearchResult:
    """A single search result with similarity score and source metadata."""
    document_id: int
    chunk_id: int
    chunk_index: int
    text: str
    similarity_score: float
    page_start: Optional[int]
    page_end: Optional[int]
    original_filename: Optional[str]


def search_similar_chunks(
    db: Session,
    user_id: int,
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
    document_id: Optional[int] = None,
) -> List[SearchResult]:
    """Search for document chunks most similar to a query embedding.

    Uses the configured vector backend for similarity search.
    Enforces user ownership — only returns chunks belonging to the user.

    Args:
        db: SQLAlchemy session.
        user_id: ID of the authenticated user. All results must belong to this user.
        query_embedding: The query vector (from EmbeddingProvider).
        top_k: Number of results to return. Must be 1..MAX_TOP_K.
        document_id: Optional. If provided, restrict search to this document.

    Returns:
        List of SearchResult objects ordered by similarity (most similar first).
        May be empty if no matching chunks exist.

    Raises:
        ValueError: If top_k is out of range or query_embedding is empty/wrong dimension.
    """
    if not query_embedding:
        raise ValueError("query_embedding must not be empty")

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}"
        )

    # Use the vector backend abstraction
    backend = get_vector_backend_singleton()
    results = backend.search(db, user_id, query_embedding, top_k, document_id)

    # Convert to SearchResult format for backward compatibility
    return [
        SearchResult(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            text=r.text,
            similarity_score=r.similarity_score,
            page_start=r.page_start,
            page_end=r.page_end,
            original_filename=r.original_filename,
        )
        for r in results
    ]


def get_vector_backend_info() -> dict:
    """Get information about the active vector backend."""
    backend = get_vector_backend_singleton()
    return {
        "backend": backend.name,
        "available": backend.is_available,
    }
