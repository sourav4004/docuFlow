"""Vector similarity search service for document chunks.

Provides semantic search over document embeddings using pgvector.
Similarity is calculated inside PostgreSQL — not in Python.

Pipeline:

User question
    ↓
EmbeddingProvider
    ↓
query embedding
    ↓
VectorSearchService (PostgreSQL pgvector)
    ↓
ranked DocumentChunks
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.document_chunk import DocumentChunk
from ..models.document import Document

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

    Uses pgvector cosine similarity computed inside PostgreSQL.
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

    # Convert embedding to pgvector format: '[0.1, 0.2, ...]'
    embedding_str = "[" + ", ".join(str(v) for v in query_embedding) + "]"

    # Build the query with ownership filtering
    # Uses raw SQL for pgvector operator (<=>) which is cosine distance
    # cosine distance = 1 - cosine similarity
    # So lower distance = more similar
    query = """
        SELECT
            dc.document_id,
            dc.id AS chunk_id,
            dc.chunk_index,
            dc.text,
            (1 - (dc.embedding <=> :query_vec)) AS similarity_score,
            dc.page_start,
            dc.page_end,
            d.original_filename
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE d.user_id = :user_id
          AND dc.embedding IS NOT NULL
    """

    params = {
        "query_vec": embedding_str,
        "user_id": user_id,
    }

    if document_id is not None:
        query += " AND dc.document_id = :document_id"
        params["document_id"] = document_id

    query += " ORDER BY dc.embedding <=> :query_vec LIMIT :top_k"
    params["top_k"] = top_k

    result = db.execute(text(query), params)
    rows = result.fetchall()

    return [
        SearchResult(
            document_id=row.document_id,
            chunk_id=row.chunk_id,
            chunk_index=row.chunk_index,
            text=row.text,
            similarity_score=float(row.similarity_score),
            page_start=row.page_start,
            page_end=row.page_end,
            original_filename=row.original_filename,
        )
        for row in rows
    ]
