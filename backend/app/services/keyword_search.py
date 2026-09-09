"""Keyword / full-text search over document chunks using PostgreSQL tsvector.

Uses PostgreSQL's built-in full-text search (FTS) with tsvector/tsquery
for lexical/keyword matching. Scoring is done via ts_rank (normalized 0-1).

Pipeline:
    User query
        ↓
    to_tsvector (on stored search_vector column)
        ↓
    plainto_tsquery (handles special characters, stop words)
        ↓
    ts_rank (normalized)
        ↓
    KeywordSearchResult list

No embedding calls. Pure database operation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD_TOP_K = 10
MAX_KEYWORD_TOP_K = 50


@dataclass
class KeywordSearchResult:
    """A single keyword search result with lexical score and metadata."""
    document_id: int
    chunk_id: int
    chunk_index: int
    text: str
    lexical_score: float  # ts_rank normalized 0-1
    page_start: Optional[int]
    page_end: Optional[int]
    original_filename: Optional[str]


def search_by_keywords(
    db: Session,
    user_id: int,
    query: str,
    top_k: int = DEFAULT_KEYWORD_TOP_K,
    document_id: Optional[int] = None,
) -> List[KeywordSearchResult]:
    """Search for document chunks matching the query using full-text search.

    Uses PostgreSQL's plainto_tsquery which:
    - handles stop words (a, the, is, etc.)
    - normalizes tokens
    - is safe against special characters (no injection possible)
    - works case-insensitively

    ts_rank returns a normalized score (0-1) based on term frequency
    and document length.

    Args:
        db: SQLAlchemy session.
        user_id: Authenticated user ID. All results scoped to this user.
        query: User's search query (plain text, not tsquery syntax).
        top_k: Number of results to return. Must be 1..MAX_KEYWORD_TOP_K.
        document_id: Optional. Restrict search to this specific document.

    Returns:
        List of KeywordSearchResult objects ordered by lexical_score DESC.
        May be empty if no matching chunks exist.

    Raises:
        ValueError: If query is empty or top_k is out of range.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if top_k < 1 or top_k > MAX_KEYWORD_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_KEYWORD_TOP_K}, got {top_k}"
        )

    cleaned = query.strip()

    # plainto_tsquery is safe: it parses plain text, not tsquery syntax.
    # Special characters, stop words, and unicode are handled safely.
    sql = """
        SELECT
            dc.document_id,
            dc.id AS chunk_id,
            dc.chunk_index,
            dc.text,
            ts_rank(dc.search_vector, plainto_tsquery('english', :query)) AS lexical_score,
            dc.page_start,
            dc.page_end,
            d.original_filename
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE d.user_id = :user_id
          AND dc.search_vector @@ plainto_tsquery('english', :query)
    """

    params = {
        "query": cleaned,
        "user_id": user_id,
    }

    if document_id is not None:
        sql += " AND dc.document_id = :document_id"
        params["document_id"] = document_id

    # Order by lexical score descending, then by chunk_id for determinism
    sql += " ORDER BY lexical_score DESC, dc.id ASC LIMIT :top_k"
    params["top_k"] = top_k

    result = db.execute(text(sql), params)
    rows = result.fetchall()

    return [
        KeywordSearchResult(
            document_id=row.document_id,
            chunk_id=row.chunk_id,
            chunk_index=row.chunk_index,
            text=row.text,
            lexical_score=float(row.lexical_score),
            page_start=row.page_start,
            page_end=row.page_end,
            original_filename=row.original_filename,
        )
        for row in rows
    ]
