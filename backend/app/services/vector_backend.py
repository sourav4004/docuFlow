"""Vector backend abstraction for DocuFlow.

Provides a clean interface for vector operations that works with:
A. pgvector when available
B. JSON-based cosine similarity fallback when pgvector is unavailable

The backend is selected via VECTOR_BACKEND config:
- "auto": Detect pgvector availability, use best available
- "pgvector": Require pgvector (fail if unavailable)
- "json": Use JSON fallback (always available, slower)
"""

import json
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.config import settings

logger = logging.getLogger(__name__)


class VectorBackendType(str, Enum):
    AUTO = "auto"
    PGVECTOR = "pgvector"
    JSON = "json"


@dataclass
class VectorSearchResult:
    """A single vector search result."""
    document_id: int
    chunk_id: int
    chunk_index: int
    text: str
    similarity_score: float
    page_start: Optional[int]
    page_end: Optional[int]
    original_filename: Optional[str]


class VectorBackend(ABC):
    """Abstract base class for vector backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name for logging."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is currently available."""

    @abstractmethod
    def search(
        self,
        db: Session,
        user_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar chunks using the vector backend."""

    @abstractmethod
    def store_embedding(
        self,
        db: Session,
        chunk_id: int,
        embedding: List[float],
    ) -> bool:
        """Store an embedding for a chunk. Returns True if successful."""


class PgvectorBackend(VectorBackend):
    """PostgreSQL pgvector backend for vector similarity search."""

    def __init__(self):
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """Check if pgvector extension is installed."""
        try:
            from ..core.database import SessionLocal
            db = SessionLocal()
            try:
                result = db.execute(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                )
                return result.fetchone() is not None
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to check pgvector availability: %s", e)
            return False

    def search(
        self,
        db: Session,
        user_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """Search using pgvector cosine similarity."""
        if not query_embedding:
            return []

        embedding_str = "[" + ", ".join(str(v) for v in query_embedding) + "]"

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

        params = {"query_vec": embedding_str, "user_id": user_id}

        if document_id is not None:
            query += " AND dc.document_id = :document_id"
            params["document_id"] = document_id

        query += " ORDER BY dc.embedding <=> :query_vec LIMIT :top_k"
        params["top_k"] = top_k

        result = db.execute(text(query), params)
        rows = result.fetchall()

        return [
            VectorSearchResult(
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

    def store_embedding(
        self,
        db: Session,
        chunk_id: int,
        embedding: List[float],
    ) -> bool:
        """Store embedding using pgvector."""
        try:
            embedding_str = "[" + ", ".join(str(v) for v in embedding) + "]"
            db.execute(
                text("UPDATE document_chunks SET embedding = :emb::vector WHERE id = :id"),
                {"emb": embedding_str, "id": chunk_id},
            )
            return True
        except Exception as e:
            logger.error("Failed to store pgvector embedding for chunk %d: %s", chunk_id, e)
            return False


class JsonFallbackBackend(VectorBackend):
    """JSON-based cosine similarity fallback backend.

    Stores embeddings as JSON arrays and computes cosine similarity in Python.
    Slower than pgvector but always available.
    """

    @property
    def name(self) -> str:
        return "json_fallback"

    @property
    def is_available(self) -> bool:
        return True

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def search(
        self,
        db: Session,
        user_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """Search using JSON-stored embeddings with Python cosine similarity."""
        if not query_embedding:
            return []

        # Build query to fetch chunks with embeddings
        query = """
            SELECT
                dc.document_id,
                dc.id AS chunk_id,
                dc.chunk_index,
                dc.text,
                dc.embedding,
                dc.page_start,
                dc.page_end,
                d.original_filename
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.user_id = :user_id
              AND dc.embedding IS NOT NULL
        """

        params = {"user_id": user_id}

        if document_id is not None:
            query += " AND dc.document_id = :document_id"
            params["document_id"] = document_id

        result = db.execute(text(query), params)
        rows = result.fetchall()

        # Compute similarity for each chunk
        scored_results = []
        for row in rows:
            try:
                # Parse embedding from JSON
                if isinstance(row.embedding, str):
                    stored_embedding = json.loads(row.embedding)
                elif isinstance(row.embedding, list):
                    stored_embedding = row.embedding
                else:
                    continue

                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                scored_results.append(
                    VectorSearchResult(
                        document_id=row.document_id,
                        chunk_id=row.chunk_id,
                        chunk_index=row.chunk_index,
                        text=row.text,
                        similarity_score=similarity,
                        page_start=row.page_start,
                        page_end=row.page_end,
                        original_filename=row.original_filename,
                    )
                )
            except (json.JSONDecodeError, TypeError):
                continue

        # Sort by similarity and return top_k
        scored_results.sort(key=lambda r: r.similarity_score, reverse=True)
        return scored_results[:top_k]

    def store_embedding(
        self,
        db: Session,
        chunk_id: int,
        embedding: List[float],
    ) -> bool:
        """Store embedding as JSON."""
        try:
            embedding_json = json.dumps(embedding)
            db.execute(
                text("UPDATE document_chunks SET embedding = :emb WHERE id = :id"),
                {"emb": embedding_json, "id": chunk_id},
            )
            return True
        except Exception as e:
            logger.error("Failed to store JSON embedding for chunk %d: %s", chunk_id, e)
            return False


class AutoVectorBackend(VectorBackend):
    """Auto-detecting vector backend that uses pgvector when available."""

    def __init__(self):
        self._pgvector = PgvectorBackend()
        self._json = JsonFallbackBackend()
        self._active_backend: Optional[VectorBackend] = None

    @property
    def name(self) -> str:
        if self._active_backend:
            return f"auto({self._active_backend.name})"
        return "auto"

    @property
    def is_available(self) -> bool:
        return True  # Auto always has a fallback

    def _get_backend(self) -> VectorBackend:
        """Get the active backend, detecting pgvector on first use."""
        if self._active_backend is None:
            if self._pgvector.is_available:
                logger.info("Vector backend: pgvector detected, using pgvector")
                self._active_backend = self._pgvector
            else:
                logger.info("Vector backend: pgvector not available, using JSON fallback")
                self._active_backend = self._json
        return self._active_backend

    def search(
        self,
        db: Session,
        user_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        return self._get_backend().search(db, user_id, query_embedding, top_k, document_id)

    def store_embedding(
        self,
        db: Session,
        chunk_id: int,
        embedding: List[float],
    ) -> bool:
        return self._get_backend().store_embedding(db, chunk_id, embedding)


def get_vector_backend() -> VectorBackend:
    """Factory: return the configured vector backend.

    Uses VECTOR_BACKEND config (default: "auto").
    """
    backend_name = getattr(settings, 'vector_backend', 'auto').lower()

    if backend_name == 'pgvector':
        backend = PgvectorBackend()
        if not backend.is_available:
            raise RuntimeError(
                "pgvector backend requested but pgvector extension is not installed"
            )
        return backend
    elif backend_name == 'json':
        return JsonFallbackBackend()
    elif backend_name == 'auto':
        return AutoVectorBackend()
    else:
        raise ValueError(f"Unknown vector backend: {backend_name!r}")


# Singleton instance
_vector_backend: Optional[VectorBackend] = None


def get_vector_backend_singleton() -> VectorBackend:
    """Get or create the singleton vector backend."""
    global _vector_backend
    if _vector_backend is None:
        _vector_backend = get_vector_backend()
    return _vector_backend
