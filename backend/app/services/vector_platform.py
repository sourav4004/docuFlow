"""Vector platform — pgvector compatibility layer, dimension validation,
embedding cache 2.0.

The JSON cosine fallback is preserved and reported honestly: startup and the
``/ops/vector-status`` endpoint state exactly which backend is active and why
— never masking pgvector absence.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase16 import EmbeddingCache

logger = logging.getLogger(__name__)

# Application-wide embedding dimension contract (must match the provider).
DEFAULT_EMBEDDING_DIMENSIONS = 384


class DimensionMismatchError(Exception):
    """Raised when embedding dimensions are inconsistent."""


class VectorConfigurationError(Exception):
    """Raised for invalid vector backend configuration."""


# ---------------------------------------------------------------------------
# Backend detection/status (no secrets exposed)
# ---------------------------------------------------------------------------

def detect_pgvector(db: Session) -> dict:
    """Detect pgvector availability and extension/column/index support.

    Returns a safe status dict — never connection strings.
    """
    dialect = db.bind.dialect.name if db.bind else "unknown"
    if dialect != "postgresql":
        return {
            "dialect": dialect,
            "pgvector_available": False,
            "extension_installed": False,
            "vector_columns_supported": False,
            "vector_index_supported": False,
            "active_backend": "json_fallback",
            "reason": "pgvector requires PostgreSQL; non-PostgreSQL databases "
                      "use the JSON cosine fallback.",
        }
    from sqlalchemy import text
    ext = db.execute(
        text("SELECT name FROM pg_available_extensions WHERE name = 'vector'")
    ).fetchone()
    installed = db.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).fetchone() is not None
    available = ext is not None
    if available and installed:
        active = "pgvector"
    else:
        active = "json_fallback"
    return {
        "dialect": dialect,
        "pgvector_available": bool(available),
        "extension_installed": bool(installed),
        "vector_columns_supported": bool(available),
        "vector_index_supported": bool(available),
        "active_backend": active,
        "reason": ("" if active == "pgvector" else
                   "pgvector extension not installed — JSON cosine fallback "
                   "is active (document_chunks.embedding stores JSON arrays)."),
    }


def validate_dimensions(
    provider: str,
    model: str,
    configured_dimensions: int,
    embedding: list,
) -> int:
    """Validate embedding dimensions against the provider/model contract.

    Returns the dimension count when valid; raises DimensionMismatchError on
    any mismatch so mismatched vectors never enter the index silently.
    """
    dims = len(embedding)
    if dims == 0:
        raise DimensionMismatchError(
            f"Empty embedding from {provider}/{model}")
    if configured_dimensions and dims != configured_dimensions:
        raise DimensionMismatchError(
            f"Embedding dimension {dims} does not match configured "
            f"{configured_dimensions} for {provider}/{model}")
    return dims


# ---------------------------------------------------------------------------
# Embedding cache 2.0
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _cache_key(provider: str, model: str, dimensions: int,
               text_hash: str, tenant_scoped: bool,
               workspace_id: Optional[int]) -> str:
    raw = "|".join([provider, model, str(dimensions), text_hash])
    if tenant_scoped:
        raw = f"{workspace_id or 0}|{raw}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def embedding_cache_get(
    db: Session,
    provider: str,
    model: str,
    dimensions: int,
    text: str,
    tenant_scoped: bool = True,
    workspace_id: Optional[int] = None,
    max_age_seconds: int = 60 * 60 * 24 * 30,
) -> Optional[list]:
    """Return a cached embedding, or None on miss.

    Cache keys are deterministic; when ``tenant_scoped`` (default) the key is
    bound to the workspace so results never cross tenant boundaries.
    """
    text_hash = content_hash(text)
    key = _cache_key(provider, model, dimensions, text_hash,
                     tenant_scoped, workspace_id)
    row = (
        db.query(EmbeddingCache)
        .filter(EmbeddingCache.cache_key == key)
        .first()
    )
    if row is None:
        return None
    expires = row.expires_at
    if expires is not None:
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            return None
    row.hits = (row.hits or 0) + 1
    db.flush()
    try:
        return json.loads(row.embedding_json)
    except (ValueError, TypeError):
        return None


def embedding_cache_set(
    db: Session,
    provider: str,
    model: str,
    dimensions: int,
    text: str,
    embedding: list,
    tenant_scoped: bool = True,
    workspace_id: Optional[int] = None,
    ttl_seconds: Optional[int] = 60 * 60 * 24 * 30,
) -> EmbeddingCache:
    """Store an embedding with a deterministic, tenant-aware key."""
    validate_dimensions(provider, model, dimensions, embedding)
    text_hash = content_hash(text)
    key = _cache_key(provider, model, dimensions, text_hash,
                     tenant_scoped, workspace_id)
    row = (
        db.query(EmbeddingCache)
        .filter(EmbeddingCache.cache_key == key)
        .first()
    )
    if row is None:
        row = EmbeddingCache(
            cache_key=key, provider=provider, model=model,
            dimensions=dimensions, content_hash=text_hash,
            workspace_id=workspace_id if tenant_scoped else None,
        )
        db.add(row)
    row.embedding_json = json.dumps(embedding)
    row.expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                      if ttl_seconds else None)
    db.flush()
    return row


def embedding_cache_stats(db: Session) -> dict:
    rows = db.query(EmbeddingCache).all()
    return {
        "entries": len(rows),
        "hits": sum(r.hits or 0 for r in rows),
        "providers": sorted({r.provider for r in rows}),
    }
