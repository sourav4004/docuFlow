"""Phase 18 vector platform — model registry, versioning, index ops, health.

Every embedding model is declared (provider/model/dimensions/version). Chunks
are never silently mixed across incompatible embeddings; retrieval validates
dimensions and model version before use. Index create/rebuild/validate
operations are audited via ``VectorIndexOp``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase18 import EmbeddingModel, VectorIndexOp

logger = logging.getLogger(__name__)

VECTOR_INDEX_PARAMS = os.getenv(
    "VECTOR_INDEX_PARAMS",
    "hnsw.m=16,hnsw.ef_construction=64")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_model(db: Session, *, provider: str, model: str,
                   dimensions: int, version: str = "v1",
                   notes: Optional[str] = None,
                   active: bool = True) -> EmbeddingModel:
    """Declare an embedding model. Idempotent on the unique tuple."""
    if dimensions <= 0 or dimensions > 65535:
        raise ValueError(f"invalid embedding dimensions: {dimensions}")
    existing = (db.query(EmbeddingModel)
                .filter(EmbeddingModel.provider == provider,
                        EmbeddingModel.model == model,
                        EmbeddingModel.dimensions == dimensions,
                        EmbeddingModel.version == version)
                .first())
    if existing is not None:
        existing.active = active
        if notes:
            existing.notes = notes
        db.flush()
        return existing
    row = EmbeddingModel(provider=provider, model=model,
                         dimensions=dimensions, version=version,
                         notes=notes, active=active)
    db.add(row)
    db.flush()
    return row


def active_model(db: Session, *, provider: Optional[str] = None,
                 model: Optional[str] = None) -> Optional[EmbeddingModel]:
    """The single active model (optionally filtered), or None."""
    q = db.query(EmbeddingModel).filter(EmbeddingModel.active.is_(True))
    if provider:
        q = q.filter(EmbeddingModel.provider == provider)
    if model:
        q = q.filter(EmbeddingModel.model == model)
    return q.order_by(EmbeddingModel.id.desc()).first()


def deactivate_model(db: Session, model_id: int) -> EmbeddingModel:
    row = db.get(EmbeddingModel, model_id)
    if row is None:
        raise KeyError(f"embedding model {model_id} not found")
    row.active = False
    db.flush()
    return row


def list_models(db: Session, limit: int = 100) -> list[EmbeddingModel]:
    return (db.query(EmbeddingModel)
            .order_by(EmbeddingModel.created_at.desc())
            .limit(min(limit, 500)).all())


def validate_embedding_dimensions(db: Session, dimensions: int,
                                  *, provider: Optional[str] = None,
                                  model: Optional[str] = None) -> dict:
    """Reject mismatched embeddings with a clean application error."""
    declared = active_model(db, provider=provider, model=model)
    if declared is None:
        # No declared model — allow but warn; registry is the source of truth
        # once populated.
        return {"valid": True, "declared": None,
                "note": "no active embedding model declared"}
    if dimensions != declared.dimensions:
        return {
            "valid": False,
            "declared": {"provider": declared.provider,
                         "model": declared.model,
                         "dimensions": declared.dimensions,
                         "version": declared.version},
            "received": dimensions,
            "note": ("embedding dimension mismatch — refusing to mix "
                     "incompatible vectors"),
        }
    return {"valid": True, "declared": declared.id, "dimensions": dimensions}


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------

def start_index_op(db: Session, *, op_type: str,
                   embedding_model_id: Optional[int] = None,
                   index_name: Optional[str] = None,
                   operator_user_id: Optional[int] = None) -> VectorIndexOp:
    if op_type not in ("create", "rebuild", "validate"):
        raise ValueError(f"invalid index op_type: {op_type}")
    row = VectorIndexOp(op_type=op_type,
                        embedding_model_id=embedding_model_id,
                        index_name=index_name,
                        operator_user_id=operator_user_id,
                        status="RUNNING")
    db.add(row)
    db.flush()
    return row


def finish_index_op(db: Session, op_id: int, *, ok: bool,
                    detail: Optional[str] = None) -> VectorIndexOp:
    row = db.get(VectorIndexOp, op_id)
    if row is None:
        raise KeyError(f"index op {op_id} not found")
    row.status = "COMPLETED" if ok else "FAILED"
    row.detail = (detail or "")[:1000]
    row.completed_at = _utcnow()
    db.flush()
    return row


def list_index_ops(db: Session, limit: int = 50) -> list[VectorIndexOp]:
    return (db.query(VectorIndexOp)
            .order_by(VectorIndexOp.started_at.desc())
            .limit(min(limit, 200)).all())


def index_plan(embedding_model: EmbeddingModel) -> dict:
    """Deterministic index strategy for the given model (declarative)."""
    model_slug = (embedding_model.model or "")[:24]
    return {
        "index_name": f"idx_vec_{embedding_model.provider}_{model_slug}_{embedding_model.dimensions}",
        "index_type": "hnsw",
        "params": VECTOR_INDEX_PARAMS,
        "dimensions": embedding_model.dimensions,
        "requires_pgvector": True,
    }


# ---------------------------------------------------------------------------
# Vector health
# ---------------------------------------------------------------------------

def vector_health(db: Session) -> dict:
    """Backend, dimensions, index status, vector coverage."""
    from .vector_platform import detect_pgvector
    from .vector_backend import get_vector_backend
    from ..models.document_chunk import DocumentChunk

    pg = detect_pgvector(db)
    backend = get_vector_backend().name
    total_chunks = db.query(DocumentChunk).count()
    vector_coverage = 0.0
    if pg.get("available") and total_chunks > 0:
        embedded = (db.query(DocumentChunk)
                    .filter(DocumentChunk.embedding.isnot(None))
                    .count())
        vector_coverage = round(embedded / total_chunks * 100, 1)
    elif total_chunks == 0:
        vector_coverage = 100.0
    else:
        # JSON-fallback mode: coverage is tracked via the embedding cache.
        from ..models.phase16 import EmbeddingCache
        cached = db.query(EmbeddingCache).count()
        vector_coverage = 100.0 if cached > 0 else 0.0
    model = active_model(db)
    return {
        "pgvector": pg,
        "backend": backend,
        "active_model": None if model is None else {
            "id": model.id, "provider": model.provider,
            "model": model.model, "dimensions": model.dimensions,
            "version": model.version},
        "total_chunks": total_chunks,
        "vector_coverage_pct": vector_coverage,
        "index_params": VECTOR_INDEX_PARAMS,
    }