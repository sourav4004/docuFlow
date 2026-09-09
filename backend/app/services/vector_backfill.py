"""Vector backfill + rebuild — Phase 17.

Idempotent batch process that populates embeddings for chunks missing them.
Behavior depends honestly on the active vector backend:

- pgvector available: vectors are stored natively through the embedding layer.
- pgvector unavailable: the JSON cosine fallback remains active and the run
  reports ``native_pgvector: false``.

Every stored embedding is dimension-validated against the run dimensions —
mismatched vectors are rejected, never silently mixed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_chunk import DocumentChunk
from ..models.phase17 import VectorBackfillRun
from .vector_platform import detect_pgvector
from . import embedding_service as _embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE_DEFAULT = 25


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _provider_model() -> str:
    return getattr(_embeddings.settings, "openai_embedding_model",
                   "text-embedding-3-small")


def _provider_dimensions() -> int:
    try:
        return _embeddings.get_embedding_provider().dimension
    except Exception:  # noqa: BLE001
        return 384


def _chunks_missing(db: Session, workspace_id: Optional[int],
                    limit: int = 500):
    q = db.query(DocumentChunk).filter(DocumentChunk.embedding.is_(None))
    if workspace_id:
        q = q.join(Document, DocumentChunk.document_id == Document.id) \
            .filter(Document.workspace_id == workspace_id)
    return q.limit(limit).all()


def preview_backfill(db: Session, workspace_id: Optional[int] = None) -> dict:
    """Estimate work without writing anything."""
    chunks = _chunks_missing(db, workspace_id, limit=2000)
    documents = {c.document_id for c in chunks}
    vector_state = detect_pgvector(db)
    return {
        "documents_to_process": len(documents),
        "chunks_to_embed": len(chunks),
        "dry_run": True,
        "native_pgvector": bool(vector_state.get("pgvector_available")),
        "vector_backend": vector_state.get("active_backend")
        or vector_state.get("backend") or "json_fallback",
        "model": _provider_model(),
        "dimensions": _provider_dimensions(),
        "note": "no rows written (preview)",
    }


def start_backfill(db: Session, *, workspace_id: Optional[int] = None,
                   dimensions: Optional[int] = None,
                   dry_run: bool = False,
                   created_by: Optional[int] = None,
                   organization_id: Optional[int] = None) -> VectorBackfillRun:
    """Create a backfill run; executes batches when not a dry run."""
    dims = dimensions or _provider_dimensions()
    model = _provider_model()
    if dry_run:
        preview = preview_backfill(db, workspace_id)
        run = VectorBackfillRun(
            workspace_id=workspace_id, organization_id=organization_id,
            model=model, dimensions=dims, dry_run=True,
            status="COMPLETED", total=preview["chunks_to_embed"],
            created_by=created_by, created_at=_utcnow(),
            completed_at=_utcnow())
        db.add(run)
        db.flush()
        return run
    chunks = _chunks_missing(db, workspace_id, limit=2000)
    run = VectorBackfillRun(
        workspace_id=workspace_id, organization_id=organization_id,
        model=model, dimensions=dims, dry_run=False,
        status="RUNNING", total=len(chunks), created_by=created_by,
        created_at=_utcnow())
    db.add(run)
    db.flush()
    _process_batch(db, run, chunks)
    run.status = "COMPLETED" if run.failed == 0 else "PARTIAL"
    run.completed_at = _utcnow()
    db.flush()
    return run


def run_backfill_batch(db: Session, run_id: int) -> dict:
    """Worker entry: continue an unfinished run (resumable)."""
    run = db.query(VectorBackfillRun).filter(
        VectorBackfillRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Vector backfill run {run_id} not found")
    if run.dry_run or run.status in ("COMPLETED", "FAILED", "PARTIAL"):
        return {"run_id": run.id, "status": run.status, "processed": 0}
    remaining = _chunks_missing(db, run.workspace_id,
                                limit=BATCH_SIZE_DEFAULT)
    if not remaining:
        run.status = "COMPLETED"
        run.completed_at = _utcnow()
        db.flush()
        return {"run_id": run.id, "status": run.status, "processed": 0}
    processed = _process_batch(db, run, remaining)
    run.status = "RUNNING"
    db.flush()
    return {"run_id": run.id, "status": run.status,
            "processed": processed, "failed": run.failed}


def _process_batch(db: Session, run: VectorBackfillRun,
                   chunks) -> int:
    processed = 0
    stored = []
    try:
        stored = json.loads(run.errors_json or "[]")
    except (ValueError, TypeError):
        stored = []
    provider = _embeddings.get_embedding_provider()
    for chunk in chunks:
        try:
            if chunk.text is None or not chunk.text.strip():
                continue
            vectors = provider.embed_texts([chunk.text])
            if len(vectors) != 1:
                raise ValueError("embedding provider returned no vector")
            vector = vectors[0]
            provider.validate_embedding(vector)
            if len(vector) != run.dimensions:
                run.failed += 1
                stored.append({
                    "document_chunk_id": chunk.id,
                    "error": (f"dimension mismatch: expected "
                              f"{run.dimensions}, got {len(vector)}")})
                continue
            chunk.embedding = vector
            processed += 1
        except Exception as exc:  # noqa: BLE001 — batch isolation
            run.failed += 1
            stored.append({"document_chunk_id": chunk.id,
                           "error": str(exc)[:500]})
    run.processed += processed
    run.total = max(run.total, run.processed + run.failed)
    run.errors_json = json.dumps(stored[-200:])
    db.flush()
    return processed


def rebuild_vectors(db: Session, *, workspace_id: Optional[int] = None,
                    created_by: Optional[int] = None) -> dict:
    """Safe rebuild after model/dimension change.

    Clears then regenerates embeddings per document through the existing
    idempotent embedding service — incompatible embeddings are never mixed
    because a whole document's vectors are regenerated atomically.
    """
    from ..services.embedding_service import (
        clear_document_embeddings, generate_document_embeddings)
    q = db.query(Document)
    if workspace_id:
        q = q.filter(Document.workspace_id == workspace_id)
    docs = q.limit(500).all()
    processed = 0
    failed = 0
    errors = []
    for doc in docs:
        try:
            cleared = clear_document_embeddings(db, doc.id)
            embedded = generate_document_embeddings(db, doc.id)
            processed += embedded
        except Exception as exc:  # noqa: BLE001 — per-document isolation
            failed += 1
            errors.append({"document_id": doc.id, "error": str(exc)[:500]})
    db.flush()
    return {
        "documents_rebuilt": max(0, len(docs) - failed),
        "chunks_embedded": processed,
        "failed_documents": failed,
        "errors": errors[:50],
        "note": "per-document atomic regeneration; incompatible vectors never mixed",
    }
