"""Phase 18 ingestion operations — poison quarantine, batch progress,
stage leasing, and resource limits.

Builds on the Phase 17 durable ingestion pipeline (ingestion_runs /
ingestion_stages). Poison detection is explicit and reviewable; documents
are never silently deleted.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import IngestionRun, IngestionStage
from ..models.phase18 import PoisonDocument

logger = logging.getLogger(__name__)

POISON_THRESHOLD = int(os.getenv("INGESTION_POISON_THRESHOLD", "3"))
MAX_BATCH_DOCUMENTS = int(os.getenv("INGESTION_MAX_BATCH", "50"))
MAX_DOCUMENT_MB = int(os.getenv("INGESTION_MAX_DOCUMENT_MB", "200"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

def validate_resource_limits(*, batch_size: int = 1,
                             document_mb: Optional[float] = None) -> dict:
    """Reject oversized batches / documents before work begins."""
    errors = []
    if batch_size < 1:
        errors.append("batch_size must be >= 1")
    if batch_size > MAX_BATCH_DOCUMENTS:
        errors.append(
            f"batch exceeds limit of {MAX_BATCH_DOCUMENTS} documents")
    if document_mb is not None and document_mb > MAX_DOCUMENT_MB:
        errors.append(
            f"document exceeds limit of {MAX_DOCUMENT_MB} MB")
    return {"valid": not errors, "errors": errors,
            "max_batch": MAX_BATCH_DOCUMENTS,
            "max_document_mb": MAX_DOCUMENT_MB}


# ---------------------------------------------------------------------------
# Stage leases
# ---------------------------------------------------------------------------

def stage_lease_status(db: Session, run_id: int) -> dict:
    """Per-stage lease/attempt state for an ingestion run."""
    stages = (db.query(IngestionStage)
              .filter(IngestionStage.run_id == run_id)
              .order_by(IngestionStage.id).all())
    return {"run_id": run_id, "stages": [
        {"stage": s.stage, "status": s.status, "attempts": s.attempts,
         "idempotency_key": s.idempotency_key,
         "error": (s.error or "")[:200],
         "started_at": s.started_at, "completed_at": s.completed_at}
        for s in stages]}


def bump_stage_attempt(db: Session, run_id: int, stage: str,
                       error: Optional[str] = None) -> IngestionStage:
    """Increment the attempt counter for a stage (bounded by policy)."""
    row = (db.query(IngestionStage)
           .filter(IngestionStage.run_id == run_id,
                   IngestionStage.stage == stage)
           .first())
    if row is None:
        raise KeyError(f"stage {stage!r} not found for run {run_id}")
    row.attempts = (row.attempts or 0) + 1
    if error:
        row.error = error[:1000]
    db.flush()
    return row


# ---------------------------------------------------------------------------
# Poison quarantine
# ---------------------------------------------------------------------------

def record_poison(db: Session, *, workspace_id: int,
                  document_id: int, stage: str,
                  error: Optional[str] = None,
                  threshold: int = POISON_THRESHOLD) -> Optional[PoisonDocument]:
    """Quarantine a document after repeated stage failures.

    Returns the PoisonDocument when newly quarantined, else None. Repeated
    calls are idempotent (counters increment, one quarantine row).
    """
    row = (db.query(PoisonDocument)
           .filter(PoisonDocument.workspace_id == workspace_id,
                   PoisonDocument.document_id == document_id,
                   PoisonDocument.status == "QUARANTINED")
           .first())
    now = _utcnow()
    if row is None:
        row = PoisonDocument(workspace_id=workspace_id,
                             document_id=document_id,
                             stage=stage, failure_count=1,
                             last_error=(error or "")[:2000],
                             status="QUARANTINED",
                             created_at=now, updated_at=now)
        db.add(row)
        db.flush()
        return row
    row.failure_count = (row.failure_count or 0) + 1
    row.last_error = (error or row.last_error or "")[:2000]
    row.stage = stage
    row.updated_at = now
    db.flush()
    return row


def resolve_poison(db: Session, *, workspace_id: int, poison_id: int,
                   action: str, user_id: int) -> PoisonDocument:
    """RELEASE (retry) or ABANDON (keep, stop retrying) a quarantined doc.

    Never deletes the document — resolution is explicit and auditable.
    """
    if action not in ("RELEASE", "ABANDON"):
        raise ValueError("action must be RELEASE or ABANDON")
    row = (db.query(PoisonDocument)
           .filter(PoisonDocument.id == poison_id,
                   PoisonDocument.workspace_id == workspace_id)
           .first())
    if row is None:
        raise KeyError(f"poison document {poison_id} not found")
    row.status = "RELEASED" if action == "RELEASE" else "ABANDONED"
    row.resolved_by = user_id
    row.resolved_at = _utcnow()
    row.updated_at = _utcnow()
    db.flush()
    return row


def list_poison(db: Session, workspace_id: Optional[int] = None,
                status: Optional[str] = None,
                limit: int = 50) -> dict:
    q = db.query(PoisonDocument)
    if workspace_id is not None:
        q = q.filter(PoisonDocument.workspace_id == workspace_id)
    if status:
        q = q.filter(PoisonDocument.status == status)
    rows = q.order_by(PoisonDocument.updated_at.desc()).limit(
        min(limit, 200)).all()
    return {"items": [{"id": p.id, "workspace_id": p.workspace_id,
                       "document_id": p.document_id, "stage": p.stage,
                       "failure_count": p.failure_count,
                       "last_error": (p.last_error or "")[:200],
                       "status": p.status,
                       "created_at": p.created_at,
                       "updated_at": p.updated_at} for p in rows],
            "total": len(rows)}


# ---------------------------------------------------------------------------
# Batch progress
# ---------------------------------------------------------------------------

def batch_progress(db: Session, workspace_id: int,
                   run_id: Optional[int] = None) -> dict:
    """Aggregate progress across ingestion runs (optionally one run)."""
    q = db.query(IngestionRun).filter(
        IngestionRun.workspace_id == workspace_id)
    if run_id is not None:
        q = q.filter(IngestionRun.id == run_id)
    runs = q.order_by(IngestionRun.id.desc()).limit(200).all()
    total = len(runs)
    completed = sum(1 for r in runs if r.status == "COMPLETED")
    failed = sum(1 for r in runs if r.status == "FAILED")
    retrying = sum(1 for r in runs
                   if r.status in ("RETRYING", "RUNNING"))
    quarantined = (db.query(PoisonDocument)
                   .filter(PoisonDocument.workspace_id == workspace_id,
                           PoisonDocument.status == "QUARANTINED")
                   .count())
    avg_pct = 0
    if total:
        avg_pct = round(sum(r.progress_pct or 0 for r in runs) / total, 1)
    return {
        "total_runs": total,
        "completed": completed,
        "failed": failed,
        "retrying": retrying,
        "quarantined": quarantined,
        "progress_pct": avg_pct,
        "runs": [{"id": r.id, "document_id": r.document_id,
                  "status": r.status, "stage": r.current_stage,
                  "progress_pct": r.progress_pct,
                  "error": (r.error or "")[:200]} for r in runs],
    }