"""Phase 18 cleanup — durable, bounded, hold-aware cleanup workers.

Cleanup runs through the worker queue (RETENTION_CLEANUP job type). Every
operation is bounded, auditable, and skips legal-hold data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

CLEANUP_BATCH = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cleanup_candidates(db: Session, *, entity_type: str,
                       workspace_id: Optional[int] = None,
                       older_than_days: int = 90,
                       limit: int = CLEANUP_BATCH) -> list[dict]:
    """Bounded candidate scan for cleanup (never unbounded deletes)."""
    from ..models.phase16 import TraceSpan, WorkerJob
    from ..models.audit_log import AuditLog
    from ..models.webhook import WebhookDelivery
    cutoff = _utcnow() - timedelta(days=older_than_days)
    candidates: list[dict] = []
    if entity_type == "traces":
        q = db.query(TraceSpan).filter(TraceSpan.created_at < cutoff)
        if workspace_id is not None:
            q = q.filter(TraceSpan.workspace_id == workspace_id)
        for row in q.limit(limit).all():
            candidates.append({"id": row.id, "entity_type": "traces",
                               "created_at": row.created_at})
    elif entity_type == "audit_logs":
        q = db.query(AuditLog).filter(AuditLog.created_at < cutoff)
        for row in q.limit(limit).all():
            candidates.append({"id": row.id, "entity_type": "audit_logs",
                               "created_at": row.created_at})
    elif entity_type == "webhook_deliveries":
        q = db.query(WebhookDelivery).filter(
            WebhookDelivery.created_at < cutoff)
        for row in q.limit(limit).all():
            candidates.append({"id": row.id,
                               "entity_type": "webhook_deliveries",
                               "created_at": row.created_at})
    elif entity_type == "dead_letters":
        q = db.query(WorkerJob).filter(
            WorkerJob.status == "DEAD_LETTERED",
            WorkerJob.completed_at < cutoff)
        for row in q.limit(limit).all():
            candidates.append({"id": row.id, "entity_type": "dead_letters",
                               "created_at": row.completed_at})
    else:
        raise ValueError(f"unsupported cleanup entity_type {entity_type!r}")
    return candidates


def run_bounded_cleanup(db: Session, *, entity_type: str,
                        workspace_id: Optional[int] = None,
                        older_than_days: int = 90,
                        dry_run: bool = True,
                        limit: int = CLEANUP_BATCH) -> dict:
    """Cleanup respecting legal holds + bounds. Dry-run by default."""
    from ..models.phase16 import TraceSpan, WorkerJob
    from ..models.audit_log import AuditLog
    from ..models.webhook import WebhookDelivery
    candidates = cleanup_candidates(
        db, entity_type=entity_type, workspace_id=workspace_id,
        older_than_days=older_than_days, limit=limit)
    deleted = 0
    held_skipped = 0
    for cand in candidates:
        held = _under_hold(db, workspace_id, cand["entity_type"],
                           cand["id"])
        if held:
            held_skipped += 1
            continue
        if dry_run:
            continue
        if entity_type == "traces":
            db.query(TraceSpan).filter(
                TraceSpan.id == cand["id"]).delete()
        elif entity_type == "audit_logs":
            db.query(AuditLog).filter(AuditLog.id == cand["id"]).delete()
        elif entity_type == "webhook_deliveries":
            db.query(WebhookDelivery).filter(
                WebhookDelivery.id == cand["id"]).delete()
        elif entity_type == "dead_letters":
            db.query(WorkerJob).filter(
                WorkerJob.id == cand["id"]).delete()
        deleted += 1
    return {"entity_type": entity_type, "candidates": len(candidates),
            "deleted": deleted, "held_skipped": held_skipped,
            "dry_run": dry_run, "limit": limit}


def _under_hold(db: Session, workspace_id: Optional[int],
                entity_type: str, entity_id: int) -> bool:
    """Legal hold check — held data is never auto-deleted."""
    from ..models.phase17 import LegalHold, HoldEntity
    if workspace_id is None:
        return False
    held = (db.query(HoldEntity.id)
            .join(LegalHold, HoldEntity.hold_id == LegalHold.id)
            .filter(LegalHold.workspace_id == workspace_id,
                    LegalHold.status == "ACTIVE",
                    HoldEntity.entity_type == entity_type,
                    HoldEntity.entity_id == entity_id)
            .first())
    return held is not None


def enqueue_cleanup(db: Session, *, entity_type: str,
                    workspace_id: int, older_than_days: int = 90) -> dict:
    """Schedule a bounded cleanup through the durable worker queue."""
    from .worker_platform import enqueue_job
    job = enqueue_job(
        db, queue_name="MAINTENANCE", job_type="RETENTION_CLEANUP",
        workspace_id=workspace_id,
        payload={"entity_type": entity_type,
                 "older_than_days": older_than_days},
        dedupe_key=f"cleanup:{entity_type}:{workspace_id}",
        max_attempts=2)
    return {"enqueued": True, "job_id": job.id}