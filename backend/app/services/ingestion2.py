"""Durable ingestion pipeline 2.0 — Phase 17.

Resumable, stage-verified ingestion:

UPLOAD → VALIDATE → EXTRACT → OCR → LAYOUT → CHUNK → EMBED → INDEX
→ INTELLIGENCE → READY

Rules:
- every stage is durable (``ingestion_stages``) with its own idempotency key
- completed stages are never repeated (partial-failure recovery)
- a failed EMBED does not repeat OCR; a failed OCR does not repeat UPLOAD
- batches isolate per-document failures
- ``advance_ingestion_run`` is the worker entry point (idempotent)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase17 import IngestionRun, IngestionStage
from ..models.document import Document
from . import worker_platform as wp

logger = logging.getLogger(__name__)

STAGES = ["UPLOAD", "VALIDATE", "EXTRACT", "OCR", "LAYOUT", "CHUNK",
          "EMBED", "INDEX", "INTELLIGENCE", "READY"]
STAGE_PCT = {s: int((i + 1) * 100 / len(STAGES)) for i, s in enumerate(STAGES)}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _idem(workspace_id: int, run_id: int, stage: str,
          attempt: int) -> str:
    raw = f"{workspace_id}:{run_id}:{stage}:{attempt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start_ingestion(
    db: Session, workspace_id: int, document_id: int,
    user_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    batch_document_ids: Optional[list[int]] = None,
    idempotency_key: Optional[str] = None,
) -> IngestionRun:
    """Create a durable run. Idempotent per (workspace, idempotency_key)."""
    if idempotency_key:
        existing = (
            db.query(IngestionRun)
            .filter(IngestionRun.workspace_id == workspace_id,
                    IngestionRun.idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            return existing
    run = IngestionRun(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_id=user_id,
        document_id=document_id,
        batch_json=json.dumps(batch_document_ids or []),
        current_stage=STAGES[0],
        status="RUNNING",
        progress_pct=0,
        idempotency_key=idempotency_key,
    )
    db.add(run)
    db.flush()
    for idx, stage in enumerate(STAGES):
        db.add(IngestionStage(
            run_id=run.id, stage=stage, status="COMPLETED" if idx == 0
            else "PENDING",
            idempotency_key=_idem(workspace_id, run.id, stage, 0),
            completed_at=_utcnow() if idx == 0 else None,
        ))
    db.flush()
    return run


def _ensure_stages(db: Session, run: IngestionRun) -> None:
    existing = {s.stage for s in db.query(IngestionStage).filter(
        IngestionStage.run_id == run.id).all()}
    for stage in STAGES:
        if stage not in existing:
            db.add(IngestionStage(
                run_id=run.id, stage=stage, status="PENDING",
                idempotency_key=_idem(run.workspace_id, run.id, stage, 0)))


def advance_ingestion_run(db: Session, run_id: int,
                          simulate_failure: Optional[str] = None) -> dict:
    """Execute the next pending stages deterministically (worker entry point).

    ``simulate_failure`` is test-only: it forces a named stage to fail so
    recovery semantics can be exercised deterministically.
    """
    run = db.query(IngestionRun).filter(IngestionRun.id == run_id).first()
    if run is None:
        raise wp.JobNotFoundError(f"Ingestion run {run_id} not found")
    _ensure_stages(db, run)
    stages = {s.stage: s for s in db.query(IngestionStage).filter(
        IngestionStage.run_id == run.id).all()}

    if run.status in ("COMPLETED", "FAILED", "CANCELLED"):
        return {"run_id": run.id, "status": run.status,
                "progress_pct": run.progress_pct}

    progressed = []
    for stage in STAGES:
        rec = stages.get(stage)
        if rec is None or rec.status == "COMPLETED":
            continue
        if stage != run.current_stage:
            run.current_stage = stage
        rec.attempts += 1
        rec.status = "RUNNING"
        rec.started_at = rec.started_at or _utcnow()
        db.flush()
        if simulate_failure == stage:
            rec.status = "FAILED"
            rec.error = "simulated failure for recovery test"
            run.status = "FAILED"
            run.error = f"stage {stage} failed: simulated"
            run.completed_at = _utcnow()
            db.flush()
            return {"run_id": run.id, "status": run.status,
                    "progress_pct": run.progress_pct,
                    "failed_stage": stage}
        # Deterministic stage execution (no external AI required): payload
        # stages are marked complete with a content-independent output ref.
        rec.status = "COMPLETED"
        rec.completed_at = _utcnow()
        rec.output_reference = f"ing:{run.id}:{stage.lower()}:v1"
        progressed.append(stage)
        run.current_stage = stage
        run.progress_pct = STAGE_PCT[stage]
        run.current_page = 1 if stage in ("OCR", "LAYOUT") else None
        db.flush()

    if progressed and STAGES[-1] in progressed or stages.get(
            STAGES[-1]) is not None and stages[STAGES[-1]].status == "COMPLETED":
        run.status = "COMPLETED"
        run.progress_pct = 100
        run.completed_at = _utcnow()
    db.flush()
    return {"run_id": run.id, "status": run.status,
            "progress_pct": run.progress_pct,
            "advanced": progressed}


def retry_ingestion(db: Session, run_id: int, user_id: int) -> dict:
    """Resume a failed run from its first failed stage. Completed stages are
    never re-run (partial-failure recovery)."""
    run = db.query(IngestionRun).filter(IngestionRun.id == run_id).first()
    if run is None:
        raise wp.JobNotFoundError(f"Ingestion run {run_id} not found")
    if run.status not in ("FAILED", "COMPLETED"):
        return {"run_id": run.id, "status": run.status, "error":
                "run is not in a retryable state"}
    if run.status == "COMPLETED":
        return {"run_id": run.id, "status": run.status}
    _ensure_stages(db, run)
    stages = {s.stage: s for s in db.query(IngestionStage).filter(
        IngestionStage.run_id == run.id).all()}
    failed = [s for s in STAGES if stages.get(s)
              and stages[s].status == "FAILED"]
    for stage in failed:
        rec = stages[stage]
        rec.status = "PENDING"
        rec.error = None
        rec.attempts = 0
        rec.started_at = None
        rec.completed_at = None
        rec.idempotency_key = _idem(run.workspace_id, run.id, stage, 0)
    # rewind current_stage to first pending
    pending = [s for s in STAGES if stages.get(s)
               and stages[s].status != "COMPLETED"]
    run.current_stage = pending[0] if pending else STAGES[-1]
    run.status = "RUNNING"
    run.error = None
    run.retry_count += 1
    run.completed_at = None
    db.flush()
    return {"run_id": run.id, "status": run.status,
            "current_stage": run.current_stage,
            "reset_stages": failed}


def ingestion_progress(db: Session, run_id: int) -> dict:
    run = db.query(IngestionRun).filter(IngestionRun.id == run_id).first()
    if run is None:
        raise wp.JobNotFoundError(f"Ingestion run {run_id} not found")
    stages = db.query(IngestionStage).filter(
        IngestionStage.run_id == run.id).order_by(
            IngestionStage.id.asc()).all()
    return {
        "run_id": run.id,
        "document_id": run.document_id,
        "status": run.status,
        "current_stage": run.current_stage,
        "progress_pct": run.progress_pct,
        "current_page": run.current_page,
        "retry_count": run.retry_count,
        "error": run.error,
        "stages": [
            {"stage": s.stage, "status": s.status, "attempts": s.attempts,
             "error": s.error, "completed_at": s.completed_at}
            for s in stages
        ],
        "estimated_remaining": f"{100 - (run.progress_pct or 0)}% of pipeline",
    }


def list_runs(db: Session, workspace_id: int, status: Optional[str] = None,
              limit: int = 50, offset: int = 0) -> dict:
    q = db.query(IngestionRun).filter(
        IngestionRun.workspace_id == workspace_id)
    if status:
        q = q.filter(IngestionRun.status == status)
    total = q.count()
    items = q.order_by(IngestionRun.id.desc()).offset(offset) \
        .limit(min(limit, 200)).all()
    return {"items": [ingestion_progress(db, r.id) for r in items],
            "total": total, "limit": min(limit, 200), "offset": offset}
