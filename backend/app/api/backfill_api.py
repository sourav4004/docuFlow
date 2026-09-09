"""Legacy workspace backfill API — admin-only, dry-run capable, audited."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.backfill_service import (
    audit_legacy, create_run, preview_batch, execute_batch,
    run_complete, BackfillError,
)
from ..services.permission_service import get_organization_role

router = APIRouter(prefix="/admin/backfill", tags=["admin-backfill"])


def _gate(db: Session, organization_id: int, principal: AuthPrincipal):
    role = get_organization_role(db, organization_id, principal.user.id)
    if role not in ("OWNER", "ADMIN"):
        raise HTTPException(status_code=403,
                            detail="Organization admin required")


def _run_dict(run) -> dict:
    return {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        "dry_run": run.dry_run,
        "workspace_id": run.workspace_id,
        "user_id": run.user_id,
        "batch_size": run.batch_size,
        "cursor_id": run.cursor_id,
        "total": run.total,
        "processed": run.processed,
        "assigned": run.assigned,
        "skipped": run.skipped,
        "ambiguous": run.ambiguous,
        "failed": run.failed,
        "error_summary": run.error_summary,
        "created_by": run.created_by,
        "created_at": run.created_at,
        "completed_at": run.completed_at,
    }


class AuditRequest(BaseModel):
    organization_id: int


@router.post("/audit")
def audit_endpoint(
    data: AuditRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Scan for legacy unassigned records (no writes)."""
    _gate(db, data.organization_id, principal)
    return audit_legacy(db)


class PreviewRequest(BaseModel):
    organization_id: int
    kind: str
    batch_size: int = 100
    cursor_id: Optional[int] = None


@router.post("/preview")
def preview_endpoint(
    data: PreviewRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Dry preview of the next batch with per-record decisions."""
    _gate(db, data.organization_id, principal)
    try:
        return preview_batch(db, data.kind, data.batch_size, data.cursor_id)
    except BackfillError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class RunCreate(BaseModel):
    organization_id: int
    kind: str
    dry_run: bool = True
    batch_size: int = 100
    workspace_id: Optional[int] = None
    user_id: Optional[int] = None
    resume_from: Optional[int] = None


@router.post("/runs", status_code=201)
def create_run_endpoint(
    data: RunCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _gate(db, data.organization_id, principal)
    try:
        run = create_run(
            db, created_by=principal.user.id, kind=data.kind,
            dry_run=data.dry_run, workspace_id=data.workspace_id,
            user_id=data.user_id, batch_size=data.batch_size,
            resume_from=data.resume_from,
        )
    except BackfillError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(run)
    return _run_dict(run)


class RunExecute(BaseModel):
    organization_id: int
    complete: bool = False


@router.post("/runs/{run_id}/execute")
def execute_run_endpoint(
    run_id: int,
    data: RunExecute,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Execute one batch (or drive to completion when ``complete``)."""
    from ..models.phase16 import BackfillRun
    run = db.query(BackfillRun).filter(BackfillRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _gate(db, data.organization_id, principal)
    try:
        if data.complete:
            result = run_complete(db, run_id)
            return result
        stats = execute_batch(db, run)
        db.commit()
        return {**stats, "run_id": run.id, "status": run.status,
                "dry_run": run.dry_run}
    except BackfillError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/runs")
def list_runs_endpoint(
    organization_id: int,
    limit: int = 50,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase16 import BackfillRun
    _gate(db, organization_id, principal)
    runs = (
        db.query(BackfillRun)
        .order_by(BackfillRun.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return {"items": [_run_dict(r) for r in runs]}
