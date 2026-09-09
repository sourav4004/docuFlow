"""Phase 17 distributed operations API — worker fleet, leases, dead-letter
management, autoscaling signals, and vector backfill.

Admin surfaces are gated by org-admin or workspace-owner (same gate as the
Phase 16 /ops endpoints). None of these endpoints expose secrets or unsafe
runtime details.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..services.permission_service import (
    require_workspace_membership, get_organization_role,
)
from ..services import distributed as fleet
from ..services.worker_platform import (
    WorkerHeartbeat, cancel_job, JobNotFoundError,
)
from ..services import vector_backfill as vb

router = APIRouter(tags=["distributed"])


def _admin_gate(db: Session, principal: AuthPrincipal,
                workspace_id: Optional[int] = None) -> None:
    if workspace_id is None:
        raise HTTPException(status_code=403,
                            detail="Workspace scope required")
    ws = require_workspace_membership(db, workspace_id,
                                      principal.user.id)
    if ws.organization_id is not None:
        role = get_organization_role(db, ws.organization_id,
                                     principal.user.id)
        if role in ("OWNER", "ADMIN"):
            return
        raise HTTPException(status_code=403,
                            detail="Organization admin required")
    if ws.owner_id == principal.user.id:
        return
    raise HTTPException(status_code=403, detail="Workspace owner required")


# ---------------------------------------------------------------------------
# Fleet + autoscaling
# ---------------------------------------------------------------------------

@router.get("/worker-fleet")
def worker_fleet(workspace_id: Optional[int] = None,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    if workspace_id is not None:
        _admin_gate(db, principal, workspace_id)
    rows = db.query(WorkerHeartbeat).order_by(
        WorkerHeartbeat.last_heartbeat.desc()).limit(200).all()
    return {"items": [
        {"worker_id": w.worker_id, "status": w.status,
         "hostname": w.hostname, "pid": w.pid, "version": w.version,
         "queue_name": w.queue_name,
         "current_job_type": w.current_job_type,
         "current_job_id": w.current_job_id,
         "started_at": w.started_at,
         "last_heartbeat": w.last_heartbeat,
         "stopped_at": w.stopped_at}
        for w in rows], "total": len(rows)}


@router.get("/autoscale-signals")
def autoscale_signals(queue_name: Optional[str] = None,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    if not principal:
        raise HTTPException(status_code=401, detail="Authentication required")
    return fleet.autoscale_signals(db, queue_name=queue_name)


@router.post("/workers/recover-stale")
def recover_stale(principal: AuthPrincipal = Depends(get_current_principal),
                  db: Session = Depends(get_db)):
    return fleet.recover_expired_leases(db)


# ---------------------------------------------------------------------------
# Dead-letter administration
# ---------------------------------------------------------------------------

@router.get("/dead-letters")
def dead_letters(workspace_id: int, queue_name: Optional[str] = None,
                 limit: int = 50, offset: int = 0,
                 principal: AuthPrincipal = Depends(get_current_principal),
                 db: Session = Depends(get_db)):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return fleet.list_dead_letters(db, workspace_id=workspace_id,
                                   queue_name=queue_name,
                                   limit=limit, offset=offset)


class RequeueBody(BaseModel):
    reset_attempts: bool = True


@router.post("/dead-letters/{job_id}/requeue")
def dead_letter_requeue(job_id: int, body: RequeueBody,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    from ..models.phase16 import WorkerJob
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _admin_gate(db, principal, job.workspace_id)
    try:
        updated = fleet.requeue_dead_letter(
            db, job_id, principal.user.id,
            reset_attempts=body.reset_attempts)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": updated.id, "status": updated.status}


@router.post("/dead-letters/{job_id}/abandon")
def dead_letter_abandon(job_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    from ..models.phase16 import WorkerJob
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _admin_gate(db, principal, job.workspace_id)
    try:
        updated = fleet.abandon_dead_letter(db, job_id,
                                            principal.user.id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": updated.id, "status": updated.status}


@router.post("/worker-jobs/{job_id}/lease-extend")
def extend_job_lease(job_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    from ..models.phase16 import WorkerJob
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    require_workspace_membership(db, job.workspace_id,
                                 principal.user.id)
    ok = fleet.extend_lease(db, job, str(principal.user.id)
                            if job.claimed_by else str(principal.user.id))
    return {"job_id": job_id, "extended": ok}


# ---------------------------------------------------------------------------
# Vector backfill
# ---------------------------------------------------------------------------

@router.get("/vector-backfill/preview")
def vector_backfill_preview(workspace_id: Optional[int] = None,
                            principal: AuthPrincipal = Depends(get_current_principal),
                            db: Session = Depends(get_db)):
    if workspace_id is not None:
        _admin_gate(db, principal, workspace_id)
    return vb.preview_backfill(db, workspace_id=workspace_id)


class BackfillBody(BaseModel):
    workspace_id: Optional[int] = None
    dimensions: Optional[int] = None
    dry_run: bool = True


@router.post("/vector-backfill")
def vector_backfill_start(body: BackfillBody,
                          principal: AuthPrincipal = Depends(get_current_principal),
                          db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _admin_gate(db, principal, body.workspace_id)
    run = vb.start_backfill(db, workspace_id=body.workspace_id,
                            dimensions=body.dimensions,
                            dry_run=body.dry_run,
                            created_by=principal.user.id)
    return {"run_id": run.id, "status": run.status,
            "processed": run.processed, "total": run.total,
            "failed": run.failed, "dry_run": run.dry_run}


class RebuildBody(BaseModel):
    workspace_id: Optional[int] = None


@router.post("/vector-backfill/rebuild")
def vector_backfill_rebuild(body: RebuildBody,
                            principal: AuthPrincipal = Depends(get_current_principal),
                            db: Session = Depends(get_db)):
    if body.workspace_id is not None:
        _admin_gate(db, principal, body.workspace_id)
    return vb.rebuild_vectors(db, workspace_id=body.workspace_id,
                              created_by=principal.user.id)
