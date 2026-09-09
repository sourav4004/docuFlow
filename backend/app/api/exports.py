"""Export job API — queue, list, download with short-lived tokens."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.export_job import ExportJob
from ..models.workspace import Workspace
from ..services.export_service import (
    EXPORT_TYPES,
    create_export_job,
    run_export,
    verify_download_token,
    issue_download_token,
)
from ..services.permission_service import require_permission
from ..services.storage_backend import retrieve_export

router = APIRouter(prefix="/exports", tags=["exports"])


class ExportCreate(BaseModel):
    workspace_id: int
    export_type: str = "all"
    format: str = "zip"


def _job_dict(job: ExportJob) -> dict:
    return {
        "id": job.id,
        "workspace_id": job.workspace_id,
        "export_type": job.export_type,
        "format": job.format,
        "status": job.status,
        "storage_path": job.storage_path,
        "file_size_bytes": job.file_size_bytes,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "download_expires_at": job.download_expires_at,
    }


def _require_job(db: Session, job_id: int, principal: AuthPrincipal) -> ExportJob:
    job = db.query(ExportJob).filter(ExportJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Export job not found")
    require_permission(db, principal.user.id, "export:create", workspace_id=job.workspace_id)
    return job


@router.post("", status_code=202)
def create_export(
    data: ExportCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Queue an asynchronous export job."""
    workspace = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "export:create", workspace_id=data.workspace_id)

    if data.export_type not in EXPORT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown export type: {data.export_type}")
    if data.format not in ("json", "zip"):
        raise HTTPException(status_code=400, detail="Format must be json or zip")

    job = create_export_job(
        db,
        data.workspace_id,
        workspace.organization_id,
        principal.user.id,
        data.export_type,
        data.format,
    )
    db.commit()
    db.refresh(job)

    # Execute immediately for small datasets (production routes to a worker)
    run_export(db, job)
    db.commit()
    db.refresh(job)
    return _job_dict(job)


@router.get("")
def list_exports(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "export:create", workspace_id=workspace_id)
    jobs = (
        db.query(ExportJob)
        .filter(ExportJob.workspace_id == workspace_id)
        .order_by(ExportJob.created_at.desc())
        .limit(100)
        .all()
    )
    return {"items": [_job_dict(j) for j in jobs]}


@router.get("/{job_id}")
def get_export(
    job_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    job = _require_job(db, job_id, principal)
    return _job_dict(job)


@router.post("/{job_id}/download-token")
def request_download_token(
    job_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Issue a short-lived download token for a completed export."""
    job = _require_job(db, job_id, principal)
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Export is not completed")
    token = issue_download_token(job)
    db.commit()
    return {"download_token": token, "expires_in_hours": 24}


@router.get("/{job_id}/download")
def download_export(
    job_id: int,
    token: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Download an export using its short-lived token."""
    job = _require_job(db, job_id, principal)
    if job.status != "COMPLETED" or not job.storage_path:
        raise HTTPException(status_code=400, detail="Export is not completed")
    if not verify_download_token(job, token):
        raise HTTPException(status_code=403, detail="Invalid or expired download token")

    try:
        data = retrieve_export(job.storage_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Export file not found")

    media_type = "application/zip" if job.format == "zip" else "application/json"
    filename = f"docuflow-export-{job.id}.{job.format}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )