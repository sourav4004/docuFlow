"""AI report builder endpoints — templates, generation, artifact listing."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.report_service import (
    list_templates,
    generate_report,
    UnknownTemplateError,
)


router = APIRouter(prefix="/reports", tags=["ai-reports"])


class ReportRequest(BaseModel):
    template: str
    organization_id: Optional[int] = None
    document_ids: Optional[list[int]] = None


@router.get("/templates")
def templates(user: User = Depends(get_current_user)):
    """List available report templates."""
    return {"templates": list_templates()}


@router.post("/generate")
def create_report(
    payload: ReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Generate a report from a template. Saved as a versioned AI artifact."""
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        report = generate_report(
            db,
            workspace_id=workspace.id,
            user_id=user.id,
            template=payload.template,
            organization_id=payload.organization_id,
            document_ids=payload.document_ids,
        )
    except UnknownTemplateError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return report


@router.get("/artifacts")
def report_artifacts(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List AI-generated reports (versioned artifacts)."""
    from ..models.ai_execution import AIArtifact
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    artifacts = (
        db.query(AIArtifact)
        .filter(
            AIArtifact.workspace_id == workspace.id,
            AIArtifact.artifact_type == "report",
        )
        .order_by(AIArtifact.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "artifacts": [
            {
                "id": a.id,
                "name": a.name,
                "version": a.version,
                "status": a.status,
                "created_at": a.created_at,
            }
            for a in artifacts
        ]
    }