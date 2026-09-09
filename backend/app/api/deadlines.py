"""Deadline API — CRUD, status, reminders."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.knowledge import Deadline
from ..models.workspace import Workspace
from ..services.deadline_service import (
    create_deadline,
    refresh_status,
    complete_deadline,
    cancel_deadline,
    check_deadline_notifications,
    deadline_actions,
    DeadlineStatusError,
    parse_deadline_from_text,
)
from ..services.permission_service import require_permission

router = APIRouter(prefix="/deadlines", tags=["deadlines"])


class DeadlineCreate(BaseModel):
    workspace_id: int
    title: str
    due_date: datetime
    description: Optional[str] = None
    document_id: Optional[int] = None
    source: str = "manual"
    source_reference: Optional[str] = None
    confidence: str = "UNKNOWN"


def _require_workspace(db, workspace_id, principal):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "document:write", workspace_id=workspace_id)
    return workspace


def _deadline_dict(d: Deadline) -> dict:
    return {
        "id": d.id,
        "workspace_id": d.workspace_id,
        "owner_id": d.owner_id,
        "document_id": d.document_id,
        "title": d.title,
        "description": d.description,
        "due_date": d.due_date,
        "source": d.source,
        "source_reference": d.source_reference,
        "confidence": d.confidence,
        "status": d.status,
        "created_at": d.created_at,
    }


@router.post("", status_code=201)
def create(
    data: DeadlineCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    workspace = _require_workspace(db, data.workspace_id, principal)
    try:
        deadline = create_deadline(
            db,
            data.workspace_id,
            principal.user.id,
            data.title,
            data.due_date,
            description=data.description,
            document_id=data.document_id,
            source=data.source,
            source_reference=data.source_reference,
            confidence=data.confidence,
            organization_id=workspace.organization_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(deadline)
    return _deadline_dict(deadline)


@router.get("")
def list_deadlines(
    workspace_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    query = db.query(Deadline).filter(Deadline.workspace_id == workspace_id)
    if status:
        query = query.filter(Deadline.status == status)
    items = query.order_by(Deadline.due_date.asc()).limit(min(limit, 200)).all()
    result = []
    for d in items:
        refresh_status(db, d)
        result.append(_deadline_dict(d))
    db.commit()
    return {"items": result}


@router.post("/{deadline_id}/complete")
def complete(
    deadline_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    deadline = db.query(Deadline).filter(Deadline.id == deadline_id).first()
    if not deadline:
        raise HTTPException(status_code=404, detail="Deadline not found")
    require_permission(db, principal.user.id, "document:write", workspace_id=deadline.workspace_id)
    try:
        complete_deadline(db, deadline)
    except DeadlineStatusError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return _deadline_dict(deadline)


@router.post("/{deadline_id}/cancel")
def cancel(
    deadline_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    deadline = db.query(Deadline).filter(Deadline.id == deadline_id).first()
    if not deadline:
        raise HTTPException(status_code=404, detail="Deadline not found")
    require_permission(db, principal.user.id, "document:write", workspace_id=deadline.workspace_id)
    try:
        cancel_deadline(db, deadline)
    except DeadlineStatusError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return _deadline_dict(deadline)


@router.post("/check-notifications")
def run_notifications(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Emit deadline reminders for the current user (deduplicated)."""
    _require_workspace(db, workspace_id, principal)
    notifications = check_deadline_notifications(
        db,
        workspace_id,
        user_ids=[principal.user.id],
    )
    db.commit()
    return {"notifications_created": len(notifications)}


@router.post("/check-actions")
def run_deadline_actions(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Create pending AI actions for overdue deadlines (once each)."""
    workspace = _require_workspace(db, workspace_id, principal)
    actions = deadline_actions(db, workspace_id, principal.user.id, workspace.organization_id)
    db.commit()
    return {"actions_created": len(actions)}


@router.post("/parse")
def parse_deadline(
    text: str,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    """Deterministic date extraction from text (bounded patterns)."""
    result = parse_deadline_from_text(text)
    if not result:
        raise HTTPException(status_code=400, detail="No date pattern found in text")
    return result