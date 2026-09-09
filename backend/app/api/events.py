"""Knowledge event outbox API — emit, process, summarize, retry dead."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.phase15 import KnowledgeEvent
from ..services.permission_service import require_permission, require_workspace_membership
from ..services.event_bus import (
    emit_event,
    process_pending_events,
    retry_dead_events,
    event_summary,
    UnknownEventTypeError,
)

router = APIRouter(prefix="/events", tags=["events"])


class EventEmit(BaseModel):
    workspace_id: int
    event_type: str
    aggregate_type: str = "unknown"
    aggregate_id: int = 0
    dedupe_key: str = ""
    payload: Optional[dict] = None


@router.post("", status_code=201)
def emit_event_endpoint(
    data: EventEmit,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, data.workspace_id, principal.user.id)
    from ..models.workspace import Workspace
    ws = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    try:
        event = emit_event(
            db,
            workspace_id=data.workspace_id,
            event_type=data.event_type,
            aggregate_type=data.aggregate_type,
            aggregate_id=data.aggregate_id,
            payload=data.payload,
            organization_id=ws.organization_id if ws else None,
            dedupe_key=data.dedupe_key,
        )
    except UnknownEventTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return {
        "id": event.id,
        "event_type": event.event_type,
        "status": event.status,
        "deduplicated": event.created_at != event.updated_at if hasattr(event, "updated_at") else False,
        "created_at": event.created_at,
    }


@router.get("")
def list_events(
    workspace_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    query = db.query(KnowledgeEvent).filter(KnowledgeEvent.workspace_id == workspace_id)
    if status:
        query = query.filter(KnowledgeEvent.status == status)
    items = query.order_by(KnowledgeEvent.created_at.desc()).limit(min(limit, 200)).all()
    return {
        "items": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "aggregate_type": e.aggregate_type,
                "aggregate_id": e.aggregate_id,
                "status": e.status,
                "retry_count": e.retry_count,
                "next_retry_at": e.next_retry_at,
                "processed_at": e.processed_at,
                "created_at": e.created_at,
            }
            for e in items
        ]
    }


@router.get("/summary")
def outbox_summary(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return event_summary(db, workspace_id=workspace_id)


@router.post("/process")
def process_events(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Worker hook: process a batch of due events (bounded)."""
    require_permission(db, principal.user.id, "ai:manage_settings")
    summary = process_pending_events(db)
    db.commit()
    return summary


@router.post("/retry-dead")
def retry_dead(
    limit: int = 50,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "ai:manage_settings")
    count = retry_dead_events(db, limit=limit)
    db.commit()
    return {"retried": count}