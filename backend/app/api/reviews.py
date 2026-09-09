"""Human review queue API — create, list, decide, escalate."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.permission_service import require_workspace_membership, require_permission
from ..services.review_service import (
    create_review_item,
    list_review_items,
    get_review_item,
    decide_review_item,
    escalate_overdue,
    review_summary,
    ReviewDecisionError,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewCreate(BaseModel):
    workspace_id: int
    item_type: str
    title: str
    description: Optional[str] = None
    payload: Optional[dict] = None
    source_type: Optional[str] = None
    source_id: Optional[int] = None
    assignee_id: Optional[int] = None
    priority: str = "NORMAL"


class ReviewDecision(BaseModel):
    decision: str  # APPROVED/REJECTED/EDITED/NEEDS_EVIDENCE/DELEGATED/DEFERRED
    note: Optional[str] = None
    reassign_to: Optional[int] = None


def _item_dict(i) -> dict:
    return {
        "id": i.id,
        "workspace_id": i.workspace_id,
        "item_type": i.item_type,
        "status": i.status,
        "priority": i.priority,
        "title": i.title,
        "description": i.description,
        "source_type": i.source_type,
        "source_id": i.source_id,
        "assignee_id": i.assignee_id,
        "due_at": i.due_at,
        "escalated": i.escalated,
        "decided_by": i.decided_by,
        "decided_at": i.decided_at,
        "created_at": i.created_at,
    }


@router.post("", status_code=201)
def create_review(
    data: ReviewCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    workspace = require_workspace_membership(db, data.workspace_id, principal.user.id)
    try:
        item = create_review_item(
            db, data.workspace_id, data.item_type, data.title, principal.user.id,
            description=data.description, payload=data.payload,
            source_type=data.source_type, source_id=data.source_id,
            assignee_id=data.assignee_id, priority=data.priority,
            organization_id=workspace.organization_id,
        )
    except ReviewDecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return _item_dict(item)


@router.get("")
def list_reviews(
    workspace_id: int,
    status: Optional[str] = None,
    assignee_id: Optional[int] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    items = list_review_items(
        db, workspace_id, status=status, assignee_id=assignee_id, limit=min(limit, 200)
    )
    return {"items": [_item_dict(i) for i in items]}


@router.get("/summary")
def reviews_summary(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return review_summary(db, workspace_id)


@router.post("/{item_id}/decide")
def decide_review(
    item_id: int,
    data: ReviewDecision,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase15 import ReviewItem
    item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    require_workspace_membership(db, item.workspace_id, principal.user.id)
    try:
        item = decide_review_item(
            db, item.workspace_id, item_id, data.decision, principal.user.id,
            note=data.note, reassign_to=data.reassign_to,
        )
    except ReviewDecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return _item_dict(item)


@router.post("/maintenance/escalate")
def escalate_reviews(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Worker hook: escalate overdue review items once (no spam)."""
    require_permission(db, principal.user.id, "ai:manage_settings")
    escalated = escalate_overdue(db)
    db.commit()
    return {"escalated": len(escalated)}