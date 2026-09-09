"""AI Action Center API — actions, suggestions, transitions."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.ai_action import AIAction, AISuggestion
from ..models.workspace import Workspace
from ..services.ai_action_service import (
    create_action,
    transition,
    approve_action,
    reject_action,
    complete_action,
    list_actions,
    create_suggestion,
    dismiss_suggestion,
    suggestion_to_action,
    InvalidTransitionError,
    UnknownActionTypeError,
    record_insight,
)
from ..services.permission_service import require_permission, require_workspace_membership
from ..services.suggestion_engine import (
    suggest_for_document_ready,
    suggest_deadline_approaching,
    suggest_duplicate_candidates,
)
from ..services.knowledge_gap import record_gap_insights

router = APIRouter(prefix="/ai-actions", tags=["ai-actions"])


class ActionCreate(BaseModel):
    workspace_id: int
    action_type: str
    title: str
    description: Optional[str] = None
    risk_level: Optional[str] = None
    priority: str = "NORMAL"
    payload: Optional[dict] = None
    status: str = "SUGGESTED"


class ActionTransition(BaseModel):
    new_status: str
    rejection_reason: Optional[str] = None


def _action_dict(a: AIAction) -> dict:
    import json
    return {
        "id": a.id,
        "workspace_id": a.workspace_id,
        "owner_id": a.owner_id,
        "action_type": a.action_type,
        "title": a.title,
        "description": a.description,
        "status": a.status,
        "risk_level": a.risk_level,
        "priority": a.priority,
        "source_type": a.source_type,
        "source_id": a.source_id,
        "execution_id": a.execution_id,
        "payload": json.loads(a.payload_json) if a.payload_json else None,
        "result": json.loads(a.result_json) if a.result_json else None,
        "estimated_cost": a.estimated_cost,
        "error_message": a.error_message,
        "created_at": a.created_at,
        "updated_at": a.updated_at,
    }


def _suggestion_dict(s: AISuggestion) -> dict:
    import json
    return {
        "id": s.id,
        "workspace_id": s.workspace_id,
        "owner_id": s.owner_id,
        "title": s.title,
        "description": s.description,
        "reason": s.reason,
        "suggestion_type": s.suggestion_type,
        "source_type": s.source_type,
        "source_id": s.source_id,
        "priority": s.priority,
        "status": s.status,
        "evidence": json.loads(s.evidence_json) if s.evidence_json else None,
        "created_at": s.created_at,
    }


def _require_workspace(db, workspace_id, principal):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=workspace_id)
    return workspace


@router.post("", status_code=201)
def create_action_endpoint(
    data: ActionCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    workspace = _require_workspace(db, data.workspace_id, principal)
    try:
        action = create_action(
            db,
            data.workspace_id,
            principal.user.id,
            data.action_type,
            data.title,
            description=data.description,
            risk_level=data.risk_level,
            priority=data.priority,
            payload=data.payload,
            organization_id=workspace.organization_id,
            status=data.status,
        )
    except (UnknownActionTypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(action)
    return _action_dict(action)


@router.get("")
def list_action_center(
    workspace_id: int,
    status: Optional[str] = None,
    action_type: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    query = db.query(AIAction).filter(AIAction.workspace_id == workspace_id)
    if status:
        query = query.filter(AIAction.status == status)
    if action_type:
        query = query.filter(AIAction.action_type == action_type)
    actions = query.order_by(AIAction.created_at.desc()).limit(min(limit, 200)).all()
    return {"items": [_action_dict(a) for a in actions]}


@router.get("/summary")
def action_center_summary(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Aggregate counts for the AI action center dashboard."""
    _require_workspace(db, workspace_id, principal)
    from sqlalchemy import func
    rows = (
        db.query(AIAction.status, func.count(AIAction.id))
        .filter(AIAction.workspace_id == workspace_id)
        .group_by(AIAction.status)
        .all()
    )
    counts = {status: count for status, count in rows}
    pending = counts.get("APPROVAL_REQUIRED", 0) + counts.get("APPROVED", 0) + counts.get("QUEUED", 0)
    return {
        "status_counts": counts,
        "pending_actions": pending,
        "running": counts.get("RUNNING", 0),
        "completed": counts.get("COMPLETED", 0),
        "failed": counts.get("FAILED", 0),
    }


@router.post("/{action_id}/transition")
def transition_action(
    action_id: int,
    data: ActionTransition,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    action = db.query(AIAction).filter(AIAction.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=action.workspace_id)
    try:
        action = transition(db, action, data.new_status, principal.user.id, rejection_reason=data.rejection_reason)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(action)
    return _action_dict(action)


@router.post("/{action_id}/approve")
def approve_action_endpoint(
    action_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    action = db.query(AIAction).filter(AIAction.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=action.workspace_id)
    try:
        action = approve_action(db, action, principal.user.id)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(action)
    return _action_dict(action)


@router.post("/{action_id}/reject")
def reject_action_endpoint(
    action_id: int,
    reason: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    action = db.query(AIAction).filter(AIAction.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=action.workspace_id)
    try:
        action = reject_action(db, action, principal.user.id, reason=reason)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(action)
    return _action_dict(action)


# --- Suggestions ---


@router.post("/suggestions/generate")
def generate_suggestions(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Generate deterministic suggestions from workspace state."""
    workspace = _require_workspace(db, workspace_id, principal)
    created = []
    created += suggest_deadline_approaching(
        db, workspace_id, principal.user.id, workspace.organization_id
    )
    created += suggest_duplicate_candidates(
        db, workspace_id, principal.user.id, workspace.organization_id
    )
    gap_insights = record_gap_insights(
        db, workspace_id, workspace.organization_id, principal.user.id
    )
    db.commit()
    return {"suggestions_created": len(created) + gap_insights}


@router.get("/suggestions")
def list_suggestions(
    workspace_id: int,
    status: Optional[str] = None,
    suggestion_type: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    query = db.query(AISuggestion).filter(AISuggestion.workspace_id == workspace_id)
    if status:
        query = query.filter(AISuggestion.status == status)
    if suggestion_type:
        query = query.filter(AISuggestion.suggestion_type == suggestion_type)
    items = query.order_by(AISuggestion.created_at.desc()).limit(min(limit, 200)).all()
    return {"items": [_suggestion_dict(s) for s in items]}


@router.post("/suggestions/{suggestion_id}/dismiss")
def dismiss_suggestion_endpoint(
    suggestion_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    suggestion = db.query(AISuggestion).filter(AISuggestion.id == suggestion_id).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=suggestion.workspace_id)
    dismiss_suggestion(db, suggestion)
    db.commit()
    return {"message": "Suggestion dismissed"}


@router.post("/suggestions/{suggestion_id}/action")
def action_on_suggestion(
    suggestion_id: int,
    action_type: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Convert a suggestion into an AI action (requires human approval)."""
    suggestion = db.query(AISuggestion).filter(AISuggestion.id == suggestion_id).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    require_permission(db, principal.user.id, "ai:execute", workspace_id=suggestion.workspace_id)
    try:
        action = suggestion_to_action(db, suggestion, action_type, principal.user.id)
    except (UnknownActionTypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(action)
    return _action_dict(action)


@router.post("/suggestions/document/{document_id}")
def suggest_for_document(
    document_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Generate suggestions for a specific document (e.g., after READY)."""
    from ..models.document import Document
    from ..models.workspace import Workspace as _WS
    from ..services.workspace_service import resolve_document_workspace
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    require_workspace_membership(db, workspace_id, principal.user.id)
    workspace = db.query(_WS).filter(_WS.id == workspace_id).first()
    organization_id = workspace.organization_id if workspace else None
    created = suggest_for_document_ready(
        db, doc, workspace_id, principal.user.id, organization_id
    )
    db.commit()
    return {"suggestions_created": len(created)}


@router.get("/insights")
def list_insights(
    workspace_id: int,
    insight_type: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.workspace import Workspace as _WS
    from ..models.knowledge import KnowledgeInsight
    _require_workspace(db, workspace_id, principal)
    query = db.query(KnowledgeInsight).filter(KnowledgeInsight.workspace_id == workspace_id)
    if insight_type:
        query = query.filter(KnowledgeInsight.insight_type == insight_type)
    items = query.order_by(KnowledgeInsight.created_at.desc()).limit(min(limit, 200)).all()
    return {
        "items": [
            {
                "id": i.id,
                "insight_type": i.insight_type,
                "title": i.title,
                "detail": i.detail,
                "importance": i.importance,
                "created_at": i.created_at,
            }
            for i in items
        ]
    }