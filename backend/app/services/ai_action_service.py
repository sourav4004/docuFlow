"""AI Action Center service — lifecycle management with explicit transitions."""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_action import AIAction, AISuggestion, ACTION_STATUSES, ACTION_TYPES, RISK_LEVELS
from ..models.knowledge import KnowledgeInsight
from ..services.audit_service import log_audit_event

# Explicit valid transitions — invalid transitions are rejected
VALID_TRANSITIONS = {
    "SUGGESTED": {"APPROVAL_REQUIRED", "APPROVED", "QUEUED", "CANCELLED", "REJECTED", "RUNNING"},
    "APPROVAL_REQUIRED": {"APPROVED", "REJECTED", "CANCELLED"},
    "APPROVED": {"QUEUED", "RUNNING", "CANCELLED"},
    "QUEUED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": {"QUEUED", "CANCELLED"},
    "CANCELLED": set(),
    "REJECTED": set(),
}

TERMINAL_STATUSES = ("COMPLETED", "FAILED", "CANCELLED", "REJECTED")

# Default risk per action type
ACTION_RISK = {
    "summarize": "LOW",
    "extract": "LOW",
    "classify": "LOW",
    "compare": "LOW",
    "tag": "MEDIUM",
    "generate_report": "LOW",
    "create_workflow": "MEDIUM",
    "send_notification": "MEDIUM",
    "request_review": "LOW",
    "detect_conflict": "LOW",
    "detect_missing_info": "LOW",
    "deadline_alert": "LOW",
    "duplicate_review": "LOW",
    "health_review": "LOW",
}


class InvalidTransitionError(Exception):
    """Raised when an action cannot transition to the requested state."""


class UnknownActionTypeError(Exception):
    """Raised for unknown action types."""


def create_action(
    db: Session,
    workspace_id: int,
    owner_id: int,
    action_type: str,
    title: str,
    description: Optional[str] = None,
    risk_level: Optional[str] = None,
    priority: str = "NORMAL",
    source_type: str = "manual",
    source_id: Optional[int] = None,
    payload: Optional[dict] = None,
    organization_id: Optional[int] = None,
    status: str = "SUGGESTED",
) -> AIAction:
    """Create a new AI action."""
    if action_type not in ACTION_TYPES:
        raise UnknownActionTypeError(f"Unknown action type: {action_type}")
    if status not in ACTION_STATUSES:
        raise ValueError(f"Invalid action status: {status}")
    risk = risk_level or ACTION_RISK.get(action_type, "LOW")
    if risk not in RISK_LEVELS:
        raise ValueError(f"Invalid risk level: {risk}")

    action = AIAction(
        workspace_id=workspace_id,
        organization_id=organization_id,
        owner_id=owner_id,
        action_type=action_type,
        title=title,
        description=description,
        status=status,
        risk_level=risk,
        priority=priority,
        source_type=source_type,
        source_id=source_id,
        payload_json=json.dumps(payload) if payload else None,
    )
    db.add(action)
    db.flush()

    log_audit_event(
        db,
        event_type="ai_action",
        event_action="create",
        user_id=owner_id,
        resource_type="ai_action",
        resource_id=action.id,
        details=f"AI action '{action_type}' created (status={status})",
    )
    return action


def transition(
    db: Session,
    action: AIAction,
    new_status: str,
    actor_id: int,
    rejection_reason: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> AIAction:
    """Transition an action to a new status, validating the transition."""
    if new_status not in ACTION_STATUSES:
        raise InvalidTransitionError(f"Unknown status: {new_status}")
    if action.status in TERMINAL_STATUSES:
        raise InvalidTransitionError(f"Cannot transition terminal status {action.status}")
    if new_status not in VALID_TRANSITIONS.get(action.status, set()):
        raise InvalidTransitionError(
            f"Invalid transition: {action.status} -> {new_status}"
        )

    action.status = new_status
    if new_status == "REJECTED":
        action.rejection_reason = rejection_reason
    if new_status == "APPROVED":
        action.approved_by = actor_id
        action.approved_at = datetime.now(timezone.utc)
    if execution_id:
        action.execution_id = execution_id
    action.updated_at = datetime.now(timezone.utc)
    db.flush()

    log_audit_event(
        db,
        event_type="ai_action",
        event_action=f"transition:{new_status.lower()}",
        user_id=actor_id,
        resource_type="ai_action",
        resource_id=action.id,
        details=f"AI action {action.id} -> {new_status}",
    )
    return action


def approve_action(db: Session, action: AIAction, actor_id: int) -> AIAction:
    """Approve an action (from APPROVAL_REQUIRED or SUGGESTED)."""
    if action.status not in ("APPROVAL_REQUIRED", "SUGGESTED"):
        raise InvalidTransitionError(f"Cannot approve action in status {action.status}")
    return transition(db, action, "APPROVED", actor_id)


def reject_action(db: Session, action: AIAction, actor_id: int, reason: Optional[str] = None) -> AIAction:
    if action.status not in ("APPROVAL_REQUIRED", "SUGGESTED"):
        raise InvalidTransitionError(f"Cannot reject action in status {action.status}")
    return transition(db, action, "REJECTED", actor_id, rejection_reason=reason)


def complete_action(
    db: Session,
    action: AIAction,
    result: Optional[dict] = None,
    execution_id: Optional[str] = None,
) -> AIAction:
    if action.status != "RUNNING":
        raise InvalidTransitionError(f"Cannot complete action in status {action.status}")
    action.result_json = json.dumps(result) if result else None
    return transition(db, action, "COMPLETED", action.owner_id, execution_id=execution_id)


def list_actions(
    db: Session,
    workspace_id: int,
    status: Optional[str] = None,
    owner_id: Optional[int] = None,
    limit: int = 100,
) -> list[AIAction]:
    query = db.query(AIAction).filter(AIAction.workspace_id == workspace_id)
    if status:
        query = query.filter(AIAction.status == status)
    if owner_id is not None:
        query = query.filter(AIAction.owner_id == owner_id)
    return query.order_by(AIAction.created_at.desc()).limit(limit).all()


# ---------------------------------------------------------------
# Suggestion engine
# ---------------------------------------------------------------

def create_suggestion(
    db: Session,
    workspace_id: int,
    owner_id: int,
    title: str,
    suggestion_type: str,
    description: Optional[str] = None,
    reason: Optional[str] = None,
    source_type: str = "system",
    source_id: Optional[int] = None,
    priority: str = "NORMAL",
    evidence: Optional[dict] = None,
    organization_id: Optional[int] = None,
) -> AISuggestion:
    """Create a deterministic, explainable suggestion."""
    suggestion = AISuggestion(
        workspace_id=workspace_id,
        organization_id=organization_id,
        owner_id=owner_id,
        title=title,
        description=description,
        reason=reason,
        suggestion_type=suggestion_type,
        source_type=source_type,
        source_id=source_id,
        priority=priority,
        evidence_json=json.dumps(evidence) if evidence else None,
    )
    db.add(suggestion)
    db.flush()
    return suggestion


def dismiss_suggestion(db: Session, suggestion: AISuggestion) -> AISuggestion:
    suggestion.status = "DISMISSED"
    db.flush()
    return suggestion


def suggestion_to_action(db: Session, suggestion: AISuggestion, action_type: str, actor_id: int) -> AIAction:
    """Convert a suggestion into an actionable AI action (never autonomous)."""
    action = create_action(
        db,
        suggestion.workspace_id,
        actor_id,
        action_type,
        suggestion.title,
        description=suggestion.description,
        source_type="suggestion",
        source_id=suggestion.id,
        payload={"suggestion_id": suggestion.id},
        organization_id=suggestion.organization_id,
        status="APPROVAL_REQUIRED",  # converting a suggestion always needs a human
    )
    suggestion.status = "ACTIONED"
    db.flush()
    return action


def record_insight(
    db: Session,
    workspace_id: int,
    insight_type: str,
    title: str,
    detail: Optional[str] = None,
    importance: str = "INFO",
    owner_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    evidence: Optional[dict] = None,
) -> KnowledgeInsight:
    """Record a workspace knowledge insight."""
    insight = KnowledgeInsight(
        workspace_id=workspace_id,
        organization_id=organization_id,
        owner_id=owner_id,
        insight_type=insight_type,
        title=title,
        detail=detail,
        importance=importance,
        evidence_json=json.dumps(evidence) if evidence else None,
    )
    db.add(insight)
    db.flush()
    return insight