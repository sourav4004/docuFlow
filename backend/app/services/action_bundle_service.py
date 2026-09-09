"""AI action prioritization + action bundles.

Ranks suggested/approved actions by business impact, risk, urgency,
confidence, effort, and cost — with explainable factors. Related actions can
be grouped into bundles with one overall status and per-child statuses.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_action import AIAction, AISuggestion, ACTION_STATUSES
from ..models.phase15 import ReviewItem
from ..services.audit_service import log_audit_event

IMPACT_RANK = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "NONE": 1}


def score_action(action: AIAction) -> dict:
    """Deterministic explainable priority score (0-100)."""
    impact = IMPACT_RANK.get(action.risk_level, 2)
    urgency = 4 if action.priority == "CRITICAL" else (3 if action.priority == "HIGH" else 2)
    confidence = 4 if action.status in ("APPROVED", "QUEUED", "RUNNING") else 3
    effort = 3 if action.action_type in ("compare", "generate_report", "create_workflow") else 2
    cost = 2 if (action.estimated_cost or 0) > 1.0 else 1

    score = (
        impact * 3
        + urgency * 2
        + confidence * 2
        + (6 - effort)
        + (5 - cost)
    )
    score = max(0, min(100, int(score / 7 * 20)))
    return {
        "score": score,
        "factors": {
            "business_impact": action.risk_level,
            "urgency": action.priority,
            "confidence": confidence,
            "estimated_effort": effort,
            "estimated_cost_usd": action.estimated_cost or 0.0,
        },
        "explanation": (
            f"Ranked {score}/100 — impact {action.risk_level}, urgency "
            f"{action.priority}, cost ${action.estimated_cost or 0:.2f}"
        ),
    }


def rank_actions(db: Session, workspace_id: int, limit: int = 50) -> list[dict]:
    """Rank actionable items (pending approvals first, then suggestions)."""
    pending = (
        db.query(AIAction)
        .filter(
            AIAction.workspace_id == workspace_id,
            AIAction.status.in_(("SUGGESTED", "APPROVAL_REQUIRED", "APPROVED")),
        )
        .limit(limit)
        .all()
    )
    suggestions = (
        db.query(AISuggestion)
        .filter(AISuggestion.workspace_id == workspace_id, AISuggestion.status == "OPEN")
        .limit(limit)
        .all()
    )
    ranked = []
    for action in pending:
        scored = score_action(action)
        ranked.append({
            "kind": "ai_action",
            "id": action.id,
            "title": action.title,
            "status": action.status,
            "risk": action.risk_level,
            **scored,
        })
    for suggestion in suggestions:
        ranked.append({
            "kind": "suggestion",
            "id": suggestion.id,
            "title": suggestion.title,
            "status": suggestion.status,
            "risk": "MEDIUM",
            "score": 50,
            "factors": {"urgency": suggestion.priority, "confidence": 3, "estimated_cost_usd": 0.0},
            "explanation": "System suggestion awaiting review",
        })
    ranked.sort(key=lambda r: (-r["score"], r["id"]))
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Action bundles
# ---------------------------------------------------------------------------

def create_action_bundle(
    db: Session,
    workspace_id: int,
    name: str,
    action_ids: list[int],
    user_id: int,
    organization_id: Optional[int] = None,
) -> dict:
    """Group related actions under one bundle (children keep their statuses)."""
    actions = (
        db.query(AIAction)
        .filter(AIAction.workspace_id == workspace_id, AIAction.id.in_(action_ids))
        .all()
    )
    if len(actions) != len(set(action_ids)):
        raise ValueError("One or more actions not found in this workspace")

    parent = AIAction(
        workspace_id=workspace_id,
        organization_id=organization_id,
        owner_id=user_id,
        action_type="bundle",
        title=name,
        description=f"Bundle of {len(actions)} related actions",
        status="QUEUED",
        risk_level="MEDIUM",
        source_type="bundle",
    )
    db.add(parent)
    db.flush()

    for action in actions:
        action.parent_id = parent.id
        action.dependency_status = "SATISFIED" if action.status == "COMPLETED" else "PENDING"
        action.blocked_reason = (
            None if action.dependency_status == "SATISFIED" else "Waiting for bundle-level approval"
        )
    db.flush()

    log_audit_event(
        db, event_type="ai_action", event_action="bundle_create",
        user_id=user_id, resource_type="ai_action", resource_id=parent.id,
        details=f"Action bundle '{name}' created with {len(actions)} children",
    )
    return {
        "bundle_id": parent.id,
        "name": name,
        "child_count": len(actions),
        "children": [
            {"id": a.id, "status": a.status, "dependency_status": a.dependency_status}
            for a in actions
        ],
    }


def bundle_status(db: Session, workspace_id: int, bundle_id: int) -> dict:
    """Overall bundle status derived from child statuses."""
    bundle = (
        db.query(AIAction)
        .filter(AIAction.workspace_id == workspace_id, AIAction.id == bundle_id)
        .first()
    )
    if not bundle or bundle.action_type != "bundle":
        raise ValueError("Bundle not found")
    children = (
        db.query(AIAction)
        .filter(AIAction.parent_id == bundle_id)
        .all()
    )
    statuses = [c.status for c in children]
    if not statuses:
        overall = "EMPTY"
    elif all(s == "COMPLETED" for s in statuses):
        overall = "COMPLETED"
    elif any(s in ("FAILED", "REJECTED") for s in statuses):
        overall = "BLOCKED"
    elif any(s in ("RUNNING", "QUEUED") for s in statuses):
        overall = "RUNNING"
    else:
        overall = "PENDING"
    bundle.status = overall if overall in ACTION_STATUSES else "QUEUED"
    db.flush()
    return {
        "bundle_id": bundle_id,
        "name": bundle.title,
        "overall_status": overall,
        "children": [{"id": c.id, "title": c.title, "status": c.status} for c in children],
    }


def dependency_satisfied(db: Session, action: AIAction) -> bool:
    """A child action's parent must be COMPLETED before it can run."""
    if action.parent_id is None:
        return True
    parent = db.query(AIAction).filter(AIAction.id == action.parent_id).first()
    if not parent:
        return True
    return parent.status == "COMPLETED"