"""Human review queue 2.0 — centralized review with SLA, escalation, reminders.

Review items (AI actions, conflicts, extraction uncertainties, policy
conflicts, recommendations, workflow approvals, AI answers) flow through
approve / reject / edit / request-more-evidence / delegate / defer.
Every decision is audited. Overdue items escalate once (no spam).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import ReviewItem, REVIEW_STATUSES
from ..services.audit_service import log_audit_event
from ..models.notification import Notification

VALID_DECISIONS = ("APPROVED", "REJECTED", "EDITED", "NEEDS_EVIDENCE", "DELEGATED", "DEFERRED")

PRIORITY_SLA_HOURS = {"CRITICAL": 4, "HIGH": 12, "NORMAL": 48, "LOW": 120}


class ReviewDecisionError(Exception):
    """Raised on invalid review decisions or transitions."""


def create_review_item(
    db: Session,
    workspace_id: int,
    item_type: str,
    title: str,
    user_id: int,
    description: Optional[str] = None,
    payload: Optional[dict] = None,
    source_type: Optional[str] = None,
    source_id: Optional[int] = None,
    assignee_id: Optional[int] = None,
    priority: str = "NORMAL",
    organization_id: Optional[int] = None,
    due_at: Optional[datetime] = None,
) -> ReviewItem:
    if priority not in ("CRITICAL", "HIGH", "NORMAL", "LOW"):
        raise ReviewDecisionError(f"Invalid priority: {priority}")
    item = ReviewItem(
        workspace_id=workspace_id,
        organization_id=organization_id,
        item_type=item_type,
        status="PENDING",
        priority=priority,
        title=title,
        description=description,
        payload_json=__import__("json").dumps(payload, default=str) if payload else None,
        source_type=source_type,
        source_id=source_id,
        assignee_id=assignee_id,
        due_at=due_at or (datetime.now(timezone.utc) + timedelta(hours=PRIORITY_SLA_HOURS.get(priority, 48))),
    )
    db.add(item)
    db.flush()
    if assignee_id:
        db.add(Notification(
            user_id=assignee_id,
            title="New review item assigned",
            message=f"{title} — due {item.due_at.date().isoformat()}",
            notification_type="review",
            resource_type="review_item",
            resource_id=item.id,
        ))
        db.flush()
    return item


def list_review_items(
    db: Session,
    workspace_id: int,
    status: Optional[str] = None,
    assignee_id: Optional[int] = None,
    limit: int = 100,
) -> list[ReviewItem]:
    query = db.query(ReviewItem).filter(ReviewItem.workspace_id == workspace_id)
    if status:
        query = query.filter(ReviewItem.status == status)
    if assignee_id is not None:
        query = query.filter(ReviewItem.assignee_id == assignee_id)
    return query.order_by(
        ReviewItem.priority == "CRITICAL", ReviewItem.created_at.asc()
    ).limit(limit).all()


def get_review_item(db: Session, workspace_id: int, item_id: int) -> Optional[ReviewItem]:
    return (
        db.query(ReviewItem)
        .filter(ReviewItem.workspace_id == workspace_id, ReviewItem.id == item_id)
        .first()
    )


def decide_review_item(
    db: Session,
    workspace_id: int,
    item_id: int,
    decision: str,
    actor_id: int,
    note: Optional[str] = None,
    reassign_to: Optional[int] = None,
) -> ReviewItem:
    """Apply a review decision (audited)."""
    item = get_review_item(db, workspace_id, item_id)
    if not item:
        raise ReviewDecisionError("Review item not found")
    if item.status != "PENDING":
        raise ReviewDecisionError(f"Cannot decide item in status {item.status}")
    if decision not in VALID_DECISIONS:
        raise ReviewDecisionError(f"Invalid decision: {decision}")

    if decision == "DELEGATED":
        if reassign_to is None:
            raise ReviewDecisionError("Delegation requires an assignee")
        item.assignee_id = reassign_to
        item.status = "PENDING"
        item.decision_note = note
        db.add(Notification(
            user_id=reassign_to,
            title="Review item delegated to you",
            message=item.title,
            notification_type="review",
            resource_type="review_item",
            resource_id=item.id,
        ))
        db.flush()
    else:
        item.status = decision
        item.decision_note = note
        item.decided_by = actor_id
        item.decided_at = datetime.now(timezone.utc)

    db.flush()
    log_audit_event(
        db, event_type="review", event_action="decision",
        user_id=actor_id, resource_type="review_item", resource_id=item.id,
        details=f"Review '{item.title}' decided: {decision}",
    )
    return item


def escalate_overdue(db: Session, now: Optional[datetime] = None) -> list[ReviewItem]:
    """Escalate overdue PENDING items once; notify assignee (no spam)."""
    now = now or datetime.now(timezone.utc)
    overdue = (
        db.query(ReviewItem)
        .filter(
            ReviewItem.status == "PENDING",
            ReviewItem.due_at.isnot(None),
            ReviewItem.due_at < now,
            ReviewItem.escalated.is_(False),
        )
        .limit(100)
        .all()
    )
    for item in overdue:
        item.escalated = True
        if item.assignee_id:
            db.add(Notification(
                user_id=item.assignee_id,
                title="Review item overdue — escalated",
                message=f"{item.title} was due {item.due_at.date().isoformat()}",
                notification_type="review",
                resource_type="review_item",
                resource_id=item.id,
            ))
            db.flush()
    if overdue:
        db.flush()
    return overdue


def review_summary(db: Session, workspace_id: int) -> dict:
    from sqlalchemy import func
    rows = (
        db.query(ReviewItem.status, func.count(ReviewItem.id))
        .filter(ReviewItem.workspace_id == workspace_id)
        .group_by(ReviewItem.status)
        .all()
    )
    counts = {status: count for status, count in rows}
    pending = counts.get("PENDING", 0)
    overdue = (
        db.query(ReviewItem)
        .filter(
            ReviewItem.workspace_id == workspace_id,
            ReviewItem.status == "PENDING",
            ReviewItem.due_at.isnot(None),
            ReviewItem.due_at < datetime.now(timezone.utc),
        )
        .count()
    )
    return {
        "pending": pending,
        "overdue": overdue,
        "by_status": counts,
    }