"""Deadline service — lifecycle, notifications, and extraction support."""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.knowledge import Deadline
from ..models.notification import Notification
from ..models.ai_action import AIAction
from ..services.ai_action_service import create_action

DEADLINE_STATUSES = ("UPCOMING", "DUE", "OVERDUE", "COMPLETED", "CANCELLED")
DEADLINE_SOURCES = ("document", "manual", "workflow")
NOTIFICATION_WINDOWS_DAYS = (30, 14, 7, 1, 0)


class DeadlineStatusError(Exception):
    """Raised on invalid deadline status transitions."""


def create_deadline(
    db: Session,
    workspace_id: int,
    owner_id: int,
    title: str,
    due_date: datetime,
    description: Optional[str] = None,
    document_id: Optional[int] = None,
    source: str = "manual",
    source_reference: Optional[str] = None,
    confidence: str = "UNKNOWN",
    organization_id: Optional[int] = None,
) -> Deadline:
    """Create a deadline."""
    if source not in DEADLINE_SOURCES:
        raise ValueError(f"Unknown deadline source: {source}")
    if confidence not in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
        raise ValueError(f"Invalid confidence: {confidence}")
    due = due_date
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)

    deadline = Deadline(
        workspace_id=workspace_id,
        organization_id=organization_id,
        owner_id=owner_id,
        document_id=document_id,
        title=title,
        description=description,
        due_date=due,
        source=source,
        source_reference=source_reference,
        confidence=confidence,
    )
    db.add(deadline)
    db.flush()
    return deadline


def _as_utc(value: datetime) -> datetime:
    """Normalize naive datetimes (e.g. SQLite storage) to aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def refresh_status(db: Session, deadline: Deadline) -> Deadline:
    """Recompute UPCOMING/DUE/OVERDUE based on the current time."""
    if deadline.status in ("COMPLETED", "CANCELLED"):
        return deadline
    now = datetime.now(timezone.utc)
    due = _as_utc(deadline.due_date)
    if now > due:
        deadline.status = "OVERDUE"
    elif now.date() == due.date():
        deadline.status = "DUE"
    else:
        deadline.status = "UPCOMING"
    db.flush()
    return deadline


def complete_deadline(db: Session, deadline: Deadline) -> Deadline:
    if deadline.status in ("COMPLETED", "CANCELLED"):
        raise DeadlineStatusError(f"Cannot complete deadline in status {deadline.status}")
    deadline.status = "COMPLETED"
    db.flush()
    return deadline


def cancel_deadline(db: Session, deadline: Deadline) -> Deadline:
    if deadline.status in ("COMPLETED", "CANCELLED"):
        raise DeadlineStatusError(f"Cannot cancel deadline in status {deadline.status}")
    deadline.status = "CANCELLED"
    db.flush()
    return deadline


def check_deadline_notifications(
    db: Session,
    workspace_id: int,
    organization_id: Optional[int] = None,
    user_ids: Optional[list[int]] = None,
) -> list[Notification]:
    """Emit one notification per deadline per window (deduplicated).

    Windows: 30, 14, 7, 1 days before, and on the due date. A deadline is
    only notified once per window (tracked via notification resource_id).
    """
    now = datetime.now(timezone.utc)
    deadlines = (
        db.query(Deadline)
        .filter(
            Deadline.workspace_id == workspace_id,
            Deadline.status.in_(("UPCOMING", "DUE")),
        )
        .all()
    )
    notifications = []
    recipients = user_ids or []

    for deadline in deadlines:
        refresh_status(db, deadline)
        if deadline.status not in ("UPCOMING", "DUE"):
            continue
        due = _as_utc(deadline.due_date)
        days_left = (due - now).days

        for window in NOTIFICATION_WINDOWS_DAYS:
            if days_left > window:
                continue
            marker = f"{deadline.id}:{window}"
            already = (
                db.query(Notification)
                .filter(
                    Notification.notification_type == "deadline",
                    Notification.resource_id == marker,
                )
                .first()
            )
            if already:
                continue
            label = "today" if window == 0 else f"in {window} day(s)"
            for user_id in recipients:
                n = Notification(
                    user_id=user_id,
                    title=f"Deadline: {deadline.title}",
                    message=f"'{deadline.title}' is due {label}.",
                    notification_type="deadline",
                    resource_type="deadline",
                    resource_id=marker,
                )
                db.add(n)
                notifications.append(n)
    if notifications:
        db.flush()
    return notifications


def deadline_actions(db: Session, workspace_id: int, owner_id: int, organization_id: Optional[int] = None) -> list[AIAction]:
    """Create pending AI actions for overdue deadlines (once each)."""
    # Recompute statuses first so deadlines past their due date are flagged
    # even when a worker has not yet refreshed them.
    candidates = (
        db.query(Deadline)
        .filter(
            Deadline.workspace_id == workspace_id,
            Deadline.status.notin_(("COMPLETED", "CANCELLED")),
        )
        .all()
    )
    overdue = []
    for deadline in candidates:
        refresh_status(db, deadline)
        if deadline.status == "OVERDUE":
            overdue.append(deadline)
    created = []
    for deadline in overdue:
        existing = (
            db.query(AIAction)
            .filter(
                AIAction.workspace_id == workspace_id,
                AIAction.source_type == "deadline",
                AIAction.source_id == deadline.id,
                AIAction.action_type == "deadline_alert",
            )
            .first()
        )
        if existing:
            continue
        created.append(create_action(
            db,
            workspace_id,
            owner_id,
            "deadline_alert",
            f"Deadline overdue: {deadline.title}",
            description=f"'{deadline.title}' was due {_as_utc(deadline.due_date).date().isoformat()}.",
            source_type="deadline",
            source_id=deadline.id,
            organization_id=organization_id,
        ))
    return created


def parse_deadline_from_text(text: str) -> Optional[dict]:
    """Deterministic date extraction from text (bounded patterns only)."""
    import re
    patterns = [
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d"),
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return {"date": datetime.strptime(match.group(0), fmt).replace(tzinfo=timezone.utc),
                        "reference": match.group(0)}
            except ValueError:
                continue
    return None