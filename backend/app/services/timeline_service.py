"""Knowledge timeline 2.0 — unified timeline across documents, versions,
policies, entities, deadlines, AI actions, workflow events, and approvals.

All entries are tenant scoped and returned newest-first with bounded
pagination.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_version import DocumentVersion
from ..models.ai_action import AIAction
from ..models.knowledge import Deadline
from ..models.phase15 import (
    KnowledgeChange, PolicyConflict, KnowledgeEvent, ReviewItem,
)
from ..models.workflow_version import WorkflowVersion

VALID_EVENT_TYPES = (
    "document", "version", "deadline", "change", "conflict", "ai_action",
    "review", "workflow", "event",
)


def unified_timeline(
    db: Session,
    workspace_id: int,
    event_type: Optional[str] = None,
    document_id: Optional[int] = None,
    entity_id: Optional[int] = None,
    severity: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
) -> list[dict]:
    """Merge tenant-scoped event streams into one ordered timeline."""
    if event_type and event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")

    entries: list[dict] = []

    def add(kind: str, occurred: datetime, severity_value: str, title: str, ref: dict) -> None:
        if event_type and kind != event_type:
            return
        if since and occurred < since:
            return
        if severity and severity_value != severity:
            return
        entries.append({
            "event_type": kind,
            "occurred_at": occurred.isoformat(),
            "severity": severity_value,
            "title": title,
            **ref,
        })

    # Document events
    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
        .limit(500)
        .all()
    )
    for d in docs:
        if document_id and d.id != document_id:
            continue
        add("document", _utc(d.created_at), "INFO", f"Document '{d.original_filename}' {d.status.lower()}", {"document_id": d.id})

    # Version events
    versions = (
        db.query(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .filter(Document.workspace_id == workspace_id)
        .order_by(DocumentVersion.created_at.desc())
        .limit(500)
        .all()
    )
    for v in versions:
        if document_id and v.document_id != document_id:
            continue
        add("version", _utc(v.created_at), "INFO", f"Version {v.version_number} of document {v.document_id}", {"document_id": v.document_id, "version": v.version_number})

    # Deadline events
    deadlines = (
        db.query(Deadline)
        .filter(Deadline.workspace_id == workspace_id)
        .order_by(Deadline.updated_at.desc())
        .limit(500)
        .all()
    )
    for dl in deadlines:
        add("deadline", _utc(dl.updated_at), "MEDIUM" if dl.status == "OVERDUE" else "LOW",
            f"Deadline '{dl.title}' — {dl.status}", {"deadline_id": dl.id, "document_id": dl.document_id})

    # Knowledge changes
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.workspace_id == workspace_id)
        .order_by(KnowledgeChange.created_at.desc())
        .limit(500)
        .all()
    )
    for c in changes:
        if document_id and c.document_id != document_id:
            continue
        add("change", _utc(c.created_at), c.severity, c.summary or c.change_type,
            {"change_id": c.id, "document_id": c.document_id})

    # Policy conflicts
    conflicts = (
        db.query(PolicyConflict)
        .filter(PolicyConflict.workspace_id == workspace_id)
        .order_by(PolicyConflict.created_at.desc())
        .limit(200)
        .all()
    )
    for pc in conflicts:
        add("conflict", _utc(pc.created_at), pc.severity, f"Policy conflict: {pc.conflict_type}",
            {"conflict_id": pc.id})

    # AI actions
    actions = (
        db.query(AIAction)
        .filter(AIAction.workspace_id == workspace_id)
        .order_by(AIAction.created_at.desc())
        .limit(500)
        .all()
    )
    for a in actions:
        add("ai_action", _utc(a.updated_at), a.risk_level, f"AI action '{a.title}' — {a.status}",
            {"action_id": a.id})

    # Review decisions
    reviews = (
        db.query(ReviewItem)
        .filter(ReviewItem.workspace_id == workspace_id)
        .order_by(ReviewItem.created_at.desc())
        .limit(500)
        .all()
    )
    for r in reviews:
        add("review", _utc(r.decided_at or r.created_at), r.priority, f"Review '{r.title}' — {r.status}",
            {"review_id": r.id})

    # Outbox events
    events = (
        db.query(KnowledgeEvent)
        .filter(KnowledgeEvent.workspace_id == workspace_id)
        .order_by(KnowledgeEvent.created_at.desc())
        .limit(500)
        .all()
    )
    for e in events:
        add("event", _utc(e.created_at), "INFO", f"Event {e.event_type}", {"event_id": e.id})

    entries.sort(key=lambda e: e["occurred_at"], reverse=True)
    return entries[:limit]


def _utc(value) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)