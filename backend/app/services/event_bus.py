"""Knowledge event outbox + reliable processor.

Domain events are persisted transactionally with their business record (the
caller commits both in one transaction) and processed asynchronously by
``process_pending_events`` — a polling worker abstraction with batch
processing, exponential backoff, dead-letter state, and idempotent handlers.
No external broker is required.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.phase15 import KnowledgeEvent, EVENT_TYPES, EVENT_STATUSES
from ..services.audit_service import log_audit_event

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BATCH_SIZE = 50
BACKOFF_BASE_SECONDS = 2  # 2^retry exponential backoff


class UnknownEventTypeError(Exception):
    """Raised when emitting an event type outside the catalog."""


class EventHandlerError(Exception):
    """Raised by handlers that must be retried (transient failures)."""


class DeadLetterError(Exception):
    """Raised by handlers for permanent failures (dead-letter immediately)."""


def emit_event(
    db: Session,
    workspace_id: int,
    event_type: str,
    aggregate_type: str = "unknown",
    aggregate_id: int = 0,
    payload: Optional[dict] = None,
    organization_id: Optional[int] = None,
    dedupe_key: str = "",
) -> KnowledgeEvent:
    """Persist an outbox event (deduplicated per tenant/type/aggregate/key).

    The business record and the event are committed by the caller in the same
    transaction, so a committed business record always has its event.
    """
    if event_type not in EVENT_TYPES:
        raise UnknownEventTypeError(f"Unknown event type: {event_type}")

    existing = (
        db.query(KnowledgeEvent)
        .filter(
            KnowledgeEvent.workspace_id == workspace_id,
            KnowledgeEvent.event_type == event_type,
            KnowledgeEvent.aggregate_type == aggregate_type,
            KnowledgeEvent.aggregate_id == aggregate_id,
            KnowledgeEvent.dedupe_key == dedupe_key,
        )
        .first()
    )
    if existing:
        return existing  # idempotent — never emit the same event twice

    event = KnowledgeEvent(
        workspace_id=workspace_id,
        organization_id=organization_id,
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        dedupe_key=dedupe_key,
        payload_json=json.dumps(payload, default=str) if payload else None,
        status="PENDING",
    )
    db.add(event)
    db.flush()
    return event


# ---------------------------------------------------------------------------
# Handler registry (idempotent by design — handlers must be safe to re-run)
# ---------------------------------------------------------------------------

_handlers: dict[str, list[Callable]] = {}


def register_handler(event_type: str, handler: Callable) -> None:
    _handlers.setdefault(event_type, []).append(handler)


def process_pending_events(db: Session, batch_size: int = BATCH_SIZE, now=None) -> dict:
    """Process a batch of pending events with retry/backoff/dead-letter.

    Returns a summary dict {processed, failed, dead_lettered, skipped}.
    """
    now = now or datetime.now(timezone.utc)
    summary = {"processed": 0, "failed": 0, "dead_lettered": 0, "skipped": 0}

    due = (
        db.query(KnowledgeEvent)
        .filter(
            KnowledgeEvent.status.in_(("PENDING", "FAILED")),
            (KnowledgeEvent.next_retry_at.is_(None)) | (KnowledgeEvent.next_retry_at <= now),
        )
        .order_by(KnowledgeEvent.created_at.asc())
        .limit(batch_size)
        .all()
    )

    for event in due:
        handlers = _handlers.get(event.event_type, [])
        if not handlers:
            # No handler registered — mark processed (the event is recorded for
            # observability/audit even when no side effect is wired up).
            event.status = "PROCESSED"
            event.processed_at = now
            summary["skipped"] += 1
            continue

        try:
            for handler in handlers:
                handler(db, event)
            event.status = "PROCESSED"
            event.processed_at = now
            summary["processed"] += 1
        except DeadLetterError:
            event.status = "DEAD"
            event.retry_count += 1
            summary["dead_lettered"] += 1
        except Exception as exc:  # noqa: BLE001 — transient failures retried
            event.retry_count += 1
            logger.warning("Event %s handler failed (attempt %d): %s",
                           event.id, event.retry_count, exc)
            if event.retry_count >= MAX_RETRIES:
                event.status = "DEAD"
                summary["dead_lettered"] += 1
            else:
                event.status = "FAILED"
                event.next_retry_at = now + timedelta(
                    seconds=BACKOFF_BASE_SECONDS ** event.retry_count
                )
                summary["failed"] += 1

    if due:
        db.flush()
    return summary


def retry_dead_events(db: Session, limit: int = 50) -> int:
    """Manually re-queue dead-lettered events (admin action)."""
    dead = (
        db.query(KnowledgeEvent)
        .filter(KnowledgeEvent.status == "DEAD")
        .limit(limit)
        .all()
    )
    for event in dead:
        event.status = "FAILED"
        event.retry_count = 0
        event.next_retry_at = datetime.now(timezone.utc)
    if dead:
        db.flush()
    return len(dead)


def event_summary(db: Session, workspace_id: Optional[int] = None) -> dict:
    from sqlalchemy import func
    query = db.query(KnowledgeEvent.status, func.count(KnowledgeEvent.id))
    if workspace_id is not None:
        query = query.filter(KnowledgeEvent.workspace_id == workspace_id)
    rows = query.group_by(KnowledgeEvent.status).all()
    counts = {status: count for status, count in rows}
    return {
        status: counts.get(status, 0)
        for status in EVENT_STATUSES
    }


def audit_event_emission(db: Session, event: KnowledgeEvent, actor_id: Optional[int] = None) -> None:
    log_audit_event(
        db, event_type="knowledge_event", event_action="emitted",
        user_id=actor_id, resource_type="knowledge_event", resource_id=event.id,
        details=f"Event '{event.event_type}' emitted",
    )