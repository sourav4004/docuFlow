"""Event handler wiring — idempotent handlers for knowledge events.

Handlers run inside the outbox processor; they must be safe to re-run
(idempotent by design). Deduplication keys on the integer outbox event id
(the Notification/AISuggestion resource_id columns are integers — string
markers never match and would defeat dedup). Registration is idempotent:
calling register_default_handlers() repeatedly never duplicates a handler.
"""

import json

from ..models.notification import Notification
from ..models.ai_action import AISuggestion
from ..models.phase15 import ReviewItem, KnowledgeEvent
from .event_bus import register_handler, _handlers


def _payload(event: KnowledgeEvent) -> dict:
    try:
        return json.loads(event.payload_json) if event.payload_json else {}
    except (ValueError, TypeError):
        return {}


def _note_already_sent(db, notification_type: str, event_id: int) -> bool:
    return (
        db.query(Notification)
        .filter(
            Notification.notification_type == notification_type,
            Notification.resource_id == event_id,
        )
        .first()
    ) is not None


def _on_document_ready(db, event: KnowledgeEvent) -> None:
    """DOCUMENT_READY → workspace digest notification (one per event)."""
    if _note_already_sent(db, "document_ready", event.id):
        return
    payload = _payload(event)
    user_id = payload.get("user_id")
    if not user_id:
        return
    db.add(Notification(
        user_id=user_id,
        title="Document ready",
        message=f"Document {payload.get('filename', '')} finished processing.",
        notification_type="document_ready",
        resource_type="document",
        resource_id=event.id,  # integer key = re-run safe, no duplicates
    ))


def _on_deadline_approaching(db, event: KnowledgeEvent) -> None:
    """DEADLINE_APPROACHING → reminder notification (one per event)."""
    if _note_already_sent(db, "deadline_reminder", event.id):
        return
    payload = _payload(event)
    user_id = payload.get("user_id")
    if not user_id:
        return
    db.add(Notification(
        user_id=user_id,
        title="Deadline approaching",
        message=f"{payload.get('title', 'Deadline')} is due {payload.get('due_date', 'soon')}.",
        notification_type="deadline_reminder",
        resource_type="deadline",
        resource_id=event.id,
    ))


def _on_policy_conflict_detected(db, event: KnowledgeEvent) -> None:
    """POLICY_CONFLICT_DETECTED → human review item (never auto-resolved)."""
    payload = _payload(event)
    already = (
        db.query(ReviewItem)
        .filter(ReviewItem.source_type == "policy_conflict", ReviewItem.source_id == event.id)
        .first()
    )
    if already:
        return
    db.add(ReviewItem(
        workspace_id=event.workspace_id,
        organization_id=event.organization_id,
        item_type="POLICY_CONFLICT",
        status="PENDING",
        priority="HIGH",
        title="Policy conflict detected",
        description=payload.get("description", "Two policy statements appear to conflict."),
        source_type="policy_conflict",
        source_id=event.id,
    ))


def _on_ai_action_completed(db, event: KnowledgeEvent) -> None:
    """AI_ACTION_COMPLETED → close the linked review item if one is open."""
    payload = _payload(event)
    review_id = payload.get("review_item_id")
    if review_id:
        item = db.query(ReviewItem).filter(ReviewItem.id == review_id).first()
        if item and item.status == "PENDING":
            item.status = "APPROVED"
            item.decision_note = "Action completed — review auto-closed"


def _on_knowledge_gap_detected(db, event: KnowledgeEvent) -> None:
    """KNOWLEDGE_GAP_DETECTED → deterministic explainable suggestion."""
    payload = _payload(event)
    already = (
        db.query(AISuggestion)
        .filter(AISuggestion.source_type == "knowledge_gap", AISuggestion.source_id == event.id)
        .first()
    )
    if already:
        return
    user_id = payload.get("user_id")
    if not user_id:
        return
    db.add(AISuggestion(
        workspace_id=event.workspace_id,
        organization_id=event.organization_id,
        owner_id=user_id,
        title=payload.get("title", "Knowledge gap detected"),
        description=payload.get("detail"),
        reason="Detected by knowledge gap engine — review and decide",
        suggestion_type="knowledge_gap",
        source_type="knowledge_gap",
        source_id=event.id,
        priority="NORMAL",
    ))


def register_default_handlers() -> None:
    """Idempotent registration — repeated calls never duplicate a handler."""
    handlers = {
        "DOCUMENT_READY": _on_document_ready,
        "DEADLINE_APPROACHING": _on_deadline_approaching,
        "POLICY_CONFLICT_DETECTED": _on_policy_conflict_detected,
        "AI_ACTION_COMPLETED": _on_ai_action_completed,
        "KNOWLEDGE_GAP_DETECTED": _on_knowledge_gap_detected,
    }
    for event_type, handler in handlers.items():
        if handler not in _handlers.get(event_type, []):
            register_handler(event_type, handler)