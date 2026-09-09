"""Phase 23 — Webhook reliability 3.0, scheduler task dedup, review center.

Webhooks (Steps 50-51) extend the Phase 21/23 notification stack with:
- HMAC-signed payloads with timestamp + replay protection
- exponential backoff retries with delivery IDs and idempotent handling
- dead-letter after bounded attempts, endpoint health tracking with
  automatic disable after repeated failures, and manual re-enable
- no secrets in payloads (signature covers body + timestamp only)

Scheduler dedup (Step 48) uses Phase 23 SchedulerTaskRun so two leaders
(or one restarted leader) cannot execute the same scheduled task twice in
the same due bucket — duplicate side effects become impossible.

Review center (Step 35) unifies human review over the Phase 15 ReviewItem
queue with audited APPROVE / REJECT / REQUEST_CHANGES / DELEGATE / EXPIRE
decisions (Phase 23 ReviewDecision).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _dumps(value) -> Optional[str]:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Webhook signing + replay protection (Step 51)
# ---------------------------------------------------------------------------

SIGNATURE_TOLERANCE_S = 5 * 60     # reject timestamps older than 5 min
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 30                # 30s, 2m, 8m, 32m, 128m


def sign_payload(secret: str, payload: dict, *,
                 timestamp: Optional[int] = None) -> dict:
    """HMAC-SHA256 over ``{ts}.{body}`` — the standard signed-webhook shape.

    The secret never enters the payload, logs, or database.
    """
    ts = timestamp if timestamp is not None else int(time.time())
    body = _dumps(payload) or ""
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body.encode(),
                   hashlib.sha256).hexdigest()
    return {"signature": f"t={ts},v1={mac}", "timestamp": ts, "body": body}


def verify_signature(secret: str, signature: str, body: str, *,
                     now_ts: Optional[int] = None) -> dict:
    """Verify signature + timestamp freshness (replay protection)."""
    try:
        parts = dict(p.split("=", 1) for p in signature.split(","))
        ts = int(parts["t"])
        mac = parts["v1"]
    except (ValueError, KeyError, AttributeError):
        return {"valid": False, "reason": "malformed signature header"}
    now_ts = now_ts if now_ts is not None else int(time.time())
    if abs(now_ts - ts) > SIGNATURE_TOLERANCE_S:
        return {"valid": False, "reason": "timestamp outside tolerance "
                                          "(possible replay)"}
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body.encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, mac):
        return {"valid": False, "reason": "signature mismatch"}
    return {"valid": True}


def backoff_delay_s(attempt: int) -> int:
    """Exponential backoff with a hard cap (attempt is 1-based)."""
    return min(BACKOFF_BASE_S * (2 ** max(attempt - 1, 0)), 24 * 3600)


def record_delivery_attempt(db: Session, *, workspace_id: int,
                            endpoint_id: int, event_type: str,
                            ok: bool, response_status: Optional[int] = None,
                            error: Optional[str] = None,
                            delivery_id: Optional[str] = None) -> dict:
    """Record one delivery attempt; manage retries + endpoint health."""
    from ..models import (WebhookDeliveryAttempt, WebhookEndpointHealth)
    from uuid import uuid4

    health = db.query(WebhookEndpointHealth).filter_by(
        endpoint_id=endpoint_id).one_or_none()
    if health is None:
        health = WebhookEndpointHealth(endpoint_id=endpoint_id,
                                       workspace_id=workspace_id,
                                       consecutive_failures=0)
        db.add(health)
        db.flush()   # populate defaults before mutation

    if health.disabled:
        return {"state": "DISABLED_ENDPOINT",
                "reason": health.disabled_reason,
                "attempt": 0, "next_retry_at": None,
                "endpoint_disabled": True}

    prior = (db.query(WebhookDeliveryAttempt)
             .filter(WebhookDeliveryAttempt.endpoint_id == endpoint_id,
                     WebhookDeliveryAttempt.event_type == event_type,
                     WebhookDeliveryAttempt.delivery_id == (delivery_id or ""))
             .order_by(WebhookDeliveryAttempt.id.desc()).first())
    attempt = (prior.attempt + 1) if prior is not None else 1

    if ok:
        state = "DELIVERED"
        health.consecutive_failures = 0
        health.last_success_at = _utcnow()
    else:
        health.consecutive_failures += 1
        health.last_failure_at = _utcnow()
        if health.consecutive_failures >= MAX_ATTEMPTS:
            health.disabled = True
            health.disabled_reason = (f"disabled after "
                                      f"{health.consecutive_failures} "
                                      "consecutive failures")
        state = "DEAD" if attempt >= MAX_ATTEMPTS else "FAILED"

    record = WebhookDeliveryAttempt(
        workspace_id=workspace_id, endpoint_id=endpoint_id,
        delivery_id=(delivery_id or str(uuid4()))[:80],
        event_type=event_type, state=state, attempt=attempt,
        response_status=response_status,
        last_error=(error or "")[:600],
        next_retry_at=None if ok or state == "DEAD"
        else _utcnow() + timedelta(seconds=backoff_delay_s(attempt)))
    db.add(record)
    db.commit()
    return {"state": state, "attempt": attempt,
            "next_retry_at": record.next_retry_at.isoformat()
            if record.next_retry_at else None,
            "endpoint_disabled": health.disabled}


def reenable_endpoint(db: Session, *, endpoint_id: int,
                      actor: Optional[str] = None) -> dict:
    from ..models import WebhookEndpointHealth

    health = db.query(WebhookEndpointHealth).filter_by(
        endpoint_id=endpoint_id).one_or_none()
    if health is None:
        return {"reenabled": False, "reason": "no health record"}
    health.disabled = False
    health.consecutive_failures = 0
    health.disabled_reason = None
    health.updated_at = _utcnow()
    db.commit()
    return {"reenabled": True, "actor": actor}


def dead_letter_deliveries(db: Session, *, workspace_id: int,
                           limit: int = 100) -> dict:
    from ..models import WebhookDeliveryAttempt

    rows = (db.query(WebhookDeliveryAttempt)
            .filter(WebhookDeliveryAttempt.workspace_id == workspace_id,
                    WebhookDeliveryAttempt.state == "DEAD")
            .order_by(WebhookDeliveryAttempt.id.desc())
            .limit(min(limit, 500)).all())
    return {"items": [{
        "id": r.id, "delivery_id": r.delivery_id,
        "event_type": r.event_type, "attempt": r.attempt,
        "last_error": r.last_error,
        "created_at": r.created_at.isoformat()} for r in rows],
        "count": len(rows)}


# ---------------------------------------------------------------------------
# Scheduler task dedup (Step 48)
# ---------------------------------------------------------------------------

def due_bucket(ts: Optional[datetime] = None, *, granularity_minutes: int = 5) -> str:
    """Deterministic bucket key for dedup (e.g. 2026-09-06T1215Z)."""
    ts = ts or _utcnow()
    minutes = (ts.minute // granularity_minutes) * granularity_minutes
    return ts.replace(minute=minutes, second=0, microsecond=0)\
        .strftime("%Y-%m-%dT%H%MZ")


def claim_scheduled_task(db: Session, *, task_key: str,
                         leader_id: str,
                         bucket: Optional[str] = None) -> dict:
    """Claim a scheduled task for this due bucket — exactly once.

    Returns ``claimed=True`` for the first caller; everyone else in the
    same bucket gets ``claimed=False`` with status SKIPPED_DUPLICATE.
    """
    from ..models import SchedulerTaskRun

    bucket = bucket or due_bucket()
    existing = db.query(SchedulerTaskRun).filter_by(
        task_key=task_key[:120], due_bucket=bucket).one_or_none()
    if existing is not None:
        existing.status = existing.status or "RUNNING"
        return {"claimed": False, "status": "SKIPPED_DUPLICATE",
                "existing_run_id": existing.id, "bucket": bucket}
    run = SchedulerTaskRun(task_key=task_key[:120], due_bucket=bucket,
                           leader_id=leader_id[:120])
    db.add(run)
    db.commit()
    return {"claimed": True, "run_id": run.id, "bucket": bucket}


def finish_scheduled_task(db: Session, run_id: int, *, ok: bool,
                          detail: Optional[dict] = None) -> dict:
    from ..models import SchedulerTaskRun

    run = db.query(SchedulerTaskRun).get(run_id)
    if run is None:
        return {"finished": False}
    run.status = "COMPLETED" if ok else "FAILED"
    run.finished_at = _utcnow()
    run.detail_json = _dumps(detail or {})
    db.commit()
    return {"finished": True, "status": run.status}


# ---------------------------------------------------------------------------
# Unified review center (Step 35)
# ---------------------------------------------------------------------------

REVIEW_ITEM_TYPES = (
    "ai_action", "policy_conflict", "knowledge_conflict",
    "extraction_uncertainty", "provider_issue", "security_finding",
    "workflow_approval", "agent_handoff",
)

REVIEW_DECISIONS = ("APPROVE", "REJECT", "REQUEST_CHANGES", "DELEGATE",
                    "EXPIRE")


def create_unified_review(db: Session, *, workspace_id: int, item_type: str,
                          title: str, user_id: int,
                          description: Optional[str] = None,
                          payload: Optional[dict] = None,
                          priority: str = "NORMAL") -> dict:
    """Create a review item in the Phase 15 queue (typed, validated)."""
    from .review_service import create_review_item

    if item_type not in REVIEW_ITEM_TYPES:
        raise ValueError(f"unknown review item type: {item_type}")
    item = create_review_item(
        db, workspace_id=workspace_id, item_type=item_type, title=title,
        user_id=user_id, description=description, payload=payload,
        priority=priority)
    # The wrapper owns the transaction: create_review_item only flushes, so
    # without an explicit commit the row would roll back at request end.
    db.commit()
    return {"item_id": item.id, "item_type": item_type,
            "status": item.status, "priority": item.priority,
            "due_at": item.due_at.isoformat() if item.due_at else None}


def decide_review(db: Session, *, workspace_id: int, item_id: int,
                  decision: str, actor_user_id: Optional[int] = None,
                  reason: Optional[str] = None,
                  delegate_to: Optional[int] = None) -> dict:
    """Apply + audit a review decision (Phase 23 ReviewDecision)."""
    from ..models import ReviewDecision
    from .review_service import decide_review_item, get_review_item

    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"unknown decision: {decision}")
    item = get_review_item(db, workspace_id=workspace_id, item_id=item_id)
    if item is None:
        raise ValueError("review item not found")

    new_status = {
        "APPROVE": "APPROVED", "REJECT": "REJECTED",
        "REQUEST_CHANGES": "NEEDS_EVIDENCE", "DELEGATE": "DELEGATED",
        "EXPIRE": "DEFERRED",
    }[decision]
    updated = decide_review_item(
        db, workspace_id=workspace_id, item_id=item_id,
        decision=new_status, actor_id=actor_user_id or 0,
        note=reason, reassign_to=delegate_to)

    record = ReviewDecision(
        item_id=item_id, workspace_id=workspace_id, decision=decision,
        actor_user_id=actor_user_id, delegate_to=delegate_to,
        reason=reason)
    db.add(record)
    db.commit()
    return {"item_id": item_id, "decision": decision,
            "status": updated.status, "audited": True}
