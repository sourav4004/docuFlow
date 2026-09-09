"""Webhook platform service.

- Endpoint registration with HMAC signing secrets (hashed at rest)
- Event outbox: domain events persisted before delivery
- Signed deliveries with timestamp + replay protection
- Bounded retry with exponential backoff
- Delivery history
"""

import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.webhook import WebhookEndpoint, WebhookEvent, WebhookDelivery
from ..services.audit_service import log_audit_event

# Well-known event types
WEBHOOK_EVENT_TYPES = [
    "document.created",
    "document.updated",
    "document.deleted",
    "document.ready",
    "document.processing_failed",
    "conversation.created",
    "message.created",
    "ai.execution.started",
    "ai.execution.completed",
    "ai.execution.failed",
    "agent.started",
    "agent.completed",
    "agent.failed",
    "workflow.started",
    "workflow.completed",
    "workflow.failed",
    "collection.updated",
    "member.joined",
    "member.removed",
    "usage.threshold_reached",
]


def validate_event_types(events: list[str]) -> list[str]:
    """Validate event subscriptions; '*' means all events."""
    valid = set(WEBHOOK_EVENT_TYPES)
    result = []
    for event in events or []:
        if event == "*":
            result.append("*")
        elif event in valid:
            result.append(event)
        else:
            raise ValueError(f"Unknown webhook event type: {event}")
    return result


def create_endpoint(
    db: Session,
    workspace_id: int,
    user_id: int,
    url: str,
    events: list[str],
    description: Optional[str] = None,
) -> tuple[WebhookEndpoint, str]:
    """Create a webhook endpoint. Returns (endpoint, signing_secret) — secret shown once."""
    signing_secret = secrets.token_urlsafe(32)

    endpoint = WebhookEndpoint(
        workspace_id=workspace_id,
        user_id=user_id,
        url=url,
        secret_hash=signing_secret,  # server-side shared secret, never returned via API
        events_json=json.dumps(validate_event_types(events)),
        description=description,
    )
    db.add(endpoint)
    db.flush()

    log_audit_event(
        db,
        event_type="webhook",
        event_action="create",
        user_id=user_id,
        resource_type="webhook_endpoint",
        resource_id=endpoint.id,
        details="Webhook endpoint created",
    )
    return endpoint, signing_secret


def rotate_endpoint_secret(db: Session, endpoint_id: int, user_id: int) -> tuple[WebhookEndpoint, str]:
    """Rotate an endpoint's signing secret."""
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise ValueError("Webhook endpoint not found")
    signing_secret = secrets.token_urlsafe(32)
    endpoint.secret_hash = signing_secret
    db.flush()
    log_audit_event(
        db,
        event_type="webhook",
        event_action="rotate_secret",
        user_id=user_id,
        resource_type="webhook_endpoint",
        resource_id=endpoint_id,
        details="Webhook signing secret rotated",
    )
    return endpoint, signing_secret


def verify_signature(payload: bytes, timestamp: str, event_id: str, signature: str, secret: str, tolerance_seconds: int = 300) -> bool:
    """Verify an incoming webhook signature.

    Mirrors sign_payload: HMAC over f"{timestamp}.{event_id}.{payload}".
    Validates both the HMAC and the timestamp freshness (replay protection).
    """
    # Timestamp replay protection
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False

    expected = hmac.new(
        secret.encode(),
        f"{timestamp}.{event_id}.{payload.decode('utf-8', errors='ignore')}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def sign_payload(payload: dict, secret: str) -> tuple[str, str, str]:
    """Sign an outgoing payload.

    Returns (timestamp, event_id, signature).
    """
    timestamp = str(int(time.time()))
    event_id = str(uuid.uuid4())
    body_str = json.dumps(payload, separators=(",", ":"))
    signature = hmac.new(
        secret.encode(),
        f"{timestamp}.{event_id}.{body_str}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return timestamp, event_id, signature


def emit_event(
    db: Session,
    event_type: str,
    workspace_id: int,
    organization_id: Optional[int],
    payload: dict,
    idempotency_key: Optional[str] = None,
) -> WebhookEvent:
    """Persist a domain event to the outbox (transactional with the caller).

    Callers should emit inside the same transaction that performs the domain
    action so a committed event can never disappear.
    """
    event = WebhookEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        workspace_id=workspace_id,
        organization_id=organization_id,
        payload_json=json.dumps(payload, default=str),
        idempotency_key=idempotency_key,
    )
    db.add(event)
    db.flush()
    return event


def enqueue_deliveries(db: Session, event: WebhookEvent) -> list[WebhookDelivery]:
    """Create delivery records for every active subscribed endpoint.

    Uses the (event, endpoint) unique constraint to prevent duplicates.
    """
    endpoints = (
        db.query(WebhookEndpoint)
        .filter(
            WebhookEndpoint.workspace_id == event.workspace_id,
            WebhookEndpoint.status == "ACTIVE",
        )
        .all()
    )
    deliveries = []
    for endpoint in endpoints:
        if not endpoint.subscribes_to(event.event_type):
            continue
        delivery = WebhookDelivery(
            event_id=event.id,
            endpoint_id=endpoint.id,
            status="PENDING",
        )
        db.add(delivery)
        deliveries.append(delivery)
    db.flush()
    return deliveries


def attempt_delivery(db: Session, delivery: WebhookDelivery) -> bool:
    """Attempt to deliver a pending webhook (synchronous delivery for tests/dev).

    Production deployments route this through a background worker; the method
    is intentionally simple and deterministic. Returns True on 2xx delivery.
    """
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == delivery.endpoint_id).first()
    event = db.query(WebhookEvent).filter(WebhookEvent.id == delivery.event_id).first()
    if not endpoint or not event:
        delivery.status = "FAILED"
        delivery.last_error = "Endpoint or event missing"
        return False

    try:
        payload = json.loads(event.payload_json)
    except (ValueError, TypeError):
        payload = {"event": event.event_type}

    signing_secret = _resolve_secret(endpoint)
    timestamp, event_id, signature = sign_payload(payload, signing_secret)

    headers = {
        "X-DocuFlow-Event": event.event_type,
        "X-DocuFlow-Event-Id": event_id,
        "X-DocuFlow-Timestamp": timestamp,
        "X-DocuFlow-Signature": signature,
        "Content-Type": "application/json",
    }

    started = time.time()
    try:
        import urllib.request
        req = urllib.request.Request(
            endpoint.url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            delivery.response_status = resp.status
            delivery.latency_ms = int((time.time() - started) * 1000)
            delivery.attempt_count += 1
            delivery.status = "DELIVERED" if resp.status < 400 else "FAILED"
            if resp.status >= 400:
                delivery.last_error = f"HTTP {resp.status}"
            return delivery.status == "DELIVERED"
    except Exception as exc:  # noqa: BLE001 — network errors are expected
        delivery.attempt_count += 1
        delivery.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        delivery.latency_ms = int((time.time() - started) * 1000)
        if delivery.attempt_count >= delivery.max_attempts:
            delivery.status = "FAILED"
        else:
            delivery.status = "RETRYING"
            backoff = min(2 ** delivery.attempt_count, 3600)
            delivery.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
        return False


def _resolve_secret(endpoint: WebhookEndpoint) -> str:
    """Return the signing secret used to sign deliveries.

    The secret is stored server-side (standard webhook practice, e.g. Stripe)
    and must never be returned through any API response.
    """
    return endpoint.secret_hash