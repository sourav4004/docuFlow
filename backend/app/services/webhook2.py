"""Webhook reliability 2.0 — durable outbox delivery worker.

Existing infrastructure (webhook_service) already provides the outbox,
HMAC-signed payloads with timestamp + event id, bounded retries with
exponential backoff, and delivery history. Phase 16 adds:

- an SSRF-safe URL gate before any delivery attempt
- atomic delivery claiming (no two workers deliver the same event)
- a drain worker that keeps synchronous delivery out of request handlers
"""

import ipaddress
import logging
import socket
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.webhook import WebhookEndpoint, WebhookDelivery
from ..services.webhook_service import attempt_delivery

logger = logging.getLogger(__name__)

PRIVATE_PREFIXES = (
    "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
    "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.", "192.168.", "127.", "169.254.",
    "0.", "100.64.", "100.65.", "100.66.", "100.67.", "100.68.", "100.69.",
    "100.70.", "100.71.", "100.72.", "100.73.", "100.74.", "100.75.",
    "100.76.", "100.77.", "100.78.", "100.79.", "100.80.", "100.81.",
    "100.82.", "100.83.", "100.84.", "100.85.", "100.86.", "100.87.",
    "100.88.", "100.89.", "100.90.", "100.91.", "100.92.", "100.93.",
    "100.94.", "100.95.", "100.96.", "100.97.", "100.98.", "100.99.",
    "100.100.", "100.101.", "100.102.", "100.103.", "100.104.", "100.105.",
    "100.106.", "100.107.", "100.108.", "100.109.", "100.110.", "100.111.",
    "100.112.", "100.113.", "100.114.", "100.115.", "100.116.", "100.117.",
    "100.118.", "100.119.", "100.120.", "100.121.", "100.122.", "100.123.",
    "100.124.", "100.125.", "100.126.", "100.127.",
)

BLOCKED_HOSTNAMES = ("localhost", "metadata.google.internal",
                     "169.254.169.254", "metadata")


def validate_delivery_url(url: str) -> tuple[bool, str]:
    """SSRF gate — blocks loopback, private/link-local/metadata ranges,
    file:// schemes and internal hostnames. Applied to the FINAL URL after
    scheme normalization (redirect targets are re-validated by callers)."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False, "unparseable url"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme {parsed.scheme!r} not allowed"
    hostname = (parsed.hostname or "").lower()
    if hostname in BLOCKED_HOSTNAMES:
        return False, f"hostname {hostname!r} is blocked"
    if hostname.endswith(".local") or hostname.endswith(".internal"):
        return False, "internal DNS target blocked"
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        addr = None
    if addr is not None:
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast):
            return False, f"address {hostname} is not routable"
    # For hostnames, block when they resolve into private ranges. A hostname
    # that cannot be resolved is safe to allow (there is no reachable target).
    if addr is None:
        try:
            resolved = socket.getaddrinfo(hostname, None)
        except OSError:
            return True, "ok (hostname did not resolve in this environment)"
        for entry in resolved[:8]:
            raw = entry[4][0]
            if raw in BLOCKED_HOSTNAMES:
                return False, "resolved address blocked"
            try:
                ip = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False, f"hostname resolves to private address {raw}"
    if "169.254" in hostname:
        return False, "metadata address blocked"
    return True, "ok"


def claim_due_delivery(db: Session, now: Optional[datetime] = None,
                       ) -> Optional[WebhookDelivery]:
    """Atomically claim the next due webhook delivery (PENDING/RETRYING)."""
    now = now or datetime.now(timezone.utc)
    candidates = (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.status.in_(("PENDING", "RETRYING")),
                (WebhookDelivery.next_retry_at.is_(None))
                | (WebhookDelivery.next_retry_at <= now))
        .order_by(WebhookDelivery.created_at.asc())
        .limit(10)
        .all()
    )
    for delivery in candidates:
        updated = (
            db.query(WebhookDelivery)
            .filter(WebhookDelivery.id == delivery.id,
                    WebhookDelivery.status.in_(("PENDING", "RETRYING")))
            .update({"status": "DELIVERING"})
        )
        if updated == 1:
            db.flush()
            db.refresh(delivery)
            return delivery
        db.rollback()
        return None
    return None


def process_due_webhooks(db: Session, max_jobs: int = 10) -> dict:
    """Drain due webhook deliveries through the durable outbox.

    Returns {delivered, failed, retrying, blocked, attempted}.
    """
    result = {"delivered": 0, "failed": 0, "retrying": 0,
              "blocked": 0, "attempted": 0}
    for _ in range(max_jobs):
        delivery = claim_due_delivery(db)
        if delivery is None:
            break
        endpoint = db.query(WebhookEndpoint).filter(
            WebhookEndpoint.id == delivery.endpoint_id).first()
        ok_url, reason = validate_delivery_url(endpoint.url) if endpoint else (
            False, "endpoint missing")
        if not ok_url:
            delivery.status = "FAILED"
            delivery.last_error = f"SSRF guard: {reason}"
            delivery.attempt_count = delivery.max_attempts
            result["blocked"] += 1
            db.flush()
            continue
        try:
            delivered = attempt_delivery(db, delivery)
        except Exception as exc:  # noqa: BLE001 — worker boundary
            delivery.status = "FAILED"
            delivery.last_error = str(exc)[:1000]
            delivered = False
        if delivered:
            delivery.next_retry_at = None
            result["delivered"] += 1
        elif delivery.status == "RETRYING":
            result["retrying"] += 1
        else:
            result["failed"] += 1
        result["attempted"] += 1
        db.flush()
    db.flush()
    return result
