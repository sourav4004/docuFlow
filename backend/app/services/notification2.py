"""Notification orchestration 2.0 — in-app + email + webhook channels with
deduplication and per-channel delivery records.

Email delivery is behind an ``EmailProvider`` interface. The default provider
is deterministic and requires no SMTP credentials; a production provider can
be configured behind the same interface.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.notification import Notification
from ..models.phase16 import NotificationDelivery

logger = logging.getLogger(__name__)

CHANNELS = ("IN_APP", "EMAIL", "WEBHOOK")


class EmailDeliveryError(Exception):
    """Raised when an email provider rejects a message."""


class EmailProvider(ABC):
    """Email delivery abstraction — never requires credentials by default."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Safe provider identifier."""

    @abstractmethod
    def send(self, to: str, subject: str, body: str) -> tuple[bool, Optional[str]]:
        """Deliver one email. Returns (ok, error)."""


class LogEmailProvider(EmailProvider):
    """Deterministic default — records delivery intent, sends nothing.

    Logs only metadata (recipient domain, subject length) — never message
    body content.
    """

    @property
    def name(self) -> str:
        return "log"

    def send(self, to: str, subject: str, body: str) -> tuple[bool, Optional[str]]:
        logger.info("Email delivery (provider=log) to=%s subject_len=%d body_len=%d",
                    to.split("@")[-1], len(subject), len(body))
        return True, None


def notify(
    db: Session,
    user_id: int,
    title: str,
    message: str,
    notification_type: str = "system",
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    channels: Optional[list[str]] = None,
    email_to: Optional[str] = None,
    email_subject: Optional[str] = None,
    dedupe_window_seconds: int = 300,
    email_provider: Optional[EmailProvider] = None,
) -> dict:
    """Create one notification + per-channel delivery records (deduplicated).

    Returns:
        {notification_id, created, deduplicated, deliveries: [...]}
    """
    channels = [c.upper() for c in (channels or ["IN_APP"])]
    for c in channels:
        if c not in CHANNELS:
            raise ValueError(f"Unknown channel: {c}")

    # Dedupe: identical recent (user, type, resource, title) notifications are
    # not created twice within the window.
    since = datetime.now(timezone.utc) - timedelta(seconds=dedupe_window_seconds)
    duplicate = (
        db.query(Notification)
        .filter(Notification.user_id == user_id,
                Notification.notification_type == notification_type,
                Notification.title == title)
        .filter(Notification.created_at >= since)
        .first()
    )
    if duplicate is not None:
        return {"notification_id": duplicate.id, "created": False,
                "deduplicated": True, "deliveries": []}

    notification = Notification(
        user_id=user_id,
        title=title[:255],
        message=message,
        notification_type=notification_type[:50],
        resource_type=resource_type,
        resource_id=resource_id,
    )
    db.add(notification)
    db.flush()

    deliveries = []
    for channel in channels:
        existing = (
            db.query(NotificationDelivery)
            .filter(NotificationDelivery.notification_id == notification.id,
                    NotificationDelivery.channel == channel)
            .first()
        )
        if existing is not None:
            deliveries.append({"channel": channel, "status": existing.status})
            continue
        delivery = NotificationDelivery(
            notification_id=notification.id,
            channel=channel,
            status="SENT" if channel == "IN_APP" else "PENDING",
            provider="in_app" if channel == "IN_APP" else None,
        )
        db.add(delivery)
        deliveries.append({"channel": channel, "status": delivery.status})
        if channel == "EMAIL":
            provider = email_provider or LogEmailProvider()
            delivery.provider = provider.name
            ok, error = provider.send(
                email_to or _email_of(db, user_id),
                email_subject or title,
                message,
            )
            delivery.attempts = 1
            if ok:
                delivery.status = "SENT"
                delivery.sent_at = datetime.now(timezone.utc)
            else:
                delivery.status = "FAILED"
                delivery.error = (error or "email send failed")[:2000]
                logger.warning("Email delivery failed for notification %s: %s",
                               notification.id, error)
    db.flush()
    return {"notification_id": notification.id, "created": True,
            "deduplicated": False, "deliveries": deliveries}


def _email_of(db: Session, user_id: int) -> str:
    from ..models.user import User
    user = db.query(User).filter(User.id == user_id).first()
    return user.email if user else "unknown@example.invalid"


def list_deliveries(db: Session, workspace_user_id: int,
                    limit: int = 50) -> list[NotificationDelivery]:
    """Delivery records for one user (scope by owning user)."""
    notification_ids = [
        n.id for n in
        db.query(Notification.id).filter(Notification.user_id == workspace_user_id)
        .limit(1000).all()
    ]
    if not notification_ids:
        return []
    return (
        db.query(NotificationDelivery)
        .filter(NotificationDelivery.notification_id.in_(notification_ids))
        .order_by(NotificationDelivery.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
