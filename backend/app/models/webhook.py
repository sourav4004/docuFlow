"""Webhook endpoint, event (outbox), and delivery models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from ..core.database import Base


class WebhookEndpoint(Base):
    """Registered webhook endpoint with event subscriptions and signing secret."""

    __tablename__ = "webhook_endpoints"
    __table_args__ = (
        Index("ix_webhook_endpoints_workspace_id", "workspace_id"),
        Index("ix_webhook_endpoints_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    url = Column(String(2000), nullable=False)
    secret_hash = Column(String(128), nullable=False)  # HMAC signing secret (stored encrypted/hashed form)
    events_json = Column(String(4000), nullable=False, default="[]")  # subscribed events or ["*"]
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, INACTIVE
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")

    @property
    def events(self) -> list[str]:
        import json
        try:
            return json.loads(self.events_json or "[]")
        except (ValueError, TypeError):
            return []

    def subscribes_to(self, event_type: str) -> bool:
        """Check if this endpoint subscribes to an event type."""
        events = self.events
        return "*" in events or event_type in events

    def __repr__(self):
        return f"<WebhookEndpoint id={self.id} status={self.status}>"


class WebhookEvent(Base):
    """Persisted domain event (outbox) awaiting delivery to subscribers."""

    __tablename__ = "webhook_events"
    __table_args__ = (
        Index("ix_webhook_events_workspace_id", "workspace_id"),
        Index("ix_webhook_events_event_type", "event_type"),
        Index("ix_webhook_events_created_at", "created_at"),
        UniqueConstraint("event_id", name="uq_webhook_event_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), nullable=False)  # unique event identifier (UUID)
    event_type = Column(String(100), nullable=False)  # document.created, ai.execution.completed, etc.
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    payload_json = Column(Text, nullable=False)
    idempotency_key = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    deliveries = relationship("WebhookDelivery", back_populates="event", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WebhookEvent id={self.event_id} type={self.event_type}>"


class WebhookDelivery(Base):
    """Delivery attempt record for a webhook event to an endpoint."""

    __tablename__ = "webhook_deliveries"
    __table_args__ = (
        Index("ix_webhook_deliveries_event_id", "event_id"),
        Index("ix_webhook_deliveries_endpoint_id", "endpoint_id"),
        Index("ix_webhook_deliveries_status", "status"),
        Index("ix_webhook_deliveries_next_retry", "next_retry_at"),
        UniqueConstraint("event_id", "endpoint_id", name="uq_delivery_event_endpoint"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("webhook_events.id", ondelete="CASCADE"), nullable=False)
    endpoint_id = Column(Integer, ForeignKey("webhook_endpoints.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING, DELIVERING, DELIVERED, FAILED, RETRYING
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    response_status = Column(Integer, nullable=True)
    response_body = Column(String(2000), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    last_error = Column(String(1000), nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    event = relationship("WebhookEvent", back_populates="deliveries")
    endpoint = relationship("WebhookEndpoint")

    def __repr__(self):
        return f"<WebhookDelivery event={self.event_id} endpoint={self.endpoint_id} status={self.status}>"