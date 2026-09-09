"""Plan and subscription models for billing/entitlement readiness.

No payment processing is implemented — these models provide the
subscription/plan abstraction layer only.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class Plan(Base):
    """Pricing plan definition with feature/limit configuration."""

    __tablename__ = "plans"
    __table_args__ = (
        Index("ix_plans_code", "code"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False)  # FREE, PRO, BUSINESS, ENTERPRISE
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    limits_json = Column(String(4000), nullable=False, default="{}")  # quota limits keyed by metric
    features_json = Column(String(4000), nullable=False, default="{}")  # feature flags
    is_default = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")

    @property
    def limits(self) -> dict:
        import json
        try:
            return json.loads(self.limits_json or "{}")
        except (ValueError, TypeError):
            return {}

    @property
    def features(self) -> dict:
        import json
        try:
            return json.loads(self.features_json or "{}")
        except (ValueError, TypeError):
            return {}

    def __repr__(self):
        return f"<Plan {self.code}>"


class Subscription(Base):
    """Organization's subscription to a plan."""

    __tablename__ = "subscriptions"
    __table_args__ = (
        Index("ix_subscriptions_organization_id", "organization_id"),
        Index("ix_subscriptions_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    plan_id = Column(Integer, ForeignKey("plans.id", ondelete="RESTRICT"), nullable=False)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, TRIALING, PAST_DUE, CANCELED, EXPIRED
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization")
    plan = relationship("Plan", back_populates="subscriptions")

    def __repr__(self):
        return f"<Subscription org={self.organization_id} plan={self.plan_id} status={self.status}>"