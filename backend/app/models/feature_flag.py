"""Feature flag model for deterministic feature rollout."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class FeatureFlag(Base):
    """Feature flag with global, organization, and workspace scopes."""

    __tablename__ = "feature_flags"
    __table_args__ = (
        Index("ix_feature_flags_name", "name"),
        Index("ix_feature_flags_scope", "scope_type", "scope_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    scope_type = Column(String(20), nullable=False, default="GLOBAL")  # GLOBAL, ORGANIZATION, WORKSPACE, USER
    scope_id = Column(Integer, nullable=True)  # ID within scope_type
    enabled = Column(Boolean, nullable=False, default=True)
    rollout_percentage = Column(Integer, nullable=False, default=100)  # 0-100 deterministic rollout
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    creator = relationship("User")

    def __repr__(self):
        return f"<FeatureFlag {self.name} scope={self.scope_type}:{self.scope_id} enabled={self.enabled}>"