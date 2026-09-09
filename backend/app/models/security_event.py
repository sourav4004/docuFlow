"""Security event model for rule-based detection and security center."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class SecurityEvent(Base):
    """Detected security-relevant event (failed logins, invalid keys, etc.)."""

    __tablename__ = "security_events"
    __table_args__ = (
        Index("ix_security_events_workspace_id", "workspace_id"),
        Index("ix_security_events_user_id", "user_id"),
        Index("ix_security_events_event_type", "event_type"),
        Index("ix_security_events_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    event_type = Column(String(100), nullable=False)  # repeated_failed_login, invalid_api_key, etc.
    severity = Column(String(20), nullable=False, default="INFO")  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    description = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)
    source_ip = Column(String(45), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization")
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<SecurityEvent id={self.id} type={self.event_type} severity={self.severity}>"