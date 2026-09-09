"""Generic third-party integration connection model."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class IntegrationConnection(Base):
    """A configured connection to an external integration provider.

    Credentials are stored as opaque encrypted references (never plaintext
    in logs or responses). Provider adapters handle the actual payload.
    """

    __tablename__ = "integration_connections"
    __table_args__ = (
        Index("ix_integration_connections_workspace_id", "workspace_id"),
        Index("ix_integration_connections_provider", "provider"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(100), nullable=False)  # slack, google_drive, s3, generic, etc.
    name = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="DISCONNECTED")  # AVAILABLE, CONNECTED, DISCONNECTED, ERROR
    config_json = Column(Text, nullable=True)  # non-secret configuration
    credentials_ref = Column(Text, nullable=True)  # encrypted credential reference — never plaintext
    last_connected_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<IntegrationConnection id={self.id} provider={self.provider} status={self.status}>"