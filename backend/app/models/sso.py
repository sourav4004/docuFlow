"""SSO (OIDC/SAML) configuration models — no real credentials required for tests."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class SSOConfiguration(Base):
    """Organization-level SSO provider configuration."""

    __tablename__ = "sso_configurations"
    __table_args__ = (
        Index("ix_sso_configurations_organization_id", "organization_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    provider_type = Column(String(20), nullable=False, default="OIDC")  # OIDC, SAML
    issuer = Column(String(500), nullable=False)
    client_id = Column(String(500), nullable=False)
    client_secret_ref = Column(Text, nullable=True)  # encrypted reference — never plaintext
    authorization_endpoint = Column(String(1000), nullable=True)
    token_endpoint = Column(String(1000), nullable=True)
    jwks_uri = Column(String(1000), nullable=True)
    redirect_uri = Column(String(1000), nullable=True)
    is_enabled = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization")

    def __repr__(self):
        return f"<SSOConfiguration org={self.organization_id} type={self.provider_type} enabled={self.is_enabled}>"


class SSOState(Base):
    """One-time OIDC authorization state with nonce, used for login CSRF protection."""

    __tablename__ = "sso_states"
    __table_args__ = (
        Index("ix_sso_states_state", "state"),
        Index("ix_sso_states_expires_at", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(128), unique=True, nullable=False)
    nonce = Column(String(128), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    redirect_to = Column(String(1000), nullable=True)
    used = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)

    def __repr__(self):
        return f"<SSOState id={self.id} used={self.used}>"