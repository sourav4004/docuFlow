"""Workspace invitation model with secure tokens."""

import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from ..core.database import Base


class WorkspaceInvitation(Base):
    """Secure workspace invitation with token-based acceptance."""
    __tablename__ = "workspace_invitations"
    __table_args__ = (
        Index("ix_invitations_workspace_id", "workspace_id"),
        Index("ix_invitations_email", "invited_email"),
        Index("ix_invitations_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    inviter_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    invited_email = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="MEMBER")
    token_hash = Column(String(128), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING, ACCEPTED, REVOKED, EXPIRED
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    workspace = relationship("Workspace")
    inviter = relationship("User")

    @classmethod
    def create_token(cls) -> tuple[str, str]:
        """Create a secure token and its hash."""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token, token_hash

    @classmethod
    def verify_token(cls, token: str, token_hash: str) -> bool:
        """Verify a token against its hash."""
        return hashlib.sha256(token.encode()).hexdigest() == token_hash

    @property
    def is_expired(self) -> bool:
        """Check if invitation is expired."""
        if self.expires_at is None:
            return True
        # Handle both naive and aware datetimes
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires

    @property
    def is_usable(self) -> bool:
        """Check if invitation can be used."""
        return (
            self.status == "PENDING"
            and not self.is_expired
        )

    def accept(self):
        """Accept the invitation."""
        if not self.is_usable:
            raise ValueError("Invitation is not usable")
        self.status = "ACCEPTED"
        self.accepted_at = datetime.now(timezone.utc)

    def revoke(self):
        """Revoke the invitation."""
        self.status = "REVOKED"
        self.revoked_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<WorkspaceInvitation workspace={self.workspace_id} email={self.invited_email} status={self.status}>"
