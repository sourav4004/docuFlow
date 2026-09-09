"""Secure API key model — only hashes are stored, never plaintext keys."""

import secrets
import hashlib
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class ApiKey(Base):
    """API key for programmatic access with scopes and expiration."""

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_workspace_id", "workspace_id"),
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_prefix", "prefix"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    prefix = Column(String(12), nullable=False)  # e.g. "df_live_ab12cd"
    key_hash = Column(String(128), unique=True, nullable=False, index=True)  # SHA-256 of full key
    scopes_json = Column(String(2000), nullable=False, default="[]")  # JSON list of scopes
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User", foreign_keys=[user_id])

    @classmethod
    def generate_key(cls) -> tuple[str, str, str]:
        """Generate a new API key.

        Returns:
            (full_key, prefix, key_hash) — full key shown only once at creation.
        """
        full_key = f"df_live_{secrets.token_urlsafe(32)}"
        prefix = full_key[:12]
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key, prefix, key_hash

    @classmethod
    def hash_key(cls, key: str) -> str:
        """Hash a full API key for comparison."""
        return hashlib.sha256(key.encode()).hexdigest()

    @property
    def scopes(self) -> list[str]:
        """Parse scopes from stored JSON."""
        import json
        try:
            return json.loads(self.scopes_json or "[]")
        except (ValueError, TypeError):
            return []

    def has_scope(self, scope: str) -> bool:
        """Check if this key has the given scope."""
        scopes = self.scopes
        return "*" in scopes or scope in scopes

    @property
    def is_active(self) -> bool:
        """Check if the key is active (not revoked, not expired)."""
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None:
            now = datetime.now(timezone.utc)
            expires = self.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if now > expires:
                return False
        return True

    def __repr__(self):
        return f"<ApiKey id={self.id} prefix={self.prefix} active={self.is_active}>"