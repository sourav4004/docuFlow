from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, Index
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserSession(Base):
    """Database-backed user session.

    Replaces the in-memory session store for persistence across restarts.
    Session IDs are cryptographically random and stored in HTTP-only cookies.

    Session lifetime is governed by two independent limits (Phase 24):
    - ``expires_at``  — absolute expiry (``session_max_age``).
    - ``last_used_at`` — last successful authentication; when now is more
      than ``session_idle_timeout`` seconds later, the session is expired
      (idle timeout) and removed. ``last_used_at`` starts at creation time.
    """

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_used_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="sessions")

    __table_args__ = (
        Index("ix_user_sessions_user_id", "user_id"),
        Index("ix_user_sessions_expires_at", "expires_at"),
    )

    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"
