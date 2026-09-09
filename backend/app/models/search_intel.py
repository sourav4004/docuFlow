"""Search intelligence and AI feedback models."""

import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from ..core.database import Base


class SavedSearch(Base):
    """A personal or workspace-shared saved search with filters."""

    __tablename__ = "saved_searches"
    __table_args__ = (
        Index("ix_saved_searches_workspace_id", "workspace_id"),
        Index("ix_saved_searches_owner_id", "owner_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    filters_json = Column(Text, nullable=True)  # structured filters
    is_shared = Column(Boolean, nullable=False, default=False)  # workspace-shared
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    owner = relationship("User")
    alerts = relationship("SearchAlert", back_populates="saved_search", cascade="all, delete-orphan")

    @property
    def filters(self) -> dict:
        try:
            return json.loads(self.filters_json or "{}")
        except (ValueError, TypeError):
            return {}

    def __repr__(self):
        return f"<SavedSearch id={self.id} name={self.name}>"


class SearchAlert(Base):
    """An alert that fires when new content matches a saved search."""

    __tablename__ = "search_alerts"
    __table_args__ = (
        Index("ix_search_alerts_workspace_id", "workspace_id"),
        Index("ix_search_alerts_saved_search_id", "saved_search_id"),
        UniqueConstraint("saved_search_id", name="uq_alert_per_search"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    saved_search_id = Column(Integer, ForeignKey("saved_searches.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    last_match_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    saved_search = relationship("SavedSearch", back_populates="alerts")

    def __repr__(self):
        return f"<SearchAlert id={self.id} active={self.is_active}>"


class AIFeedback(Base):
    """User feedback on AI answers (never used to train models automatically)."""

    __tablename__ = "ai_feedback"
    __table_args__ = (
        Index("ix_ai_feedback_workspace_id", "workspace_id"),
        Index("ix_ai_feedback_execution_id", "execution_id"),
        Index("ix_ai_feedback_user_id", "user_id"),
        Index("ix_ai_feedback_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    execution_id = Column(String(36), nullable=True)  # linked AI execution
    rating = Column(String(20), nullable=False)  # thumbs_up, thumbs_down
    category = Column(String(50), nullable=True)  # correct, incorrect, missing_info, citation_issue
    comment = Column(Text, nullable=True)  # optional free text
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<AIFeedback id={self.id} rating={self.rating}>"