"""Knowledge intelligence models — health scores, deadlines, insights."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class DocumentHealth(Base):
    """Heuristic health score for a document (0–100, never falsely precise)."""

    __tablename__ = "document_health"
    __table_args__ = (
        Index("ix_document_health_workspace_id", "workspace_id"),
        Index("ix_document_health_document_id", "document_id"),
        Index("ix_document_health_score", "score"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)
    score = Column(Integer, nullable=False, default=0)  # 0–100 heuristic
    factors_json = Column(Text, nullable=True)  # factor name -> contribution
    reasons_json = Column(Text, nullable=True)  # human-readable reasons
    computed_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    document = relationship("Document")

    def __repr__(self):
        return f"<DocumentHealth doc={self.document_id} score={self.score}>"


class Deadline(Base):
    """A deadline derived from documents or created manually."""

    __tablename__ = "deadlines"
    __table_args__ = (
        Index("ix_deadlines_workspace_id", "workspace_id"),
        Index("ix_deadlines_owner_id", "owner_id"),
        Index("ix_deadlines_status", "status"),
        Index("ix_deadlines_due_date", "due_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=False)
    source = Column(String(50), nullable=False, default="manual")  # document, manual, workflow
    source_reference = Column(Text, nullable=True)  # e.g. extracted quote/page
    confidence = Column(String(20), nullable=False, default="UNKNOWN")  # HIGH, MEDIUM, LOW, UNKNOWN
    status = Column(String(20), nullable=False, default="UPCOMING")  # UPCOMING, DUE, OVERDUE, COMPLETED, CANCELLED
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    owner = relationship("User")
    document = relationship("Document")

    def __repr__(self):
        return f"<Deadline id={self.id} title={self.title} status={self.status}>"


class KnowledgeInsight(Base):
    """A workspace-scoped knowledge insight (used by dashboards/digests)."""

    __tablename__ = "knowledge_insights"
    __table_args__ = (
        Index("ix_insights_workspace_id", "workspace_id"),
        Index("ix_insights_type", "insight_type"),
        Index("ix_insights_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    insight_type = Column(String(50), nullable=False)  # health, change, deadline, conflict, digest, recommendation
    title = Column(String(255), nullable=False)
    detail = Column(Text, nullable=True)
    importance = Column(String(20), nullable=False, default="INFO")  # INFO, LOW, MEDIUM, HIGH
    evidence_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    owner = relationship("User")

    def __repr__(self):
        return f"<KnowledgeInsight id={self.id} type={self.insight_type}>"