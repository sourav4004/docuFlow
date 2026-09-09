"""Workflow versioning model."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class WorkflowVersion(Base):
    """An immutable version of a workflow definition.

    Workflows are identified by a stable workflow_id; each edit creates a new
    version row. Executions reference the version they started with.
    """

    __tablename__ = "workflow_versions"
    __table_args__ = (
        Index("ix_workflow_versions_workflow_id", "workflow_id"),
        Index("ix_workflow_versions_workspace_id", "workspace_id"),
        Index("ix_workflow_versions_is_active", "is_active"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(64), nullable=False)  # stable logical workflow ID
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    definition_json = Column(Text, nullable=False)  # trigger, conditions, nodes, approval gates
    status = Column(String(20), nullable=False, default="DRAFT")  # DRAFT, ACTIVE, INACTIVE
    is_active = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    creator = relationship("User")

    def __repr__(self):
        return f"<WorkflowVersion workflow={self.workflow_id} v{self.version} status={self.status}>"