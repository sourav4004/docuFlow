"""Workspace and membership models for multi-tenant architecture."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class Workspace(Base):
    """Workspace/tenant model for multi-tenant isolation."""
    __tablename__ = "workspaces"
    __table_args__ = (
        Index("ix_workspaces_owner_id", "owner_id"),
        Index("ix_workspaces_name", "name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000), default="")
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    owner = relationship("User", back_populates="owned_workspaces")
    organization = relationship("Organization", back_populates="workspaces")
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Workspace {self.name} (id={self.id})>"


class WorkspaceMember(Base):
    """Workspace membership with role-based access control."""
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_user"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default="MEMBER")  # OWNER, ADMIN, MEMBER, VIEWER
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="workspace_memberships")

    def __repr__(self):
        return f"<WorkspaceMember workspace={self.workspace_id} user={self.user_id} role={self.role}>"

    @property
    def is_owner(self) -> bool:
        return self.role == "OWNER"

    @property
    def is_admin(self) -> bool:
        return self.role in ("OWNER", "ADMIN")

    @property
    def can_write(self) -> bool:
        return self.role in ("OWNER", "ADMIN", "MEMBER")

    @property
    def can_read(self) -> bool:
        return True  # All roles can read


# Role hierarchy for permission checks
ROLE_HIERARCHY = {
    "OWNER": 4,
    "ADMIN": 3,
    "MEMBER": 2,
    "VIEWER": 1,
}


def has_permission(user_role: str, required_role: str) -> bool:
    """Check if user role has sufficient permissions."""
    return ROLE_HIERARCHY.get(user_role, 0) >= ROLE_HIERARCHY.get(required_role, 0)
