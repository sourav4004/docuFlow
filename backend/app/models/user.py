from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class User(Base):
    """User model for authentication and account management."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Workspace relationships
    owned_workspaces = relationship("Workspace", back_populates="owner", lazy="select")
    workspace_memberships = relationship("WorkspaceMember", back_populates="user", lazy="select")

    # Organization relationships
    owned_organizations = relationship("Organization", foreign_keys="Organization.owner_id", back_populates="owner", lazy="select")
    organization_memberships = relationship("OrganizationMember", back_populates="user", lazy="select")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
