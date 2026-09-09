"""Organization (top-level tenant) models for enterprise multi-tenancy."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class Organization(Base):
    """Top-level tenant that owns workspaces, users, and enterprise configuration."""

    __tablename__ = "organizations"
    __table_args__ = (
        Index("ix_organizations_owner_id", "owner_id"),
        Index("ix_organizations_name", "name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(1000), default="")
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, SUSPENDED, DEACTIVATED
    settings_json = Column(String(4000), nullable=True)  # organization-level configuration
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    owner = relationship("User", foreign_keys=[owner_id])
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    workspaces = relationship("Workspace", back_populates="organization", lazy="dynamic")

    def __repr__(self):
        return f"<Organization {self.name} (id={self.id})>"


class OrganizationMember(Base):
    """Organization membership with role-based access control."""

    __tablename__ = "organization_members"
    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_org_user"),
        Index("ix_org_members_user_id", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default="MEMBER")  # OWNER, ADMIN, MEMBER, VIEWER
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization", back_populates="members")
    user = relationship("User", back_populates="organization_memberships")

    def __repr__(self):
        return f"<OrganizationMember org={self.organization_id} user={self.user_id} role={self.role}>"


class OrganizationInvitation(Base):
    """Organization invitation with secure token-based acceptance."""

    __tablename__ = "organization_invitations"
    __table_args__ = (
        Index("ix_org_invitations_organization_id", "organization_id"),
        Index("ix_org_invitations_email", "invited_email"),
        Index("ix_org_invitations_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
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
    organization = relationship("Organization")
    inviter = relationship("User")

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return True
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires

    @property
    def is_usable(self) -> bool:
        return self.status == "PENDING" and not self.is_expired

    def accept(self):
        if not self.is_usable:
            raise ValueError("Invitation is not usable")
        self.status = "ACCEPTED"
        self.accepted_at = datetime.now(timezone.utc)

    def revoke(self):
        self.status = "REVOKED"
        self.revoked_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<OrganizationInvitation org={self.organization_id} email={self.invited_email} status={self.status}>"


class VerifiedDomain(Base):
    """Email domain verified as belonging to an organization."""

    __tablename__ = "verified_domains"
    __table_args__ = (
        UniqueConstraint("organization_id", "domain", name="uq_org_domain"),
        Index("ix_verified_domains_domain", "domain"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    domain = Column(String(255), nullable=False)
    verification_token = Column(String(128), nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization")

    def __repr__(self):
        return f"<VerifiedDomain org={self.organization_id} domain={self.domain} verified={self.is_verified}>"