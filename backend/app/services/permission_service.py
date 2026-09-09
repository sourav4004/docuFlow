"""Centralized permission engine for enterprise role-based access control.

Consolidates permission checks so application code never scatters
ad-hoc role comparisons. Supports three levels:

- PLATFORM  (platform operator, not granted to ordinary users)
- ORGANIZATION (OWNER, ADMIN, MEMBER, VIEWER)
- WORKSPACE  (OWNER, ADMIN, MEMBER, VIEWER)
"""

from typing import Optional, Dict, Set, Callable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..models.workspace import Workspace, WorkspaceMember, has_permission as ws_has_permission
from ..models.organization import Organization, OrganizationMember

# Role hierarchy: higher number = more privilege
ROLE_HIERARCHY = {
    "VIEWER": 1,
    "MEMBER": 2,
    "ADMIN": 3,
    "OWNER": 4,
}


def role_at_least(role: str, required: str) -> bool:
    """True if role is equal or above required in the hierarchy."""
    return ROLE_HIERARCHY.get(role, 0) >= ROLE_HIERARCHY.get(required, 0)


# Permission catalog. Each permission maps to the minimum role required.
# Permissions are workspace-scoped unless prefixed with "org:" (organization)
# or "platform:" (platform operator).
PERMISSIONS: Dict[str, str] = {
    # Workspace administration
    "workspace:view": "VIEWER",
    "workspace:manage": "ADMIN",
    "workspace:delete": "OWNER",
    "workspace:invite": "ADMIN",
    "workspace:manage_members": "ADMIN",
    # Document management
    "document:read": "VIEWER",
    "document:write": "MEMBER",
    "document:delete": "ADMIN",
    "document:export": "MEMBER",
    # AI
    "ai:execute": "MEMBER",
    "ai:manage_settings": "ADMIN",
    "agent:execute": "MEMBER",
    "workflow:execute": "MEMBER",
    "workflow:manage": "ADMIN",
    # API platform
    "apikey:manage": "ADMIN",
    "webhook:manage": "ADMIN",
    "integration:manage": "ADMIN",
    # Usage / billing
    "usage:view": "MEMBER",
    "billing:view": "ADMIN",
    "billing:manage": "OWNER",
    # Enterprise
    "audit:view": "ADMIN",
    "security:view": "ADMIN",
    "security:manage": "OWNER",
    "sso:manage": "OWNER",
    "retention:manage": "ADMIN",
    "export:create": "MEMBER",
    "import:create": "ADMIN",
    # Organization-level permissions
    "org:manage": "ADMIN",
    "org:delete": "OWNER",
    "org:manage_members": "ADMIN",
    "org:view": "MEMBER",
}

# Platform operator permissions (never granted via roles)
PLATFORM_PERMISSIONS: Set[str] = {
    "platform:admin",
    "platform:manage_organizations",
    "platform:view_all",
}


class PermissionDeniedError(Exception):
    """Raised when a user lacks permission."""


def get_workspace_role(db: Session, workspace_id: int, user_id: int) -> Optional[str]:
    """Return the user's role in a workspace, or None if not a member."""
    membership = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
    )
    return membership.role if membership else None


def get_organization_role(db: Session, organization_id: int, user_id: int) -> Optional[str]:
    """Return the user's role in an organization, or None if not a member."""
    membership = (
        db.query(OrganizationMember)
        .filter(
            OrganizationMember.organization_id == organization_id,
            OrganizationMember.user_id == user_id,
        )
        .first()
    )
    return membership.role if membership else None


def has_permission(
    db: Session,
    user_id: int,
    permission: str,
    workspace_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    platform_role: Optional[str] = None,
) -> bool:
    """Check whether a user has a permission.

    Args:
        db: Database session
        user_id: Acting user
        permission: Permission key from PERMISSIONS
        workspace_id: Workspace context (required for workspace permissions)
        organization_id: Organization context (required for org: permissions)
        platform_role: Platform operator role (only from trusted internal context)

    Returns:
        True if permitted.
    """
    # Platform permissions are never granted through normal roles
    if permission in PLATFORM_PERMISSIONS:
        return bool(platform_role)

    required_role = PERMISSIONS.get(permission)
    if required_role is None:
        return False

    if permission.startswith("org:"):
        if organization_id is None:
            return False
        role = get_organization_role(db, organization_id, user_id)
        return role is not None and role_at_least(role, required_role)

    # Workspace permission — organization ADMIN/OWNER may inherit workspace access
    if workspace_id is None:
        return False
    role = get_workspace_role(db, workspace_id, user_id)
    if role is not None:
        return role_at_least(role, required_role)
    return False


def require_permission(
    db: Session,
    user_id: int,
    permission: str,
    workspace_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    platform_role: Optional[str] = None,
) -> None:
    """Require a permission or raise HTTP 403."""
    if not has_permission(
        db,
        user_id,
        permission,
        workspace_id=workspace_id,
        organization_id=organization_id,
        platform_role=platform_role,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}",
        )


def check_permission_or_404(
    db: Session,
    user_id: int,
    permission: str,
    workspace_id: Optional[int] = None,
    organization_id: Optional[int] = None,
) -> None:
    """Require permission but report 404 to avoid resource enumeration."""
    if not has_permission(
        db,
        user_id,
        permission,
        workspace_id=workspace_id,
        organization_id=organization_id,
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")


def require_workspace_membership(db: Session, workspace_id: int, user_id: int) -> Workspace:
    """Ensure the user belongs to the workspace; raise 404 otherwise (no enumeration)."""
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    role = get_workspace_role(db, workspace_id, user_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def require_organization_membership(db: Session, organization_id: int, user_id: int) -> Organization:
    """Ensure the user belongs to the organization; raise 404 otherwise (no enumeration)."""
    organization = db.query(Organization).filter(Organization.id == organization_id).first()
    if not organization:
        raise HTTPException(status_code=404, detail="Organization not found")
    role = get_organization_role(db, organization_id, user_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return organization


# Backward-compatible alias for existing callers
def workspace_has_permission(user_role: str, required_role: str) -> bool:
    """Backward-compatible wrapper around workspace role hierarchy."""
    return ws_has_permission(user_role, required_role)