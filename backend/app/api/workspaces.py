"""Workspace management API endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.workspace import Workspace, WorkspaceMember, has_permission, ROLE_HIERARCHY
from ..schemas.auth import MessageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=1000)


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class MemberInvite(BaseModel):
    email: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="MEMBER", pattern="^(ADMIN|MEMBER|VIEWER)$")


class MemberRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(ADMIN|MEMBER|VIEWER)$")


def _get_owned_workspace(db: Session, user_id: int, workspace_id: int) -> Workspace:
    """Get workspace with ownership/membership check."""
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Check membership
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this workspace")
    
    return workspace


def _get_membership(db: Session, workspace_id: int, user_id: int) -> Optional[WorkspaceMember]:
    """Get user's membership in a workspace."""
    return db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    ).first()


@router.post("", status_code=201, summary="Create a workspace")
def create_workspace(
    body: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new workspace. Creator becomes OWNER."""
    workspace = Workspace(
        name=body.name.strip(),
        description=body.description.strip() if body.description else "",
        owner_id=current_user.id,
    )
    db.add(workspace)
    db.flush()
    
    # Add creator as OWNER
    membership = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=current_user.id,
        role="OWNER",
    )
    db.add(membership)
    db.commit()
    db.refresh(workspace)
    
    return {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "created_at": workspace.created_at.isoformat() if workspace.created_at else None,
    }


@router.get("", summary="List user's workspaces")
def list_workspaces(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all workspaces the user is a member of."""
    memberships = db.query(WorkspaceMember).filter(
        WorkspaceMember.user_id == current_user.id,
    ).all()
    
    workspaces = []
    for membership in memberships:
        workspace = db.query(Workspace).filter(Workspace.id == membership.workspace_id).first()
        if workspace:
            workspaces.append({
                "id": workspace.id,
                "name": workspace.name,
                "description": workspace.description,
                "role": membership.role,
                "created_at": workspace.created_at.isoformat() if workspace.created_at else None,
            })
    
    return {"items": workspaces, "total": len(workspaces)}


@router.get("/{workspace_id}", summary="Get workspace details")
def get_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get workspace details including member count."""
    _get_owned_workspace(db, current_user.id, workspace_id)
    
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    membership_count = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id
    ).count()
    
    return {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "member_count": membership_count,
        "created_at": workspace.created_at.isoformat() if workspace.created_at else None,
    }


@router.patch("/{workspace_id}", summary="Update workspace")
def update_workspace(
    workspace_id: int,
    body: WorkspaceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update workspace settings. Requires ADMIN or OWNER role."""
    workspace = _get_owned_workspace(db, current_user.id, workspace_id)
    membership = _get_membership(db, workspace_id, current_user.id)
    
    if not membership or not membership.is_admin:
        raise HTTPException(status_code=403, detail="Admin or Owner role required")
    
    if body.name is not None:
        workspace.name = body.name.strip()
    if body.description is not None:
        workspace.description = body.description.strip()
    
    db.commit()
    db.refresh(workspace)
    
    return {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
    }


@router.delete("/{workspace_id}", response_model=MessageResponse, summary="Delete workspace")
def delete_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete workspace. Only OWNER can delete."""
    workspace = _get_owned_workspace(db, current_user.id, workspace_id)
    membership = _get_membership(db, workspace_id, current_user.id)
    
    if not membership or not membership.is_owner:
        raise HTTPException(status_code=403, detail="Owner role required to delete workspace")
    
    db.delete(workspace)
    db.commit()
    
    return MessageResponse(message="Workspace deleted successfully")


@router.post("/{workspace_id}/members", status_code=201, summary="Invite member")
def invite_member(
    workspace_id: int,
    body: MemberInvite,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Invite a user to workspace by email. Requires ADMIN or OWNER role."""
    workspace = _get_owned_workspace(db, current_user.id, workspace_id)
    membership = _get_membership(db, workspace_id, current_user.id)
    
    if not membership or not membership.is_admin:
        raise HTTPException(status_code=403, detail="Admin or Owner role required")
    
    # Find user by email
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found with that email")
    
    # Check if already a member
    existing = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="User is already a member")
    
    new_member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=user.id,
        role=body.role,
    )
    db.add(new_member)
    db.commit()
    
    return {"message": "Member added successfully", "user_id": user.id, "role": body.role}


@router.get("/{workspace_id}/members", summary="List workspace members")
def list_members(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all members of a workspace."""
    _get_owned_workspace(db, current_user.id, workspace_id)
    
    members = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id
    ).all()
    
    result = []
    for member in members:
        user = db.query(User).filter(User.id == member.user_id).first()
        if user:
            result.append({
                "user_id": user.id,
                "name": user.name,
                "email": user.email,
                "role": member.role,
                "joined_at": member.created_at.isoformat() if member.created_at else None,
            })
    
    return {"items": result, "total": len(result)}


@router.patch("/{workspace_id}/members/{user_id}", summary="Update member role")
def update_member_role(
    workspace_id: int,
    user_id: int,
    body: MemberRoleUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a member's role. Requires OWNER role."""
    _get_owned_workspace(db, current_user.id, workspace_id)
    membership = _get_membership(db, workspace_id, current_user.id)
    
    if not membership or not membership.is_owner:
        raise HTTPException(status_code=403, detail="Owner role required")
    
    target_membership = _get_membership(db, workspace_id, user_id)
    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if target_membership.is_owner:
        raise HTTPException(status_code=400, detail="Cannot change owner's role")
    
    target_membership.role = body.role
    db.commit()
    
    return {"message": "Role updated successfully", "user_id": user_id, "role": body.role}


@router.delete("/{workspace_id}/members/{user_id}", response_model=MessageResponse, summary="Remove member")
def remove_member(
    workspace_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a member from workspace. Requires ADMIN or OWNER role."""
    _get_owned_workspace(db, current_user.id, workspace_id)
    membership = _get_membership(db, workspace_id, current_user.id)
    
    if not membership or not membership.is_admin:
        raise HTTPException(status_code=403, detail="Admin or Owner role required")
    
    target_membership = _get_membership(db, workspace_id, user_id)
    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if target_membership.is_owner:
        raise HTTPException(status_code=400, detail="Cannot remove workspace owner")
    
    db.delete(target_membership)
    db.commit()
    
    return MessageResponse(message="Member removed successfully")
