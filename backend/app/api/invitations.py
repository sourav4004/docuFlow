"""Workspace invitation API endpoints."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.workspace import Workspace, WorkspaceMember, has_permission
from ..models.invitation import WorkspaceInvitation
from ..models.notification import Notification

router = APIRouter(prefix="/invitations", tags=["invitations"])


class InvitationCreate(BaseModel):
    email: EmailStr
    role: str = "MEMBER"


class InvitationResponse(BaseModel):
    id: int
    workspace_id: int
    invited_email: str
    role: str
    status: str
    created_at: datetime
    expires_at: datetime

    class Config:
        from_attributes = True


class InvitationAccept(BaseModel):
    token: str


@router.post("/workspaces/{workspace_id}/invite", response_model=InvitationResponse)
def create_invitation(
    workspace_id: int,
    invitation: InvitationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a workspace invitation."""
    # Check workspace exists
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Check user has ADMIN or OWNER role
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == current_user.id
    ).first()

    if not membership or not has_permission(membership.role, "ADMIN"):
        raise HTTPException(status_code=403, detail="Not authorized to invite members")

    # Check email not already a member
    from ..models.user import User as UserModel
    existing_user = db.query(UserModel).filter(UserModel.email == invitation.email).first()
    if existing_user:
        existing_member = db.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == existing_user.id
        ).first()
        if existing_member:
            raise HTTPException(status_code=400, detail="User is already a member")

    # Check for existing pending invitation
    existing_invite = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.workspace_id == workspace_id,
        WorkspaceInvitation.invited_email == invitation.email,
        WorkspaceInvitation.status == "PENDING"
    ).first()
    if existing_invite:
        raise HTTPException(status_code=400, detail="Invitation already pending")

    # Create invitation with secure token
    token, token_hash = WorkspaceInvitation.create_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    db_invitation = WorkspaceInvitation(
        workspace_id=workspace_id,
        inviter_id=current_user.id,
        invited_email=invitation.email,
        role=invitation.role,
        token_hash=token_hash,
        expires_at=expires_at
    )
    db.add(db_invitation)

    # Create notification for inviter
    notification = Notification(
        user_id=current_user.id,
        title="Invitation Sent",
        message=f"Invitation sent to {invitation.email}",
        notification_type="invitation_sent",
        resource_type="workspace",
        resource_id=workspace_id
    )
    db.add(notification)

    db.commit()
    db.refresh(db_invitation)

    # Return invitation with token (only shown once)
    return InvitationResponse(
        id=db_invitation.id,
        workspace_id=db_invitation.workspace_id,
        invited_email=db_invitation.invited_email,
        role=db_invitation.role,
        status=db_invitation.status,
        created_at=db_invitation.created_at,
        expires_at=db_invitation.expires_at
    )


@router.get("/workspaces/{workspace_id}/invitations")
def list_invitations(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List pending invitations for a workspace."""
    # Check user has ADMIN or OWNER role
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == current_user.id
    ).first()

    if not membership or not has_permission(membership.role, "ADMIN"):
        raise HTTPException(status_code=403, detail="Not authorized")

    invitations = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.workspace_id == workspace_id,
        WorkspaceInvitation.status == "PENDING"
    ).all()

    return {
        "items": [
            {
                "id": inv.id,
                "invited_email": inv.invited_email,
                "role": inv.role,
                "status": inv.status,
                "created_at": inv.created_at,
                "expires_at": inv.expires_at
            }
            for inv in invitations
        ]
    }


@router.post("/accept")
def accept_invitation(
    invitation_data: InvitationAccept,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept a workspace invitation using token."""
    # Find invitation by token hash
    import hashlib
    token_hash = hashlib.sha256(invitation_data.token.encode()).hexdigest()

    invitation = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.token_hash == token_hash
    ).first()

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")

    if not invitation.is_usable:
        raise HTTPException(status_code=400, detail="Invitation is expired or already used")

    # Check email matches
    if invitation.invited_email != current_user.email:
        raise HTTPException(status_code=403, detail="This invitation is for a different email")

    # Add user to workspace
    membership = WorkspaceMember(
        workspace_id=invitation.workspace_id,
        user_id=current_user.id,
        role=invitation.role
    )
    db.add(membership)

    # Mark invitation as accepted
    invitation.accept()

    db.commit()

    return {"message": "Invitation accepted", "workspace_id": invitation.workspace_id}
