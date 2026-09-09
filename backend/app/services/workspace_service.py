"""Workspace resolution helpers.

Phase 14 copilots, feedback, reports, and research endpoints operate in a
workspace context. These helpers resolve the user's workspace deterministically
(membership first, then ownership) without requiring a workspace_id parameter.
"""

from typing import Optional

from sqlalchemy.orm import Session

from ..models.workspace import Workspace, WorkspaceMember


class WorkspaceResolutionError(Exception):
    """Raised when no workspace can be resolved for a user."""


def get_current_workspace(db: Session, user_id: int) -> Workspace:
    """Resolve the user's current workspace.

    Prefers the most recently joined membership; falls back to owned
    workspaces. Raises WorkspaceResolutionError when the user has none.
    """
    membership = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.user_id == user_id)
        .order_by(WorkspaceMember.id.desc())
        .first()
    )
    if membership:
        workspace = db.query(Workspace).filter(Workspace.id == membership.workspace_id).first()
        if workspace:
            return workspace

    workspace = (
        db.query(Workspace)
        .filter(Workspace.owner_id == user_id)
        .order_by(Workspace.id.asc())
        .first()
    )
    if workspace:
        return workspace

    raise WorkspaceResolutionError("No workspace available for this user")


def get_user_workspaces(db: Session, user_id: int) -> list[Workspace]:
    """List all workspaces the user can access (membership or ownership)."""
    member_rows = (
        db.query(WorkspaceMember.workspace_id)
        .filter(WorkspaceMember.user_id == user_id)
        .all()
    )
    member_ids = [r[0] for r in member_rows]

    owned_ids = [
        w.id
        for w in db.query(Workspace).filter(Workspace.owner_id == user_id).all()
    ]

    all_ids = list(dict.fromkeys(member_ids + owned_ids))
    if not all_ids:
        return []
    return (
        db.query(Workspace)
        .filter(Workspace.id.in_(all_ids))
        .order_by(Workspace.id.asc())
        .all()
    )


def resolve_document_workspace(db: Session, document, user_id: int) -> int:
    """Resolve the effective workspace for a document, verifying authorization.

    Workspace-scoped documents (Phase 14+) use their workspace membership.
    Legacy user-scoped documents (workspace_id NULL) require document
    ownership and fall back to the user's current workspace. Never grants
    cross-tenant access.
    """
    from fastapi import HTTPException
    if document.workspace_id is not None:
        return document.workspace_id
    if getattr(document, "user_id", None) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        workspace = get_current_workspace(db, user_id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return workspace.id


def workspace_membership_role(db: Session, workspace_id: int, user_id: int) -> Optional[str]:
    """Return the user's role in a workspace, or None if not a member."""
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
    )
    if member:
        return member.role
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace and workspace.owner_id == user_id:
        return "OWNER"
    return None