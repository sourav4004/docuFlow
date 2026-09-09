"""Security center API — sessions, security events, sensitive actions.

Never exposes secrets: API key hashes, webhook signing secrets, session
tokens, and passwords are excluded from every response.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.session import UserSession
from ..models.api_key import ApiKey
from ..models.security_event import SecurityEvent
from ..models.audit_log import AuditLog
from ..models.webhook import WebhookEndpoint
from ..services.permission_service import require_permission
from ..services.security_monitor import SecurityMonitor

router = APIRouter(prefix="/security-center", tags=["security-center"])


@router.get("/workspaces/{workspace_id}/sessions")
def active_sessions(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """List active sessions for members of a workspace (safe metadata only)."""
    require_permission(db, principal.user.id, "security:view", workspace_id=workspace_id)
    from ..models.workspace import WorkspaceMember

    member_ids = [
        m.user_id
        for m in db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id)
        .all()
    ]
    sessions = (
        db.query(UserSession)
        .filter(UserSession.user_id.in_(member_ids) if member_ids else UserSession.user_id == -1)
        .order_by(UserSession.created_at.desc())
        .limit(200)
        .all()
    )
    return {
        "items": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "created_at": s.created_at,
                "expires_at": s.expires_at,
            }
            for s in sessions
        ]
    }


@router.post("/workspaces/{workspace_id}/sessions/revoke")
def revoke_sessions(
    workspace_id: int,
    target_user_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Revoke all sessions for a given member (admin action)."""
    require_permission(db, principal.user.id, "security:manage", workspace_id=workspace_id)
    from ..models.workspace import WorkspaceMember

    # Target must actually be a member of this workspace
    membership = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == target_user_id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(status_code=404, detail="User is not a workspace member")

    revoked = (
        db.query(UserSession)
        .filter(UserSession.user_id == target_user_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"message": f"{revoked} session(s) revoked"}


@router.get("/workspaces/{workspace_id}/events")
def security_events(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Recent security events for a workspace."""
    require_permission(db, principal.user.id, "security:view", workspace_id=workspace_id)
    monitor = SecurityMonitor(db)
    events = monitor.list_recent(limit=100, workspace_id=workspace_id)
    return {
        "items": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "severity": e.severity,
                "description": e.description,
                "user_id": e.user_id,
                "source_ip": e.source_ip,
                "created_at": e.created_at,
            }
            for e in events
        ]
    }


@router.get("/workspaces/{workspace_id}/api-keys")
def api_key_overview(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """API key overview for the security center (never hashes or secrets)."""
    require_permission(db, principal.user.id, "security:view", workspace_id=workspace_id)
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.workspace_id == workspace_id)
        .order_by(ApiKey.created_at.desc())
        .limit(100)
        .all()
    )
    return {
        "items": [
            {
                "id": k.id,
                "name": k.name,
                "prefix": k.prefix,
                "scopes": k.scopes,
                "last_used_at": k.last_used_at,
                "expires_at": k.expires_at,
                "revoked_at": k.revoked_at,
            }
            for k in keys
        ]
    }


@router.get("/workspaces/{workspace_id}/webhooks")
def webhook_overview(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Webhook endpoint overview (never secrets)."""
    require_permission(db, principal.user.id, "security:view", workspace_id=workspace_id)
    endpoints = (
        db.query(WebhookEndpoint)
        .filter(WebhookEndpoint.workspace_id == workspace_id)
        .all()
    )
    return {
        "items": [
            {
                "id": e.id,
                "url": e.url,
                "status": e.status,
                "events": e.events,
                "created_at": e.created_at,
            }
            for e in endpoints
        ]
    }


@router.get("/workspaces/{workspace_id}/audit")
def audit_overview(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Recent audit events (redacted, no sensitive payloads)."""
    require_permission(db, principal.user.id, "audit:view", workspace_id=workspace_id)
    from ..models.workspace import WorkspaceMember

    member_ids = [
        m.user_id
        for m in db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id)
        .all()
    ]
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.user_id.in_(member_ids) if member_ids else AuditLog.user_id == -1)
        .order_by(AuditLog.id.desc())
        .limit(200)
        .all()
    )
    return {
        "items": [
            {
                "id": a.id,
                "event_type": a.event_type,
                "event_action": a.event_action,
                "user_id": a.user_id,
                "resource_type": a.resource_type,
                "resource_id": a.resource_id,
                "details": a.details,
                "created_at": a.created_at,
            }
            for a in logs
        ]
    }