"""API key service — secure key lifecycle, hashing, scopes, and verification."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.api_key import ApiKey
from ..models.audit_log import AuditLog
from ..services.audit_service import log_audit_event

# Well-known API key scopes
API_KEY_SCOPES = [
    "documents:read",
    "documents:write",
    "collections:read",
    "collections:write",
    "search:read",
    "conversations:read",
    "conversations:write",
    "ai:execute",
    "agents:execute",
    "workflows:execute",
    "usage:read",
    "webhooks:manage",
]


def validate_scopes(scopes: list[str]) -> list[str]:
    """Validate requested scopes against the well-known set."""
    valid = set(API_KEY_SCOPES)
    result = []
    for scope in scopes or []:
        if scope == "*":
            result.append("*")
        elif scope in valid:
            result.append(scope)
        else:
            raise ValueError(f"Unknown API key scope: {scope}")
    return result


def create_api_key(
    db: Session,
    workspace_id: int,
    user_id: int,
    name: str,
    scopes: list[str],
    expires_in_days: Optional[int] = None,
) -> tuple[ApiKey, str]:
    """Create a new API key.

    Returns:
        (api_key_model, full_key) — the full key is shown exactly once.
    """
    full_key, prefix, key_hash = ApiKey.generate_key()
    expires_at = None
    if expires_in_days:
        expires_at = datetime.now(timezone.utc).replace(microsecond=0)
        from datetime import timedelta
        expires_at = expires_at + timedelta(days=expires_in_days)

    key = ApiKey(
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        prefix=prefix,
        key_hash=key_hash,
        scopes_json=json.dumps(validate_scopes(scopes)),
        expires_at=expires_at,
    )
    db.add(key)
    db.flush()

    log_audit_event(
        db,
        event_type="api_key",
        event_action="create",
        user_id=user_id,
        resource_type="api_key",
        resource_id=key.id,
        details=f"API key '{name}' created",
    )
    return key, full_key


def authenticate_api_key(db: Session, key: str) -> Optional[ApiKey]:
    """Look up an API key by its hash.

    Returns the key if valid and active, otherwise None. Revoked/expired
    keys return None so they can never consume expensive resources.
    """
    key_hash = ApiKey.hash_key(key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if not api_key or not api_key.is_active:
        return None
    return api_key


def touch_api_key(db: Session, api_key: ApiKey) -> None:
    """Update last_used_at without re-committing the caller's transaction."""
    api_key.last_used_at = datetime.now(timezone.utc)
    db.flush()


def revoke_api_key(db: Session, api_key_id: int, revoked_by: int) -> Optional[ApiKey]:
    """Revoke an API key. Idempotent — revoking twice is safe."""
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not api_key:
        return None
    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.now(timezone.utc)
        api_key.revoked_by = revoked_by
        log_audit_event(
            db,
            event_type="api_key",
            event_action="revoke",
            user_id=revoked_by,
            resource_type="api_key",
            resource_id=api_key_id,
            details=f"API key '{api_key.name}' revoked",
        )
    return api_key


def rotate_api_key(
    db: Session,
    api_key_id: int,
    user_id: int,
    scopes: Optional[list[str]] = None,
    expires_in_days: Optional[int] = None,
) -> tuple[ApiKey, str]:
    """Rotate an API key: revoke the old one and issue a new key."""
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not api_key:
        raise ValueError("API key not found")
    new_scopes = scopes if scopes is not None else api_key.scopes
    revoke_api_key(db, api_key_id, user_id)
    return create_api_key(
        db,
        api_key.workspace_id,
        user_id,
        api_key.name,
        new_scopes,
        expires_in_days=expires_in_days,
    )


def key_has_scope(api_key: ApiKey, required_scope: str) -> bool:
    """Check whether an API key has a required scope."""
    if not api_key.is_active:
        return False
    return api_key.has_scope(required_scope)