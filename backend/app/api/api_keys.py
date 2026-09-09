"""API key management endpoints — create, list, revoke, rotate."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.api_key import ApiKey
from ..models.workspace import Workspace
from ..services.api_key_service import (
    API_KEY_SCOPES,
    create_api_key,
    revoke_api_key,
    rotate_api_key,
    validate_scopes,
)
from ..services.permission_service import require_permission

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class ApiKeyCreate(BaseModel):
    workspace_id: int
    name: str
    scopes: list[str] = ["documents:read", "search:read"]
    expires_in_days: Optional[int] = None


class ApiKeyRotate(BaseModel):
    scopes: Optional[list[str]] = None
    expires_in_days: Optional[int] = None


def _key_dict(key: ApiKey, include_secret: bool = False, secret: Optional[str] = None) -> dict:
    result = {
        "id": key.id,
        "name": key.name,
        "prefix": key.prefix,
        "workspace_id": key.workspace_id,
        "scopes": key.scopes,
        "last_used_at": key.last_used_at,
        "expires_at": key.expires_at,
        "revoked_at": key.revoked_at,
        "created_at": key.created_at,
    }
    if include_secret and secret:
        result["key"] = secret  # full key, shown exactly once
    return result


@router.get("/scopes")
def list_scopes(principal: AuthPrincipal = Depends(get_current_principal)):
    """List available API key scopes."""
    return {"scopes": API_KEY_SCOPES}


@router.post("", status_code=201)
def create_key(
    data: ApiKeyCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Create an API key. The full key is returned exactly once."""
    workspace = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    require_permission(db, principal.user.id, "apikey:manage", workspace_id=data.workspace_id)

    try:
        scopes = validate_scopes(data.scopes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    key, full_key = create_api_key(
        db,
        data.workspace_id,
        principal.user.id,
        data.name,
        scopes,
        expires_in_days=data.expires_in_days,
    )
    db.commit()
    db.refresh(key)
    return _key_dict(key, include_secret=True, secret=full_key)


@router.get("")
def list_keys(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """List API keys for a workspace (metadata only, never secrets)."""
    require_permission(db, principal.user.id, "apikey:manage", workspace_id=workspace_id)
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.workspace_id == workspace_id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return {"items": [_key_dict(k) for k in keys]}


@router.post("/{key_id}/revoke")
def revoke_key(
    key_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Revoke an API key immediately (idempotent)."""
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    require_permission(db, principal.user.id, "apikey:manage", workspace_id=key.workspace_id)

    revoke_api_key(db, key_id, principal.user.id)
    db.commit()
    return {"message": "API key revoked"}


@router.post("/{key_id}/rotate")
def rotate_key(
    key_id: int,
    data: ApiKeyRotate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Rotate an API key: revoke old, issue new with same or updated scopes."""
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    require_permission(db, principal.user.id, "apikey:manage", workspace_id=key.workspace_id)

    try:
        if data.scopes is not None:
            validate_scopes(data.scopes)
        new_key, full_key = rotate_api_key(
            db,
            key_id,
            principal.user.id,
            scopes=data.scopes,
            expires_in_days=data.expires_in_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    db.commit()
    db.refresh(new_key)
    return _key_dict(new_key, include_secret=True, secret=full_key)