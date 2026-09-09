"""Knowledge intelligence API — health scores and workspace analytics."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.document import Document
from ..models.collection import Collection
from ..models.workspace import Workspace
from ..services.health_service import (
    compute_document_health,
    get_document_health,
    workspace_health,
    collection_health,
    ai_usage_summary,
)
from ..services.permission_service import require_permission, require_workspace_membership

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _require_workspace(db, workspace_id, principal) -> Workspace:
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "document:read", workspace_id=workspace_id)
    return workspace


def _health_dict(h) -> dict:
    import json
    return {
        "document_id": h.document_id,
        "score": h.score,
        "factors": json.loads(h.factors_json) if h.factors_json else {},
        "reasons": json.loads(h.reasons_json) if h.reasons_json else [],
        "computed_at": h.computed_at,
    }


@router.post("/documents/{document_id}/health")
def refresh_document_health(
    document_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Compute/refresh the health score for a document."""
    from ..services.workspace_service import resolve_document_workspace
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    require_workspace_membership(db, workspace_id, principal.user.id)
    health = compute_document_health(db, doc)
    db.commit()
    db.refresh(health)
    return _health_dict(health)


@router.get("/documents/{document_id}/health")
def get_health(
    document_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..services.workspace_service import resolve_document_workspace
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace_id = resolve_document_workspace(db, doc, principal.user.id)
    require_workspace_membership(db, workspace_id, principal.user.id)
    health = get_document_health(db, document_id)
    if not health:
        raise HTTPException(status_code=404, detail="Health not computed yet")
    return _health_dict(health)


@router.get("/workspaces/{workspace_id}/health")
def get_workspace_health(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    health = workspace_health(db, workspace_id)
    health["ai_usage"] = ai_usage_summary(db, workspace_id)
    return health


@router.get("/collections/{collection_id}/health")
def get_collection_health(
    collection_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    require_workspace_membership(db, collection.workspace_id, principal.user.id)
    return collection_health(db, collection)


@router.get("/workspaces/{workspace_id}/collections-health")
def all_collection_health(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    collections = db.query(Collection).filter(Collection.workspace_id == workspace_id).limit(100).all()
    return {"items": [collection_health(db, c) for c in collections]}