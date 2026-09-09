"""Search intelligence API — NL search, saved searches, search alerts."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.search_intel import SavedSearch, SearchAlert
from ..models.workspace import Workspace
from ..services.nl_search import (
    detect_intent,
    extract_filters,
    explain_matches,
    create_saved_search,
    list_saved_searches,
    delete_saved_search,
    create_search_alert,
    run_search_alerts,
)
from ..services.permission_service import require_permission

router = APIRouter(prefix="/search", tags=["search-intel"])


class SavedSearchCreate(BaseModel):
    workspace_id: int
    name: str
    query: str
    filters: Optional[dict] = None
    is_shared: bool = False


def _require_workspace(db, workspace_id, principal):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "document:read", workspace_id=workspace_id)
    return workspace


@router.post("/intent")
def analyze_query(
    query: str,
    principal: AuthPrincipal = Depends(get_current_principal),
):
    """Deterministic intent + filter analysis (no LLM)."""
    return {
        "query": query,
        "intents": detect_intent(query),
        "filters": extract_filters(query),
        "explanations": explain_matches(query, "semantic"),
    }


@router.post("/explain")
def explain(
    query: str,
    result_kind: str = "semantic",
    principal: AuthPrincipal = Depends(get_current_principal),
):
    return {"query": query, "reasons": explain_matches(query, result_kind)}


# --- Saved searches ---


@router.post("/saved", status_code=201)
def save_search(
    data: SavedSearchCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, data.workspace_id, principal)
    saved = create_saved_search(
        db,
        data.workspace_id,
        principal.user.id,
        data.name,
        data.query,
        filters=data.filters,
        is_shared=data.is_shared,
    )
    db.commit()
    db.refresh(saved)
    return {
        "id": saved.id,
        "name": saved.name,
        "query": saved.query,
        "filters": saved.filters,
        "is_shared": saved.is_shared,
        "created_at": saved.created_at,
    }


@router.get("/saved")
def list_saved(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    items = list_saved_searches(db, workspace_id, principal.user.id)
    return {
        "items": [
            {
                "id": s.id,
                "name": s.name,
                "query": s.query,
                "filters": s.filters,
                "is_shared": s.is_shared,
                "owner_id": s.owner_id,
                "created_at": s.created_at,
            }
            for s in items
        ]
    }


@router.delete("/saved/{search_id}")
def delete_saved(
    search_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    saved = db.query(SavedSearch).filter(SavedSearch.id == search_id).first()
    if not saved:
        raise HTTPException(status_code=404, detail="Saved search not found")
    require_permission(db, principal.user.id, "document:read", workspace_id=saved.workspace_id)
    if saved.owner_id != principal.user.id and not saved.is_shared:
        raise HTTPException(status_code=403, detail="Not allowed to delete this search")
    delete_saved_search(db, saved)
    db.commit()
    return {"message": "Saved search deleted"}


# --- Search alerts ---


@router.post("/saved/{search_id}/alert", status_code=201)
def enable_alert(
    search_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    saved = db.query(SavedSearch).filter(SavedSearch.id == search_id).first()
    if not saved:
        raise HTTPException(status_code=404, detail="Saved search not found")
    require_permission(db, principal.user.id, "document:read", workspace_id=saved.workspace_id)
    if saved.owner_id != principal.user.id:
        raise HTTPException(status_code=403, detail="Only the owner can enable alerts")
    alert = create_search_alert(db, saved.workspace_id, search_id)
    db.commit()
    db.refresh(alert)
    return {"id": alert.id, "saved_search_id": alert.saved_search_id, "is_active": alert.is_active}


@router.post("/alerts/{alert_id}/toggle")
def toggle_alert(
    alert_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    alert = db.query(SearchAlert).filter(SearchAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    require_permission(db, principal.user.id, "document:read", workspace_id=alert.workspace_id)
    alert.is_active = not alert.is_active
    db.commit()
    return {"id": alert.id, "is_active": alert.is_active}


@router.post("/alerts/run")
def run_alerts(
    workspace_id: int,
    match_counts: dict,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Process alert matches (saved_search_id -> match count) and notify."""
    _require_workspace(db, workspace_id, principal)
    int_counts = {int(k): int(v) for k, v in match_counts.items()}
    notifications = run_search_alerts(
        db,
        workspace_id,
        int_counts,
        user_ids=[principal.user.id],
    )
    db.commit()
    return {"notifications_created": len(notifications)}