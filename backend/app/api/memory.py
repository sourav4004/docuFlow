"""AI memory governance API — inspect, delete, expire, export, reset."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.permission_service import require_workspace_membership, require_permission
from ..services.memory_service import (
    store_memory,
    list_memories,
    get_memory,
    delete_memory,
    expire_memories,
    export_memories,
    reset_workspace_memory,
    memory_summary,
    MemoryValidationError,
)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    workspace_id: int
    memory_type: str
    content: str
    scope: str = "WORKSPACE"
    source: Optional[str] = None
    confidence: str = "MEDIUM"
    expires_in_days: Optional[int] = None


@router.post("", status_code=201)
def create_memory(
    data: MemoryCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, data.workspace_id, principal.user.id)
    expires_at = None
    if data.expires_in_days:
        from datetime import datetime, timezone, timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_in_days)
    try:
        memory = store_memory(
            db, data.workspace_id, principal.user.id,
            data.memory_type, data.content, scope=data.scope,
            source=data.source, confidence=data.confidence, expires_at=expires_at,
        )
    except MemoryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "scope": memory.scope,
        "content": memory.content,
        "confidence": memory.confidence,
        "expires_at": memory.expires_at,
    }


@router.get("")
def list_memory(
    workspace_id: int,
    scope: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    items = list_memories(
        db, workspace_id, user_id=principal.user.id,
        scope=scope, memory_type=memory_type, limit=min(limit, 200),
    )
    return {
        "items": [
            {
                "id": m.id,
                "memory_type": m.memory_type,
                "scope": m.scope,
                "content": m.content,
                "source": m.source,
                "confidence": m.confidence,
                "created_at": m.created_at,
                "expires_at": m.expires_at,
            }
            for m in items
        ]
    }


@router.get("/summary")
def memory_summary_endpoint(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return memory_summary(db, workspace_id)


@router.delete("/{memory_id}")
def delete_memory_endpoint(
    memory_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    from ..models.phase15 import AIMemory
    memory = db.query(AIMemory).filter(AIMemory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    require_workspace_membership(db, memory.workspace_id, principal.user.id)
    deleted = delete_memory(db, memory.workspace_id, memory_id, principal.user.id)
    db.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory deleted"}


@router.get("/export")
def export_memory(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return {"items": export_memories(db, workspace_id, user_id=principal.user.id)}


@router.post("/maintenance/expire")
def expire_memory_endpoint(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "ai:manage_settings")
    count = expire_memories(db)
    db.commit()
    return {"expired": count}


@router.post("/workspaces/{workspace_id}/reset")
def reset_memory(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Admin action: delete ALL workspace memory (audited)."""
    require_permission(db, principal.user.id, "ai:manage_settings", workspace_id=workspace_id)
    count = reset_workspace_memory(db, workspace_id, principal.user.id)
    db.commit()
    return {"deleted": count}