"""Integration management API — provider listing and connections."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.integration import IntegrationConnection
from ..models.workspace import Workspace
from ..services.integration_service import (
    list_integrations,
    create_connection,
    disconnect_connection,
)
from ..services.permission_service import require_permission

router = APIRouter(prefix="/integrations", tags=["integrations"])


class ConnectionCreate(BaseModel):
    workspace_id: int
    provider: str
    name: str
    config: dict = {}


def _connection_dict(conn: IntegrationConnection) -> dict:
    return {
        "id": conn.id,
        "provider": conn.provider,
        "name": conn.name,
        "status": conn.status,
        "last_connected_at": conn.last_connected_at,
        "last_error": conn.last_error,
        "created_at": conn.created_at,
    }


@router.get("/providers")
def providers(principal: AuthPrincipal = Depends(get_current_principal)):
    """List integration providers with honest availability status."""
    return {"items": list_integrations()}


@router.post("", status_code=201)
def connect(
    data: ConnectionCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Create an integration connection."""
    workspace = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "integration:manage", workspace_id=data.workspace_id)

    try:
        connection = create_connection(
            db,
            data.workspace_id,
            principal.user.id,
            data.provider,
            data.name,
            data.config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    db.commit()
    db.refresh(connection)
    return _connection_dict(connection)


@router.get("")
def list_connections(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_permission(db, principal.user.id, "integration:manage", workspace_id=workspace_id)
    connections = (
        db.query(IntegrationConnection)
        .filter(IntegrationConnection.workspace_id == workspace_id)
        .order_by(IntegrationConnection.created_at.desc())
        .all()
    )
    return {"items": [_connection_dict(c) for c in connections]}


@router.delete("/{connection_id}")
def disconnect(
    connection_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    connection = db.query(IntegrationConnection).filter(IntegrationConnection.id == connection_id).first()
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    require_permission(db, principal.user.id, "integration:manage", workspace_id=connection.workspace_id)
    disconnect_connection(db, connection_id, principal.user.id)
    db.commit()
    return {"message": "Connection disconnected"}