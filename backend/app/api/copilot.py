"""AI Copilot API — document, collection, and workspace contexts."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.copilot_service import copilot_ask, CopilotScopeError, COPILOT_SCOPES

router = APIRouter(prefix="/copilot", tags=["copilot"])


class CopilotRequest(BaseModel):
    question: str
    scope: str = "WORKSPACE"
    document_id: Optional[int] = None
    collection_id: Optional[int] = None
    workspace_id: Optional[int] = None


@router.post("/ask")
def ask(
    data: CopilotRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Ask the copilot within a scope. Authorization is enforced server-side."""
    if data.scope not in COPILOT_SCOPES:
        raise HTTPException(status_code=400, detail=f"Unknown scope: {data.scope}")
    try:
        result = copilot_ask(
            db,
            principal.user.id,
            data.scope,
            data.question,
            document_id=data.document_id,
            collection_id=data.collection_id,
            workspace_id=data.workspace_id,
        )
    except CopilotScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.post("/document/{document_id}/ask")
def ask_document(
    document_id: int,
    data: CopilotRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    try:
        result = copilot_ask(
            db,
            principal.user.id,
            "DOCUMENT",
            data.question,
            document_id=document_id,
        )
    except CopilotScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.post("/collection/{collection_id}/ask")
def ask_collection(
    collection_id: int,
    data: CopilotRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    try:
        result = copilot_ask(
            db,
            principal.user.id,
            "COLLECTION",
            data.question,
            collection_id=collection_id,
        )
    except CopilotScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.post("/workspace/{workspace_id}/ask")
def ask_workspace(
    workspace_id: int,
    data: CopilotRequest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    try:
        result = copilot_ask(
            db,
            principal.user.id,
            "WORKSPACE",
            data.question,
            workspace_id=workspace_id,
        )
    except CopilotScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result