"""Multi-document research mode endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.research_service import run_research, list_research_briefs
from ..services.llm.service import LLMService


router = APIRouter(prefix="/research", tags=["ai-research"])


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    organization_id: Optional[int] = None
    save_artifact: bool = True


@router.post("")
def research(
    payload: ResearchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Run a multi-document research query scoped to the current workspace."""
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    llm_service = LLMService()
    result = run_research(
        db,
        user_id=user.id,
        workspace_id=workspace.id,
        question=payload.question,
        organization_id=payload.organization_id,
        llm_service=llm_service,
        save_artifact=payload.save_artifact,
    )
    db.commit()
    return result


@router.get("/briefs")
def research_briefs(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List saved research briefs."""
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    briefs = list_research_briefs(db, workspace.id, limit=limit)
    return {
        "briefs": [
            {
                "id": b.id,
                "name": b.name,
                "version": b.version,
                "status": b.status,
                "created_at": b.created_at,
            }
            for b in briefs
        ]
    }