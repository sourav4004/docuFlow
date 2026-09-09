"""AI feedback endpoints — thumbs up/down, categories, workspace analytics."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.feedback_service import submit_feedback, workspace_feedback_analytics


router = APIRouter(prefix="/feedback", tags=["ai-feedback"])


class FeedbackRequest(BaseModel):
    rating: str = Field(..., pattern="^(thumbs_up|thumbs_down)$")
    execution_id: Optional[str] = None
    category: Optional[str] = None
    comment: Optional[str] = None


@router.post("")
def create_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Submit feedback on an AI answer. Tenant scoped to the current workspace."""
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        feedback = submit_feedback(
            db,
            workspace_id=workspace.id,
            user_id=user.id,
            rating=payload.rating,
            execution_id=payload.execution_id,
            category=payload.category,
            comment=payload.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"id": feedback.id, "rating": feedback.rating, "status": "recorded"}


@router.get("/analytics")
def feedback_analytics(
    days: int = 30,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Workspace-level feedback analytics (tenant scoped)."""
    from ..services.workspace_service import get_current_workspace, WorkspaceResolutionError

    try:
        workspace = get_current_workspace(db, user.id)
    except WorkspaceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return workspace_feedback_analytics(db, workspace.id, days=days)