"""Document comment API endpoints."""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.document import Document
from ..models.document_comment import DocumentComment

router = APIRouter(prefix="/documents/{document_id}/comments", tags=["comments"])


class CommentCreate(BaseModel):
    content: str
    parent_id: Optional[int] = None


class CommentUpdate(BaseModel):
    content: str


@router.get("")
def list_comments(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List comments for a document."""
    # Check document exists and user has access
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    comments = db.query(DocumentComment).filter(
        DocumentComment.document_id == document_id,
        DocumentComment.parent_id.is_(None)  # Top-level comments only
    ).order_by(DocumentComment.created_at.desc()).all()

    def format_comment(comment):
        return {
            "id": comment.id,
            "content": comment.content,
            "user_id": comment.user_id,
            "is_resolved": comment.is_resolved,
            "created_at": comment.created_at,
            "replies": [format_comment(reply) for reply in comment.replies]
        }

    return {"items": [format_comment(c) for c in comments]}


@router.post("", status_code=201)
def create_comment(
    document_id: int,
    comment: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a comment on a document."""
    # Check document exists and user has access
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Check parent comment if provided
    if comment.parent_id:
        parent = db.query(DocumentComment).filter(
            DocumentComment.id == comment.parent_id,
            DocumentComment.document_id == document_id
        ).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent comment not found")

    db_comment = DocumentComment(
        document_id=document_id,
        user_id=current_user.id,
        parent_id=comment.parent_id,
        content=comment.content
    )
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)

    return {
        "id": db_comment.id,
        "content": db_comment.content,
        "user_id": db_comment.user_id,
        "created_at": db_comment.created_at
    }


@router.put("/{comment_id}")
def update_comment(
    document_id: int,
    comment_id: int,
    update: CommentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update own comment."""
    comment = db.query(DocumentComment).filter(
        DocumentComment.id == comment_id,
        DocumentComment.document_id == document_id,
        DocumentComment.user_id == current_user.id
    ).first()

    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    comment.content = update.content
    comment.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"message": "Comment updated"}


@router.delete("/{comment_id}")
def delete_comment(
    document_id: int,
    comment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete own comment."""
    comment = db.query(DocumentComment).filter(
        DocumentComment.id == comment_id,
        DocumentComment.document_id == document_id,
        DocumentComment.user_id == current_user.id
    ).first()

    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    db.delete(comment)
    db.commit()

    return {"message": "Comment deleted"}


@router.post("/{comment_id}/resolve")
def resolve_comment(
    document_id: int,
    comment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Resolve a comment."""
    comment = db.query(DocumentComment).filter(
        DocumentComment.id == comment_id,
        DocumentComment.document_id == document_id
    ).first()

    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    # Only document owner or comment author can resolve
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc.user_id != current_user.id and comment.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    comment.resolve()
    db.commit()

    return {"message": "Comment resolved"}
