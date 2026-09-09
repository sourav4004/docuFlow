"""Collection (workspace) API endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.collection import Collection
from ..models.document import Document
from ..schemas.auth import MessageResponse as SuccessMessageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


class CollectionCreate(BaseModel):
    """Schema for collection creation."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=500)


class CollectionUpdate(BaseModel):
    """Schema for collection update."""
    name: str = Field(default=None, min_length=1, max_length=255)
    description: str = Field(default=None, max_length=500)


class CollectionResponse:
    """Response schema for a collection."""
    def __init__(self, collection: Collection, document_ids: list = None):
        self.id = collection.id
        self.name = collection.name
        self.description = collection.description or ""
        self.document_ids = document_ids or [d.id for d in collection.documents]
        self.document_count = len(self.document_ids)
        self.created_at = collection.created_at
        self.updated_at = collection.updated_at

    def model_dump(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "document_ids": self.document_ids,
            "document_count": self.document_count,
            "created_at": str(self.created_at) if self.created_at else None,
            "updated_at": str(self.updated_at) if self.updated_at else None,
        }


class CollectionListResponse:
    """Response schema for collection list."""
    def __init__(self, items, total, limit, offset, has_next, has_previous):
        self.items = items
        self.total = total
        self.limit = limit
        self.offset = offset
        self.has_next = has_next
        self.has_previous = has_previous

    def model_dump(self):
        return {
            "items": [i.model_dump() for i in self.items],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_next": self.has_next,
            "has_previous": self.has_previous,
        }


def _get_owned_collection(db, user_id, collection_id):
    """Get a collection with ownership check."""
    col = db.query(Collection).filter(
        Collection.id == collection_id,
        Collection.user_id == user_id,
    ).first()
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


@router.post("", status_code=201, summary="Create a collection")
def create_collection(body: CollectionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new collection for the authenticated user."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name is required")

    col = Collection(
        user_id=current_user.id,
        name=name,
        description=body.description.strip() if body.description else "",
    )
    db.add(col)
    db.commit()
    db.refresh(col)
    return CollectionResponse(col).model_dump()


@router.get("", summary="List collections")
def list_collections(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List collections belonging to the authenticated user."""
    query = db.query(Collection).filter(Collection.user_id == current_user.id)
    total = query.count()
    items = query.order_by(Collection.updated_at.desc()).offset(offset).limit(limit).all()
    return CollectionListResponse(
        items=[CollectionResponse(c) for c in items],
        total=total, limit=limit, offset=offset,
        has_next=offset + limit < total,
        has_previous=offset > 0,
    ).model_dump()


@router.get("/{collection_id}", summary="Get a collection")
def get_collection(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a collection with its document IDs."""
    col = _get_owned_collection(db, current_user.id, collection_id)
    return CollectionResponse(col).model_dump()


@router.get("/{collection_id}/details", summary="Get collection details with documents")
def get_collection_details(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get collection with full document details and statistics."""
    col = _get_owned_collection(db, current_user.id, collection_id)

    documents = []
    total_size = 0
    ready_count = 0
    failed_count = 0

    for doc in col.documents:
        documents.append({
            "id": doc.id,
            "filename": doc.original_filename,
            "status": doc.status,
            "file_size": doc.file_size,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        })
        total_size += doc.file_size or 0
        if doc.status == "READY":
            ready_count += 1
        elif doc.status == "FAILED":
            failed_count += 1

    return {
        "id": col.id,
        "name": col.name,
        "description": col.description or "",
        "document_count": len(documents),
        "documents": documents,
        "statistics": {
            "total_size_bytes": total_size,
            "ready_count": ready_count,
            "failed_count": failed_count,
            "processing_count": len(documents) - ready_count - failed_count,
        },
        "created_at": str(col.created_at) if col.created_at else None,
        "updated_at": str(col.updated_at) if col.updated_at else None,
    }


@router.patch("/{collection_id}", summary="Update a collection")
def update_collection(
    collection_id: int,
    body: CollectionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a collection's name or description."""
    col = _get_owned_collection(db, current_user.id, collection_id)
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")
        col.name = name
    if body.description is not None:
        col.description = body.description.strip()
    db.commit()
    db.refresh(col)
    return CollectionResponse(col).model_dump()


@router.delete("/{collection_id}", summary="Delete a collection")
def delete_collection(
    collection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a collection. Documents are NOT deleted — only the association."""
    col = _get_owned_collection(db, current_user.id, collection_id)
    db.delete(col)
    db.commit()
    return {"message": "Collection deleted"}


@router.post("/{collection_id}/documents/{document_id}", summary="Add document to collection")
def add_document_to_collection(
    collection_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a document to a collection. Both must belong to the user."""
    col = _get_owned_collection(db, current_user.id, collection_id)
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc in col.documents:
        return {"message": "Document already in collection"}

    col.documents.append(doc)
    db.commit()
    return {"message": "Document added to collection"}


@router.delete("/{collection_id}/documents/{document_id}", summary="Remove document from collection")
def remove_document_from_collection(
    collection_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a document from a collection. Document is NOT deleted."""
    col = _get_owned_collection(db, current_user.id, collection_id)
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc not in col.documents:
        raise HTTPException(status_code=404, detail="Document not in collection")

    col.documents.remove(doc)
    db.commit()
    return {"message": "Document removed from collection"}
