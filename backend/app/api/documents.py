"""Document API endpoints."""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Response, status, BackgroundTasks
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.document import Document
from ..schemas.document import DocumentResponse, DocumentListResponse, DocumentStatusResponse, DocumentContentResponse
from ..schemas.auth import MessageResponse
from ..services.storage import storage_service
from ..services.document_processor import process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def validate_pdf_content(content: bytes) -> bool:
    """Validate that the binary content has a valid PDF magic signature."""
    if not content:
        return False
    # PDF magic signature %PDF- must appear within the beginning of the file
    return content.startswith(b"%PDF-") or b"%PDF-" in content[:1024]


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document"
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload and store a document for the authenticated user.

    Validates file presence, size limits, and PDF file signature.
    Stores the physical file securely using StorageService and records metadata in PostgreSQL.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided or invalid filename."
        )

    # Sanitize original filename (prevent path injection in metadata)
    original_filename = Path(file.filename).name.strip()
    if not original_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty."
        )

    # Read file content safely
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file."
        )

    # 1. Validate file size (not empty)
    file_size = len(content)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 2. Validate max upload size
    if file_size > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed limit of {settings.max_upload_size} bytes."
        )

    # 3. Validate file signature / type
    lower_filename = original_filename.lower()
    is_pdf_ext = lower_filename.endswith(".pdf")
    is_pdf_content_type = file.content_type == "application/pdf"

    if not is_pdf_ext and not is_pdf_content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents (.pdf) are supported."
        )

    # Verify actual PDF magic bytes signature (do not trust client Content-Type)
    if not validate_pdf_content(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF content: file content signature does not match PDF format."
        )

    mime_type = "application/pdf"

    # 4. Save to StorageService
    storage_key = storage_service.generate_storage_key()
    try:
        saved_key = storage_service.save(content, storage_key=storage_key)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document file."
        )

    # 5. Save metadata record to Database and trigger processing
    try:
        document = Document(
            user_id=current_user.id,
            original_filename=original_filename,
            storage_key=saved_key,
            mime_type=mime_type,
            file_size=file_size,
            status="UPLOADED"
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # Queue for background processing
        doc_id = document.id
        background_tasks.add_task(_background_process, doc_id)

        return document
    except Exception:
        db.rollback()
        # Clean up physical file on database failure to prevent orphaned files
        try:
            storage_service.delete(saved_key)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save document metadata."
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List authenticated user's documents"
)
def list_documents(
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve a paginated list of documents owned by the authenticated user.
    Database-level filtering ensures user isolation.
    """
    query = db.query(Document).filter(Document.user_id == current_user.id)
    total = query.count()
    items = query.order_by(Document.created_at.desc(), Document.id.desc()).offset(offset).limit(limit).all()

    return DocumentListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document metadata by ID"
)
def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve metadata for a specific document owned by the authenticated user.
    Returns 404 if document does not exist or belongs to another user.
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return document


@router.get(
    "/{document_id}/file",
    status_code=status.HTTP_200_OK,
    summary="Download/stream document file content"
)
def get_document_file(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve the physical file content for a specific document owned by the authenticated user.
    Returns 404 if document not found, unauthorized, or physical file missing.
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        content = storage_service.retrieve(document.storage_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found in storage"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key detected"
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file from storage"
        )

    return Response(
        content=content,
        media_type=document.mime_type,
        headers={
            "Content-Disposition": f'inline; filename="{document.original_filename}"',
            "Content-Length": str(len(content)),
        }
    )


@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document"
)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a document and its underlying storage file.
    Returns 404 if document does not exist or belongs to another user.
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    storage_key = document.storage_key

    # Delete from database first in transaction
    try:
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document from database."
        )

    # Delete physical file gracefully
    try:
        storage_service.delete(storage_key)
    except Exception:
        # Physical file may already be removed or missing; do not fail the request
        pass

    return MessageResponse(message="Document deleted successfully")


# ---------------------------------------------------------------------------
# Background processing helper
# ---------------------------------------------------------------------------

def _background_process(doc_id: int) -> None:
    """Run document processing in a background thread.

    Uses a fresh DB session since the request session will be closed.
    This keeps the API route thin and non-blocking.
    """
    try:
        from ..core.database import SessionLocal
        db = SessionLocal()
    except Exception:
        logger.warning("Background processing skipped for doc %s: cannot create DB session", doc_id)
        return
    try:
        process_document(db, doc_id)
    except Exception:
        logger.exception("Unhandled error in background processing for doc %s", doc_id)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Processing status endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document processing status"
)
def get_document_status(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return processing status for a document owned by the authenticated user."""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return DocumentStatusResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


# ---------------------------------------------------------------------------
# Extracted content endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/{document_id}/content",
    response_model=DocumentContentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get extracted text content"
)
def get_document_content(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return extracted text for a document owned by the authenticated user."""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    if document.status == "FAILED":
        return DocumentContentResponse(
            document_id=document.id,
            status="FAILED",
            extracted_text=None,
            error_message="Document processing failed. You may retry processing.",
        )

    if document.status in ("UPLOADED", "QUEUED", "PROCESSING"):
        return DocumentContentResponse(
            document_id=document.id,
            status=document.status,
            extracted_text=None,
            error_message=None,
        )

    # Status is READY — retrieve content
    if not document.content:
        return DocumentContentResponse(
            document_id=document.id,
            status=document.status,
            extracted_text=None,
            error_message="No extracted content available.",
        )

    return DocumentContentResponse(
        document_id=document.id,
        status=document.status,
        extracted_text=document.content.extracted_text,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# Retry processing endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{document_id}/process",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retry document processing"
)
def retry_processing(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Re-queue a document for processing. Owner-only."""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Prevent duplicate processing
    if document.status == "PROCESSING":
        return DocumentStatusResponse(
            document_id=document.id,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    if document.status == "QUEUED":
        return DocumentStatusResponse(
            document_id=document.id,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    # Transition to QUEUED
    document.status = "QUEUED"
    db.commit()
    db.refresh(document)

    # Start background processing
    background_tasks.add_task(_background_process, document.id)

    return DocumentStatusResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )
