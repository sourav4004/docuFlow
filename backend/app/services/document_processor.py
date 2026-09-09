"""Document processing orchestration service.

Coordinates the full lifecycle: validation → extraction → storage → status update.
Uses database-backed ProcessingJob for durable job tracking.
"""

import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.document_content import DocumentContent
from ..models.job import ProcessingJob, JobStatus
from .pdf_extractor import extract_text_from_pdf, PDFExtractionError, PDFNotFoundError
from .text_normalizer import normalize_text
from .chunker import chunk_text
from .chunk_service import persist_chunks

logger = logging.getLogger(__name__)

# Status constants
STATUS_UPLOADED = "UPLOADED"
STATUS_QUEUED = "QUEUED"
STATUS_PROCESSING = "PROCESSING"
STATUS_READY = "READY"
STATUS_FAILED = "FAILED"


def process_document(db: Session, document_id: int) -> None:
    """Orchestrate document text extraction for a single document.

    Creates/updates a ProcessingJob record and manages the full lifecycle:
        1. Fetch document from DB
        2. Validate state
        3. Create/update job record
        4. Transition to PROCESSING
        5. Extract text via PDF extractor
        6. Create/update DocumentContent row
        7. Transition to READY (or FAILED on error)

    Args:
        db: SQLAlchemy session (caller owns commit/close).
        document_id: The document to process.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.warning("process_document: document %s not found, skipping", document_id)
        return

    # Guard: only process documents in a processable state
    if document.status not in (STATUS_UPLOADED, STATUS_QUEUED, STATUS_FAILED):
        logger.info(
            "process_document: document %s has status '%s', skipping",
            document_id, document.status,
        )
        return

    # Get or create job record
    job = db.query(ProcessingJob).filter(
        ProcessingJob.document_id == document_id,
        ProcessingJob.status.in_([JobStatus.QUEUED.value, JobStatus.PROCESSING.value])
    ).first()

    if not job:
        # Create new job
        job = ProcessingJob(
            document_id=document_id,
            user_id=document.user_id,
            status=JobStatus.QUEUED.value,
            attempts=0,
        )
        db.add(job)
        db.flush()

    # Increment attempts
    job.attempts += 1
    job.started_at = datetime.now(timezone.utc)
    job.error_message = None

    # Transition to PROCESSING
    document.status = STATUS_PROCESSING
    job.status = JobStatus.PROCESSING.value
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("process_document: failed to set PROCESSING for doc %s", document_id)
        return

    # Attempt extraction
    try:
        result = extract_text_from_pdf(document.storage_key)
    except (PDFExtractionError, PDFNotFoundError) as exc:
        _mark_failed(db, document, job, str(exc))
        return
    except Exception as exc:
        _mark_failed(db, document, job, f"Unexpected error: {exc}")
        return

    # Normalize extracted text
    normalized_text = normalize_text(result.text)
    normalized_char_count = len(normalized_text)

    # Chunk the normalized text
    try:
        chunks = chunk_text(normalized_text)
    except Exception as exc:
        _mark_failed(db, document, job, f"Chunking failed: {exc}")
        return

    # Save extracted content + chunks (upsert — replace if retry)
    try:
        existing = db.query(DocumentContent).filter(
            DocumentContent.document_id == document.id
        ).first()

        if existing:
            existing.extracted_text = normalized_text
            existing.page_count = result.page_count
            existing.char_count = normalized_char_count
        else:
            content = DocumentContent(
                document_id=document.id,
                extracted_text=normalized_text,
                page_count=result.page_count,
                char_count=normalized_char_count,
            )
            db.add(content)

        # Persist chunks (atomic replace)
        persist_chunks(db, document.id, chunks)

        document.status = STATUS_READY
        job.status = JobStatus.COMPLETED.value
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            "process_document: document %s processed successfully (%d pages, %d chars, %d chunks)",
            document_id, result.page_count, normalized_char_count, len(chunks),
        )
    except Exception:
        db.rollback()
        _mark_failed(db, document, job, "Failed to save extracted content and chunks")


def _mark_failed(db: Session, document: Document, job: ProcessingJob, error_message: str) -> None:
    """Set document and job status to FAILED and log the error safely."""
    logger.error(
        "process_document: document %s failed — %s",
        document.id, error_message,
    )
    try:
        document.status = STATUS_FAILED
        job.status = JobStatus.FAILED.value
        job.error_message = error_message
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("process_document: could not even mark doc %s as FAILED", document.id)
