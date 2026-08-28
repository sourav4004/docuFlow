"""Chunk persistence service for document processing.

Handles creation, replacement, retrieval, and deletion of DocumentChunk
records. Ensures atomic replacement on reprocessing.
"""

import logging
from typing import List

from sqlalchemy.orm import Session

from ..models.document_chunk import DocumentChunk
from .chunker import TextChunk

logger = logging.getLogger(__name__)


def persist_chunks(db: Session, document_id: int, chunks: List[TextChunk]) -> None:
    """Atomically replace all chunks for a document.

    Deletes existing chunks (if any) and inserts the new set in a single
    transaction. If the caller's session commits successfully, the
    document's chunks are fully updated. On failure the caller should
    rollback — this function does NOT commit.

    Args:
        db: SQLAlchemy session (caller owns commit/rollback).
        document_id: The document these chunks belong to.
        chunks: Ordered list of TextChunk objects from the chunker.
    """
    # Delete existing chunks for this document
    existing_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).delete()

    if existing_count > 0:
        logger.info(
            "persist_chunks: removed %d existing chunks for doc %s",
            existing_count, document_id,
        )

    # Insert new chunks in bulk
    if not chunks:
        logger.debug("persist_chunks: no chunks to persist for doc %s", document_id)
        return

    db_objects = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk in chunks
    ]
    db.add_all(db_objects)

    logger.info(
        "persist_chunks: persisted %d chunks for doc %s",
        len(db_objects), document_id,
    )


def get_chunks_for_document(db: Session, document_id: int) -> List[DocumentChunk]:
    """Retrieve all chunks for a document, ordered by chunk_index.

    Args:
        db: SQLAlchemy session.
        document_id: The document to retrieve chunks for.

    Returns:
        Ordered list of DocumentChunk objects.
    """
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )


def delete_chunks_for_document(db: Session, document_id: int) -> int:
    """Delete all chunks for a document.

    Args:
        db: SQLAlchemy session (caller owns commit).
        document_id: The document whose chunks to delete.

    Returns:
        Number of chunks deleted.
    """
    count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).delete()
    if count > 0:
        logger.debug("delete_chunks_for_document: removed %d chunks for doc %s", count, document_id)
    return count
