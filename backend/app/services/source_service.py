"""Message source persistence service.

Converts trusted RAG SourceReference results into MessageSource
database records for persistence alongside assistant messages.

Design decisions:
- Sources are created ONLY from server-side RAG results (never from client input)
- Sources reference existing Document and DocumentChunk records
- Ordering is preserved from RAG results, stored with chunk_index for deterministic sort
- Zero sources = zero records (no fabrication)
- Duplicate (document_id, chunk_id) pairs are deduplicated (first occurrence wins)
- Source count is bounded by MAX_MESSAGE_SOURCES
"""

import logging
from typing import List, Set, Tuple

from sqlalchemy.orm import Session

from ..models.message_source import MessageSource
from ..services.rag_service import SourceReference

logger = logging.getLogger(__name__)

# Maximum number of sources to persist per assistant message.
# Matches the retrieval pipeline's upper bound.
MAX_MESSAGE_SOURCES = 20


def persist_sources(
    db: Session,
    message_id: int,
    sources: List[SourceReference],
    owner_user_id: int,
) -> List[MessageSource]:
    """Persist RAG source references as MessageSource records.

    Creates MessageSource records for each source returned by RAG.
    Invalid references (non-existent document/chunk) are skipped.
    Duplicate (document_id, chunk_id) pairs are deduplicated.
    Source count is bounded by MAX_MESSAGE_SOURCES.

    Args:
        db: SQLAlchemy session.
        message_id: The assistant message ID these sources belong to.
        sources: List of SourceReference from RAG results.
        owner_user_id: The user ID that owns the conversation/message.
            Used to verify document ownership before persisting.

    Returns:
        List of created MessageSource records.
    """
    from ..models.document import Document
    from ..models.document_chunk import DocumentChunk

    if not sources:
        return []

    records: List[MessageSource] = []
    seen_chunks: Set[Tuple[int, int]] = set()  # (document_id, chunk_id)

    for source in sources:
        # Enforce source count limit
        if len(records) >= MAX_MESSAGE_SOURCES:
            logger.warning(
                "Source limit reached (%d) for message %d, skipping remaining",
                MAX_MESSAGE_SOURCES, message_id,
            )
            break

        # Deduplicate: skip if same (document_id, chunk_id) already persisted
        chunk_key = (source.document_id, source.chunk_id)
        if chunk_key in seen_chunks:
            logger.debug(
                "Skipping duplicate source: doc=%d chunk=%d (message %d)",
                source.document_id, source.chunk_id, message_id,
            )
            continue

        # Validate document exists AND belongs to the owner
        doc = db.query(Document).filter(Document.id == source.document_id).first()
        if not doc:
            logger.warning(
                "Skipping source: document %d not found (message %d)",
                source.document_id, message_id,
            )
            continue

        if doc.user_id != owner_user_id:
            logger.warning(
                "Skipping source: document %d not owned by user %d (message %d)",
                source.document_id, owner_user_id, message_id,
            )
            continue

        # Validate chunk exists
        chunk_exists = db.query(DocumentChunk.id).filter(DocumentChunk.id == source.chunk_id).first()
        if not chunk_exists:
            logger.warning(
                "Skipping source: chunk %d not found (message %d)",
                source.chunk_id, message_id,
            )
            continue

        msg_source = MessageSource(
            message_id=message_id,
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            chunk_index=source.chunk_index,
            page_start=source.page_start,
            page_end=source.page_end,
            similarity_score=source.similarity_score,
        )
        db.add(msg_source)
        records.append(msg_source)
        seen_chunks.add(chunk_key)

    db.flush()  # Assign IDs without committing

    logger.debug(
        "Persisted %d/%d sources for message %d",
        len(records), len(sources), message_id,
    )

    return records
