"""Embedding generation service for document chunks.

Coordinates embedding generation and persistence for DocumentChunk records.
Designed as a separate service — NOT integrated into the Phase 3 processing
pipeline to avoid breaking existing functionality.

Pipeline (separate from Phase 3):

DocumentChunk
    ↓
EmbeddingProvider
    ↓
Embedding vector
    ↓
Persist to document_chunks.embedding
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.document_chunk import DocumentChunk
from .chunk_service import get_chunks_for_document
from .embeddings.base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)


def get_embedding_provider() -> EmbeddingProvider:
    """Factory: return the configured embedding provider.

    Uses settings.embedding_provider to select the concrete implementation.
    Defaults to FakeEmbeddingProvider for development/testing.
    """
    provider_name = settings.embedding_provider.lower()

    if provider_name == "fake":
        from .embeddings.fake_provider import FakeEmbeddingProvider
        return FakeEmbeddingProvider(dimension=settings.embedding_dimension)
    elif provider_name == "openai":
        from .embeddings.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            timeout=settings.openai_embedding_timeout,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name!r}")


def generate_document_embeddings(
    db: Session,
    document_id: int,
    provider: Optional[EmbeddingProvider] = None,
) -> int:
    """Generate and persist embeddings for all chunks of a document.

    Idempotent: running twice produces the same result (embeddings replaced).
    Does NOT commit — caller owns the transaction.

    Args:
        db: SQLAlchemy session (caller owns commit/rollback).
        document_id: The document whose chunks to embed.
        provider: Optional override (for testing). Uses config default if None.

    Returns:
        Number of chunks embedded.

    Raises:
        EmbeddingError: If embedding generation fails.
    """
    if provider is None:
        provider = get_embedding_provider()

    # Retrieve chunks in order
    chunks = get_chunks_for_document(db, document_id)
    if not chunks:
        logger.info(
            "generate_document_embeddings: no chunks for doc %s, skipping",
            document_id,
        )
        return 0

    # Extract texts (preserve order)
    texts = [chunk.text for chunk in chunks]

    # Generate embeddings in batch
    try:
        embeddings = provider.embed_texts(texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Embedding provider failed: {exc}") from exc

    # Validate count matches
    if len(embeddings) != len(chunks):
        raise EmbeddingError(
            f"Provider returned {len(embeddings)} embeddings "
            f"for {len(chunks)} chunks"
        )

    # Validate each embedding
    for i, emb in enumerate(embeddings):
        provider.validate_embedding(emb)

    # Persist — atomic replace of embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    logger.info(
        "generate_document_embeddings: embedded %d chunks for doc %s (dim=%d)",
        len(chunks), document_id, provider.dimension,
    )
    return len(chunks)


def clear_document_embeddings(db: Session, document_id: int) -> int:
    """Remove embeddings from all chunks of a document.

    Does NOT commit — caller owns the transaction.

    Args:
        db: SQLAlchemy session.
        document_id: The document whose embeddings to clear.

    Returns:
        Number of chunks cleared.
    """
    chunks = get_chunks_for_document(db, document_id)
    count = 0
    for chunk in chunks:
        if chunk.embedding is not None:
            chunk.embedding = None
            count += 1
    if count > 0:
        logger.debug(
            "clear_document_embeddings: cleared %d embeddings for doc %s",
            count, document_id,
        )
    return count
