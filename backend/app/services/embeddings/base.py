"""Abstract embedding provider interface.

All concrete embedding providers must implement this interface.
The rest of DocuFlow interacts with EmbeddingProvider, not vendor SDKs.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement embed_text and embed_texts.
    The provider must be stateless and thread-safe.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider/model."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed. Must not be empty.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed. Must not be empty.
                Each text must be non-empty.

        Returns:
            List of embedding vectors, one per input text, in the same order.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    def validate_embedding(self, embedding: List[float]) -> None:
        """Validate that an embedding has the correct dimension and type.

        Args:
            embedding: The embedding vector to validate.

        Raises:
            EmbeddingError: If the embedding is invalid.
        """
        if not isinstance(embedding, list):
            raise EmbeddingError(
                f"Embedding must be a list, got {type(embedding).__name__}"
            )
        if len(embedding) != self.dimension:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding)}"
            )
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise EmbeddingError(
                    f"Embedding value at index {i} must be numeric, got {type(val).__name__}"
                )
