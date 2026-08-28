"""Deterministic fake embedding provider for testing.

Generates consistent, reproducible embeddings from text content using
hashing. No network calls, no ML models, no paid APIs.

The fake embeddings are NOT semantically meaningful — they exist solely
to exercise the embedding pipeline with deterministic, testable output.
"""

import hashlib
import logging
import struct
from typing import List

from .base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

# Default dimension for the fake provider
FAKE_DIMENSION = 384


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for unit/integration tests.

    Generates vectors by hashing the input text and expanding
    the hash into the target dimension. The same input always
    produces the same output.
    """

    def __init__(self, dimension: int = FAKE_DIMENSION):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic fake embedding for a single text."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        return self._hash_to_vector(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic fake embeddings for multiple texts."""
        if not texts:
            raise EmbeddingError("Cannot embed empty list of texts")
        return [self.embed_text(t) for t in texts]

    def _hash_to_vector(self, text: str) -> List[float]:
        """Convert text to a deterministic vector using iterative hashing."""
        # Use text content as seed for reproducibility
        seed = text.encode("utf-8")
        vector: List[float] = []

        # Generate enough hash bytes to fill the dimension
        # Each float needs 4 bytes
        needed_bytes = self._dimension * 4
        current_hash = hashlib.sha256(seed).digest()
        all_bytes = current_hash

        while len(all_bytes) < needed_bytes:
            current_hash = hashlib.sha256(current_hash + seed).digest()
            all_bytes += current_hash

        # Convert bytes to floats in [-1.0, 1.0]
        for i in range(self._dimension):
            raw = struct.unpack_from(">f", all_bytes, i * 4)[0]
            # Clamp to [-1, 1] range
            val = max(-1.0, min(1.0, raw))
            vector.append(val)

        # Normalize to unit vector
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector
