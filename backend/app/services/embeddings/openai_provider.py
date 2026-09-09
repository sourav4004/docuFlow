"""OpenAI embedding provider.

Uses the official OpenAI Python SDK to generate embeddings via the
OpenAI API. Supports text-embedding-3-small, text-embedding-3-large,
and text-embedding-ada-002 models.

For text-embedding-3 models, supports the `dimensions` parameter
to truncate embeddings to a target dimensionality (e.g. 384 for
compatibility with the existing pgvector Vector(384) column).

No migration required when using text-embedding-3 models with the
configured embedding_dimension setting.
"""

import logging
from typing import List, Optional

from .base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

# Models that support the `dimensions` parameter for truncation
_DIMENSION_CAPABLE_MODELS = {"text-embedding-3-small", "text-embedding-3-large"}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the OpenAI Embeddings API.

    Requires the `openai` Python package and a valid API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 384,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise EmbeddingError(
                "OpenAI API key is required. Set the OPENAI_API_KEY environment variable."
            )
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._client = None  # Lazy-initialized

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        """Lazy-initialize the OpenAI client to avoid import at module level."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise EmbeddingError(
                    "The 'openai' package is required for OpenAI embeddings. "
                    "Install it with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        results = self._call_api([text])
        return results[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)."""
        if not texts:
            raise EmbeddingError("Cannot embed empty list of texts")
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise EmbeddingError(f"Cannot embed empty text at index {i}")
        return self._call_api(texts)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI embeddings API and return the results."""
        client = self._get_client()

        kwargs = {
            "model": self._model,
            "input": texts,
        }

        # text-embedding-3 models support the dimensions parameter
        # for output dimensionality truncation
        if self._model in _DIMENSION_CAPABLE_MODELS:
            kwargs["dimensions"] = self._dimension

        try:
            response = client.embeddings.create(**kwargs)
        except ImportError:
            raise EmbeddingError(
                "The 'openai' package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )
        except Exception as exc:
            # Do not expose API keys or internal details
            error_msg = str(exc)
            # Sanitize: remove any API key fragments that might appear
            if self._api_key and self._api_key in error_msg:
                error_msg = error_msg.replace(self._api_key, "[REDACTED]")
            logger.error("OpenAI embedding API error: %s", error_msg)
            raise EmbeddingError(f"OpenAI embedding request failed: {error_msg}") from exc

        # Extract embeddings in order
        # OpenAI returns embeddings indexed by their position in the input.
        # Malformed/foreign responses must fail as EmbeddingError, never leak
        # raw attribute errors to callers.
        try:
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]
        except AttributeError as exc:
            raise EmbeddingError(
                f"Malformed embedding response from provider: {exc}") from exc

        # Validate count
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"OpenAI returned {len(embeddings)} embeddings "
                f"for {len(texts)} inputs"
            )

        # Validate dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != self._dimension:
                raise EmbeddingError(
                    f"OpenAI returned embedding of dimension {len(emb)} "
                    f"at index {i}, expected {self._dimension}"
                )

        return embeddings
