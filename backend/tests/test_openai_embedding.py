"""Phase 5.9 Step 2: OpenAI embedding provider tests.

All tests mock the OpenAI SDK — no real API calls are made.
No API keys are required to run this test suite.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass
from typing import List

from app.services.embeddings.base import EmbeddingProvider, EmbeddingError
from app.services.embeddings.openai_provider import (
    OpenAIEmbeddingProvider,
    _DIMENSION_CAPABLE_MODELS,
)
from app.services.embeddings.fake_provider import FakeEmbeddingProvider
from app.services.embedding_service import get_embedding_provider
from app.core.config import settings


# ---------------------------------------------------------------------------
# Helpers — mock OpenAI response objects
# ---------------------------------------------------------------------------

@dataclass
class MockEmbeddingObject:
    index: int
    embedding: List[float]


@dataclass
class MockEmbeddingResponse:
    data: list


def _make_embedding(dimension: int = 384, value: float = 0.1) -> List[float]:
    """Create a fake embedding vector of the given dimension."""
    return [value] * dimension


def _make_openai_response(texts: List[str], dimension: int = 384) -> MockEmbeddingResponse:
    """Create a mock OpenAI embedding API response."""
    data = [
        MockEmbeddingObject(index=i, embedding=_make_embedding(dimension, value=0.1 + i * 0.01))
        for i in range(len(texts))
    ]
    return MockEmbeddingResponse(data=data)


# ==========================================================================
# 1. BASIC FUNCTIONALITY (mocked)
# ==========================================================================

class TestOpenAIBasic:
    def test_single_text_embedding(self):
        """Single text produces correct dimension embedding."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-fake-key",
            model="text-embedding-3-small",
            dimension=384,
        )

        mock_response = _make_openai_response(["hello"], dimension=384)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            emb = provider.embed_text("hello")

        assert len(emb) == 384
        assert emb == [0.1] * 384

    def test_batch_embedding(self):
        """Batch embedding produces correct count and dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-fake-key",
            model="text-embedding-3-small",
            dimension=128,
        )

        texts = ["first", "second", "third"]
        mock_response = _make_openai_response(texts, dimension=128)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 128

    def test_dimension_property(self):
        """Provider dimension property returns configured value."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=256
        )
        assert provider.dimension == 256

    def test_api_key_stored(self):
        """API key is stored for later use."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key-123", model="text-embedding-3-small", dimension=64
        )
        assert provider._api_key == "sk-test-key-123"


# ==========================================================================
# 2. DIMENSION TRUNCATION (text-embedding-3 models)
# ==========================================================================

class TestDimensionTruncation:
    def test_dimensions_param_sent_for_text_embedding_3(self):
        """text-embedding-3 models get the dimensions parameter."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-3-small",
            dimension=384,
        )

        mock_response = _make_openai_response(["test"], dimension=384)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            provider.embed_text("test")

        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 384
        assert call_kwargs["model"] == "text-embedding-3-small"

    def test_dimensions_param_for_text_embedding_3_large(self):
        """text-embedding-3-large also gets the dimensions parameter."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-3-large",
            dimension=512,
        )

        mock_response = _make_openai_response(["test"], dimension=512)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            provider.embed_text("test")

        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 512

    def test_no_dimensions_param_for_ada_002(self):
        """text-embedding-ada-002 does NOT get the dimensions parameter."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-ada-002",
            dimension=1536,
        )

        mock_response = _make_openai_response(["test"], dimension=1536)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            provider.embed_text("test")

        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert "dimensions" not in call_kwargs

    def test_dimension_capable_models_constant(self):
        """Verify the set of dimension-capable models."""
        assert "text-embedding-3-small" in _DIMENSION_CAPABLE_MODELS
        assert "text-embedding-3-large" in _DIMENSION_CAPABLE_MODELS
        assert "text-embedding-ada-002" not in _DIMENSION_CAPABLE_MODELS


# ==========================================================================
# 3. RESPONSE ORDERING
# ==========================================================================

class TestResponseOrdering:
    def test_embeddings_returned_in_input_order(self):
        """Embeddings are returned in the same order as input texts."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        # Create response with shuffled indices
        mock_response = MockEmbeddingResponse(data=[
            MockEmbeddingObject(index=2, embedding=[0.3] * 64),
            MockEmbeddingObject(index=0, embedding=[0.1] * 64),
            MockEmbeddingObject(index=1, embedding=[0.2] * 64),
        ])
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            embeddings = provider.embed_texts(["a", "b", "c"])

        # Should be reordered to match input: index 0, 1, 2
        assert embeddings[0] == [0.1] * 64
        assert embeddings[1] == [0.2] * 64
        assert embeddings[2] == [0.3] * 64


# ==========================================================================
# 4. VALIDATION
# ==========================================================================

class TestValidation:
    def test_validate_correct_embedding(self):
        """Valid embedding passes validation."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=4
        )
        provider.validate_embedding([0.1, 0.2, 0.3, 0.4])  # Should not raise

    def test_validate_wrong_dimension(self):
        """Wrong dimension raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=4
        )
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            provider.validate_embedding([0.1, 0.2])

    def test_validate_non_list(self):
        """Non-list raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=4
        )
        with pytest.raises(EmbeddingError, match="must be a list"):
            provider.validate_embedding("not a list")

    def test_validate_non_numeric(self):
        """Non-numeric values raise EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=3
        )
        with pytest.raises(EmbeddingError, match="must be numeric"):
            provider.validate_embedding([0.1, "bad", 0.3])


# ==========================================================================
# 5. INPUT VALIDATION
# ==========================================================================

class TestInputValidation:
    def test_empty_text_raises_error(self):
        """Empty text raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_text("")

    def test_whitespace_only_text_raises_error(self):
        """Whitespace-only text raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_text("   ")

    def test_empty_list_raises_error(self):
        """Empty list raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_texts([])

    def test_empty_text_in_batch_raises_error(self):
        """Empty text in batch raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )
        with pytest.raises(EmbeddingError, match="empty text at index 1"):
            provider.embed_texts(["valid", ""])

    def test_missing_api_key_raises_error(self):
        """Missing API key raises EmbeddingError at construction."""
        with pytest.raises(EmbeddingError, match="API key is required"):
            OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small")

    def test_invalid_dimension_raises(self):
        """Zero or negative dimension raises ValueError."""
        with pytest.raises(ValueError):
            OpenAIEmbeddingProvider(api_key="sk-test", dimension=0)
        with pytest.raises(ValueError):
            OpenAIEmbeddingProvider(api_key="sk-test", dimension=-1)


# ==========================================================================
# 6. ERROR HANDLING
# ==========================================================================

class TestErrorHandling:
    def test_api_error_propagates_as_embedding_error(self):
        """OpenAI API errors are wrapped in EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("Rate limit exceeded")

        with patch.object(provider, '_get_client', return_value=mock_client):
            with pytest.raises(EmbeddingError, match="OpenAI embedding request failed"):
                provider.embed_text("hello")

    def test_api_key_redacted_from_error_message(self):
        """API key is removed from error messages."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-super-secret-key-12345",
            model="text-embedding-3-small",
            dimension=64,
        )

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception(
            "Error with key sk-super-secret-key-12345 in message"
        )

        with patch.object(provider, '_get_client', return_value=mock_client):
            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed_text("hello")

        assert "sk-super-secret-key-12345" not in str(exc_info.value)

    def test_wrong_count_raises_error(self):
        """Wrong number of embeddings returned raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        # Return 1 embedding for 2 texts
        mock_response = MockEmbeddingResponse(data=[
            MockEmbeddingObject(index=0, embedding=[0.1] * 64),
        ])
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            with pytest.raises(EmbeddingError, match="returned 1 embeddings"):
                provider.embed_texts(["a", "b"])

    def test_wrong_dimension_raises_error(self):
        """Embedding with wrong dimension raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        mock_response = MockEmbeddingResponse(data=[
            MockEmbeddingObject(index=0, embedding=[0.1] * 32),  # 32 instead of 64
        ])
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, '_get_client', return_value=mock_client):
            with pytest.raises(EmbeddingError, match="dimension 32.*expected 64"):
                provider.embed_text("hello")

    def test_import_error_for_missing_openai_package(self):
        """Missing openai package raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        with patch.dict('sys.modules', {'openai': None}):
            with pytest.raises(EmbeddingError, match="openai.*package is required"):
                provider._get_client()


# ==========================================================================
# 7. PROVIDER FACTORY INTEGRATION
# ==========================================================================

class TestProviderFactory:
    def test_factory_returns_openai_provider(self):
        """Factory returns OpenAIEmbeddingProvider when configured."""
        with patch.object(settings, "embedding_provider", "openai"):
            with patch.object(settings, "openai_api_key", "sk-fake-key"):
                with patch.object(settings, "openai_embedding_model", "text-embedding-3-small"):
                    with patch.object(settings, "embedding_dimension", 384):
                        provider = get_embedding_provider()
                        assert isinstance(provider, OpenAIEmbeddingProvider)
                        assert provider.dimension == 384

    def test_factory_returns_fake_provider(self):
        """Factory still returns FakeEmbeddingProvider when configured."""
        with patch.object(settings, "embedding_provider", "fake"):
            provider = get_embedding_provider()
            assert isinstance(provider, FakeEmbeddingProvider)

    def test_factory_unknown_provider_raises(self):
        """Factory raises ValueError for unknown provider."""
        with patch.object(settings, "embedding_provider", "unknown"):
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                get_embedding_provider()

    def test_factory_uses_config_values(self):
        """Factory passes config values to OpenAI provider."""
        with patch.object(settings, "embedding_provider", "openai"):
            with patch.object(settings, "openai_api_key", "sk-config-key"):
                with patch.object(settings, "openai_embedding_model", "text-embedding-3-large"):
                    with patch.object(settings, "embedding_dimension", 512):
                        with patch.object(settings, "openai_embedding_timeout", 30.0):
                            provider = get_embedding_provider()
                            assert provider._api_key == "sk-config-key"
                            assert provider._model == "text-embedding-3-large"
                            assert provider.dimension == 512
                            assert provider._timeout == 30.0


# ==========================================================================
# 8. CONFIGURATION
# ==========================================================================

class TestConfiguration:
    def test_config_has_openai_fields(self):
        """Config has all required OpenAI embedding fields."""
        assert hasattr(settings, "openai_api_key")
        assert hasattr(settings, "openai_embedding_model")
        assert hasattr(settings, "openai_embedding_timeout")
        assert hasattr(settings, "embedding_provider")
        assert hasattr(settings, "embedding_dimension")

    def test_config_defaults(self):
        """Default config values are sensible."""
        assert settings.embedding_provider in ("fake", "openai")
        assert settings.embedding_dimension > 0
        assert isinstance(settings.openai_api_key, str)
        assert isinstance(settings.openai_embedding_model, str)
        assert settings.openai_embedding_timeout > 0

    def test_api_key_not_in_logs(self, caplog):
        """API key should never appear in log output."""
        with patch.object(settings, "embedding_provider", "fake"):
            provider = get_embedding_provider()
            # Even with an OpenAI key set, the fake provider doesn't use it
            assert "sk-" not in caplog.text


# ==========================================================================
# 9. LAZY CLIENT INITIALIZATION
# ==========================================================================

class TestLazyInit:
    def test_client_not_created_at_init(self):
        """OpenAI client is not created until first API call."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )
        assert provider._client is None

    def test_client_created_on_first_call(self):
        """OpenAI client is created on first API call."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        mock_response = _make_openai_response(["test"], dimension=64)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
            provider.embed_text("test")
            mock_cls.assert_called_once_with(api_key="sk-test", timeout=60.0)

    def test_client_reused_across_calls(self):
        """Same client instance is reused for multiple calls."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimension=64
        )

        mock_response = _make_openai_response(["test"], dimension=64)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            provider.embed_text("first")
            provider.embed_text("second")

        # Client should only be created once
        assert provider._client is mock_client


# ==========================================================================
# 10. TIMEOUT CONFIGURATION
# ==========================================================================

class TestTimeout:
    def test_custom_timeout_passed_to_client(self):
        """Custom timeout is passed to the OpenAI client."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small",
            dimension=64, timeout=30.0,
        )

        mock_response = _make_openai_response(["test"], dimension=64)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
            provider.embed_text("test")
            mock_cls.assert_called_with(api_key="sk-test", timeout=30.0)
