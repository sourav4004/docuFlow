"""Phase 4.4 tests: embedding provider abstraction, fake provider, and embedding service.

All tests use the FakeEmbeddingProvider — no paid APIs required.
"""

import math
import pytest
from typing import List
from unittest.mock import patch, MagicMock

from app.core import auth
from app.core.config import settings
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.embeddings.base import EmbeddingProvider, EmbeddingError
from app.services.embeddings.fake_provider import FakeEmbeddingProvider
from app.services.embedding_service import (
    generate_document_embeddings,
    clear_document_embeddings,
    get_embedding_provider,
)
from app.services.chunk_service import persist_chunks
from app.services.chunker import TextChunk
from tests.test_auth import client, TestingSessionLocal


# ---------------------------------------------------------------------------
# Minimal valid PDF for integration tests
# ---------------------------------------------------------------------------

TEXT_PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
    b"(Hello DocuFlow Phase 3) Tj ET\nendstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n440\n%%EOF"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Isolate storage and sessions before each test."""
    auth._sessions.clear()
    original_storage_dir = storage_service.base_dir
    test_storage_dir = tmp_path / "test_embed_storage"
    test_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_service.base_dir = test_storage_dir
    yield
    storage_service.base_dir = original_storage_dir


# ==========================================================================
# 1. FAKE PROVIDER — BASIC FUNCTIONALITY
# ==========================================================================

class TestFakeEmbeddingProvider:
    def test_single_text_embedding(self):
        """Test embedding a single text produces correct dimension."""
        provider = FakeEmbeddingProvider(dimension=384)
        emb = provider.embed_text("Hello world")
        assert len(emb) == 384
        assert all(isinstance(v, float) for v in emb)

    def test_batch_embedding(self):
        """Test batch embedding produces correct count and dimensions."""
        provider = FakeEmbeddingProvider(dimension=128)
        texts = ["First text", "Second text", "Third text"]
        embeddings = provider.embed_texts(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 128

    def test_deterministic_output(self):
        """Same input always produces the same output."""
        provider = FakeEmbeddingProvider(dimension=64)
        emb1 = provider.embed_text("deterministic test")
        emb2 = provider.embed_text("deterministic test")
        assert emb1 == emb2

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different embeddings."""
        provider = FakeEmbeddingProvider(dimension=64)
        emb1 = provider.embed_text("cats are great")
        emb2 = provider.embed_text("dogs are great")
        assert emb1 != emb2

    def test_unit_vector_normalization(self):
        """Embeddings should be approximately unit-length."""
        provider = FakeEmbeddingProvider(dimension=384)
        emb = provider.embed_text("normalize me")
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 1e-5

    def test_dimension_property(self):
        """Provider dimension property returns correct value."""
        provider = FakeEmbeddingProvider(dimension=256)
        assert provider.dimension == 256

    def test_custom_dimension(self):
        """Provider works with non-default dimensions."""
        provider = FakeEmbeddingProvider(dimension=16)
        emb = provider.embed_text("short dim")
        assert len(emb) == 16

    def test_empty_text_raises_error(self):
        """Empty text should raise EmbeddingError."""
        provider = FakeEmbeddingProvider()
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_text("")

    def test_whitespace_only_text_raises_error(self):
        """Whitespace-only text should raise EmbeddingError."""
        provider = FakeEmbeddingProvider()
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_text("   ")

    def test_empty_list_raises_error(self):
        """Empty list should raise EmbeddingError."""
        provider = FakeEmbeddingProvider()
        with pytest.raises(EmbeddingError, match="empty"):
            provider.embed_texts([])

    def test_invalid_dimension_raises(self):
        """Zero or negative dimension should raise ValueError."""
        with pytest.raises(ValueError):
            FakeEmbeddingProvider(dimension=0)
        with pytest.raises(ValueError):
            FakeEmbeddingProvider(dimension=-1)


# ==========================================================================
# 2. EMBEDDING VALIDATION
# ==========================================================================

class TestEmbeddingValidation:
    def test_validate_correct_embedding(self):
        """Valid embedding should pass validation."""
        provider = FakeEmbeddingProvider(dimension=4)
        emb = [0.1, 0.2, 0.3, 0.4]
        provider.validate_embedding(emb)  # should not raise

    def test_validate_wrong_dimension(self):
        """Wrong dimension should raise EmbeddingError."""
        provider = FakeEmbeddingProvider(dimension=4)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            provider.validate_embedding([0.1, 0.2])

    def test_validate_non_list(self):
        """Non-list should raise EmbeddingError."""
        provider = FakeEmbeddingProvider(dimension=4)
        with pytest.raises(EmbeddingError, match="must be a list"):
            provider.validate_embedding("not a list")

    def test_validate_non_numeric_values(self):
        """Non-numeric values should raise EmbeddingError."""
        provider = FakeEmbeddingProvider(dimension=3)
        with pytest.raises(EmbeddingError, match="must be numeric"):
            provider.validate_embedding([0.1, "bad", 0.3])


# ==========================================================================
# 3. EMBEDDING SERVICE — GENERATE
# ==========================================================================

class TestEmbeddingService:
    def _create_doc_with_chunks(self, db, user, texts=None):
        """Helper: create document with chunks."""
        if texts is None:
            texts = ["First chunk text", "Second chunk text", "Third chunk text"]
        storage_key = storage_service.save(TEXT_PDF_CONTENT)
        doc = Document(
            user_id=user.id, original_filename="test.pdf",
            storage_key=storage_key, mime_type="application/pdf",
            file_size=len(TEXT_PDF_CONTENT), status="READY",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        chunks = [
            TextChunk(chunk_index=i, text=t, char_start=i * 100, char_end=(i + 1) * 100)
            for i, t in enumerate(texts)
        ]
        persist_chunks(db, doc.id, chunks)
        db.commit()
        return doc

    def test_generate_embeddings_for_document(self):
        """Test generating embeddings for a document's chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="EmbTest", email="emb_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = self._create_doc_with_chunks(db, user)
            provider = FakeEmbeddingProvider(dimension=64)

            count = generate_document_embeddings(db, doc.id, provider=provider)
            assert count == 3

            # Verify embeddings are stored
            from app.services.chunk_service import get_chunks_for_document
            chunks = get_chunks_for_document(db, doc.id)
            for c in chunks:
                assert c.embedding is not None
                assert len(c.embedding) == 64
        finally:
            db.close()

    def test_embeddings_preserve_chunk_order(self):
        """Embeddings must correspond to chunks in order."""
        db = TestingSessionLocal()
        try:
            user = User(name="OrderTest", email="order_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            texts = ["Alpha text", "Beta text", "Gamma text"]
            doc = self._create_doc_with_chunks(db, user, texts)
            provider = FakeEmbeddingProvider(dimension=32)

            generate_document_embeddings(db, doc.id, provider=provider)

            from app.services.chunk_service import get_chunks_for_document
            chunks = get_chunks_for_document(db, doc.id)

            # Each chunk's embedding should match embedding its text directly
            for chunk in chunks:
                expected = provider.embed_text(chunk.text)
                assert chunk.embedding == expected
        finally:
            db.close()

    def test_re_embed_idempotent(self):
        """Running embedding generation twice replaces (not duplicates)."""
        db = TestingSessionLocal()
        try:
            user = User(name="IdempTest", email="idemp_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = self._create_doc_with_chunks(db, user)
            provider = FakeEmbeddingProvider(dimension=32)

            # First run
            generate_document_embeddings(db, doc.id, provider=provider)
            from app.services.chunk_service import get_chunks_for_document
            chunks_v1 = get_chunks_for_document(db, doc.id)
            embeddings_v1 = [c.embedding for c in chunks_v1]

            # Second run
            generate_document_embeddings(db, doc.id, provider=provider)
            chunks_v2 = get_chunks_for_document(db, doc.id)
            embeddings_v2 = [c.embedding for c in chunks_v2]

            # Same embeddings, no extra records
            assert embeddings_v1 == embeddings_v2
            assert len(chunks_v1) == len(chunks_v2)
        finally:
            db.close()

    def test_no_chunks_returns_zero(self):
        """Document with no chunks returns 0."""
        db = TestingSessionLocal()
        try:
            user = User(name="NoChunkTest", email="nochunk_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="empty.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            count = generate_document_embeddings(db, doc.id)
            assert count == 0
        finally:
            db.close()


# ==========================================================================
# 4. FAILURE HANDLING
# ==========================================================================

class TestEmbeddingFailures:
    def test_provider_failure_propagates(self):
        """Provider failure should raise EmbeddingError."""
        db = TestingSessionLocal()
        try:
            user = User(name="FailTest", email="fail_emb@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="fail.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            chunks = [TextChunk(chunk_index=0, text="test", char_start=0, char_end=4)]
            persist_chunks(db, doc.id, chunks)
            db.commit()

            # Mock provider that fails
            mock_provider = MagicMock(spec=EmbeddingProvider)
            mock_provider.dimension = 64
            mock_provider.embed_texts.side_effect = EmbeddingError("Provider timeout")

            with pytest.raises(EmbeddingError, match="Provider timeout"):
                generate_document_embeddings(db, doc.id, provider=mock_provider)

            # Embedding should NOT be set on the chunk
            from app.services.chunk_service import get_chunks_for_document
            chunks = get_chunks_for_document(db, doc.id)
            assert chunks[0].embedding is None
        finally:
            db.close()

    def test_wrong_dimension_count_raises(self):
        """Provider returning wrong number of embeddings should raise."""
        db = TestingSessionLocal()
        try:
            user = User(name="DimTest", email="dim_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="dim.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            chunks = [
                TextChunk(chunk_index=0, text="a", char_start=0, char_end=1),
                TextChunk(chunk_index=1, text="b", char_start=1, char_end=2),
            ]
            persist_chunks(db, doc.id, chunks)
            db.commit()

            # Mock provider returning wrong count
            mock_provider = MagicMock(spec=EmbeddingProvider)
            mock_provider.dimension = 64
            mock_provider.embed_texts.return_value = [[0.0] * 64]  # 1 instead of 2

            with pytest.raises(EmbeddingError, match="returned 1 embeddings"):
                generate_document_embeddings(db, doc.id, provider=mock_provider)
        finally:
            db.close()

    def test_wrong_embedding_dimension_raises(self):
        """Provider returning wrong dimension vectors should raise."""
        db = TestingSessionLocal()
        try:
            user = User(name="WrongDim", email="wrong_dim@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="wdim.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            chunks = [TextChunk(chunk_index=0, text="test", char_start=0, char_end=4)]
            persist_chunks(db, doc.id, chunks)
            db.commit()

            # Create a real provider subclass that returns wrong dimensions
            class BadDimensionProvider(FakeEmbeddingProvider):
                def embed_texts(self, texts):
                    return [[0.0] * 32 for _ in texts]  # 32 instead of 64

            bad_provider = BadDimensionProvider(dimension=64)

            with pytest.raises(EmbeddingError, match="dimension mismatch"):
                generate_document_embeddings(db, doc.id, provider=bad_provider)
        finally:
            db.close()


# ==========================================================================
# 5. CLEAR EMBEDDINGS
# ==========================================================================

class TestClearEmbeddings:
    def test_clear_embeddings(self):
        """Test clearing embeddings from chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="ClearTest", email="clear_emb@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="clear.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            chunks = [TextChunk(chunk_index=0, text="test", char_start=0, char_end=4)]
            persist_chunks(db, doc.id, chunks)
            db.commit()

            # Set embeddings
            provider = FakeEmbeddingProvider(dimension=32)
            generate_document_embeddings(db, doc.id, provider=provider)

            # Clear
            count = clear_document_embeddings(db, doc.id)
            assert count == 1

            # Verify cleared
            from app.services.chunk_service import get_chunks_for_document
            refreshed = get_chunks_for_document(db, doc.id)
            assert refreshed[0].embedding is None
        finally:
            db.close()


# ==========================================================================
# 6. PROVIDER FACTORY
# ==========================================================================

class TestProviderFactory:
    def test_get_fake_provider(self):
        """Factory returns FakeEmbeddingProvider when configured."""
        with patch.object(settings, "embedding_provider", "fake"):
            provider = get_embedding_provider()
            assert isinstance(provider, FakeEmbeddingProvider)

    def test_get_provider_unknown_raises(self):
        """Factory raises ValueError for unknown provider."""
        with patch.object(settings, "embedding_provider", "nonexistent"):
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                get_embedding_provider()

    def test_fake_provider_uses_config_dimension(self):
        """Fake provider uses EMBEDDING_DIMENSION from config."""
        with patch.object(settings, "embedding_provider", "fake"):
            with patch.object(settings, "embedding_dimension", 128):
                provider = get_embedding_provider()
                assert provider.dimension == 128


# ==========================================================================
# 7. UNICODE TEXT
# ==========================================================================

class TestUnicodeEmbedding:
    def test_embed_unicode_text(self):
        """Test embedding text with various Unicode scripts."""
        provider = FakeEmbeddingProvider(dimension=64)

        texts = [
            "English text",
            "日本語テキスト",
            "مرحبا بالعالم",
            "Über naïve résumé",
            "🌍 Hello world!",
        ]
        embeddings = provider.embed_texts(texts)
        assert len(embeddings) == 5
        for emb in embeddings:
            assert len(emb) == 64
            norm = math.sqrt(sum(v * v for v in emb))
            assert abs(norm - 1.0) < 1e-5


# ==========================================================================
# 8. LARGE BATCH
# ==========================================================================

class TestLargeBatch:
    def test_large_number_of_chunks(self):
        """Test embedding a large number of chunks."""
        provider = FakeEmbeddingProvider(dimension=32)
        texts = [f"Chunk number {i} with some content" for i in range(100)]
        embeddings = provider.embed_texts(texts)
        assert len(embeddings) == 100
        for emb in embeddings:
            assert len(emb) == 32


# ==========================================================================
# 9. CONFIGURATION
# ==========================================================================

class TestConfiguration:
    def test_embedding_config_defaults(self):
        """Verify default embedding config values."""
        assert settings.embedding_provider in ("fake", "openai")
        assert settings.embedding_dimension > 0
        assert isinstance(settings.embedding_api_key, str)

    def test_api_key_not_in_logs(self, caplog):
        """API key should never appear in log output."""
        with patch.object(settings, "embedding_api_key", "sk-test-secret-key"):
            provider = get_embedding_provider()
            # The fake provider doesn't use the key, but verify it's not logged
            assert "sk-test-secret-key" not in caplog.text
