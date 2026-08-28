"""Phase 4.6 tests: retrieval pipeline.

Uses FakeEmbeddingProvider for all tests — no paid API required.
Some tests require real PostgreSQL (pgvector operators).
"""

import math
import pytest
from typing import List
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core import auth
from app.core.config import settings
from app.core.database import Base, get_db, SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.embeddings.base import EmbeddingProvider, EmbeddingError
from app.services.retrieval_service import (
    validate_query,
    build_context,
    retrieve_context,
    RetrievalResult,
    RetrievalResponse,
    RetrievalError,
    QueryValidationError,
    MAX_QUERY_LENGTH,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_MAX_CONTEXT_CHARS,
)
from app.services.vector_search import search_similar_chunks, SearchResult


# ---------------------------------------------------------------------------
# Test embedding provider — deterministic, no external API
# ---------------------------------------------------------------------------

class FakeTestEmbedding(EmbeddingProvider):
    """Deterministic fake provider for retrieval tests.

    Maps keywords to known vectors in 384D space so ranking can be verified.
    Uses sparse embedding: each keyword activates a specific dimension.
    """

    _DIM = settings.embedding_dimension  # 384 — must match DocumentChunk.embedding

    # Keyword → dimension index mapping (sparse in 384D)
    _KEYWORD_DIMS = {
        "france": 0,
        "python": 1,
        "mountain": 2,
        "policy": 3,
        "remote": 4,
        "work": 5,
    }

    @property
    def dimension(self) -> int:
        return self._DIM

    def embed_text(self, text: str) -> List[float]:
        lower = text.lower()
        vec = [0.0] * self._DIM
        for kw, dim_idx in self._KEYWORD_DIMS.items():
            if kw in lower:
                vec[dim_idx] = 1.0
        # Normalize if non-zero
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 384D unit vectors for known keyword vectors (sparse: each keyword at a specific dim)
_DIM = settings.embedding_dimension  # 384
VEC_FRANCE = [1.0] + [0.0] * (_DIM - 1)
VEC_PYTHON = [0.0, 1.0] + [0.0] * (_DIM - 2)
VEC_MOUNTAIN = [0.0, 0.0, 1.0] + [0.0] * (_DIM - 3)
VEC_POLICY = [0.0, 0.0, 0.0, 1.0] + [0.0] * (_DIM - 4)
VEC_REMOTE_WORK = [0.0, 0.0, 0.0, 0.0, 1 / math.sqrt(2), 1 / math.sqrt(2)] + [0.0] * (_DIM - 6)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Isolate auth sessions and clean up test data."""
    auth._sessions.clear()
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'ret_%@example.com'))"))
        db.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'ret_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'ret_%@example.com'"))
        db.commit()
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'ret_%@example.com'))"))
        db.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'ret_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'ret_%@example.com'"))
        db.commit()
    finally:
        db.close()


def create_doc_with_embeddings(db, user, chunks_data, embeddings):
    """Helper: create document, chunks, and set embeddings."""
    storage_key = storage_service.save(b"%PDF-1.4 test")
    doc = Document(
        user_id=user.id, original_filename="test.pdf",
        storage_key=storage_key, mime_type="application/pdf",
        file_size=100, status="READY",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    for i, ((text_val, cs, ce), emb) in enumerate(zip(chunks_data, embeddings)):
        chunk = DocumentChunk(
            document_id=doc.id, chunk_index=i,
            text=text_val, char_start=cs, char_end=ce,
            embedding=emb,
        )
        db.add(chunk)
    db.commit()
    return doc


# ==========================================================================
# 1. QUERY VALIDATION
# ==========================================================================

class TestQueryValidation:
    def test_valid_query(self):
        q = validate_query("What is the remote work policy?")
        assert q == "What is the remote work policy?"

    def test_strips_whitespace(self):
        q = validate_query("  hello world  ")
        assert q == "hello world"

    def test_empty_string_raises(self):
        with pytest.raises(QueryValidationError, match="must not be empty"):
            validate_query("")

    def test_whitespace_only_raises(self):
        with pytest.raises(QueryValidationError, match="must not be empty"):
            validate_query("   ")

    def test_non_string_raises(self):
        with pytest.raises(QueryValidationError, match="must be a string"):
            validate_query(123)

    def test_too_long_raises(self):
        with pytest.raises(QueryValidationError, match="must not exceed"):
            validate_query("a" * (MAX_QUERY_LENGTH + 1))

    def test_exact_max_length_ok(self):
        q = validate_query("a" * MAX_QUERY_LENGTH)
        assert len(q) == MAX_QUERY_LENGTH

    def test_unicode_query(self):
        q = validate_query("什么是远程工作政策？")
        assert q == "什么是远程工作政策？"


# ==========================================================================
# 2. CONTEXT BUILDER
# ==========================================================================

class TestBuildContext:
    def _make_result(self, **kwargs):
        defaults = dict(
            document_id=1, chunk_id=10, chunk_index=5,
            text="Sample text", similarity_score=0.9,
            original_filename="test.pdf",
            page_start=3, page_end=4,
        )
        defaults.update(kwargs)
        return RetrievalResult(**defaults)

    def test_empty_results_returns_empty(self):
        assert build_context([]) == ""

    def test_single_result(self):
        r = self._make_result(text="Hello world")
        ctx = build_context([r])
        assert "[Source 1]" in ctx
        assert "Document: test.pdf" in ctx
        assert "Chunk: 5" in ctx
        assert "Page: 3–4" in ctx
        assert "Hello world" in ctx

    def test_multiple_results_separated(self):
        r1 = self._make_result(text="First chunk")
        r2 = self._make_result(chunk_index=6, text="Second chunk")
        ctx = build_context([r1, r2])
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "---" in ctx  # Separator
        assert "First chunk" in ctx
        assert "Second chunk" in ctx

    def test_source_order_preserved(self):
        results = [self._make_result(chunk_index=i, text=f"Chunk {i}") for i in range(5)]
        ctx = build_context(results)
        for i in range(5):
            assert f"Chunk {i}" in ctx

    def test_max_chars_respects_limit(self):
        # Create results that exceed the limit
        results = [
            self._make_result(chunk_index=i, text="x" * 500)
            for i in range(20)
        ]
        ctx = build_context(results, max_chars=1000)
        assert len(ctx) <= 1100  # Allow some overhead for headers

    def test_first_chunk_always_included(self):
        # Even if first chunk alone exceeds limit, it should be included
        r = self._make_result(text="x" * 2000)
        ctx = build_context([r], max_chars=1000)
        assert "x" * 2000 in ctx

    def test_no_filename_omits_document_line(self):
        r = self._make_result(original_filename=None)
        ctx = build_context([r])
        assert "Document:" not in ctx
        assert "[Source 1]" in ctx

    def test_same_page_start_end_shows_single_page(self):
        r = self._make_result(page_start=5, page_end=5)
        ctx = build_context([r])
        assert "Page: 5" in ctx
        assert "–" not in ctx.split("Page:")[1].split("\n")[0]

    def test_no_page_omits_page_line(self):
        r = self._make_result(page_start=None, page_end=None)
        ctx = build_context([r])
        assert "Page:" not in ctx


# ==========================================================================
# 3. BASIC RETRIEVAL (using FakeTestEmbedding + real PostgreSQL)
# ==========================================================================

class TestBasicRetrieval:
    def test_retrieve_returns_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetBasic", email="ret_basic@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = create_doc_with_embeddings(
                db, user,
                [("France is great", 0, 16), ("Python is fun", 16, 30), ("Mountains are tall", 30, 50)],
                [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN],
            )

            resp = retrieve_context(db, user.id, "What about France?", provider=provider)
            assert resp.total_results > 0
            assert isinstance(resp, RetrievalResponse)
            assert resp.query == "What about France?"
        finally:
            db.close()

    def test_retrieve_context_not_empty(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetCtx", email="ret_ctx@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = create_doc_with_embeddings(
                db, user,
                [("France is beautiful", 0, 20)],
                [VEC_FRANCE],
            )

            resp = retrieve_context(db, user.id, "Tell me about France", provider=provider)
            assert resp.context != ""
            assert "France" in resp.context
        finally:
            db.close()

    def test_retrieve_with_empty_database(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetEmpty", email="ret_empty@example.com", password_hash="h")
            db.add(user); db.commit()

            resp = retrieve_context(db, user.id, "anything", provider=provider)
            assert resp.total_results == 0
            assert resp.context == ""
            assert resp.results == []
        finally:
            db.close()


# ==========================================================================
# 4. QUERY VALIDATION IN RETRIEVAL
# ==========================================================================

class TestRetrievalQueryValidation:
    def test_empty_query_raises(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetQV", email="ret_qv@example.com", password_hash="h")
            db.add(user); db.commit()

            with pytest.raises(QueryValidationError, match="must not be empty"):
                retrieve_context(db, user.id, "", provider=provider)
        finally:
            db.close()

    def test_whitespace_query_raises(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetQW", email="ret_qw@example.com", password_hash="h")
            db.add(user); db.commit()

            with pytest.raises(QueryValidationError, match="must not be empty"):
                retrieve_context(db, user.id, "   ", provider=provider)
        finally:
            db.close()

    def test_long_query_raises(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetQL", email="ret_ql@example.com", password_hash="h")
            db.add(user); db.commit()

            with pytest.raises(QueryValidationError, match="must not exceed"):
                retrieve_context(db, user.id, "a" * (MAX_QUERY_LENGTH + 1), provider=provider)
        finally:
            db.close()

    def test_unicode_query_works(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetQU", email="ret_qu@example.com", password_hash="h")
            db.add(user); db.commit()

            doc = create_doc_with_embeddings(
                db, user,
                [("Some text", 0, 9)],
                [VEC_FRANCE],
            )

            resp = retrieve_context(db, user.id, "什么是远程工作？", provider=provider)
            assert resp.query == "什么是远程工作？"
        finally:
            db.close()


# ==========================================================================
# 5. SIMILARITY THRESHOLD FILTERING
# ==========================================================================

class TestSimilarityFiltering:
    def test_low_similarity_filtered_out(self):
        """Chunks below min_similarity are excluded."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetFilter", email="ret_filter@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            # France and Mountain are orthogonal — searching for France
            # should rank France high, Mountain low
            doc = create_doc_with_embeddings(
                db, user,
                [("France is great", 0, 16), ("Mountains are tall", 16, 34)],
                [VEC_FRANCE, VEC_MOUNTAIN],
            )

            # With high threshold, only France should pass
            resp = retrieve_context(
                db, user.id, "What about France?",
                provider=provider, min_similarity=0.9,
            )
            # Only France chunk should be in results
            texts = [r.text for r in resp.results]
            assert any("France" in t for t in texts)
        finally:
            db.close()

    def test_high_threshold_no_results(self):
        """Impossibly high threshold returns empty results."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetHigh", email="ret_high@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = create_doc_with_embeddings(
                db, user,
                [("Some text", 0, 9)],
                [VEC_FRANCE],
            )

            resp = retrieve_context(
                db, user.id, "anything",
                provider=provider, min_similarity=0.99,
            )
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()


# ==========================================================================
# 6. TOP-K
# ==========================================================================

class TestTopK:
    def test_top_k_limits_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetTopK", email="ret_topk@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [(f"Chunk {i}", i * 10, (i + 1) * 10) for i in range(10)]
            embeddings = [VEC_FRANCE if i % 2 == 0 else VEC_PYTHON for i in range(10)]
            doc = create_doc_with_embeddings(db, user, chunks_data, embeddings)

            resp = retrieve_context(db, user.id, "France", provider=provider, top_k=3)
            assert resp.total_results <= 3
        finally:
            db.close()


# ==========================================================================
# 7. DOCUMENT-SPECIFIC RETRIEVAL
# ==========================================================================

class TestDocumentFiltering:
    def test_search_specific_document(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetDocF", email="ret_docf@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc_a = create_doc_with_embeddings(db, user, [("France text", 0, 11)], [VEC_FRANCE])
            doc_b = create_doc_with_embeddings(db, user, [("Python text", 0, 11)], [VEC_PYTHON])

            resp = retrieve_context(
                db, user.id, "What about France?",
                provider=provider, document_id=doc_a.id,
            )
            # Results should only be from doc_a
            for r in resp.results:
                assert r.document_id == doc_a.id
        finally:
            db.close()


# ==========================================================================
# 8. CROSS-USER ISOLATION
# ==========================================================================

class TestCrossUserIsolation:
    def test_user_a_cannot_see_user_b_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = User(name="RetIsoA", email="ret_iso_a@example.com", password_hash="h")
            user_b = User(name="RetIsoB", email="ret_iso_b@example.com", password_hash="h")
            db.add_all([user_a, user_b]); db.commit()
            db.refresh(user_a); db.refresh(user_b)

            create_doc_with_embeddings(db, user_a, [("France secret", 0, 13)], [VEC_FRANCE])
            create_doc_with_embeddings(db, user_b, [("Python secret", 0, 13)], [VEC_PYTHON])

            # User A searches — should NOT see Python
            resp_a = retrieve_context(db, user_a.id, "Tell me about Python", provider=provider)
            for r in resp_a.results:
                assert "Python" not in r.text

            # User B searches — should NOT see France
            resp_b = retrieve_context(db, user_b.id, "Tell me about France", provider=provider)
            for r in resp_b.results:
                assert "France" not in r.text
        finally:
            db.close()

    def test_document_id_cannot_bypass_ownership(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = User(name="RetIsoA2", email="ret_iso_a2@example.com", password_hash="h")
            user_b = User(name="RetIsoB2", email="ret_iso_b2@example.com", password_hash="h")
            db.add_all([user_a, user_b]); db.commit()
            db.refresh(user_a); db.refresh(user_b)

            doc_b = create_doc_with_embeddings(
                db, user_b, [("Python secret data", 0, 17)], [VEC_PYTHON],
            )

            # User A tries to access User B's document
            resp = retrieve_context(
                db, user_a.id, "Tell me about Python",
                provider=provider, document_id=doc_b.id,
            )
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()


# ==========================================================================
# 9. DETERMINISTIC ORDERING
# ==========================================================================

class TestDeterministicOrdering:
    def test_same_query_same_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetDet", email="ret_det@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc = create_doc_with_embeddings(
                db, user,
                [("France text", 0, 11), ("Python text", 11, 22), ("Mountain text", 22, 34)],
                [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN],
            )

            resp1 = retrieve_context(db, user.id, "What about France?", provider=provider)
            resp2 = retrieve_context(db, user.id, "What about France?", provider=provider)

            assert [r.chunk_id for r in resp1.results] == [r.chunk_id for r in resp2.results]
            assert [r.similarity_score for r in resp1.results] == [r.similarity_score for r in resp2.results]
        finally:
            db.close()


# ==========================================================================
# 10. FAILURE HANDLING
# ==========================================================================

class TestFailureHandling:
    def test_embedding_provider_failure(self):
        """Embedding provider error raises RetrievalError."""
        db = SessionLocal()
        try:
            user = User(name="RetFail", email="ret_fail@example.com", password_hash="h")
            db.add(user); db.commit()

            bad_provider = MagicMock(spec=EmbeddingProvider)
            bad_provider.embed_text.side_effect = EmbeddingError("Provider down")

            with pytest.raises(RetrievalError, match="embedding"):
                retrieve_context(db, user.id, "test query", provider=bad_provider)
        finally:
            db.close()

    def test_embedding_unexpected_exception(self):
        """Unexpected embedding error raises RetrievalError."""
        db = SessionLocal()
        try:
            user = User(name="RetFail2", email="ret_fail2@example.com", password_hash="h")
            db.add(user); db.commit()

            bad_provider = MagicMock(spec=EmbeddingProvider)
            bad_provider.embed_text.side_effect = RuntimeError("Unexpected")

            with pytest.raises(RetrievalError, match="Embedding provider failed"):
                retrieve_context(db, user.id, "test query", provider=bad_provider)
        finally:
            db.close()


# ==========================================================================
# 11. CONTEXT SIZE CONTROL
# ==========================================================================

class TestContextSizeControl:
    def test_max_context_chars_respected(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetSize", email="ret_size@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            # Many chunks with long text
            chunks_data = [(f"Chunk {i} " + "x" * 500, i * 510, (i + 1) * 510) for i in range(10)]
            embeddings = [VEC_FRANCE] * 10
            doc = create_doc_with_embeddings(db, user, chunks_data, embeddings)

            resp = retrieve_context(
                db, user.id, "France",
                provider=provider, max_context_chars=1000,
            )
            assert len(resp.context) <= 1200  # Some overhead for headers
        finally:
            db.close()


# ==========================================================================
# 12. EDGE CASES
# ==========================================================================

class TestEdgeCases:
    def test_no_embeddings_in_database(self):
        """Chunks without embeddings are skipped by vector search."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetNoEmb", email="ret_noemb@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            storage_key = storage_service.save(b"%PDF-1.4 test")
            doc = Document(
                user_id=user.id, original_filename="noemb.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc); db.commit(); db.refresh(doc)

            chunk = DocumentChunk(
                document_id=doc.id, chunk_index=0,
                text="No embedding", char_start=0, char_end=12,
                embedding=None,
            )
            db.add(chunk); db.commit()

            resp = retrieve_context(db, user.id, "anything", provider=provider)
            assert resp.total_results == 0
        finally:
            db.close()

    def test_missing_document_returns_empty(self):
        """Non-existent document_id returns empty (ownership check fails)."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = User(name="RetMiss", email="ret_miss@example.com", password_hash="h")
            db.add(user); db.commit()

            resp = retrieve_context(
                db, user.id, "test",
                provider=provider, document_id=99999,
            )
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()

    def test_retrieval_result_dataclass(self):
        """RetrievalResult has expected fields."""
        r = RetrievalResult(
            document_id=1, chunk_id=10, chunk_index=5,
            text="Hello", similarity_score=0.9,
            original_filename="test.pdf",
            page_start=3, page_end=4,
        )
        assert r.document_id == 1
        assert r.chunk_id == 10
        assert r.similarity_score == 0.9
        assert r.original_filename == "test.pdf"
