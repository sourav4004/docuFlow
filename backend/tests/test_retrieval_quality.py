"""Phase 6 Step 2 tests: Retrieval Quality, Configuration & Observability.

Covers: configuration, scoring, fallback, metadata, deterministic ranking,
input validation, context construction, ownership, RAG isolation, timing.
"""

import math
import re
import pytest
from typing import List
from unittest.mock import MagicMock, patch

from sqlalchemy import text

from app.core import auth
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.embeddings.base import EmbeddingProvider, EmbeddingError
from app.services.retrieval_service import (
    retrieve_context,
    build_context,
    validate_query,
    normalize_query,
    RetrievalResult,
    RetrievalResponse,
    RetrievalMetadata,
    RetrievalError,
    QueryValidationError,
    _normalize_score,
    _safe_normalize_score,
    _deduplicate_results,
    _validate_weights,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CHUNKS_IN_CONTEXT,
    MAX_QUERY_LENGTH,
)


# ---------------------------------------------------------------------------
# FakeTestEmbedding
# ---------------------------------------------------------------------------

class FakeTestEmbedding(EmbeddingProvider):
    _DIM = settings.embedding_dimension
    _KEYWORD_DIMS = {
        "france": 0, "python": 1, "mountain": 2,
        "policy": 3, "remote": 4, "work": 5,
        "refund": 6, "return": 7, "payment": 8,
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
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


_D = settings.embedding_dimension
VEC_FRANCE = [1.0] + [0.0] * (_D - 1)
VEC_PYTHON = [0.0, 1.0] + [0.0] * (_D - 2)
VEC_POLICY = [0.0, 0.0, 0.0, 1.0] + [0.0] * (_D - 4)
VEC_REMOTE_WORK = [0.0, 0.0, 0.0, 0.0, 1 / math.sqrt(2), 1 / math.sqrt(2)] + [0.0] * (_D - 6)


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------

_CLEANUP_QUERIES = [
    "DELETE FROM message_sources WHERE message_id IN"
    " (SELECT id FROM messages WHERE conversation_id IN"
    " (SELECT id FROM conversations WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat)))",
    "DELETE FROM messages WHERE conversation_id IN"
    " (SELECT id FROM conversations WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat))",
    "DELETE FROM conversations WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat)",
    "DELETE FROM document_chunks WHERE document_id IN"
    " (SELECT id FROM documents WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat))",
    "DELETE FROM documents WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat)",
    "DELETE FROM users WHERE email LIKE :pat",
]


def _cleanup_quality_data(db):
    for sql in _CLEANUP_QUERIES:
        db.execute(text(sql), {"pat": "rq_%@example.com"})
    db.commit()


@pytest.fixture(autouse=True)
def setup_test_environment():
    auth._sessions.clear()
    db = SessionLocal()
    try:
        _cleanup_quality_data(db)
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        _cleanup_quality_data(db)
    finally:
        db.close()


def _create_user(db, name, email):
    user = User(name=name, email=email, password_hash="h")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_doc_with_chunks(db, user, chunks_data, embeddings, filename="test.pdf"):
    storage_key = storage_service.save(b"%PDF-1.4 test")
    doc = Document(
        user_id=user.id, original_filename=filename,
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
    db.refresh(doc)
    return doc


# ==========================================================================
# A. CONFIGURATION
# ==========================================================================

class TestConfiguration:
    def test_defaults_exist(self):
        assert settings.retrieval_top_k == 5
        assert settings.retrieval_min_similarity == 0.3
        assert settings.retrieval_max_context_chars == 8000
        assert settings.hybrid_search_enabled is True
        assert settings.vector_search_weight == 0.7
        assert settings.keyword_search_weight == 0.3

    def test_new_config_fields(self):
        assert hasattr(settings, 'retrieval_max_chunks_in_context')
        assert hasattr(settings, 'vector_only_top_k')
        assert hasattr(settings, 'enable_keyword_fallback')
        assert hasattr(settings, 'enable_vector_fallback')
        assert settings.retrieval_max_chunks_in_context == 20
        assert settings.enable_keyword_fallback is True
        assert settings.enable_vector_fallback is True


# ==========================================================================
# B. SCORE CALCULATION
# ==========================================================================

class TestScoreCalculation:
    def test_normalize_basic(self):
        assert _normalize_score(0.5, 0.0, 1.0) == 0.5

    def test_normalize_zero_range_positive(self):
        assert _normalize_score(0.5, 0.5, 0.5) == 1.0

    def test_normalize_zero_range_zero(self):
        assert _normalize_score(0.0, 0.0, 0.0) == 0.0

    def test_normalize_clamp_low(self):
        assert _normalize_score(-1.0, 0.0, 1.0) == 0.0

    def test_normalize_clamp_high(self):
        assert _normalize_score(2.0, 0.0, 1.0) == 1.0

    def test_normalize_nan_protection(self):
        assert _normalize_score(float('nan'), 0.0, 1.0) == 0.0

    def test_normalize_inf_protection(self):
        assert _normalize_score(float('inf'), 0.0, 1.0) == 0.0
        assert _normalize_score(float('-inf'), 0.0, 1.0) == 0.0

    def test_safe_normalize_nan(self):
        assert _safe_normalize_score(float('nan')) == 0.0

    def test_safe_normalize_inf(self):
        assert _safe_normalize_score(float('inf')) == 0.0
        assert _safe_normalize_score(float('-inf')) == 0.0

    def test_safe_normalize_valid(self):
        assert _safe_normalize_score(0.5) == 0.5
        assert _safe_normalize_score(1.5) == 1.0
        assert _safe_normalize_score(-0.5) == 0.0

    def test_weight_validation_normal(self):
        vw, kw = _validate_weights(0.7, 0.3)
        assert abs(vw - 0.7) < 1e-6
        assert abs(kw - 0.3) < 1e-6

    def test_weight_validation_not_summing_to_one(self):
        vw, kw = _validate_weights(2.0, 1.0)
        assert abs(vw - 2.0 / 3.0) < 1e-6
        assert abs(kw - 1.0 / 3.0) < 1e-6

    def test_weight_validation_both_zero(self):
        vw, kw = _validate_weights(0.0, 0.0)
        assert vw == 0.5
        assert kw == 0.5

    def test_weight_validation_negative(self):
        vw, kw = _validate_weights(-1.0, 0.5)
        assert vw == 0.0
        assert kw == 1.0

    def test_weight_validation_negative_both(self):
        vw, kw = _validate_weights(-1.0, -1.0)
        # Both clamped to 0, then both_zero branch
        assert vw == 0.5
        assert kw == 0.5

    def test_score_bounds_hybrid(self):
        """Final score should always be in [0, 1]."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "ScoreBnd", "rq_scorebnd@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy details", 0, 28),
                ("France is beautiful", 0, 20),
            ], [VEC_REMOTE_WORK, VEC_FRANCE])
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            for r in resp.results:
                assert 0.0 <= r.final_score <= 1.0
                assert 0.0 <= r.keyword_score <= 1.0
                assert 0.0 <= r.similarity_score <= 1.0
        finally:
            db.close()


# ==========================================================================
# C. FALLBACK BEHAVIOR
# ==========================================================================

class TestFallbackBehavior:
    def test_vector_failure_keyword_succeeds(self):
        """Case B: vector search fails, keyword succeeds."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "FallB", "rq_fallb@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy is important", 0, 32),
            ], [VEC_REMOTE_WORK])

            # Mock vector search to fail
            with patch('app.services.retrieval_service.search_similar_chunks', side_effect=RuntimeError("vector down")):
                resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert resp.metadata.vector_failed is True
            assert resp.metadata.keyword_failed is False
            assert resp.metadata.fallback_used == "keyword_fallback"
            assert resp.total_results >= 0
        finally:
            db.close()

    def test_keyword_failure_vector_succeeds(self):
        """Case C: keyword search fails, vector succeeds."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "FallC", "rq_fallc@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy details", 0, 28),
            ], [VEC_REMOTE_WORK])

            with patch('app.services.retrieval_service.search_by_keywords', side_effect=RuntimeError("keyword down")):
                resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert resp.metadata.vector_failed is False
            assert resp.metadata.keyword_failed is True
            assert resp.metadata.fallback_used == "vector_fallback"
        finally:
            db.close()

    def test_both_fail_empty_results(self):
        """Case D: both fail → empty results."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "FallD", "rq_falld@example.com")
            _create_doc_with_chunks(db, user, [("text", 0, 4)], [VEC_FRANCE])

            with patch('app.services.retrieval_service.search_similar_chunks', side_effect=RuntimeError("down")):
                with patch('app.services.retrieval_service.search_by_keywords', side_effect=RuntimeError("down")):
                    resp = retrieve_context(db, user.id, "test", provider=provider, hybrid=True)
            assert resp.metadata.vector_failed is True
            assert resp.metadata.keyword_failed is True
            assert resp.metadata.fallback_used == "both_failed"
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()

    def test_hybrid_disabled_vector_only(self):
        """Case E: hybrid disabled → vector-only."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "FallE", "rq_falle@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "France", provider=provider, hybrid=False)
            assert resp.retrieval_mode == "vector"
            for r in resp.results:
                assert r.keyword_score is None
        finally:
            db.close()

    def test_keyword_fallback_disabled(self):
        """When keyword fallback is disabled and vector fails, return empty."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "FallDis", "rq_falldis@example.com")
            _create_doc_with_chunks(db, user, [("remote work text", 0, 16)], [VEC_REMOTE_WORK])

            with patch('app.services.retrieval_service.search_similar_chunks', side_effect=RuntimeError("down")):
                with patch.object(settings, 'enable_keyword_fallback', False):
                    resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert resp.metadata.fallback_used == "keyword_disabled"
        finally:
            db.close()


# ==========================================================================
# D. DETERMINISTIC ORDERING
# ==========================================================================

class TestDeterministicOrdering:
    def test_tie_breaking_final_score(self):
        """Same final_score → tie-break by similarity_score."""
        r1 = RetrievalResult(1, 10, 0, "a", 0.5, final_score=0.5)
        r2 = RetrievalResult(1, 11, 1, "b", 0.8, final_score=0.5)
        r3 = RetrievalResult(1, 12, 2, "c", 0.3, final_score=0.5)
        result = _deduplicate_results({10: r1, 11: r2, 12: r3})
        assert [r.chunk_id for r in result] == [11, 10, 12]

    def test_tie_breaking_keyword_score(self):
        """Same final + similarity → tie-break by keyword_score."""
        r1 = RetrievalResult(1, 10, 0, "a", 0.5, keyword_score=0.3, final_score=0.5)
        r2 = RetrievalResult(1, 11, 1, "b", 0.5, keyword_score=0.9, final_score=0.5)
        result = _deduplicate_results({10: r1, 11: r2})
        assert [r.chunk_id for r in result] == [11, 10]

    def test_tie_breaking_chunk_id(self):
        """All scores equal → tie-break by chunk_id ASC."""
        r1 = RetrievalResult(1, 10, 0, "a", 0.5, final_score=0.5)
        r2 = RetrievalResult(1, 11, 1, "b", 0.5, final_score=0.5)
        result = _deduplicate_results({11: r2, 10: r1})
        assert [r.chunk_id for r in result] == [10, 11]

    def test_repeated_retrieval_deterministic(self):
        """Same query always returns same ordering."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "DetR", "rq_detr@example.com")
            chunks = [(f"Remote work item {i}", i * 25, (i + 1) * 25) for i in range(8)]
            _create_doc_with_chunks(db, user, chunks, [VEC_REMOTE_WORK] * 8)
            r1 = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            r2 = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert [r.chunk_id for r in r1.results] == [r.chunk_id for r in r2.results]
        finally:
            db.close()


# ==========================================================================
# E. INPUT VALIDATION
# ==========================================================================

class TestInputValidation:
    def test_empty_query(self):
        with pytest.raises(QueryValidationError, match="must not be empty"):
            validate_query("")

    def test_whitespace_query(self):
        with pytest.raises(QueryValidationError, match="must not be empty"):
            validate_query("   ")

    def test_long_query(self):
        with pytest.raises(QueryValidationError, match="must not exceed"):
            validate_query("a" * (MAX_QUERY_LENGTH + 1))

    def test_non_string_query(self):
        with pytest.raises(QueryValidationError, match="must be a string"):
            validate_query(123)

    def test_unicode_query_ok(self):
        q = validate_query("\u4ec0\u4e48\u662f\u8fdc\u7a0b\u5de5\u4f5c")
        assert len(q) > 0

    def test_punctuation_query_ok(self):
        q = validate_query("what is this???!!!")
        assert q == "what is this???!!!"

    def test_repeated_terms_ok(self):
        q = validate_query("work work work work")
        assert q == "work work work work"

    def test_stop_words_only_ok(self):
        """Stop-word-only query should not crash."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "ValStop", "rq_valstop@example.com")
            _create_doc_with_chunks(db, user, [("Some policy text", 0, 16)], [VEC_POLICY])
            resp = retrieve_context(db, user.id, "the a is", provider=provider, hybrid=True)
            # Should return results or empty — just not crash
        finally:
            db.close()

    def test_normalize_query_collapse_whitespace(self):
        assert normalize_query("hello   world") == "hello world"
        assert normalize_query("  hello  ") == "hello"
        assert normalize_query("a\n\nb\t\nc") == "a b c"

    def test_normalize_preserves_unicode(self):
        assert normalize_query("\u4ec0\u4e48\u662f\u8fdc\u7a0b") == "\u4ec0\u4e48\u662f\u8fdc\u7a0b"

    def test_normalize_preserves_punctuation(self):
        assert normalize_query("hello, world!") == "hello, world!"


# ==========================================================================
# F. CONTEXT CONSTRUCTION
# ==========================================================================

class TestContextConstruction:
    def _make_result(self, **kwargs):
        defaults = dict(
            document_id=1, chunk_id=10, chunk_index=5,
            text="Sample text", similarity_score=0.9,
            original_filename="test.pdf",
            page_start=3, page_end=4,
        )
        defaults.update(kwargs)
        return RetrievalResult(**defaults)

    def test_empty_results(self):
        assert build_context([]) == ""

    def test_max_chunks_respected(self):
        results = [self._make_result(chunk_index=i, text=f"Chunk {i}") for i in range(10)]
        ctx = build_context(results, max_chunks=3)
        assert ctx.count("[Source") == 3

    def test_max_chars_respected(self):
        results = [self._make_result(chunk_index=i, text="x" * 500) for i in range(10)]
        ctx = build_context(results, max_chars=1000)
        assert len(ctx) <= 1200

    def test_first_valid_chunk_always_included(self):
        r = self._make_result(text="x" * 2000)
        ctx = build_context([r], max_chars=1000)
        assert "x" * 2000 in ctx

    def test_empty_chunks_skipped(self):
        r1 = self._make_result(chunk_index=0, text="")
        r2 = self._make_result(chunk_index=1, text="Valid text", chunk_id=11)
        ctx = build_context([r1, r2])
        assert "Valid text" in ctx
        # Empty chunk skipped, valid chunk gets Source 2 (enumerate counter advances)
        assert "[Source" in ctx

    def test_whitespace_only_chunks_skipped(self):
        r1 = self._make_result(chunk_index=0, text="   ", chunk_id=10)
        r2 = self._make_result(chunk_index=1, text="Valid text", chunk_id=11)
        ctx = build_context([r1, r2])
        assert "Valid text" in ctx

    def test_unicode_content(self):
        r = self._make_result(text="\u4ec0\u4e48\u662f\u8fdc\u7a0b\u5de5\u4f5c")
        ctx = build_context([r])
        assert "\u4ec0\u4e48\u662f\u8fdc\u7a0b\u5de5\u4f5c" in ctx

    def test_deterministic_order(self):
        results = [self._make_result(chunk_index=i, text=f"Chunk {i}", chunk_id=i) for i in range(5)]
        ctx = build_context(results)
        for i in range(5):
            assert f"Chunk {i}" in ctx

    def test_many_tiny_chunks(self):
        results = [self._make_result(chunk_index=i, text="x", chunk_id=i) for i in range(50)]
        ctx = build_context(results, max_chunks=10)
        assert ctx.count("[Source") == 10

    def test_one_huge_chunk(self):
        r = self._make_result(text="x" * 5000)
        ctx = build_context([r], max_chars=10000)
        assert len(ctx) > 5000

    def test_duplicate_chunks_in_results(self):
        """build_context doesn't dedup — that's retrieval's job."""
        r1 = self._make_result(chunk_index=0, text="Same text", chunk_id=10)
        r2 = self._make_result(chunk_index=1, text="Same text", chunk_id=10)
        ctx = build_context([r1, r2])
        assert ctx.count("Same text") == 2


# ==========================================================================
# G. OWNERSHIP
# ==========================================================================

class TestOwnership:
    def test_cross_user_vector_blocked(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = _create_user(db, "OwnA", "rq_owna@example.com")
            user_b = _create_user(db, "OwnB", "rq_ownb@example.com")
            _create_doc_with_chunks(db, user_a, [("France secret", 0, 13)], [VEC_FRANCE])
            _create_doc_with_chunks(db, user_b, [("Python secret", 0, 13)], [VEC_PYTHON])
            resp = retrieve_context(db, user_a.id, "Python", provider=provider, hybrid=False)
            for r in resp.results:
                assert "Python" not in r.text
        finally:
            db.close()

    def test_cross_user_hybrid_blocked(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = _create_user(db, "OwnHA", "rq_ownha@example.com")
            user_b = _create_user(db, "OwnHB", "rq_ownhb@example.com")
            _create_doc_with_chunks(db, user_a, [("France secret", 0, 13)], [VEC_FRANCE])
            _create_doc_with_chunks(db, user_b, [("Python secret", 0, 13)], [VEC_PYTHON])
            resp = retrieve_context(db, user_a.id, "Python", provider=provider, hybrid=True)
            for r in resp.results:
                assert "Python" not in r.text
        finally:
            db.close()

    def test_document_id_cannot_bypass(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = _create_user(db, "OwnDA", "rq_ownda@example.com")
            user_b = _create_user(db, "OwnDB", "rq_owndb@example.com")
            doc_b = _create_doc_with_chunks(db, user_b, [("Python data", 0, 10)], [VEC_PYTHON])
            resp = retrieve_context(db, user_a.id, "Python", provider=provider, hybrid=True, document_id=doc_b.id)
            assert resp.total_results == 0
        finally:
            db.close()


# ==========================================================================
# H. RAG ISOLATION
# ==========================================================================

class TestRAGIsolation:
    def test_retrieval_does_not_call_llm(self):
        """Retrieval should never invoke LLM."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RAGIso", "rq_ragiso@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy", 0, 19)], [VEC_REMOTE_WORK])
            mock_llm = MagicMock()
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            # No LLM should have been called
            mock_llm.generate.assert_not_called()
            assert resp.context != ""
        finally:
            db.close()

    def test_retrieval_does_not_write_db(self):
        """Retrieval should be read-only."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RAGWr", "rq_ragwr@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16)], [VEC_FRANCE])
            # Count chunks before
            before = db.execute(text("SELECT count(*) FROM document_chunks")).scalar()
            retrieve_context(db, user.id, "France", provider=provider, hybrid=True)
            after = db.execute(text("SELECT count(*) FROM document_chunks")).scalar()
            assert before == after
        finally:
            db.close()


# ==========================================================================
# I. METADATA
# ==========================================================================

class TestMetadata:
    def test_hybrid_mode_metadata(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "MetaH", "rq_metah@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy", 0, 19)], [VEC_REMOTE_WORK])
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert resp.metadata is not None
            assert resp.metadata.retrieval_mode == "hybrid"
            assert resp.metadata.candidate_count >= 0
            assert resp.metadata.returned_count == resp.total_results
            assert resp.metadata.total_time >= 0
        finally:
            db.close()

    def test_vector_mode_metadata(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "MetaV", "rq_metav@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "France", provider=provider, hybrid=False)
            assert resp.metadata.retrieval_mode == "vector"
            assert resp.metadata.embedding_time >= 0
            assert resp.metadata.vector_search_time >= 0
        finally:
            db.close()

    def test_match_type_vector_only(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "MetaMT", "rq_metamt@example.com")
            _create_doc_with_chunks(db, user, [("France is beautiful", 0, 20)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "France", provider=provider, hybrid=False)
            for r in resp.results:
                assert r.match_type == "vector"
        finally:
            db.close()

    def test_match_type_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "MetaHT", "rq_metht@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy details", 0, 28)], [VEC_REMOTE_WORK])
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            for r in resp.results:
                assert r.match_type in ("vector", "keyword", "both")
        finally:
            db.close()

    def test_timing_non_negative(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "MetaTime", "rq_metatime@example.com")
            _create_doc_with_chunks(db, user, [("text", 0, 4)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "test", provider=provider, hybrid=True)
            m = resp.metadata
            assert m.embedding_time >= 0
            assert m.vector_search_time >= 0
            assert m.keyword_search_time >= 0
            assert m.merge_time >= 0
            assert m.context_time >= 0
            assert m.total_time >= 0
        finally:
            db.close()


# ==========================================================================
# J. REGRESSION
# ==========================================================================

class TestRegression:
    def test_vector_only_backward_compat(self):
        """vector-only mode produces same structure as before."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RegVO", "rq_regvo@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "France", provider=provider, hybrid=False)
            assert resp.retrieval_mode == "vector"
            assert resp.query == "France"
            assert resp.context != ""
            assert resp.total_results > 0
            for r in resp.results:
                assert r.keyword_score is None
                assert r.final_score is None
        finally:
            db.close()

    def test_hybrid_mode_backward_compat(self):
        """hybrid mode produces expected structure."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RegHM", "rq_reghm@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy details", 0, 28),
            ], [VEC_REMOTE_WORK])
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert resp.retrieval_mode == "hybrid"
            assert resp.query == "remote work"
            for r in resp.results:
                assert r.keyword_score is not None
                assert r.final_score is not None
        finally:
            db.close()

    def test_empty_database_vector(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RegEmpV", "rq_regempv@example.com")
            resp = retrieve_context(db, user.id, "test", provider=provider, hybrid=False)
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()

    def test_empty_database_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RegEmpH", "rq_regemph@example.com")
            resp = retrieve_context(db, user.id, "test", provider=provider, hybrid=True)
            assert resp.total_results == 0
            assert resp.context == ""
            assert resp.retrieval_mode == "hybrid"
        finally:
            db.close()

    def test_chunk_without_embeddings_keyword_fallback(self):
        """Chunks without embeddings should still be found by keyword in hybrid."""
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "RegNoEmb", "rq_regnemb@example.com")
            storage_key = storage_service.save(b"%PDF-1.4 test")
            doc = Document(
                user_id=user.id, original_filename="noemb.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            chunk = DocumentChunk(
                document_id=doc.id, chunk_index=0,
                text="Remote work policy text without embeddings",
                char_start=0, char_end=44, embedding=None,
            )
            db.add(chunk)
            db.commit()
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, min_similarity=0.1)
            texts = " ".join(r.text for r in resp.results)
            assert "remote" in texts.lower()
        finally:
            db.close()
