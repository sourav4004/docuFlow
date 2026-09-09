"""Phase 6 Step 1 tests: Hybrid Retrieval (vector + keyword/full-text search).

Uses FakeTestEmbedding for vector tests and real PostgreSQL for FTS.
"""

import math
import pytest
from typing import List

from sqlalchemy import text

from app.core import auth
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.embeddings.base import EmbeddingProvider
from app.services.vector_search import search_similar_chunks
from app.services.keyword_search import (
    search_by_keywords,
    KeywordSearchResult,
    DEFAULT_KEYWORD_TOP_K,
    MAX_KEYWORD_TOP_K,
)
from app.services.retrieval_service import (
    retrieve_context,
    RetrievalResult,
    RetrievalResponse,
    RetrievalError,
    QueryValidationError,
    _normalize_score,
    _deduplicate_results,
    _hybrid_retrieve,
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
# Cleanup helper — uses parameterized LIKE to avoid %-escaping issues
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


def _cleanup_hybrid_data(db):
    """Remove all test data for hyb_* emails."""
    for sql in _CLEANUP_QUERIES:
        db.execute(text(sql), {"pat": "hyb_%@example.com"})
    db.commit()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Isolate auth sessions and clean up test data."""
    auth._sessions.clear()
    db = SessionLocal()
    try:
        _cleanup_hybrid_data(db)
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        _cleanup_hybrid_data(db)
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
# 1. KEYWORD SEARCH — BASIC
# ==========================================================================

class TestKeywordSearchBasic:
    def test_exact_match(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSExact", "hyb_ks_exact@example.com")
            _create_doc_with_chunks(db, user, [
                ("The remote work policy allows flexible hours", 0, 46),
                ("France is a beautiful country in Europe", 0, 41),
            ], [VEC_REMOTE_WORK, VEC_FRANCE])
            results = search_by_keywords(db, user.id, "remote work policy")
            assert len(results) >= 1
            texts = " ".join(r.text for r in results)
            assert "remote" in texts.lower()
        finally:
            db.close()

    def test_partial_match(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSPartial", "hyb_ks_partial@example.com")
            _create_doc_with_chunks(db, user, [
                ("The refund policy states full refunds within 30 days", 0, 52),
                ("Python is a popular programming language", 0, 39),
            ], [VEC_POLICY, VEC_PYTHON])
            results = search_by_keywords(db, user.id, "refund")
            assert len(results) >= 1
            assert "refund" in results[0].text.lower()
        finally:
            db.close()

    def test_case_insensitive(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSCase", "hyb_ks_case@example.com")
            _create_doc_with_chunks(db, user, [
                ("France is beautiful and diverse", 0, 32),
            ], [VEC_FRANCE])
            r1 = search_by_keywords(db, user.id, "france")
            r2 = search_by_keywords(db, user.id, "FRANCE")
            r3 = search_by_keywords(db, user.id, "France")
            assert len(r1) == len(r2) == len(r3)
            assert all("France" in r.text for r in r1)
        finally:
            db.close()

    def test_no_results(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSNone", "hyb_ks_none@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16)], [VEC_FRANCE])
            results = search_by_keywords(db, user.id, "quantum computing")
            assert results == []
        finally:
            db.close()

    def test_stop_words_ignored(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSStop", "hyb_ks_stop@example.com")
            _create_doc_with_chunks(db, user, [
                ("The document covers important policy information", 0, 49),
            ], [VEC_POLICY])
            # Query with only stop words — should not crash
            search_by_keywords(db, user.id, "the a is")
        finally:
            db.close()

    def test_ranking_by_relevance(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSRank", "hyb_ks_rank@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy allows three days per week", 0, 48),
                ("The policy document covers insurance benefits", 0, 46),
                ("Work from home on Fridays", 0, 27),
            ], [VEC_REMOTE_WORK, VEC_POLICY, VEC_REMOTE_WORK])
            # plainto_tsquery requires ALL terms; only first chunk has all 3
            results = search_by_keywords(db, user.id, "remote work policy")
            assert len(results) >= 1
            assert "remote" in results[0].text.lower()
            # Partial match: search for just "work" should find 2 chunks
            results2 = search_by_keywords(db, user.id, "work")
            assert len(results2) >= 2
        finally:
            db.close()


# ==========================================================================
# 2. KEYWORD SEARCH — INPUT VALIDATION
# ==========================================================================

class TestKeywordSearchValidation:
    def test_empty_query_raises(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSEmpty", "hyb_ks_empty@example.com")
            with pytest.raises(ValueError, match="must not be empty"):
                search_by_keywords(db, user.id, "")
        finally:
            db.close()

    def test_whitespace_query_raises(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSWS", "hyb_ks_ws@example.com")
            with pytest.raises(ValueError, match="must not be empty"):
                search_by_keywords(db, user.id, "   ")
        finally:
            db.close()

    def test_top_k_zero_raises(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSTopK0", "hyb_ks_topk0@example.com")
            with pytest.raises(ValueError, match="top_k"):
                search_by_keywords(db, user.id, "test", top_k=0)
        finally:
            db.close()

    def test_top_k_too_large_raises(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSTopKB", "hyb_ks_topkb@example.com")
            with pytest.raises(ValueError, match="top_k"):
                search_by_keywords(db, user.id, "test", top_k=MAX_KEYWORD_TOP_K + 1)
        finally:
            db.close()


# ==========================================================================
# 3. KEYWORD SEARCH — UNICODE AND SPECIAL CHARACTERS
# ==========================================================================

class TestKeywordSearchUnicode:
    def test_unicode_text_searchable(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSUnicode", "hyb_ks_unicode@example.com")
            _create_doc_with_chunks(db, user, [
                ("\u8fdc\u7a0b\u5de5\u4f5c\u653f\u7b56\u5141\u8bb8\u6bcf\u5468\u4e09\u5929\u5c45\u5bb6\u529e\u516c", 0, 30),
            ], [VEC_REMOTE_WORK])
            # Should not raise
            search_by_keywords(db, user.id, "\u8fdc\u7a0b\u5de5\u4f5c")
        finally:
            db.close()

    def test_special_characters_safe(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSSpecial", "hyb_ks_special@example.com")
            _create_doc_with_chunks(db, user, [
                ("Normal document text about policies", 0, 35),
            ], [VEC_POLICY])
            # SQL injection attempts — should not raise
            search_by_keywords(db, user.id, "'; DROP TABLE users; --")
            search_by_keywords(db, user.id, "test OR 1=1")
            search_by_keywords(db, user.id, "test AND true")
        finally:
            db.close()

    def test_empty_text_chunks_handled(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSEmptyTxt", "hyb_ks_emptytxt@example.com")
            _create_doc_with_chunks(db, user, [
                ("", 0, 0),
                ("Valid text content", 0, 18),
            ], [VEC_FRANCE, VEC_FRANCE])
            results = search_by_keywords(db, user.id, "valid")
            assert len(results) >= 1
        finally:
            db.close()


# ==========================================================================
# 4. KEYWORD SEARCH — CROSS-USER ISOLATION
# ==========================================================================

class TestKeywordSearchIsolation:
    def test_user_a_cannot_see_user_b_chunks(self):
        db = SessionLocal()
        try:
            user_a = _create_user(db, "KSIsA", "hyb_ks_isa@example.com")
            user_b = _create_user(db, "KSIsB", "hyb_ks_isb@example.com")
            _create_doc_with_chunks(db, user_a, [("France secret", 0, 13)], [VEC_FRANCE])
            _create_doc_with_chunks(db, user_b, [("Python confidential", 0, 19)], [VEC_PYTHON])
            results_a = search_by_keywords(db, user_a.id, "Python")
            for r in results_a:
                assert "Python" not in r.text
            results_b = search_by_keywords(db, user_b.id, "France")
            for r in results_b:
                assert "France" not in r.text
        finally:
            db.close()

    def test_document_id_cannot_bypass_ownership(self):
        db = SessionLocal()
        try:
            user_a = _create_user(db, "KSIsA2", "hyb_ks_isa2@example.com")
            user_b = _create_user(db, "KSIsB2", "hyb_ks_isb2@example.com")
            doc_b = _create_doc_with_chunks(db, user_b, [("Python data", 0, 10)], [VEC_PYTHON])
            results = search_by_keywords(db, user_a.id, "Python", document_id=doc_b.id)
            assert results == []
        finally:
            db.close()


# ==========================================================================
# 5. KEYWORD SEARCH — DOCUMENT FILTERING
# ==========================================================================

class TestKeywordSearchDocumentFilter:
    def test_search_specific_document(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSDocF", "hyb_ks_docf@example.com")
            doc_a = _create_doc_with_chunks(db, user, [("France is beautiful", 0, 20)], [VEC_FRANCE], filename="france.pdf")
            _create_doc_with_chunks(db, user, [("France has great cuisine", 0, 25)], [VEC_FRANCE], filename="cuisine.pdf")
            results = search_by_keywords(db, user.id, "France", document_id=doc_a.id)
            for r in results:
                assert r.document_id == doc_a.id
        finally:
            db.close()


# ==========================================================================
# 6. KEYWORD SEARCH — RESULT STRUCTURE
# ==========================================================================

class TestKeywordSearchResult:
    def test_result_fields(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSRes", "hyb_ks_res@example.com")
            doc = _create_doc_with_chunks(db, user, [("Remote work policy details", 0, 28)], [VEC_REMOTE_WORK], filename="handbook.pdf")
            results = search_by_keywords(db, user.id, "remote work")
            assert len(results) >= 1
            r = results[0]
            assert isinstance(r, KeywordSearchResult)
            assert r.document_id == doc.id
            assert r.chunk_id > 0
            assert r.chunk_index == 0
            assert r.text == "Remote work policy details"
            assert 0 <= r.lexical_score <= 1.0
            assert r.original_filename == "handbook.pdf"
        finally:
            db.close()

    def test_deterministic_ordering(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "KSDet", "hyb_ks_det@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy is important", 0, 33)], [VEC_REMOTE_WORK])
            r1 = search_by_keywords(db, user.id, "remote work")
            r2 = search_by_keywords(db, user.id, "remote work")
            assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        finally:
            db.close()


# ==========================================================================
# 7. SCORE NORMALIZATION (unit tests)
# ==========================================================================

class TestScoreNormalization:
    def test_normalize_zero_range_positive(self):
        # Single non-zero result: normalize to 1.0
        assert _normalize_score(0.5, 0.5, 0.5) == 1.0

    def test_normalize_zero_range_zero(self):
        # Single zero result: normalize to 0.0
        assert _normalize_score(0.0, 0.0, 0.0) == 0.0

    def test_normalize_basic(self):
        assert _normalize_score(0.5, 0.0, 1.0) == 0.5

    def test_normalize_clamp_low(self):
        assert _normalize_score(-1.0, 0.0, 1.0) == 0.0

    def test_normalize_clamp_high(self):
        assert _normalize_score(2.0, 0.0, 1.0) == 1.0

    def test_normalize_custom_range(self):
        result = _normalize_score(0.7, 0.4, 1.0)
        assert abs(result - 0.5) < 0.01


# ==========================================================================
# 8. DEDUPLICATION (unit tests)
# ==========================================================================

class TestDeduplication:
    def test_no_duplicates(self):
        r1 = RetrievalResult(1, 10, 0, "text1", 0.9, final_score=0.9)
        r2 = RetrievalResult(1, 11, 1, "text2", 0.8, final_score=0.8)
        result = _deduplicate_results({10: r1, 11: r2})
        assert len(result) == 2

    def test_ordering_by_score_desc(self):
        r1 = RetrievalResult(1, 10, 0, "a", 0.5, final_score=0.3)
        r2 = RetrievalResult(1, 11, 1, "b", 0.5, final_score=0.9)
        r3 = RetrievalResult(1, 12, 2, "c", 0.5, final_score=0.6)
        result = _deduplicate_results({10: r1, 11: r2, 12: r3})
        assert [r.chunk_id for r in result] == [11, 12, 10]

    def test_tie_breaking_by_chunk_id(self):
        r1 = RetrievalResult(1, 10, 0, "a", 0.5, final_score=0.5)
        r2 = RetrievalResult(1, 11, 1, "b", 0.5, final_score=0.5)
        result = _deduplicate_results({11: r2, 10: r1})
        assert [r.chunk_id for r in result] == [10, 11]

    def test_empty(self):
        result = _deduplicate_results({})
        assert result == []


# ==========================================================================
# 9. HYBRID RETRIEVAL — INTEGRATION
# ==========================================================================

class TestHybridRetrieval:
    def test_hybrid_finds_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybBasic", "hyb_basic@example.com")
            _create_doc_with_chunks(db, user, [
                ("The remote work policy allows employees to work from home", 0, 61),
                ("France is a beautiful country in Europe", 0, 41),
                ("Python programming language is widely used", 0, 43),
            ], [VEC_REMOTE_WORK, VEC_FRANCE, VEC_PYTHON])
            resp = retrieve_context(db, user.id, "remote work policy", provider=provider, hybrid=True)
            assert resp.retrieval_mode == "hybrid"
            assert resp.total_results > 0
            assert resp.context != ""
        finally:
            db.close()

    def test_hybrid_deduplicates_chunks(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybDedup", "hyb_dedup@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy allows flexible schedules", 0, 46)], [VEC_REMOTE_WORK])
            resp = retrieve_context(db, user.id, "remote work policy", provider=provider, hybrid=True)
            chunk_ids = [r.chunk_id for r in resp.results]
            assert len(chunk_ids) == len(set(chunk_ids))
        finally:
            db.close()

    def test_hybrid_score_fields_populated(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybScore", "hyb_score@example.com")
            _create_doc_with_chunks(db, user, [("The refund policy provides full refunds", 0, 39)], [VEC_POLICY])
            resp = retrieve_context(db, user.id, "refund policy", provider=provider, hybrid=True)
            if resp.total_results > 0:
                r = resp.results[0]
                assert r.keyword_score is not None
                assert r.final_score is not None
                assert 0.0 <= r.keyword_score <= 1.0
                assert 0.0 <= r.final_score <= 1.0
        finally:
            db.close()

    def test_hybrid_ranks_relevant_chunks_higher(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybRank", "hyb_rank@example.com")
            _create_doc_with_chunks(db, user, [
                ("Remote work policy allows 3 days per week", 0, 44),
                ("The document discusses insurance benefits", 0, 42),
                ("France is a beautiful European country", 0, 38),
            ], [VEC_REMOTE_WORK, VEC_POLICY, VEC_FRANCE])
            resp = retrieve_context(db, user.id, "remote work policy", provider=provider, hybrid=True)
            assert resp.total_results >= 1
            assert "remote" in resp.results[0].text.lower()
        finally:
            db.close()


# ==========================================================================
# 10. HYBRID vs VECTOR-ONLY
# ==========================================================================

class TestHybridVsVectorOnly:
    def test_vector_only_mode_preserved(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "VecOnly", "hyb_veconly@example.com")
            _create_doc_with_chunks(db, user, [("France is great", 0, 16), ("Python is fun", 13, 26)], [VEC_FRANCE, VEC_PYTHON])
            resp = retrieve_context(db, user.id, "France", provider=provider, hybrid=False)
            assert resp.retrieval_mode == "vector"
            for r in resp.results:
                assert r.keyword_score is None
                assert r.final_score is None
        finally:
            db.close()

    def test_hybrid_finds_keyword_matches_vector_misses(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybMiss", "hyb_miss@example.com")
            _create_doc_with_chunks(db, user, [("The company reimbursement policy covers travel expenses", 0, 58)], [VEC_POLICY])
            resp = retrieve_context(db, user.id, "reimbursement", provider=provider, hybrid=True, min_similarity=0.1)
            texts = " ".join(r.text for r in resp.results)
            assert "reimbursement" in texts.lower()
        finally:
            db.close()


# ==========================================================================
# 11. HYBRID — THRESHOLD FILTERING
# ==========================================================================

class TestHybridThresholdFiltering:
    def test_high_threshold_filters_out(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybThresh", "hyb_thresh@example.com")
            _create_doc_with_chunks(db, user, [("France is beautiful", 0, 20)], [VEC_FRANCE])
            resp = retrieve_context(db, user.id, "quantum computing", provider=provider, hybrid=True, min_similarity=0.99)
            assert resp.total_results == 0
            assert resp.context == ""
        finally:
            db.close()

    def test_low_threshold_includes_more(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybLowT", "hyb_lowt@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy details here", 0, 33), ("France is nice", 0, 15)], [VEC_REMOTE_WORK, VEC_FRANCE])
            resp_high = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, min_similarity=0.9)
            resp_low = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, min_similarity=0.1)
            assert resp_low.total_results >= resp_high.total_results
        finally:
            db.close()


# ==========================================================================
# 12. HYBRID — CROSS-USER ISOLATION
# ==========================================================================

class TestHybridCrossUserIsolation:
    def test_user_a_cannot_see_user_b_via_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = _create_user(db, "HybIsoA", "hyb_isoa@example.com")
            user_b = _create_user(db, "HybIsoB", "hyb_isob@example.com")
            _create_doc_with_chunks(db, user_a, [("France secret document", 0, 22)], [VEC_FRANCE])
            _create_doc_with_chunks(db, user_b, [("Python confidential information", 0, 31)], [VEC_PYTHON])
            resp_a = retrieve_context(db, user_a.id, "Python", provider=provider, hybrid=True)
            for r in resp_a.results:
                assert "Python" not in r.text
            resp_b = retrieve_context(db, user_b.id, "France", provider=provider, hybrid=True)
            for r in resp_b.results:
                assert "France" not in r.text
        finally:
            db.close()

    def test_document_id_cannot_bypass_ownership_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user_a = _create_user(db, "HybIsoA2", "hyb_isoa2@example.com")
            user_b = _create_user(db, "HybIsoB2", "hyb_isob2@example.com")
            doc_b = _create_doc_with_chunks(db, user_b, [("Python confidential data", 0, 25)], [VEC_PYTHON])
            resp = retrieve_context(db, user_a.id, "Python", provider=provider, hybrid=True, document_id=doc_b.id)
            assert resp.total_results == 0
        finally:
            db.close()


# ==========================================================================
# 13. HYBRID — CONFIGURATION
# ==========================================================================

class TestHybridConfiguration:
    def test_custom_weights(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybConf", "hyb_conf@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy allows flexibility", 0, 38), ("France is a country", 0, 20)], [VEC_REMOTE_WORK, VEC_FRANCE])
            resp_v = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, vector_weight=0.9, keyword_weight=0.1)
            resp_k = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, vector_weight=0.1, keyword_weight=0.9)
            assert resp_v.total_results > 0
            assert resp_k.total_results > 0
        finally:
            db.close()

    def test_top_k_limits_results(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybTopK", "hyb_topk@example.com")
            chunks = [(f"Chunk {i} about remote work", i * 30, (i + 1) * 30) for i in range(10)]
            _create_doc_with_chunks(db, user, chunks, [VEC_REMOTE_WORK] * 10)
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True, top_k=3)
            assert resp.total_results <= 3
        finally:
            db.close()


# ==========================================================================
# 14. HYBRID — EMPTY DATABASE
# ==========================================================================

class TestHybridEmptyDatabase:
    def test_empty_database(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybEmpty", "hyb_empty@example.com")
            resp = retrieve_context(db, user.id, "anything", provider=provider, hybrid=True)
            assert resp.total_results == 0
            assert resp.context == ""
            assert resp.retrieval_mode == "hybrid"
        finally:
            db.close()

    def test_chunks_without_embeddings_found_by_keyword(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybNoEmb", "hyb_noemb@example.com")
            storage_key = storage_service.save(b"%PDF-1.4 test")
            doc = Document(user_id=user.id, original_filename="noemb.pdf", storage_key=storage_key, mime_type="application/pdf", file_size=100, status="READY")
            db.add(doc)
            db.commit()
            db.refresh(doc)
            chunk = DocumentChunk(document_id=doc.id, chunk_index=0, text="Remote work policy text without embeddings", char_start=0, char_end=44, embedding=None)
            db.add(chunk)
            db.commit()
            resp = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            texts = " ".join(r.text for r in resp.results)
            assert "remote" in texts.lower()
        finally:
            db.close()


# ==========================================================================
# 15. HYBRID — RESPONSE METADATA
# ==========================================================================

class TestHybridResponseMetadata:
    def test_retrieval_mode_in_response(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybMeta", "hyb_meta@example.com")
            _create_doc_with_chunks(db, user, [("Some text", 0, 9)], [VEC_FRANCE])
            resp_h = retrieve_context(db, user.id, "something", provider=provider, hybrid=True)
            assert resp_h.retrieval_mode == "hybrid"
            resp_v = retrieve_context(db, user.id, "something", provider=provider, hybrid=False)
            assert resp_v.retrieval_mode == "vector"
        finally:
            db.close()

    def test_score_property_uses_final_score(self):
        r = RetrievalResult(document_id=1, chunk_id=10, chunk_index=0, text="test", similarity_score=0.5, keyword_score=0.8, final_score=0.7)
        assert r.score == 0.7

    def test_score_property_falls_back_to_similarity(self):
        r = RetrievalResult(document_id=1, chunk_id=10, chunk_index=0, text="test", similarity_score=0.5)
        assert r.score == 0.5


# ==========================================================================
# 16. HYBRID — EDGE CASES
# ==========================================================================

class TestHybridEdgeCases:
    def test_single_chunk_both_vector_and_keyword(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybEdge1", "hyb_edge1@example.com")
            _create_doc_with_chunks(db, user, [("Remote work policy details", 0, 28)], [VEC_REMOTE_WORK])
            resp = retrieve_context(db, user.id, "remote work policy", provider=provider, hybrid=True)
            assert resp.total_results == 1
            r = resp.results[0]
            assert r.keyword_score is not None
            assert r.final_score is not None
            assert 0.0 <= r.final_score <= 1.0
        finally:
            db.close()

    def test_many_chunks_deterministic(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybDet", "hyb_det@example.com")
            chunks = [(f"Remote work item {i}", i * 25, (i + 1) * 25) for i in range(8)]
            _create_doc_with_chunks(db, user, chunks, [VEC_REMOTE_WORK] * 8)
            r1 = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            r2 = retrieve_context(db, user.id, "remote work", provider=provider, hybrid=True)
            assert [r.chunk_id for r in r1.results] == [r.chunk_id for r in r2.results]
        finally:
            db.close()

    def test_unicode_query_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybUni", "hyb_uni@example.com")
            _create_doc_with_chunks(db, user, [("Some text content about policies", 0, 32)], [VEC_POLICY])
            # Should not raise
            retrieve_context(db, user.id, "\u4ec0\u4e48\u662f\u8fdc\u7a0b\u5de5\u4f5c\uff1f", provider=provider, hybrid=True)
        finally:
            db.close()

    def test_long_query_hybrid(self):
        db = SessionLocal()
        try:
            provider = FakeTestEmbedding()
            user = _create_user(db, "HybLong", "hyb_long@example.com")
            with pytest.raises(QueryValidationError, match="must not exceed"):
                retrieve_context(db, user.id, "a" * 2001, provider=provider, hybrid=True)
        finally:
            db.close()

    def test_min_similarity_uses_score(self):
        r = RetrievalResult(document_id=1, chunk_id=10, chunk_index=0, text="test", similarity_score=0.9, final_score=0.2)
        assert r.score == 0.2


# ==========================================================================
# 17. TRIGGER VERIFICATION
# ==========================================================================

class TestSearchVectorTrigger:
    def test_insert_populates_search_vector(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "TrigIns", "hyb_trig_ins@example.com")
            doc = _create_doc_with_chunks(db, user, [("Trigger test text", 0, 18)], [VEC_FRANCE])
            row = db.execute(text("SELECT search_vector IS NOT NULL FROM document_chunks WHERE document_id = :doc_id AND chunk_index = 0"), {"doc_id": doc.id}).fetchone()
            assert row[0] is True
        finally:
            db.close()

    def test_update_populates_search_vector(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "TrigUpd", "hyb_trig_upd@example.com")
            doc = _create_doc_with_chunks(db, user, [("Original text", 0, 14)], [VEC_FRANCE])
            db.execute(text("UPDATE document_chunks SET text = 'Updated remote work content' WHERE document_id = :doc_id AND chunk_index = 0"), {"doc_id": doc.id})
            db.commit()
            row = db.execute(text("SELECT search_vector IS NOT NULL, text FROM document_chunks WHERE document_id = :doc_id AND chunk_index = 0"), {"doc_id": doc.id}).fetchone()
            assert row[0] is True
            assert "Updated" in row[1]
        finally:
            db.close()

    def test_keyword_search_finds_newly_inserted_chunks(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "TrigNew", "hyb_trig_new@example.com")
            _create_doc_with_chunks(db, user, [("Quantum computing breakthrough announced today", 0, 46)], [VEC_FRANCE])
            results = search_by_keywords(db, user.id, "quantum computing")
            assert len(results) == 1
            assert "quantum" in results[0].text.lower()
        finally:
            db.close()
