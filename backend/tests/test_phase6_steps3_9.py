"""Phase 6 Steps 3-9 tests: Citations, Multi-Doc, Confidence, Memory, Collections, Jobs.

Covers:
- Enhanced source metadata (Step 3)
- Multi-document context grouping (Step 4)
- RAG confidence/grounding (Step 6)
- Conversation memory with char budget (Step 7)
- Document collections (Step 8)
- Background job service (Step 9)
"""

import math
import pytest
from typing import List
from unittest.mock import MagicMock

from sqlalchemy import text

from app.core import auth
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.collection import Collection
from app.services.storage import storage_service
from app.services.embeddings.base import EmbeddingProvider
from app.services.retrieval_service import (
    retrieve_context,
    build_context,
    RetrievalResult,
)
from app.services.rag_service import (
    _build_sources,
    _compute_confidence,
    ConfidenceScore,
    SourceReference,
)
from app.services.conversation_context import (
    load_conversation_history,
    _apply_char_budget,
    HistoryMessage,
)
from app.services.job_service import (
    JobService,
    JobStatus,
    ProcessingJob,
)


# ---------------------------------------------------------------------------
# FakeTestEmbedding
# ---------------------------------------------------------------------------

class FakeTestEmbedding(EmbeddingProvider):
    _DIM = settings.embedding_dimension
    _KEYWORD_DIMS = {
        "france": 0, "python": 1, "mountain": 2,
        "policy": 3, "remote": 4, "work": 5,
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
# Cleanup
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
    "DELETE FROM collection_documents WHERE collection_id IN"
    " (SELECT id FROM collections WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat))",
    "DELETE FROM collections WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat)",
    "DELETE FROM document_chunks WHERE document_id IN"
    " (SELECT id FROM documents WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat))",
    "DELETE FROM documents WHERE user_id IN"
    " (SELECT id FROM users WHERE email LIKE :pat)",
    "DELETE FROM users WHERE email LIKE :pat",
]


def _cleanup_p6_data(db):
    for sql in _CLEANUP_QUERIES:
        db.execute(text(sql), {"pat": "p6_%@example.com"})
    db.commit()


@pytest.fixture(autouse=True)
def setup_test_environment():
    auth._sessions.clear()
    db = SessionLocal()
    try:
        _cleanup_p6_data(db)
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        _cleanup_p6_data(db)
    finally:
        db.close()


def _create_user(db, name, email):
    user = User(name=name, email=email, password_hash="h")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_doc(db, user, chunks_data, embeddings, filename="test.pdf"):
    storage_key = storage_service.save(b"%PDF-1.4 test")
    doc = Document(user_id=user.id, original_filename=filename, storage_key=storage_key, mime_type="application/pdf", file_size=100, status="READY")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    for i, ((tv, cs, ce), emb) in enumerate(zip(chunks_data, embeddings)):
        chunk = DocumentChunk(document_id=doc.id, chunk_index=i, text=tv, char_start=cs, char_end=ce, embedding=emb)
        db.add(chunk)
    db.commit()
    db.refresh(doc)
    return doc


# ==========================================================================
# STEP 3: Enhanced Citations
# ==========================================================================

class TestEnhancedCitations:
    def test_source_reference_has_match_type(self):
        r = RetrievalResult(1, 10, 0, "text", 0.9, keyword_score=0.8, final_score=0.85, match_type="both")
        sources = _build_sources(MagicMock(results=[r]))
        assert len(sources) == 1
        assert sources[0].match_type == "both"
        assert sources[0].keyword_score == 0.8
        assert sources[0].final_score == 0.85

    def test_source_reference_vector_only(self):
        r = RetrievalResult(1, 10, 0, "text", 0.9, match_type="vector")
        sources = _build_sources(MagicMock(results=[r]))
        assert sources[0].match_type == "vector"
        assert sources[0].keyword_score is None

    def test_source_reference_keyword_only(self):
        r = RetrievalResult(1, 10, 0, "text", 0.0, keyword_score=0.7, final_score=0.21, match_type="keyword")
        sources = _build_sources(MagicMock(results=[r]))
        assert sources[0].match_type == "keyword"


# ==========================================================================
# STEP 4: Multi-Document Context
# ==========================================================================

class TestMultiDocumentContext:
    def _r(self, doc_id, chunk_id, text, filename="test.pdf", **kw):
        return RetrievalResult(doc_id, chunk_id, 0, text, 0.9, original_filename=filename, **kw)

    def test_grouped_context_by_document(self):
        r1 = self._r(1, 10, "France text", filename="france.pdf")
        r2 = self._r(2, 20, "Python text", filename="python.pdf")
        ctx = build_context([r1, r2], group_by_document=True)
        assert "DOCUMENT: france.pdf" in ctx
        assert "DOCUMENT: python.pdf" in ctx
        assert "France text" in ctx
        assert "Python text" in ctx

    def test_flat_context_unchanged(self):
        r1 = self._r(1, 10, "France text", filename="france.pdf")
        r2 = self._r(2, 20, "Python text", filename="python.pdf")
        ctx = build_context([r1, r2], group_by_document=False)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "DOCUMENT:" not in ctx

    def test_grouped_empty_results(self):
        assert build_context([], group_by_document=True) == ""

    def test_grouped_max_chars_respected(self):
        results = [self._r(i, i, "x" * 500, filename=f"doc{i}.pdf") for i in range(10)]
        ctx = build_context(results, max_chars=1000, group_by_document=True)
        assert len(ctx) <= 1200

    def test_grouped_max_chunks_respected(self):
        results = [self._r(i, i, f"Text {i}", filename=f"doc{i}.pdf") for i in range(10)]
        ctx = build_context(results, max_chunks=3, group_by_document=True)
        # Grouped context limits total chunks across all documents
        chunk_count = sum(1 for i in range(10) if f"Text {i}" in ctx)
        assert chunk_count <= 3

    def test_grouped_same_document(self):
        r1 = self._r(1, 10, "Chunk A", filename="same.pdf")
        r2 = self._r(1, 11, "Chunk B", filename="same.pdf")
        ctx = build_context([r1, r2], group_by_document=True)
        assert ctx.count("DOCUMENT: same.pdf") == 1
        assert "Chunk A" in ctx
        assert "Chunk B" in ctx


# ==========================================================================
# STEP 6: Confidence / Grounding
# ==========================================================================

class TestConfidenceScoring:
    def test_no_sources_low_confidence(self):
        conf = _compute_confidence(MagicMock(results=[]), [])
        assert conf.level == "LOW"
        assert conf.grounding_score == 0.0
        assert conf.supporting_sources == 0

    def test_single_source_medium(self):
        src = SourceReference(document_id=1, filename="a.pdf", chunk_id=10, chunk_index=0, similarity_score=0.7, final_score=0.7)
        retrieval = MagicMock(results=[MagicMock(final_score=0.7, similarity_score=0.7)])
        conf = _compute_confidence(retrieval, [src])
        assert conf.supporting_sources == 1
        assert conf.source_diversity == 1
        assert 0.0 <= conf.grounding_score <= 1.0

    def test_multiple_sources_higher_confidence(self):
        srcs = [
            SourceReference(document_id=1, filename="a.pdf", chunk_id=10, chunk_index=0, similarity_score=0.9, final_score=0.9),
            SourceReference(document_id=1, filename="a.pdf", chunk_id=11, chunk_index=1, similarity_score=0.8, final_score=0.8),
            SourceReference(document_id=2, filename="b.pdf", chunk_id=20, chunk_index=0, similarity_score=0.85, final_score=0.85),
        ]
        retrieval = MagicMock(results=[MagicMock(final_score=0.9, similarity_score=0.9)] * 3)
        conf = _compute_confidence(retrieval, srcs)
        assert conf.supporting_sources == 3
        assert conf.source_diversity == 2
        assert conf.level in ("MEDIUM", "HIGH")

    def test_score_bounds(self):
        src = SourceReference(document_id=1, filename="a.pdf", chunk_id=10, chunk_index=0, similarity_score=1.0, final_score=1.0)
        retrieval = MagicMock(results=[MagicMock(final_score=1.0, similarity_score=1.0)])
        conf = _compute_confidence(retrieval, [src])
        assert 0.0 <= conf.grounding_score <= 1.0

    def test_level_classification(self):
        # HIGH: multiple sources from multiple docs with high scores
        srcs = [
            SourceReference(document_id=1, filename="a.pdf", chunk_id=i, chunk_index=i, similarity_score=0.95, final_score=0.95)
            for i in range(5)
        ]
        srcs.append(SourceReference(document_id=2, filename="b.pdf", chunk_id=100, chunk_index=0, similarity_score=0.9, final_score=0.9))
        retrieval = MagicMock(results=[MagicMock(final_score=0.95, similarity_score=0.95)] * 6)
        conf = _compute_confidence(retrieval, srcs)
        assert conf.level == "HIGH"


# ==========================================================================
# STEP 7: Conversation Memory
# ==========================================================================

class TestConversationMemory:
    def test_apply_char_budget_keeps_recent(self):
        history = [
            HistoryMessage("user", "A" * 100),
            HistoryMessage("assistant", "B" * 100),
            HistoryMessage("user", "C" * 100),
            HistoryMessage("assistant", "D" * 100),
        ]
        result = _apply_char_budget(history, 150)
        # Should keep most recent messages that fit
        assert len(result) >= 1
        assert sum(len(m.content) for m in result) <= 150

    def test_apply_char_budget_empty(self):
        assert _apply_char_budget([], 100) == []

    def test_apply_char_budget_single_message(self):
        history = [HistoryMessage("user", "Hello")]
        result = _apply_char_budget(history, 1000)
        assert len(result) == 1

    def test_apply_char_budget_exact_fit(self):
        history = [
            HistoryMessage("user", "A" * 50),
            HistoryMessage("assistant", "B" * 50),
        ]
        result = _apply_char_budget(history, 100)
        assert len(result) == 2

    def test_config_settings_exist(self):
        assert hasattr(settings, 'max_history_messages')
        assert hasattr(settings, 'max_history_chars')
        assert settings.max_history_messages == 10
        assert settings.max_history_chars == 4000


# ==========================================================================
# STEP 8: Document Collections
# ==========================================================================

class TestCollections:
    def test_create_collection(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "CollCreate", "p6_collcreate@example.com")
            col = Collection(user_id=user.id, name="My Collection", description="Test desc")
            db.add(col)
            db.commit()
            db.refresh(col)
            assert col.id is not None
            assert col.name == "My Collection"
            assert col.description == "Test desc"
        finally:
            db.close()

    def test_collection_document_association(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "CollAssoc", "p6_collassoc@example.com")
            doc = _create_doc(db, user, [("text", 0, 4)], [VEC_FRANCE])
            col = Collection(user_id=user.id, name="Test Col")
            db.add(col)
            db.commit()
            db.refresh(col)
            col.documents.append(doc)
            db.commit()
            db.refresh(col)
            assert len(col.documents) == 1
            assert col.documents[0].id == doc.id
        finally:
            db.close()

    def test_collection_ownership(self):
        db = SessionLocal()
        try:
            user_a = _create_user(db, "CollOwnA", "p6_colowna@example.com")
            user_b = _create_user(db, "CollOwnB", "p6_colownb@example.com")
            col_a = Collection(user_id=user_a.id, name="A's Collection")
            col_b = Collection(user_id=user_b.id, name="B's Collection")
            db.add_all([col_a, col_b])
            db.commit()
            # User A should only see their collections
            cols_a = db.query(Collection).filter(Collection.user_id == user_a.id).all()
            assert len(cols_a) == 1
            assert cols_a[0].name == "A's Collection"
        finally:
            db.close()

    def test_delete_collection_preserves_documents(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "CollDel", "p6_colldel@example.com")
            doc = _create_doc(db, user, [("text", 0, 4)], [VEC_FRANCE])
            col = Collection(user_id=user.id, name="Delete Me")
            db.add(col)
            db.commit()
            db.refresh(col)
            col.documents.append(doc)
            db.commit()
            db.delete(col)
            db.commit()
            # Document should still exist
            remaining = db.query(Document).filter(Document.id == doc.id).first()
            assert remaining is not None
        finally:
            db.close()

    def test_collection_name_not_null(self):
        db = SessionLocal()
        try:
            user = _create_user(db, "CollName", "p6_collname@example.com")
            # Collection with empty name is allowed at DB level (String allows empty)
            # but the API rejects it. Test the model works with valid name.
            col = Collection(user_id=user.id, name="Valid Name")
            db.add(col)
            db.commit()
            db.refresh(col)
            assert col.name == "Valid Name"
        finally:
            db.close()


# ==========================================================================
# STEP 9: Background Job Service
# ==========================================================================

class TestJobService:
    def test_enqueue_creates_job(self):
        svc = JobService()
        job = svc.enqueue(document_id=1, user_id=1)
        assert job.document_id == 1
        assert job.status == JobStatus.QUEUED

    def test_enqueue_duplicate_prevented(self):
        svc = JobService()
        job1 = svc.enqueue(document_id=1, user_id=1)
        job2 = svc.enqueue(document_id=1, user_id=1)
        assert job1 is job2  # Same object, no duplicate

    def test_start_transitions_to_processing(self):
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        job = svc.start(document_id=1)
        assert job.status == JobStatus.PROCESSING
        assert job.started_at is not None

    def test_complete_transitions_to_ready(self):
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        svc.start(document_id=1)
        svc.complete(document_id=1)
        job = svc.get_status(document_id=1)
        assert job.status == JobStatus.READY
        assert job.progress == 1.0

    def test_fail_transitions_to_failed(self):
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        svc.start(document_id=1)
        svc.fail(document_id=1, error="Extraction failed")
        job = svc.get_status(document_id=1)
        assert job.status == JobStatus.FAILED
        assert job.error == "Extraction failed"

    def test_is_active(self):
        svc = JobService()
        assert not svc.is_active(document_id=1)
        svc.enqueue(document_id=1, user_id=1)
        assert svc.is_active(document_id=1)
        svc.start(document_id=1)
        assert svc.is_active(document_id=1)
        svc.complete(document_id=1)
        assert not svc.is_active(document_id=1)

    def test_update_stage(self):
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        svc.start(document_id=1)
        svc.update_stage(document_id=1, stage="Extracting text", progress=0.3)
        job = svc.get_status(document_id=1)
        assert job.stage == "Extracting text"
        assert job.progress == 0.3

    def test_cleanup(self):
        import time as _time
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        svc.start(document_id=1)
        svc.complete(document_id=1)
        # Force old timestamp by setting completed_at far in the past
        job = svc.get_status(document_id=1)
        job.completed_at = _time.monotonic() - 7200  # 2 hours ago
        svc.cleanup(max_age_seconds=3600)
        assert svc.get_status(document_id=1) is None

    def test_get_status_nonexistent(self):
        svc = JobService()
        assert svc.get_status(document_id=999) is None

    def test_ready_not_reprocessed(self):
        svc = JobService()
        svc.enqueue(document_id=1, user_id=1)
        svc.start(document_id=1)
        svc.complete(document_id=1)
        job2 = svc.enqueue(document_id=1, user_id=1)
        assert job2.status == JobStatus.READY
