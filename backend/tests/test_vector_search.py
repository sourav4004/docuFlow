"""Phase 4.5 tests: vector similarity search with pgvector.

These tests MUST use real PostgreSQL because pgvector operators (<=>)
are not available in SQLite.

Uses manually-set known embeddings for controlled ranking tests.
No paid API required.

Similarity metric: cosine similarity via pgvector <=> operator.
cosine_distance = 1 - cosine_similarity.
Lower distance = more similar.
"""

import math
import pytest
from typing import List

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core import auth
from app.core.config import settings
from app.core.database import Base, get_db, SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.vector_search import (
    search_similar_chunks,
    SearchResult,
    MAX_TOP_K,
)


# ---------------------------------------------------------------------------
# Known embedding vectors for controlled ranking tests
# ---------------------------------------------------------------------------

# Using 384D unit vectors (matching EMBEDDING_DIMENSION=384)
# Vectors are orthogonal for clean ranking tests.
_EMB_DIM = 384
VEC_FRANCE = [1.0] + [0.0] * (_EMB_DIM - 1)
VEC_PYTHON = [0.0, 1.0] + [0.0] * (_EMB_DIM - 2)
VEC_MOUNTAIN = [0.0, 0.0, 1.0] + [0.0] * (_EMB_DIM - 3)

QUERY_FRANCE = [0.9, 0.1] + [0.0] * (_EMB_DIM - 2)
QUERY_PYTHON = [0.1, 0.9] + [0.0] * (_EMB_DIM - 2)
QUERY_MOUNTAIN = [0.0, 0.1, 0.9] + [0.0] * (_EMB_DIM - 3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Isolate storage and sessions before each test."""
    auth._sessions.clear()
    # Clean up any leftover test data from previous runs
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'vec_%@example.com'))"))
        db.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'vec_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'vec_%@example.com'"))
        db.commit()
    finally:
        db.close()
    yield
    # Cleanup after test
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'vec_%@example.com'))"))
        db.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE 'vec_%@example.com')"))
        db.execute(text("DELETE FROM users WHERE email LIKE 'vec_%@example.com'"))
        db.commit()
    finally:
        db.close()


def create_doc_with_embeddings(db, user, chunks_data, embeddings):
    """Helper: create document, chunks, and manually set embeddings.

    chunks_data: list of (text, char_start, char_end)
    embeddings: list of vectors matching chunks_data
    """
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
# 1. BASIC SIMILARITY SEARCH
# ==========================================================================

class TestBasicSearch:
    def test_search_returns_results(self):
        """Search returns results when embeddings exist."""
        db = SessionLocal()
        try:
            user = User(name="SearchTest", email="vec_search_basic@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [
                ("France is a country", 0, 20),
                ("Python is a language", 20, 40),
                ("Mountains are tall", 40, 60),
            ]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=3)
            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
        finally:
            db.close()

    def test_search_returns_structured_results(self):
        """Results contain all expected fields."""
        db = SessionLocal()
        try:
            user = User(name="StructTest", email="vec_struct_result@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [("Test text", 0, 9)]
            doc = create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=1)
            assert len(results) == 1
            r = results[0]
            assert r.document_id == doc.id
            assert r.text == "Test text"
            assert r.similarity_score > 0
            assert r.original_filename == "test.pdf"
        finally:
            db.close()


# ==========================================================================
# 2. CORRECT RANKING
# ==========================================================================

class TestRanking:
    def test_france_query_ranks_france_first(self):
        """Query about France should rank France chunk highest."""
        db = SessionLocal()
        try:
            user = User(name="RankTest", email="vec_rank_france@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [
                ("France is beautiful", 0, 18),
                ("Python is great", 18, 34),
                ("Mountains are tall", 34, 52),
            ]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=3)
            # HNSW is approximate; may return 2-3 results for small datasets
            assert len(results) >= 2
            assert len(results) <= 3
            assert "France" in results[0].text
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score
        finally:
            db.close()

    def test_python_query_ranks_python_first(self):
        """Query about Python should rank Python chunk highest."""
        db = SessionLocal()
        try:
            user = User(name="RankTest2", email="vec_rank_python@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [
                ("France is beautiful", 0, 18),
                ("Python is great", 18, 34),
                ("Mountains are tall", 34, 52),
            ]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN])

            results = search_similar_chunks(db, user.id, QUERY_PYTHON, top_k=3)
            assert "Python" in results[0].text
        finally:
            db.close()


# ==========================================================================
# 3. TOP-K
# ==========================================================================

class TestTopK:
    def test_top_k_1(self):
        """top_k=1 returns only the best match."""
        db = SessionLocal()
        try:
            user = User(name="TopK1", email="vec_topk_1@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [
                ("France text", 0, 11),
                ("Python text", 11, 22),
                ("Mountain text", 22, 34),
            ]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=1)
            assert len(results) == 1
        finally:
            db.close()

    def test_top_k_larger_than_chunks(self):
        """top_k larger than available chunks returns all chunks."""
        db = SessionLocal()
        try:
            user = User(name="TopKBig", email="vec_topk_big@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [("Only chunk", 0, 10)]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=50)
            assert len(results) == 1
        finally:
            db.close()

    def test_top_k_validation(self):
        """Invalid top_k values raise ValueError."""
        db = SessionLocal()
        try:
            user = User(name="TopKVal", email="vec_topk_val@example.com", password_hash="h")
            db.add(user); db.commit()

            with pytest.raises(ValueError, match="top_k"):
                search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=0)
            with pytest.raises(ValueError, match="top_k"):
                search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=MAX_TOP_K + 1)
        finally:
            db.close()


# ==========================================================================
# 4. EMPTY / MISSING EMBEDDINGS
# ==========================================================================

class TestEmptyAndMissing:
    def test_empty_database(self):
        """Search on empty database returns empty list."""
        db = SessionLocal()
        try:
            user = User(name="EmptyDB", email="vec_empty_db@example.com", password_hash="h")
            db.add(user); db.commit()

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=5)
            assert results == []
        finally:
            db.close()

    def test_chunks_without_embeddings(self):
        """Chunks with NULL embeddings are skipped."""
        db = SessionLocal()
        try:
            user = User(name="NoEmb", email="vec_no_embed@example.com", password_hash="h")
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
                text="No embedding here", char_start=0, char_end=18,
                embedding=None,
            )
            db.add(chunk); db.commit()

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=5)
            assert results == []
        finally:
            db.close()

    def test_empty_query_embedding_raises(self):
        """Empty query embedding raises ValueError."""
        db = SessionLocal()
        try:
            user = User(name="EmptyQ", email="vec_empty_q@example.com", password_hash="h")
            db.add(user); db.commit()

            with pytest.raises(ValueError, match="must not be empty"):
                search_similar_chunks(db, user.id, [], top_k=5)
        finally:
            db.close()


# ==========================================================================
# 5. DOCUMENT FILTERING
# ==========================================================================

class TestDocumentFiltering:
    def test_search_specific_document(self):
        """Searching with document_id restricts to that document."""
        db = SessionLocal()
        try:
            user = User(name="DocFilter", email="vec_doc_filter@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            doc_a = create_doc_with_embeddings(db, user, [("France text", 0, 11)], [VEC_FRANCE])
            doc_b = create_doc_with_embeddings(db, user, [("Python text", 0, 11)], [VEC_PYTHON])

            results = search_similar_chunks(
                db, user.id, QUERY_FRANCE, top_k=10, document_id=doc_a.id
            )
            assert len(results) == 1
            assert results[0].document_id == doc_a.id
            assert "France" in results[0].text
        finally:
            db.close()

    def test_search_all_documents(self):
        """Searching without document_id returns results from all documents."""
        db = SessionLocal()
        try:
            user = User(name="AllDocs", email="vec_all_docs@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            create_doc_with_embeddings(db, user, [("France", 0, 6)], [VEC_FRANCE])
            create_doc_with_embeddings(db, user, [("Python", 0, 6)], [VEC_PYTHON])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=10)
            assert len(results) == 2
        finally:
            db.close()


# ==========================================================================
# 6. CROSS-USER ISOLATION
# ==========================================================================

class TestCrossUserIsolation:
    def test_user_a_cannot_see_user_b_chunks(self):
        """User A search must never return User B's chunks."""
        db = SessionLocal()
        try:
            user_a = User(name="UserA", email="vec_iso_a@example.com", password_hash="h")
            user_b = User(name="UserB", email="vec_iso_b@example.com", password_hash="h")
            db.add_all([user_a, user_b]); db.commit()
            db.refresh(user_a); db.refresh(user_b)

            create_doc_with_embeddings(db, user_a, [("France", 0, 6)], [VEC_FRANCE])
            create_doc_with_embeddings(db, user_b, [("Python", 0, 6)], [VEC_PYTHON])

            # User A searches — should NOT see Python
            results_a = search_similar_chunks(db, user_a.id, QUERY_PYTHON, top_k=10)
            assert len(results_a) == 1
            assert "France" in results_a[0].text

            # User B searches — should NOT see France
            results_b = search_similar_chunks(db, user_b.id, QUERY_FRANCE, top_k=10)
            assert len(results_b) == 1
            assert "Python" in results_b[0].text
        finally:
            db.close()

    def test_document_id_cannot_bypass_ownership(self):
        """Providing another user's document_id should return empty."""
        db = SessionLocal()
        try:
            user_a = User(name="UserA2", email="vec_iso_a2@example.com", password_hash="h")
            user_b = User(name="UserB2", email="vec_iso_b2@example.com", password_hash="h")
            db.add_all([user_a, user_b]); db.commit()
            db.refresh(user_a); db.refresh(user_b)

            doc_b = create_doc_with_embeddings(
                db, user_b, [("Python secret", 0, 13)], [VEC_PYTHON]
            )

            results = search_similar_chunks(
                db, user_a.id, QUERY_PYTHON, top_k=10, document_id=doc_b.id
            )
            assert results == []
        finally:
            db.close()


# ==========================================================================
# 7. SIMILARITY SCORE
# ==========================================================================

class TestSimilarityScore:
    def test_identical_vectors_high_score(self):
        """Identical vectors should have similarity close to 1.0."""
        db = SessionLocal()
        try:
            user = User(name="ScoreTest", email="vec_score_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            create_doc_with_embeddings(db, user, [("text", 0, 4)], [VEC_FRANCE])

            results = search_similar_chunks(db, user.id, VEC_FRANCE, top_k=1)
            assert len(results) == 1
            assert results[0].similarity_score > 0.99
        finally:
            db.close()

    def test_orthogonal_vectors_low_score(self):
        """Orthogonal vectors should have similarity close to 0.0."""
        db = SessionLocal()
        try:
            user = User(name="OrthoTest", email="vec_ortho_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            create_doc_with_embeddings(db, user, [("text", 0, 4)], [VEC_FRANCE])

            results = search_similar_chunks(db, user.id, VEC_PYTHON, top_k=1)
            assert len(results) == 1
            assert abs(results[0].similarity_score) < 0.01
        finally:
            db.close()


# ==========================================================================
# 8. RESULTS ORDERING
# ==========================================================================

class TestOrdering:
    def test_results_ordered_by_similarity_desc(self):
        """Results must be ordered from most to least similar."""
        db = SessionLocal()
        try:
            user = User(name="OrderTest", email="vec_order_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = [
                ("France", 0, 6),
                ("Python", 6, 12),
                ("Mountain", 12, 20),
            ]
            create_doc_with_embeddings(db, user, chunks_data, [VEC_FRANCE, VEC_PYTHON, VEC_MOUNTAIN])

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=3)
            # HNSW is approximate; may return 2-3 results for small datasets
            assert len(results) >= 2
            assert len(results) <= 3
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score
        finally:
            db.close()


# ==========================================================================
# 9. LARGE NUMBER OF CHUNKS
# ==========================================================================

class TestLargeDataset:
    def test_many_chunks_searchable(self):
        """Search works with a larger number of chunks."""
        db = SessionLocal()
        try:
            user = User(name="LargeTest", email="vec_large_test@example.com", password_hash="h")
            db.add(user); db.commit(); db.refresh(user)

            chunks_data = []
            embeddings = []
            for i in range(20):
                chunks_data.append((f"Chunk {i} content", i * 10, (i + 1) * 10))
                # Alternate between France and Python vectors
                embeddings.append(VEC_FRANCE if i % 2 == 0 else VEC_PYTHON)

            create_doc_with_embeddings(db, user, chunks_data, embeddings)

            results = search_similar_chunks(db, user.id, QUERY_FRANCE, top_k=5)
            assert len(results) == 5
            # All results should have France-like similarity (higher than Python-like)
            for r in results:
                assert r.similarity_score > 0.5
        finally:
            db.close()
