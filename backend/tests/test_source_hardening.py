"""Phase 5.5 Step 2 tests: Source replay, ordering, dedup, deletion robustness.

Tests cover:
- Deterministic source ordering
- Duplicate source deduplication
- Source count limit
- Deleted document/chunk robustness
- Ownership validation
- Conversation detail + message list source exposure
- No RAG/LLM on reload
- Empty sources consistency
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.message_source import MessageSource
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.rag_service import RAGResponse, SourceReference
from app.services.source_service import MAX_MESSAGE_SOURCES
from sqlalchemy import text
import uuid as _uuid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def setup_test_environment(db_session):
    auth._sessions.clear()
    previous_overrides = dict(app.dependency_overrides)

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db

    try:
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM document_content WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "hard_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "hard_%%@example.com"})
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

    yield

    auth._sessions.clear()
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous_overrides)


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def register_and_login(name, email, password="testpass123"):
    resp = client.post("/auth/register", json={"name": name, "email": email, "password": password})
    assert resp.status_code == 201
    user = resp.json()
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return user


def create_conversation(title="Test Conv"):
    resp = client.post("/conversations", json={"title": title})
    assert resp.status_code == 201
    return resp.json()


def create_doc(db_session, user_id, filename=None):
    if filename is None:
        filename = f"doc_{_uuid.uuid4().hex[:6]}.pdf"
    doc = Document(
        user_id=user_id, original_filename=filename,
        storage_key=f"test/{user_id}/{_uuid.uuid4().hex[:8]}",
        mime_type="application/pdf", file_size=1024, status="ready",
    )
    db_session.add(doc)
    db_session.flush()
    return doc


def create_chunk(db_session, doc, idx=0, text=None):
    if text is None:
        text = f"Chunk {idx} content."
    chunk = DocumentChunk(
        document_id=doc.id, chunk_index=idx,
        text=text, char_start=idx * 50, char_end=(idx + 1) * 50,
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def make_source(doc, chunk, score=0.9, page_start=1, page_end=1):
    return SourceReference(
        document_id=doc.id, filename=doc.original_filename,
        chunk_id=chunk.id, chunk_index=chunk.chunk_index,
        page_start=page_start, page_end=page_end,
        similarity_score=score,
    )


def rag_response(sources, answer="Answer.", grounded=True):
    return RAGResponse(
        answer=answer, sources=sources, grounded=grounded,
        retrieval_count=len(sources), model="fake", provider="fake",
    )


def send_and_get_assistant(conv_id, sources, answer="Answer.", user_content="Q?"):
    with patch("app.api.conversations.answer_question_with_history") as m:
        m.return_value = rag_response(sources, answer=answer)
        resp = client.post(f"/conversations/{conv_id}/messages", json={"content": user_content})
        assert resp.status_code == 201
        return resp.json()


# ==========================================================================
# 1. DETERMINISTIC SOURCE ORDERING
# ==========================================================================

class TestDeterministicOrdering:
    def test_ordering_by_chunk_index(self, db_session):
        """Sources are ordered by chunk_index, not insertion order."""
        user = register_and_login("HardOrd1", "hard_ord1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])

        # Create chunks in reverse order of similarity
        c2 = create_chunk(db_session, doc, idx=2)
        c0 = create_chunk(db_session, doc, idx=0)
        c1 = create_chunk(db_session, doc, idx=1)

        # RAG returns them out of order (by similarity, not chunk index)
        sources = [
            make_source(doc, c2, score=0.95),  # chunk 2 first in RAG
            make_source(doc, c0, score=0.70),  # chunk 0 second
            make_source(doc, c1, score=0.80),  # chunk 1 third
        ]

        data = send_and_get_assistant(conv["id"], sources)
        assistant = [m for m in data["assistant_message"]["sources"] or [] if True]
        # Reload from DB to verify ordering
        resp = client.get(f"/conversations/{conv['id']}")
        msgs = [m for m in resp.json()["messages"] if m["role"] == "assistant"]
        srcs = msgs[0]["sources"]
        assert len(srcs) == 3
        # Ordered by chunk_index ascending (0, 1, 2)
        assert srcs[0]["chunk_index"] == 0
        assert srcs[1]["chunk_index"] == 1
        assert srcs[2]["chunk_index"] == 2

    def test_equal_scores_deterministic(self, db_session):
        """Sources with equal scores still have deterministic ordering."""
        user = register_and_login("HardOrd2", "hard_ord2@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])

        c0 = create_chunk(db_session, doc, idx=0)
        c1 = create_chunk(db_session, doc, idx=1)

        sources = [
            make_source(doc, c1, score=0.90),
            make_source(doc, c0, score=0.90),
        ]

        send_and_get_assistant(conv["id"], sources)
        resp = client.get(f"/conversations/{conv['id']}")
        srcs = [m for m in resp.json()["messages"] if m["role"] == "assistant"][0]["sources"]
        assert len(srcs) == 2
        # chunk_index deterministic
        assert srcs[0]["chunk_index"] == 0
        assert srcs[1]["chunk_index"] == 1


# ==========================================================================
# 2. DUPLICATE SOURCE DEDUPLICATION
# ==========================================================================

class TestDuplicateDedup:
    def test_duplicate_chunks_deduplicated(self, db_session):
        """RAG returning the same chunk twice only persists one record."""
        user = register_and_login("HardDedup1", "hard_dedup1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)

        # RAG returns the same chunk twice (duplicates)
        sources = [
            make_source(doc, chunk, score=0.90),
            make_source(doc, chunk, score=0.85),  # duplicate
        ]

        data = send_and_get_assistant(conv["id"], sources)
        msg_id = data["assistant_message"]["id"]
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 1  # deduplicated

    def test_different_chunks_not_deduplicated(self, db_session):
        """Different chunks are not deduplicated."""
        user = register_and_login("HardDedup2", "hard_dedup2@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        c0 = create_chunk(db_session, doc, idx=0)
        c1 = create_chunk(db_session, doc, idx=1)

        sources = [make_source(doc, c0, score=0.9), make_source(doc, c1, score=0.8)]
        data = send_and_get_assistant(conv["id"], sources)
        msg_id = data["assistant_message"]["id"]
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 2


# ==========================================================================
# 3. SOURCE COUNT LIMIT
# ==========================================================================

class TestSourceCountLimit:
    def test_source_limit_enforced(self, db_session):
        """Source count is bounded by MAX_MESSAGE_SOURCES."""
        user = register_and_login("HardLim1", "hard_lim1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])

        # Create more chunks than the limit
        chunks = [create_chunk(db_session, doc, idx=i) for i in range(MAX_MESSAGE_SOURCES + 5)]
        sources = [make_source(doc, c, score=0.9 - i * 0.01) for i, c in enumerate(chunks)]

        data = send_and_get_assistant(conv["id"], sources)
        msg_id = data["assistant_message"]["id"]
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == MAX_MESSAGE_SOURCES


# ==========================================================================
# 4. DELETED DOCUMENT ROBUSTNESS
# ==========================================================================

class TestDeletedDocument:
    def test_deleted_document_sources_removed(self, db_session):
        """Deleting a document cascades to remove MessageSource records."""
        user = register_and_login("HardDel1", "hard_del1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)

        sources = [make_source(doc, chunk)]
        data = send_and_get_assistant(conv["id"], sources)
        msg_id = data["assistant_message"]["id"]

        # Verify source exists
        assert db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count() == 1

        # Delete the document (FK CASCADE should remove the source)
        db_session.delete(doc)
        db_session.commit()

        # Verify source removed by cascade
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 0

        # Conversation retrieval still works
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        # Assistant message still exists, but its sources are gone
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["sources"] == []

    def test_deleted_chunk_sources_removed(self, db_session):
        """Deleting a chunk cascades to remove MessageSource records."""
        user = register_and_login("HardDel2", "hard_del2@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)

        sources = [make_source(doc, chunk)]
        data = send_and_get_assistant(conv["id"], sources)
        msg_id = data["assistant_message"]["id"]

        # Delete the chunk
        db_session.delete(chunk)
        db_session.commit()

        # Source removed by cascade
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 0

        # Conversation retrieval still works
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200


# ==========================================================================
# 5. OWNERSHIP VALIDATION
# ==========================================================================

class TestOwnership:
    def test_cross_user_document_rejected(self, db_session):
        """Source referencing another user's document is skipped."""
        user_a = register_and_login("HardOwnA", "hard_owna@example.com")
        conv_a = create_conversation()

        user_b = register_and_login("HardOwnB", "hard_ownb@example.com")
        doc_b = create_doc(db_session, user_b["id"])
        chunk_b = create_chunk(db_session, doc_b, idx=0)

        # Switch back to User A to send message
        auth._sessions.clear()
        client.post("/auth/login", json={"email": "hard_owna@example.com", "password": "testpass123"})

        # RAG incorrectly returns a source from User B's document
        sources = [make_source(doc_b, chunk_b)]
        data = send_and_get_assistant(conv_a["id"], sources)
        msg_id = data["assistant_message"]["id"]

        # Source should be skipped (wrong owner)
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 0


# ==========================================================================
# 6. CONVERSATION DETAIL + MESSAGE LIST CONSISTENCY
# ==========================================================================

class TestAPIConsistency:
    def test_conversation_detail_has_sources(self, db_session):
        """GET /conversations/{id} returns persisted sources on assistant messages."""
        user = register_and_login("HardAPI1", "hard_api1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        c0 = create_chunk(db_session, doc, idx=0)
        c1 = create_chunk(db_session, doc, idx=1)

        sources = [make_source(doc, c0, score=0.9), make_source(doc, c1, score=0.8)]
        send_and_get_assistant(conv["id"], sources)

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assistant = [m for m in resp.json()["messages"] if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert len(assistant[0]["sources"]) == 2

    def test_message_list_has_sources(self, db_session):
        """GET /conversations/{id}/messages returns persisted sources."""
        user = register_and_login("HardAPI2", "hard_api2@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)

        sources = [make_source(doc, chunk)]
        send_and_get_assistant(conv["id"], sources)

        resp = client.get(f"/conversations/{conv['id']}/messages")
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        assistant = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert len(assistant[0]["sources"]) == 1

    def test_user_message_has_empty_sources(self, db_session):
        """User messages always have empty sources."""
        user = register_and_login("HardAPI3", "hard_api3@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)
        sources = [make_source(doc, chunk)]
        send_and_get_assistant(conv["id"], sources)

        resp = client.get(f"/conversations/{conv['id']}")
        user_msgs = [m for m in resp.json()["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["sources"] == []


# ==========================================================================
# 7. NO RAG/LLM ON RELOAD
# ==========================================================================

class TestNoRAGOnReload:
    def test_conversation_reload_no_rag(self, db_session):
        """GET /conversations/{id} never calls RAG."""
        user = register_and_login("HardReload1", "hard_reload1@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)
        sources = [make_source(doc, chunk)]
        send_and_get_assistant(conv["id"], sources)

        with patch("app.api.conversations.answer_question_with_history") as m:
            resp = client.get(f"/conversations/{conv['id']}")
            assert resp.status_code == 200
            m.assert_not_called()

    def test_message_list_reload_no_rag(self, db_session):
        """GET /conversations/{id}/messages never calls RAG."""
        user = register_and_login("HardReload2", "hard_reload2@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)
        sources = [make_source(doc, chunk)]
        send_and_get_assistant(conv["id"], sources)

        with patch("app.api.conversations.answer_question_with_history") as m:
            resp = client.get(f"/conversations/{conv['id']}/messages")
            assert resp.status_code == 200
            m.assert_not_called()

    def test_reload_no_llm(self, db_session):
        """GET /conversations/{id} never calls the LLM service."""
        user = register_and_login("HardReload3", "hard_reload3@example.com")
        conv = create_conversation()
        doc = create_doc(db_session, user["id"])
        chunk = create_chunk(db_session, doc, idx=0)
        sources = [make_source(doc, chunk)]
        send_and_get_assistant(conv["id"], sources)

        with patch("app.services.llm.service.LLMService.generate") as m:
            resp = client.get(f"/conversations/{conv['id']}")
            assert resp.status_code == 200
            m.assert_not_called()
