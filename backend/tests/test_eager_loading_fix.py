"""Phase 5.9 Step 1 fix: Regression tests for N+1 eager loading fix.

Verifies:
- Paginated messages return sources correctly
- Multiple messages with sources work correctly
- Pagination metadata remains correct
- Source ordering is preserved
- Conversation detail endpoint returns sources correctly
- No RAG/LLM/retrieval is called during message retrieval
- Cross-user access remains blocked
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.message_source import MessageSource
from sqlalchemy import text


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
        db_session.execute(text(
            "DELETE FROM message_sources WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)))"
        ), {"p1": "en1_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1))"
        ), {"p1": "en1_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "en1_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM documents WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "en1_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM users WHERE email LIKE :p1"
        ), {"p1": "en1_%%@example.com"})
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

    yield

    auth._sessions.clear()
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous_overrides)


client = TestClient(app)


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


def add_message(db_session, conversation_id, role, content):
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return {"id": msg.id, "role": msg.role, "content": msg.content}


def create_doc(db_session, user_id, filename="test.pdf"):
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"en1/{user_id}/{filename}",
        mime_type="application/pdf",
        file_size=1024,
        status="ready",
    )
    db_session.add(doc)
    db_session.flush()
    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        text="Test chunk.",
        char_start=0,
        char_end=10,
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(doc)
    return {"id": doc.id, "chunk_id": chunk.id}


def add_source(db_session, message_id, doc_id, chunk_id,
               page_start=1, page_end=1, score=0.85, chunk_index=0):
    src = MessageSource(
        message_id=message_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        similarity_score=score,
    )
    db_session.add(src)
    db_session.commit()
    db_session.refresh(src)
    return src


# ==========================================================================
# 1. PAGINATED MESSAGES RETURN SOURCES CORRECTLY
# ==========================================================================

class TestEagerLoadedSources:
    def test_paginated_messages_include_sources(self, db_session):
        """Paginated messages endpoint returns sources on assistant messages."""
        user = register_and_login("en1_Src1", "en1_src1@example.com")
        conv = create_conversation("Source Test")
        doc = create_doc(db_session, user["id"])

        add_message(db_session, conv["id"], "user", "Question?")
        asst = add_message(db_session, conv["id"], "assistant", "Answer with sources")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=3, page_end=5, score=0.92)

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()

        # Find the assistant message
        asst_msg = [m for m in data["messages"] if m["role"] == "assistant"][0]
        assert len(asst_msg["sources"]) == 1
        assert asst_msg["sources"][0]["page_start"] == 3
        assert asst_msg["sources"][0]["page_end"] == 5

    def test_multiple_messages_with_multiple_sources(self, db_session):
        """Multiple assistant messages with multiple sources each."""
        user = register_and_login("en1_Src2", "en1_src2@example.com")
        conv = create_conversation("Multi Source")
        doc = create_doc(db_session, user["id"], "multi.pdf")

        for i in range(5):
            add_message(db_session, conv["id"], "user", f"Q{i}")
            asst = add_message(db_session, conv["id"], "assistant", f"A{i}")
            add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                       page_start=i + 1, page_end=i + 1, score=0.8 + i * 0.02,
                       chunk_index=i)

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=20")
        assert resp.status_code == 200
        data = resp.json()

        asst_msgs = [m for m in data["messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 5
        for msg in asst_msgs:
            assert len(msg["sources"]) == 1

    def test_user_messages_have_empty_sources(self, db_session):
        """User messages have an empty sources list."""
        register_and_login("en1_Src3", "en1_src3@example.com")
        conv = create_conversation("User Msg")
        add_message(db_session, conv["id"], "user", "Hello")

        resp = client.get(f"/conversations/{conv['id']}/messages")
        assert resp.status_code == 200
        user_msg = resp.json()["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["sources"] == []


# ==========================================================================
# 2. PAGINATION METADATA REMAINS CORRECT
# ==========================================================================

class TestPaginationMetadata:
    def test_pagination_metadata_with_sources(self, db_session):
        """Pagination metadata is correct when messages have sources."""
        user = register_and_login("en1_Meta1", "en1_meta1@example.com")
        conv = create_conversation("Meta Test")
        doc = create_doc(db_session, user["id"])

        for i in range(10):
            add_message(db_session, conv["id"], "user", f"Q{i}")
            asst = add_message(db_session, conv["id"], "assistant", f"A{i}")
            add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                       page_start=i + 1, page_end=i + 1, score=0.8)

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=5")
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 5
        assert data["total"] == 20
        assert data["has_next"] is True
        assert data["has_previous"] is False
        assert len(data["messages"]) == 5

        resp2 = client.get(f"/conversations/{conv['id']}/messages?page=2&page_size=5")
        data2 = resp2.json()
        assert data2["page"] == 2
        assert data2["has_next"] is True
        assert data2["has_previous"] is True

        resp4 = client.get(f"/conversations/{conv['id']}/messages?page=4&page_size=5")
        data4 = resp4.json()
        assert data4["has_next"] is False
        assert data4["has_previous"] is True


# ==========================================================================
# 3. SOURCE ORDERING IS PRESERVED
# ==========================================================================

class TestSourceOrdering:
    def test_sources_ordered_by_chunk_index(self, db_session):
        """Sources are returned ordered by chunk_index."""
        user = register_and_login("en1_Ord1", "en1_ord1@example.com")
        conv = create_conversation("Order Test")
        doc = create_doc(db_session, user["id"])

        asst = add_message(db_session, conv["id"], "assistant", "Multi-source")
        # Add sources in reverse order
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   chunk_index=5, page_start=5, page_end=5, score=0.7)
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   chunk_index=1, page_start=1, page_end=1, score=0.9)
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   chunk_index=3, page_start=3, page_end=3, score=0.8)

        resp = client.get(f"/conversations/{conv['id']}/messages")
        asst_msg = resp.json()["messages"][0]
        indices = [s["chunk_index"] for s in asst_msg["sources"]]
        assert indices == [1, 3, 5]


# ==========================================================================
# 4. CONVERSATION DETAIL RETURNS SOURCES
# ==========================================================================

class TestConversationDetailSources:
    def test_detail_endpoint_includes_sources(self, db_session):
        """GET /conversations/{id} returns messages with sources."""
        user = register_and_login("en1_Det1", "en1_det1@example.com")
        conv = create_conversation("Detail Source")
        doc = create_doc(db_session, user["id"])

        add_message(db_session, conv["id"], "user", "Q")
        asst = add_message(db_session, conv["id"], "assistant", "A with src")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=2, page_end=4, score=0.88)

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        data = resp.json()
        asst_msg = [m for m in data["messages"] if m["role"] == "assistant"][0]
        assert len(asst_msg["sources"]) == 1
        assert asst_msg["sources"][0]["page_start"] == 2


# ==========================================================================
# 5. NO RAG/LLM/RETRIEVAL DURING MESSAGE RETRIEVAL
# ==========================================================================

class TestNoSideEffects:
    def test_no_rag_during_pagination(self, db_session):
        """Paginated message retrieval does not call RAG."""
        user = register_and_login("en1_NoR1", "en1_nor1@example.com")
        conv = create_conversation("No RAG")
        add_message(db_session, conv["id"], "user", "Q")

        with patch("app.api.conversations.answer_question_with_history") as mock:
            resp = client.get(f"/conversations/{conv['id']}/messages")
            assert resp.status_code == 200
            mock.assert_not_called()

    def test_no_rag_during_detail(self, db_session):
        """Conversation detail does not call RAG."""
        user = register_and_login("en1_NoR2", "en1_nor2@example.com")
        conv = create_conversation("No RAG Detail")
        add_message(db_session, conv["id"], "assistant", "A")

        with patch("app.api.conversations.answer_question_with_history") as mock:
            resp = client.get(f"/conversations/{conv['id']}")
            assert resp.status_code == 200
            mock.assert_not_called()


# ==========================================================================
# 6. CROSS-USER ACCESS REMAINS BLOCKED
# ==========================================================================

class TestCrossUser:
    def test_cross_user_paginated_messages_blocked(self, db_session):
        """User B cannot access User A's paginated messages."""
        user_a = register_and_login("en1_XUsr1", "en1_xusr1@example.com")
        conv_a = create_conversation("A's Conv")
        add_message(db_session, conv_a["id"], "user", "Secret")

        register_and_login("en1_XUsr2", "en1_xusr2@example.com")
        resp = client.get(f"/conversations/{conv_a['id']}/messages")
        assert resp.status_code == 404

    def test_cross_user_detail_blocked(self, db_session):
        """User B cannot access User A's conversation detail."""
        user_a = register_and_login("en1_XUsr3", "en1_xusr3@example.com")
        conv_a = create_conversation("A's Detail")
        add_message(db_session, conv_a["id"], "user", "Secret")

        register_and_login("en1_XUsr4", "en1_xusr4@example.com")
        resp = client.get(f"/conversations/{conv_a['id']}")
        assert resp.status_code == 404

    def test_unauthenticated_messages_blocked(self, db_session):
        """Unauthenticated user cannot access messages."""
        auth._sessions.clear()
        resp = client.get("/conversations/1/messages")
        assert resp.status_code == 401

    def test_unauthenticated_detail_blocked(self, db_session):
        """Unauthenticated user cannot access conversation detail."""
        auth._sessions.clear()
        resp = client.get("/conversations/1")
        assert resp.status_code == 401
