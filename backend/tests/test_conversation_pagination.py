"""Phase 5.6 Step 1 tests: Conversation message pagination.

Tests cover:
- Database-level pagination
- Edge cases (empty, beyond last page, invalid params)
- Deterministic ordering
- Source persistence with pagination
- No RAG/LLM on reload
- Ownership isolation
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.message import Message
from app.models.message_source import MessageSource
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.rag_service import RAGResponse, SourceReference
from sqlalchemy import text
import uuid as _uuid


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
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM document_content WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "pg_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "pg_%%@example.com"})
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


def add_messages(db_session, conversation_id, count, prefix="Msg"):
    """Add N messages to a conversation."""
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=f"{prefix} {i}",
        )
        db_session.add(msg)
    db_session.commit()


def get_messages(conv_id, page=1, page_size=50):
    resp = client.get(f"/conversations/{conv_id}/messages?page={page}&page_size={page_size}")
    assert resp.status_code == 200
    return resp.json()


# ==========================================================================
# 1. EMPTY CONVERSATION
# ==========================================================================

class TestEmptyConversation:
    def test_empty_conversation(self):
        register_and_login("PgEmpty", "pg_empty@example.com")
        conv = create_conversation()
        data = get_messages(conv["id"])
        assert data["messages"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["has_next"] is False
        assert data["has_previous"] is False


# ==========================================================================
# 2. SINGLE MESSAGE
# ==========================================================================

class TestSingleMessage:
    def test_single_message(self, db_session):
        register_and_login("PgSingle", "pg_single@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 1)
        data = get_messages(conv["id"])
        assert len(data["messages"]) == 1
        assert data["total"] == 1
        assert data["has_next"] is False


# ==========================================================================
# 3. MULTIPLE MESSAGES
# ==========================================================================

class TestMultipleMessages:
    def test_first_page(self, db_session):
        register_and_login("PgFirst", "pg_first@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 10)
        data = get_messages(conv["id"], page=1, page_size=3)
        assert len(data["messages"]) == 3
        assert data["total"] == 10
        assert data["has_next"] is True
        assert data["has_previous"] is False

    def test_second_page(self, db_session):
        register_and_login("PgSecond", "pg_second@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 10)
        data = get_messages(conv["id"], page=2, page_size=3)
        assert len(data["messages"]) == 3
        assert data["has_next"] is True
        assert data["has_previous"] is True

    def test_last_page(self, db_session):
        register_and_login("PgLast", "pg_last@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 10)
        data = get_messages(conv["id"], page=4, page_size=3)
        assert len(data["messages"]) == 1  # 10 = 3+3+3+1
        assert data["has_next"] is False
        assert data["has_previous"] is True

    def test_page_beyond_last(self, db_session):
        register_and_login("PgBeyond", "pg_beyond@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 5)
        data = get_messages(conv["id"], page=10, page_size=5)
        assert data["messages"] == []
        assert data["total"] == 5
        assert data["has_next"] is False
        assert data["has_previous"] is True


# ==========================================================================
# 4. PAGE SIZE EDGE CASES
# ==========================================================================

class TestPageSizeEdgeCases:
    def test_page_size_one(self, db_session):
        register_and_login("PgSize1", "pg_size1@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 3)
        data = get_messages(conv["id"], page=1, page_size=1)
        assert len(data["messages"]) == 1
        assert data["total"] == 3
        assert data["has_next"] is True

    def test_maximum_page_size(self, db_session):
        register_and_login("PgMax", "pg_max@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 5)
        data = get_messages(conv["id"], page=1, page_size=200)
        assert len(data["messages"]) == 5

    def test_invalid_page_rejected(self):
        register_and_login("PgInvPage", "pg_inv_page@example.com")
        conv = create_conversation()
        resp = client.get(f"/conversations/{conv['id']}/messages?page=0&page_size=10")
        assert resp.status_code == 422

    def test_invalid_page_size_rejected(self):
        register_and_login("PgInvSize", "pg_inv_size@example.com")
        conv = create_conversation()
        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=0")
        assert resp.status_code == 422

    def test_page_size_over_max_rejected(self):
        register_and_login("PgOverMax", "pg_over_max@example.com")
        conv = create_conversation()
        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=201")
        assert resp.status_code == 422


# ==========================================================================
# 5. TOTAL COUNT
# ==========================================================================

class TestTotalCount:
    def test_correct_total(self, db_session):
        register_and_login("PgTotal", "pg_total@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 7)
        data = get_messages(conv["id"], page=1, page_size=3)
        assert data["total"] == 7

    def test_total_respects_page_size(self, db_session):
        register_and_login("PgTotalRes", "pg_total_res@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 20)
        data = get_messages(conv["id"], page=1, page_size=5)
        assert data["total"] == 20
        assert len(data["messages"]) == 5


# ==========================================================================
# 6. DETERMINISTIC ORDERING
# ==========================================================================

class TestDeterministicOrdering:
    def test_messages_ordered_by_id(self, db_session):
        register_and_login("PgOrder", "pg_order@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 5)
        data = get_messages(conv["id"], page=1, page_size=50)
        for i, msg in enumerate(data["messages"]):
            assert msg["content"] == f"Msg {i}"

    def test_pagination_preserves_order(self, db_session):
        register_and_login("PgOrdPag", "pg_ord_pag@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 6)
        page1 = get_messages(conv["id"], page=1, page_size=2)
        page2 = get_messages(conv["id"], page=2, page_size=2)
        page3 = get_messages(conv["id"], page=3, page_size=2)
        all_msgs = page1["messages"] + page2["messages"] + page3["messages"]
        for i, msg in enumerate(all_msgs):
            assert msg["content"] == f"Msg {i}"


# ==========================================================================
# 7. SOURCES WITH PAGINATION
# ==========================================================================

class TestSourcesWithPagination:
    def test_assistant_messages_include_sources(self, db_session):
        """Paginated assistant messages still include persisted sources."""
        user = register_and_login("PgSrc", "pg_src@example.com")
        conv = create_conversation()

        doc = Document(
            user_id=user["id"], original_filename="test.pdf",
            storage_key=f"test/{_uuid.uuid4().hex[:8]}",
            mime_type="application/pdf", file_size=1024, status="ready",
        )
        db_session.add(doc)
        db_session.flush()

        chunk = DocumentChunk(
            document_id=doc.id, chunk_index=0,
            text="Test content.", char_start=0, char_end=12,
        )
        db_session.add(chunk)
        db_session.commit()

        # Send message via API (creates user + assistant messages with sources)
        with patch("app.api.conversations.answer_question_with_history") as m:
            m.return_value = RAGResponse(
                answer="Answer.", grounded=True, retrieval_count=1,
                model="fake", provider="fake",
                sources=[SourceReference(
                    document_id=doc.id, filename="test.pdf",
                    chunk_id=chunk.id, chunk_index=0,
                    similarity_score=0.9,
                )],
            )
            resp = client.post(f"/conversations/{conv['id']}/messages", json={"content": "Q?"})
            assert resp.status_code == 201

        # Reload via pagination — sources should be present
        data = get_messages(conv["id"])
        assistant_msgs = [m for m in data["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["sources"]) == 1


# ==========================================================================
# 8. NO RAG/LLM ON RELOAD
# ==========================================================================

class TestNoExternalCalls:
    def test_pagination_no_rag(self, db_session):
        register_and_login("PgNoRag", "pg_no_rag@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 3)

        with patch("app.api.conversations.answer_question_with_history") as m:
            data = get_messages(conv["id"])
            m.assert_not_called()
        assert len(data["messages"]) == 3

    def test_pagination_no_llm(self, db_session):
        register_and_login("PgNoLLM", "pg_no_llm@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 3)

        with patch("app.services.llm.service.LLMService.generate") as m:
            data = get_messages(conv["id"])
            m.assert_not_called()

    def test_pagination_no_retrieval(self, db_session):
        register_and_login("PgNoRet", "pg_no_ret@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 3)

        with patch("app.services.retrieval_service.retrieve_context") as m:
            data = get_messages(conv["id"])
            m.assert_not_called()


# ==========================================================================
# 9. OWNERSHIP
# ==========================================================================

class TestOwnership:
    def test_cannot_access_other_users_messages(self, db_session):
        register_and_login("PgOwnA", "pg_owna@example.com")
        conv_a = create_conversation()
        add_messages(db_session, conv_a["id"], 3)
        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        register_and_login("PgOwnB", "pg_ownb@example.com")

        resp = client.get(f"/conversations/{conv_a_id}/messages")
        assert resp.status_code == 404

    def test_no_cross_user_messages(self, db_session):
        register_and_login("PgCrossA", "pg_crossa@example.com")
        conv_a = create_conversation()
        add_messages(db_session, conv_a["id"], 3, prefix="A's msg")

        auth._sessions.clear()
        register_and_login("PgCrossB", "pg_crossb@example.com")

        resp = client.get(f"/conversations/1/messages")
        # Should get 404 or empty (no conversation with id=1 for this user)
        if resp.status_code == 200:
            data = resp.json()
            for msg in data["messages"]:
                assert "A's msg" not in msg["content"]


# ==========================================================================
# 10. EXISTING CONVERSATION DETAIL STILL WORKS
# ==========================================================================

class TestBackwardCompatibility:
    def test_conversation_detail_still_returns_all_messages(self, db_session):
        """GET /conversations/{id} still returns all messages (unchanged)."""
        register_and_login("PgCompat", "pg_compat@example.com")
        conv = create_conversation()
        add_messages(db_session, conv["id"], 5)

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 5
