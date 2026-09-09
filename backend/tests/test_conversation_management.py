"""Phase 5.7 Step 1 tests: Conversation management lifecycle.

Tests cover:
- Conversation creation (auth, ownership, side effects)
- Conversation listing (user isolation, ordering, no unnecessary loading)
- Conversation retrieval (ownership, correct data)
- Conversation deletion (auth, ownership, cascade, document preservation)
- Cross-user security (full isolation)
- No RAG/LLM/retrieval on management operations
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.rag_service import RAGResponse, SourceReference
from app.core.security import hash_password
from sqlalchemy import text


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
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "mgmt_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "mgmt_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "mgmt_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "mgmt_%%@example.com"})
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

def register_and_login(name: str, email: str, password: str = "testpass123") -> dict:
    resp = client.post("/auth/register", json={"name": name, "email": email, "password": password})
    assert resp.status_code == 201
    user = resp.json()
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return user


def create_conversation(title: str = "Test Conv") -> dict:
    resp = client.post("/conversations", json={"title": title})
    assert resp.status_code == 201
    return resp.json()


def create_document_with_chunk(db_session: Session, user_id: int, filename: str = "test.pdf") -> dict:
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"mgmt/{user_id}/{filename}",
        mime_type="application/pdf",
        file_size=1024,
        status="ready",
    )
    db_session.add(doc)
    db_session.flush()
    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        text="Test chunk text.",
        char_start=0,
        char_end=17,
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(doc)
    return {"id": doc.id, "filename": doc.original_filename, "chunk_id": chunk.id}


def add_messages_with_sources(db_session: Session, conversation_id: int, doc_id: int, chunk_id: int):
    """Add a user message + assistant message with a MessageSource."""
    from app.models.message_source import MessageSource

    user_msg = Message(conversation_id=conversation_id, role="user", content="Test question?")
    db_session.add(user_msg)
    db_session.flush()

    assistant_msg = Message(conversation_id=conversation_id, role="assistant", content="Test answer.")
    db_session.add(assistant_msg)
    db_session.flush()

    source = MessageSource(
        message_id=assistant_msg.id,
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        page_start=1,
        page_end=1,
        similarity_score=0.85,
    )
    db_session.add(source)
    db_session.commit()

    return {"user_msg_id": user_msg.id, "assistant_msg_id": assistant_msg.id, "source_id": source.id}


# ==========================================================================
# 1. CONVERSATION CREATION
# ==========================================================================

class TestCreateConversation:
    def test_create_authenticated(self):
        register_and_login("MgmtUser1", "mgmt_1@example.com")
        conv = create_conversation("My Conversation")
        assert conv["title"] == "My Conversation"
        assert "id" in conv
        assert "created_at" in conv
        assert "updated_at" in conv

    def test_unauthenticated_create_rejected(self):
        resp = client.post("/conversations", json={"title": "Test"})
        assert resp.status_code == 401

    def test_conversation_belongs_to_creator(self):
        user = register_and_login("MgmtUser2", "mgmt_2@example.com")
        conv = create_conversation("Ownership Test")
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        # The conversation is accessible (ownership verified by GET succeeding)

    def test_no_messages_fabricated_on_create(self, db_session):
        register_and_login("MgmtUser3", "mgmt_3@example.com")
        conv = create_conversation("No Messages")
        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()
        assert len(data["messages"]) == 0

    def test_no_rag_on_create(self):
        register_and_login("MgmtUser4", "mgmt_4@example.com")
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            create_conversation("No RAG")
            mock_rag.assert_not_called()

    def test_no_retrieval_on_create(self):
        register_and_login("MgmtUser5", "mgmt_5@example.com")
        with patch("app.services.retrieval_service.retrieve_context") as mock_retrieval:
            create_conversation("No Retrieval")
            mock_retrieval.assert_not_called()

    def test_empty_title_rejected(self):
        register_and_login("MgmtUser6", "mgmt_6@example.com")
        resp = client.post("/conversations", json={"title": ""})
        assert resp.status_code == 422

    def test_whitespace_title_rejected(self):
        register_and_login("MgmtUser7", "mgmt_7@example.com")
        resp = client.post("/conversations", json={"title": "   "})
        assert resp.status_code == 422

    def test_default_title(self):
        register_and_login("MgmtUser8", "mgmt_8@example.com")
        resp = client.post("/conversations", json={})
        assert resp.status_code == 201
        assert resp.json()["title"] == "New Conversation"


# ==========================================================================
# 2. CONVERSATION LISTING
# ==========================================================================

class TestListConversations:
    def test_list_own_conversations(self):
        register_and_login("MgmtUser9", "mgmt_9@example.com")
        create_conversation("Conv A")
        create_conversation("Conv B")
        resp = client.get("/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_unauthenticated_list_rejected(self):
        auth._sessions.clear()
        resp = client.get("/conversations")
        assert resp.status_code == 401

    def test_user_isolation(self):
        register_and_login("MgmtListA", "mgmt_lista@example.com")
        create_conversation("A's Conv")

        auth._sessions.clear()
        register_and_login("MgmtListB", "mgmt_listb@example.com")
        create_conversation("B's Conv")

        resp = client.get("/conversations")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "B's Conv"

    def test_ordering_newest_first(self):
        register_and_login("MgmtUser10", "mgmt_10@example.com")
        create_conversation("First")
        create_conversation("Second")
        resp = client.get("/conversations")
        items = resp.json()["items"]
        assert items[0]["title"] == "Second"
        assert items[1]["title"] == "First"

    def test_no_messages_loaded(self, db_session):
        register_and_login("MgmtUser11", "mgmt_11@example.com")
        conv = create_conversation("With Messages")
        # Add messages directly
        msg = Message(conversation_id=conv["id"], role="user", content="Hello")
        db_session.add(msg)
        db_session.commit()

        resp = client.get("/conversations")
        # List endpoint should NOT include messages
        for item in resp.json()["items"]:
            assert "messages" not in item

    def test_no_rag_on_list(self):
        register_and_login("MgmtUser12", "mgmt_12@example.com")
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            client.get("/conversations")
            mock_rag.assert_not_called()

    def test_pagination(self):
        register_and_login("MgmtUser13", "mgmt_13@example.com")
        for i in range(5):
            create_conversation(f"Conv {i}")

        resp = client.get("/conversations?limit=2&offset=0")
        data = resp.json()
        assert data["limit"] == 2
        assert data["total"] == 5
        assert len(data["items"]) == 2

        resp2 = client.get("/conversations?limit=2&offset=4")
        data2 = resp2.json()
        assert len(data2["items"]) == 1


# ==========================================================================
# 3. CONVERSATION RETRIEVAL
# ==========================================================================

class TestGetConversation:
    def test_get_own_conversation(self):
        register_and_login("MgmtUser14", "mgmt_14@example.com")
        conv = create_conversation("Get Me")
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Get Me"

    def test_unauthenticated_get_rejected(self):
        auth._sessions.clear()
        resp = client.get("/conversations/1")
        assert resp.status_code == 401

    def test_cross_user_get_rejected(self):
        register_and_login("MgmtGetA", "mgmt_geta@example.com")
        conv_a = create_conversation("A Private")

        auth._sessions.clear()
        register_and_login("MgmtGetB", "mgmt_getb@example.com")

        resp = client.get(f"/conversations/{conv_a['id']}")
        assert resp.status_code == 404

    def test_nonexistent_returns_404(self):
        register_and_login("MgmtUser15", "mgmt_15@example.com")
        resp = client.get("/conversations/99999")
        assert resp.status_code == 404

    def test_no_rag_on_get(self):
        register_and_login("MgmtUser16", "mgmt_16@example.com")
        conv = create_conversation("No RAG Get")
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            client.get(f"/conversations/{conv['id']}")
            mock_rag.assert_not_called()


# ==========================================================================
# 4. CONVERSATION DELETION
# ==========================================================================

class TestDeleteConversation:
    def test_delete_own_conversation(self):
        register_and_login("MgmtUser17", "mgmt_17@example.com")
        conv = create_conversation("To Delete")
        resp = client.delete(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

        # Verify gone
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 404

    def test_unauthenticated_delete_rejected(self):
        auth._sessions.clear()
        resp = client.delete("/conversations/1")
        assert resp.status_code == 401

    def test_cross_user_delete_rejected(self):
        user_a = register_and_login("MgmtDelA", "mgmt_dela@example.com")
        conv_a = create_conversation("A's Private")
        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        register_and_login("MgmtDelB", "mgmt_delb@example.com")

        resp = client.delete(f"/conversations/{conv_a_id}")
        assert resp.status_code == 404

        # Verify A's conversation still exists (login as A again)
        auth._sessions.clear()
        client.post("/auth/login", json={"email": "mgmt_dela@example.com", "password": "testpass123"})
        resp = client.get(f"/conversations/{conv_a_id}")
        assert resp.status_code == 200

    def test_nonexistent_returns_404(self):
        register_and_login("MgmtUser18", "mgmt_18@example.com")
        resp = client.delete("/conversations/99999")
        assert resp.status_code == 404


# ==========================================================================
# 5. CASCADE DELETION
# ==========================================================================

class TestCascadeDeletion:
    def test_delete_conversation_removes_messages_and_sources(self, db_session):
        register_and_login("MgmtUser19", "mgmt_19@example.com")
        user = register_and_login("MgmtUser19b", "mgmt_19b@example.com")
        conv = create_conversation("Cascade Test")
        doc = create_document_with_chunk(db_session, user["id"], "cascade.pdf")

        ids = add_messages_with_sources(db_session, conv["id"], doc["id"], doc["chunk_id"])
        conv_id = conv["id"]

        # Verify data exists
        assert db_session.query(Message).filter(Message.conversation_id == conv_id).count() == 2
        from app.models.message_source import MessageSource
        assert db_session.query(MessageSource).filter(MessageSource.message_id == ids["assistant_msg_id"]).count() == 1

        # Delete
        resp = client.delete(f"/conversations/{conv_id}")
        assert resp.status_code == 200

        # Verify messages and sources are gone
        assert db_session.query(Message).filter(Message.conversation_id == conv_id).count() == 0
        assert db_session.query(MessageSource).filter(MessageSource.message_id == ids["assistant_msg_id"]).count() == 0

    def test_documents_preserved_after_conversation_delete(self, db_session):
        register_and_login("MgmtUser20", "mgmt_20@example.com")
        user = register_and_login("MgmtUser20b", "mgmt_20b@example.com")
        conv = create_conversation("Preserve Docs")
        doc = create_document_with_chunk(db_session, user["id"], "preserve.pdf")

        add_messages_with_sources(db_session, conv["id"], doc["id"], doc["chunk_id"])

        # Delete conversation
        client.delete(f"/conversations/{conv['id']}")

        # Verify document still exists
        remaining_doc = db_session.query(Document).filter(Document.id == doc["id"]).first()
        assert remaining_doc is not None

        # Verify chunk still exists
        remaining_chunk = db_session.query(DocumentChunk).filter(DocumentChunk.id == doc["chunk_id"]).first()
        assert remaining_chunk is not None

    def test_deleting_one_conversation_does_not_affect_another(self, db_session):
        register_and_login("MgmtUser21", "mgmt_21@example.com")
        conv_a = create_conversation("Conv A")
        conv_b = create_conversation("Conv B")

        # Add message to A
        msg = Message(conversation_id=conv_a["id"], role="user", content="A's message")
        db_session.add(msg)
        db_session.commit()

        # Delete A
        client.delete(f"/conversations/{conv_a['id']}")

        # B should still be accessible with no messages of its own
        resp = client.get(f"/conversations/{conv_b['id']}")
        assert resp.status_code == 200
        assert len(resp.json()["messages"]) == 0


# ==========================================================================
# 6. DELETION AFTER NAVIGATION
# ==========================================================================

class TestDeleteAfterNavigation:
    def test_cannot_access_deleted_conversation(self):
        register_and_login("MgmtUser22", "mgmt_22@example.com")
        conv = create_conversation("Will Delete")

        # Delete it
        client.delete(f"/conversations/{conv['id']}")

        # All operations should fail
        assert client.get(f"/conversations/{conv['id']}").status_code == 404
        assert client.get(f"/conversations/{conv['id']}/messages").status_code == 404
        assert client.post(f"/conversations/{conv['id']}/messages", json={"content": "Hi"}).status_code == 404
        assert client.patch(f"/conversations/{conv['id']}", json={"title": "Hacked"}).status_code == 404


# ==========================================================================
# 7. USER ISOLATION (COMPREHENSIVE)
# ==========================================================================

class TestUserIsolation:
    def test_full_isolation(self, db_session):
        register_and_login("MgmtIsoA", "mgmt_iso_a@example.com")
        conv_a = create_conversation("A's Conv")
        user_a = register_and_login("MgmtIsoAb", "mgmt_iso_ab@example.com")
        doc_a = create_document_with_chunk(db_session, user_a["id"], "a_doc.pdf")
        add_messages_with_sources(db_session, conv_a["id"], doc_a["id"], doc_a["chunk_id"])

        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        register_and_login("MgmtIsoB", "mgmt_iso_b@example.com")

        # B cannot access A's conversation
        assert client.get(f"/conversations/{conv_a_id}").status_code == 404
        assert client.get(f"/conversations/{conv_a_id}/messages").status_code == 404
        assert client.delete(f"/conversations/{conv_a_id}").status_code == 404
        assert client.patch(f"/conversations/{conv_a_id}", json={"title": "Hacked"}).status_code == 404
        assert client.post(f"/conversations/{conv_a_id}/messages", json={"content": "Hi"}).status_code == 404

        # B's list does not contain A's conversation
        resp = client.get("/conversations")
        ids = [c["id"] for c in resp.json()["items"]]
        assert conv_a_id not in ids
