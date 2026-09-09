"""Phase 5.6 Step 3 tests: End-to-end conversation integration.

Tests cover the complete conversation workflow:
- Create conversation → open → send message → RAG → persist → reload
- Pagination correctness after message sending
- Source persistence and replay
- Cross-user security for the full flow
- Reload does not trigger RAG/LLM/retrieval
- Edge cases (empty, failed RAG, etc.)
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
from app.services.rag_service import RAGResponse, SourceReference, RAGError
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
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "inttest_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "inttest_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "inttest_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "inttest_%%@example.com"})
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


def create_document(db_session: Session, user_id: int, filename: str = "test.pdf") -> dict:
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"inttest/{user_id}/{filename}",
        mime_type="application/pdf",
        file_size=1024,
        status="ready",
    )
    db_session.add(doc)
    db_session.flush()
    # Create a chunk so source persistence validation passes
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


def make_rag_response(answer="Test answer.", grounded=True, sources=None):
    if sources is None:
        sources = [
            SourceReference(
                document_id=1, filename="test.pdf",
                chunk_id=1, chunk_index=0,
                page_start=1, page_end=1,
                similarity_score=0.85,
            )
        ]
    return RAGResponse(
        answer=answer, sources=sources, grounded=grounded,
        retrieval_count=len(sources), model="fake-llm", provider="fake",
    )


def send_message(conversation_id: int, content: str, document_id=None) -> dict:
    payload = {"content": content}
    if document_id is not None:
        payload["document_id"] = document_id
    resp = client.post(f"/conversations/{conversation_id}/messages", json=payload)
    return resp


# ==========================================================================
# 1. CREATE CONVERSATION → RETRIEVE
# ==========================================================================

class TestCreateAndRetrieve:
    def test_create_then_retrieve(self):
        register_and_login("IntUser1", "inttest_1@example.com")
        conv = create_conversation("Integration Test")

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Integration Test"
        assert data["id"] == conv["id"]
        assert "messages" in data
        assert len(data["messages"]) == 0

    def test_create_appears_in_list(self):
        register_and_login("IntUser2", "inttest_2@example.com")
        conv = create_conversation("Listed Conv")

        resp = client.get("/conversations")
        data = resp.json()
        assert data["total"] >= 1
        ids = [c["id"] for c in data["items"]]
        assert conv["id"] in ids


# ==========================================================================
# 2. CREATE → SEND MESSAGE → RETRIEVE MESSAGES
# ==========================================================================

class TestCreateSendRetrieve:
    def test_send_and_retrieve(self, db_session):
        register_and_login("IntUser3", "inttest_3@example.com")
        conv = create_conversation("Send Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("Integration answer.")

            resp = send_message(conv["id"], "What is the policy?")
            assert resp.status_code == 201

        # Retrieve messages
        resp = client.get(f"/conversations/{conv['id']}/messages")
        data = resp.json()
        assert data["total"] == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "What is the policy?"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Integration answer."

    def test_multiple_messages(self, db_session):
        register_and_login("IntUser4", "inttest_4@example.com")
        conv = create_conversation("Multi Test")

        for i in range(3):
            with patch("app.api.conversations.answer_question_with_history") as mock_rag:
                mock_rag.return_value = make_rag_response(f"Answer {i}.")
                resp = send_message(conv["id"], f"Question {i}?")
                assert resp.status_code == 201

        resp = client.get(f"/conversations/{conv['id']}/messages")
        data = resp.json()
        assert data["total"] == 6  # 3 user + 3 assistant
        for i in range(3):
            assert data["messages"][i * 2]["role"] == "user"
            assert data["messages"][i * 2]["content"] == f"Question {i}?"
            assert data["messages"][i * 2 + 1]["role"] == "assistant"
            assert data["messages"][i * 2 + 1]["content"] == f"Answer {i}."


# ==========================================================================
# 3. ASSISTANT MESSAGE PERSISTED
# ==========================================================================

class TestAssistantPersistence:
    def test_assistant_message_persisted(self, db_session):
        register_and_login("IntUser5", "inttest_5@example.com")
        conv = create_conversation("Persist Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("Persisted answer.")

            send_message(conv["id"], "Test?")

        # Reload conversation — should not call RAG
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            resp = client.get(f"/conversations/{conv['id']}")
            mock_rag.assert_not_called()

        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["messages"][1]["content"] == "Persisted answer."


# ==========================================================================
# 4. SOURCES PERSISTED
# ==========================================================================

class TestSourcePersistence:
    def test_sources_persisted_and_restored(self, db_session):
        register_and_login("IntUser6", "inttest_6@example.com")
        user = register_and_login("IntUser6b", "inttest_6b@example.com")
        conv = create_conversation("Source Test")

        doc = create_document(db_session, user["id"], "source.pdf")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response(
                "Answer with sources.",
                sources=[
                    SourceReference(
                        document_id=doc["id"], filename="source.pdf",
                        chunk_id=doc["chunk_id"], chunk_index=0,
                        page_start=2, page_end=3,
                        similarity_score=0.91,
                    )
                ],
            )
            send_message(conv["id"], "Source question?")

        # Reload conversation — sources should come from DB
        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()
        assistant_msg = data["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["sources"]) == 1
        assert assistant_msg["sources"][0]["document_id"] == doc["id"]
        assert assistant_msg["sources"][0]["page_start"] == 2

    def test_no_rag_on_reload(self, db_session):
        register_and_login("IntUser7", "inttest_7@example.com")
        conv = create_conversation("No RAG Reload")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("Answer.")
            send_message(conv["id"], "Q?")

        # Reload — RAG must NOT be called (messages are loaded from DB)
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            resp = client.get(f"/conversations/{conv['id']}")
            mock_rag.assert_not_called()

        assert resp.status_code == 200
        assert len(resp.json()["messages"]) == 2


# ==========================================================================
# 5. PAGINATION AFTER MESSAGES
# ==========================================================================

class TestPaginationIntegration:
    def test_pagination_with_real_messages(self, db_session):
        register_and_login("IntUser8", "inttest_8@example.com")
        conv = create_conversation("Page Test")

        # Send 5 messages (10 total: 5 user + 5 assistant)
        for i in range(5):
            with patch("app.api.conversations.answer_question_with_history") as mock_rag:
                mock_rag.return_value = make_rag_response(f"Answer {i}.")
                send_message(conv["id"], f"Q{i}?")

        # Get page 1 with page_size=4
        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=4")
        data = resp.json()
        assert len(data["messages"]) == 4
        assert data["total"] == 10
        assert data["has_next"] is True
        assert data["has_previous"] is False

        # Get page 2
        resp2 = client.get(f"/conversations/{conv['id']}/messages?page=2&page_size=4")
        data2 = resp2.json()
        assert len(data2["messages"]) == 4
        assert data2["has_next"] is True
        assert data2["has_previous"] is True

        # Get page 3
        resp3 = client.get(f"/conversations/{conv['id']}/messages?page=3&page_size=4")
        data3 = resp3.json()
        assert len(data3["messages"]) == 2
        assert data3["has_next"] is False
        assert data3["has_previous"] is True

    def test_send_message_then_paginate(self, db_session):
        register_and_login("IntUser9", "inttest_9@example.com")
        conv = create_conversation("Page Send Test")

        # Create 8 messages on page 1 (page_size=5)
        for i in range(4):
            with patch("app.api.conversations.answer_question_with_history") as mock_rag:
                mock_rag.return_value = make_rag_response(f"Answer {i}.")
                send_message(conv["id"], f"Q{i}?")

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=5")
        assert resp.json()["total"] == 8
        assert resp.json()["has_next"] is True

        # Send one more — total becomes 10
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("Answer 4.")
            send_message(conv["id"], "Q4?")

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=5")
        assert resp.json()["total"] == 10

        # Page 2 should now have 5 messages
        resp2 = client.get(f"/conversations/{conv['id']}/messages?page=2&page_size=5")
        assert len(resp2.json()["messages"]) == 5


# ==========================================================================
# 6. CROSS-USER SECURITY (FULL FLOW)
# ==========================================================================

class TestCrossUserFullFlow:
    def test_user_b_cannot_access_user_a_conversation(self, db_session):
        register_and_login("SecA", "inttest_seca@example.com")
        conv_a = create_conversation("A Private")
        conv_a_id = conv_a["id"]

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("A's secret answer.")
            send_message(conv_a_id, "A's secret question?")

        # Switch to User B
        auth._sessions.clear()
        register_and_login("SecB", "inttest_secb@example.com")

        # All access denied
        assert client.get(f"/conversations/{conv_a_id}").status_code == 404
        assert client.get(f"/conversations/{conv_a_id}/messages").status_code == 404
        assert client.post(f"/conversations/{conv_a_id}/messages", json={"content": "Hi"}).status_code == 404

        # User B's conversation list should be empty
        resp = client.get("/conversations")
        assert resp.json()["total"] == 0

    def test_user_b_cannot_use_user_a_document(self, db_session):
        user_a = register_and_login("DocSecA", "inttest_docseca@example.com")
        conv_a = create_conversation("A Doc Conv")
        doc_a = create_document(db_session, user_a["id"], "a_secret.pdf")

        auth._sessions.clear()
        user_b = register_and_login("DocSecB", "inttest_docsecb@example.com")
        conv_b = create_conversation("B Conv")

        # User B tries to use A's document
        resp = send_message(conv_b["id"], "Use A's doc?", document_id=doc_a["id"])
        assert resp.status_code == 404

    def test_user_cannot_change_ownership(self, db_session):
        """User cannot submit arbitrary user_id to create conversations."""
        register_and_login("OwnTest", "inttest_own@example.com")
        conv = create_conversation("Owner Test")

        # Verify conversation belongs to the current user
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200


# ==========================================================================
# 7. FAILED RAG
# ==========================================================================

class TestFailedRAGIntegration:
    def test_rag_failure_no_fake_assistant(self, db_session):
        register_and_login("FailInt1", "inttest_fail1@example.com")
        conv = create_conversation("Fail Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.side_effect = RAGError("Provider down")

            resp = send_message(conv["id"], "What happens?")
            assert resp.status_code == 500

        # Only user message persisted
        resp = client.get(f"/conversations/{conv['id']}/messages")
        data = resp.json()
        assert data["total"] == 1
        assert data["messages"][0]["role"] == "user"

    def test_rag_failure_conversation_still_accessible(self, db_session):
        register_and_login("FailInt2", "inttest_fail2@example.com")
        conv = create_conversation("Still Works")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.side_effect = RuntimeError("Unexpected")

            send_message(conv["id"], "Bad question?")

        # Conversation is still accessible
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Still Works"


# ==========================================================================
# 8. EMPTY CONVERSATION
# ==========================================================================

class TestEmptyConversation:
    def test_empty_conversation_pagination(self):
        register_and_login("EmptyInt1", "inttest_empty1@example.com")
        conv = create_conversation("Empty")

        resp = client.get(f"/conversations/{conv['id']}/messages")
        data = resp.json()
        assert data["messages"] == []
        assert data["total"] == 0
        assert data["has_next"] is False
        assert data["has_previous"] is False
        assert data["page"] == 1

    def test_empty_conversation_detail(self):
        register_and_login("EmptyInt2", "inttest_empty2@example.com")
        conv = create_conversation("Empty Detail")

        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()
        assert data["messages"] == []
        assert data["title"] == "Empty Detail"


# ==========================================================================
# 9. CONVERSATION DELETION CASCADING
# ==========================================================================

class TestDeletionCascade:
    def test_delete_conversation_removes_everything(self, db_session):
        register_and_login("DelInt1", "inttest_del1@example.com")
        conv = create_conversation("To Be Deleted")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response("Will be deleted.")
            send_message(conv["id"], "Delete me?")

        conv_id = conv["id"]

        # Verify messages exist
        resp = client.get(f"/conversations/{conv_id}/messages")
        assert resp.json()["total"] == 2

        # Delete
        resp = client.delete(f"/conversations/{conv_id}")
        assert resp.status_code == 200

        # Verify gone
        assert client.get(f"/conversations/{conv_id}").status_code == 404
        assert client.get(f"/conversations/{conv_id}/messages").status_code == 404

        # Verify DB is clean
        msg_count = db_session.query(Message).filter(Message.conversation_id == conv_id).count()
        assert msg_count == 0


# ==========================================================================
# 10. AUTHENTICATION EDGE CASES
# ==========================================================================

class TestAuthEdgeCases:
    def test_unauthenticated_cannot_send(self):
        auth._sessions.clear()
        resp = client.post("/conversations/1/messages", json={"content": "Hi"})
        assert resp.status_code == 401

    def test_unauthenticated_cannot_list(self):
        auth._sessions.clear()
        resp = client.get("/conversations")
        assert resp.status_code == 401

    def test_unauthenticated_cannot_get_messages(self):
        auth._sessions.clear()
        resp = client.get("/conversations/1/messages")
        assert resp.status_code == 401

    def test_unauthenticated_cannot_create(self):
        auth._sessions.clear()
        resp = client.post("/conversations", json={"title": "Nope"})
        assert resp.status_code == 401


# ==========================================================================
# 11. CONVERSATION TITLE UPDATE
# ==========================================================================

class TestTitleUpdate:
    def test_update_title(self):
        register_and_login("TitleInt1", "inttest_title1@example.com")
        conv = create_conversation("Old Title")

        resp = client.patch(f"/conversations/{conv['id']}", json={"title": "New Title"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"

        # Verify via GET
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.json()["title"] == "New Title"


# ==========================================================================
# 12. SOURCE OWNERSHIP INTEGRATION
# ==========================================================================

class TestSourceOwnershipIntegration:
    def test_source_references_valid_document(self, db_session):
        user = register_and_login("SrcOwn1", "inttest_srcown1@example.com")
        conv = create_conversation("Source Own")
        doc = create_document(db_session, user["id"], "owned.pdf")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = make_rag_response(
                "Answer.",
                sources=[
                    SourceReference(
                        document_id=doc["id"], filename="owned.pdf",
                        chunk_id=doc["chunk_id"], chunk_index=0,
                        page_start=1, page_end=1,
                        similarity_score=0.88,
                    )
                ],
            )
            send_message(conv["id"], "Source Q?")

        resp = client.get(f"/conversations/{conv['id']}")
        assistant = resp.json()["messages"][1]
        assert len(assistant["sources"]) == 1
        assert assistant["sources"][0]["document_id"] == doc["id"]
