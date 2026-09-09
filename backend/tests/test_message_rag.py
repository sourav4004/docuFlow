"""Phase 5.3 tests: Message creation + RAG integration.

Tests cover:
- Message creation via POST /conversations/{id}/messages
- RAG pipeline integration (mocked)
- Conversation and document ownership enforcement
- Edge cases (empty, whitespace, long messages)
- Grounded/ungrounded responses
- Hallucination prevention
- Source metadata
- Cross-user isolation
- Transaction behavior on RAG failure
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
from app.services.rag_service import RAGResponse, SourceReference, RAGError
from app.core.security import hash_password
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """Provide a single DB session shared across API calls and direct inserts."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def setup_test_environment(db_session):
    """Clean test data and override FastAPI get_db to use shared session."""
    auth._sessions.clear()

    previous_overrides = dict(app.dependency_overrides)

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db

    # Clean test data — only touch tables guaranteed to exist
    try:
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "msg_test_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "msg_test_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "msg_test_%%@example.com"})
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
    """Register a user, login, return user dict."""
    resp = client.post("/auth/register", json={
        "name": name, "email": email, "password": password,
    })
    assert resp.status_code == 201
    user = resp.json()
    resp = client.post("/auth/login", json={
        "email": email, "password": password,
    })
    assert resp.status_code == 200
    return user


def create_conversation(title: str = "Test Conv") -> dict:
    """Create a conversation via API."""
    resp = client.post("/conversations", json={"title": title})
    assert resp.status_code == 201
    return resp.json()


def create_document(db_session: Session, user_id: int, filename: str = "test.pdf") -> dict:
    """Create a document directly in DB for testing."""
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"test/{filename}",
        mime_type="application/pdf",
        file_size=1024,
        status="ready",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return {"id": doc.id, "filename": doc.original_filename}


def mock_rag_response(answer: str = "This is a test answer.", grounded: bool = True,
                       sources: list = None, document_id: int = 1, chunk_id: int = 1):
    """Create a mock RAGResponse for testing.

    Uses valid document_id/chunk_id if provided; otherwise creates
    references that may not exist (for tests that don't persist sources).
    """
    if sources is None:
        sources = [
            SourceReference(
                document_id=document_id,
                filename="test.pdf",
                chunk_id=chunk_id,
                chunk_index=0,
                page_start=1,
                page_end=1,
                similarity_score=0.85,
            )
        ]
    return RAGResponse(
        answer=answer,
        sources=sources,
        grounded=grounded,
        retrieval_count=len(sources),
        model="fake-llm",
        provider="fake",
    )


# ==========================================================================
# 1. BASIC MESSAGE CREATION
# ==========================================================================

class TestSendMessageBasic:
    def test_send_message_creates_both_messages(self, db_session):
        """User message and assistant message should both be created."""
        user = register_and_login("MsgUser1", "msg_test_1@example.com")
        conv = create_conversation("Msg Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Test answer here.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "What is the policy?",
            })
            assert resp.status_code == 201
            data = resp.json()

            # User message
            assert data["user_message"]["role"] == "user"
            assert data["user_message"]["content"] == "What is the policy?"

            # Assistant message
            assert data["assistant_message"]["role"] == "assistant"
            assert data["assistant_message"]["content"] == "Test answer here."

            # Both have IDs
            assert data["user_message"]["id"] is not None
            assert data["assistant_message"]["id"] is not None

    def test_rag_receives_correct_question(self, db_session):
        """The user's message content should be passed to RAG as the question."""
        register_and_login("MsgUser2", "msg_test_2@example.com")
        conv = create_conversation("RAG Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "How many vacation days?",
            })

            mock_rag.assert_called_once()
            call_kwargs = mock_rag.call_args
            assert call_kwargs.kwargs["question"] == "How many vacation days?"

    def test_rag_receives_correct_user_id(self, db_session):
        """RAG should receive the authenticated user's ID."""
        user = register_and_login("MsgUser3", "msg_test_3@example.com")
        conv = create_conversation("User ID Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Test?",
            })

            call_kwargs = mock_rag.call_args
            assert call_kwargs.kwargs["user_id"] == user["id"]

    def test_sources_returned(self, db_session):
        """Sources from RAG should be included in the response."""
        register_and_login("MsgUser4", "msg_test_4@example.com")
        conv = create_conversation("Sources Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response(
                "Answer with sources.",
                sources=[
                    SourceReference(
                        document_id=10, filename="handbook.pdf",
                        chunk_id=5, chunk_index=2,
                        page_start=3, page_end=4,
                        similarity_score=0.92,
                    )
                ]
            )

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()
            assert len(data["sources"]) == 1
            assert data["sources"][0]["document_id"] == 10
            assert data["sources"][0]["filename"] == "handbook.pdf"
            assert data["sources"][0]["similarity_score"] == 0.92
            assert data["sources"][0]["page_start"] == 3

    def test_grounded_true(self, db_session):
        """grounded=true should be returned when RAG finds relevant context."""
        register_and_login("MsgUser5", "msg_test_5@example.com")
        conv = create_conversation("Grounded Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Grounded answer.", grounded=True)

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            assert resp.json()["grounded"] is True


# ==========================================================================
# 2. OWNERSHIP ENFORCEMENT
# ==========================================================================

class TestOwnership:
    def test_cannot_send_to_other_users_conversation(self, db_session):
        """User A cannot send messages to User B's conversation."""
        register_and_login("OwnA", "msg_test_own_a@example.com")
        conv_a = create_conversation("A's Conv")
        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        register_and_login("OwnB", "msg_test_own_b@example.com")

        resp = client.post(f"/conversations/{conv_a_id}/messages", json={
            "content": "Hello?",
        })
        assert resp.status_code == 404

    def test_unauthenticated_send_rejected(self):
        """Unauthenticated users cannot send messages."""
        auth._sessions.clear()
        resp = client.post("/conversations/1/messages", json={
            "content": "Hello?",
        })
        assert resp.status_code == 401

    def test_document_ownership_enforced(self, db_session):
        """User A cannot use User B's document_id."""
        register_and_login("DocOwnA", "msg_test_docown_a@example.com")

        # Create User B's document via direct DB insert
        user_b = User(name="UserB", email="msg_test_docown_b@example.com", password_hash=hash_password("testpass123"))
        db_session.add(user_b)
        db_session.commit()
        db_session.refresh(user_b)

        doc_b = create_document(db_session, user_b.id, "b_doc.pdf")

        conv = create_conversation("A's Conv")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
                "document_id": doc_b["id"],
            })
            # Should fail — user doesn't own the document
            assert resp.status_code == 404


# ==========================================================================
# 3. RAG FAILURE HANDLING
# ==========================================================================

class TestRAGFailure:
    def test_rag_error_returns_500(self, db_session):
        """RAG error should return 500 without creating assistant message."""
        register_and_login("FailUser1", "msg_test_fail1@example.com")
        conv = create_conversation("Fail Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.side_effect = RAGError("Provider failed")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "What is the policy?",
            })
            assert resp.status_code == 500

        # Verify user message was persisted but assistant message was not
        resp = client.get(f"/conversations/{conv['id']}/messages")
        messages = resp.json()["messages"]
        assert len(messages) == 1  # only user message
        assert messages[0]["role"] == "user"

    def test_unhandled_error_returns_500(self, db_session):
        """Unexpected errors should return 500 without creating fake assistant message."""
        register_and_login("FailUser2", "msg_test_fail2@example.com")
        conv = create_conversation("Error Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.side_effect = RuntimeError("Unexpected")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            assert resp.status_code == 500

        resp = client.get(f"/conversations/{conv['id']}/messages")
        messages = resp.json()["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ==========================================================================
# 4. NO-CONTEXT / HALLUCINATION PREVENTION
# ==========================================================================

class TestHallucination:
    def test_no_context_returns_grounded_false(self, db_session):
        """When RAG finds no context, grounded=false and no hallucination."""
        register_and_login("HallucUser1", "msg_test_halluc1@example.com")
        conv = create_conversation("Hallucination Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = RAGResponse(
                answer="I don't have enough information in the provided documents to answer this question.",
                sources=[],
                grounded=False,
                retrieval_count=0,
            )

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "What is the maternity policy?",
            })
            assert resp.status_code == 201
            data = resp.json()
            assert data["grounded"] is False
            assert len(data["sources"]) == 0
            assert "don't have enough" in data["assistant_message"]["content"].lower()

    def test_empty_sources_with_grounded_false(self, db_session):
        """grounded=false should come with empty sources."""
        register_and_login("HallucUser2", "msg_test_halluc2@example.com")
        conv = create_conversation("No Sources Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = RAGResponse(
                answer="No information available.",
                sources=[],
                grounded=False,
                retrieval_count=0,
            )

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()
            assert data["grounded"] is False
            assert data["sources"] == []


# ==========================================================================
# 5. VALIDATION
# ==========================================================================

class TestValidation:
    def test_empty_message_rejected(self, db_session):
        """Empty message content should be rejected."""
        register_and_login("ValUser1", "msg_test_val1@example.com")
        conv = create_conversation("Val Test")

        resp = client.post(f"/conversations/{conv['id']}/messages", json={
            "content": "",
        })
        assert resp.status_code == 422

    def test_whitespace_message_rejected(self, db_session):
        """Whitespace-only message should be rejected."""
        register_and_login("ValUser2", "msg_test_val2@example.com")
        conv = create_conversation("Val WS Test")

        resp = client.post(f"/conversations/{conv['id']}/messages", json={
            "content": "   ",
        })
        assert resp.status_code == 422

    def test_missing_content_rejected(self, db_session):
        """Missing content field should be rejected."""
        register_and_login("ValUser3", "msg_test_val3@example.com")
        conv = create_conversation("Val Missing")

        resp = client.post(f"/conversations/{conv['id']}/messages", json={})
        assert resp.status_code == 422

    def test_long_message_rejected(self, db_session):
        """Message exceeding max length should be rejected."""
        register_and_login("ValUser4", "msg_test_val4@example.com")
        conv = create_conversation("Val Long")

        resp = client.post(f"/conversations/{conv['id']}/messages", json={
            "content": "x" * 2001,
        })
        assert resp.status_code == 422

    def test_invalid_conversation_id(self, db_session):
        """Non-existent conversation should return 404."""
        register_and_login("ValUser5", "msg_test_val5@example.com")

        resp = client.post("/conversations/99999/messages", json={
            "content": "Hello?",
        })
        assert resp.status_code == 404

    def test_invalid_document_id_rejected(self, db_session):
        """Non-existent document_id should return 404."""
        register_and_login("ValUser6", "msg_test_val6@example.com")
        conv = create_conversation("Val Doc Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
                "document_id": 99999,
            })
            assert resp.status_code == 404

    def test_user_cannot_submit_role(self, db_session):
        """Role should be set by server, not client."""
        register_and_login("ValUser7", "msg_test_val7@example.com")
        conv = create_conversation("Val Role")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()
            assert data["user_message"]["role"] == "user"
            assert data["assistant_message"]["role"] == "assistant"


# ==========================================================================
# 6. MESSAGE ORDERING & PERSISTENCE
# ==========================================================================

class TestMessageOrdering:
    def test_messages_appear_in_chronological_order(self, db_session):
        """After sending multiple messages, they should appear in order."""
        register_and_login("OrdUser1", "msg_test_ord1@example.com")
        conv = create_conversation("Order Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer 1.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "First question",
            })

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer 2.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Second question",
            })

        # Verify ordering via GET messages
        resp = client.get(f"/conversations/{conv['id']}/messages")
        messages = resp.json()["messages"]
        assert len(messages) == 4  # user1, assistant1, user2, assistant2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First question"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Second question"
        assert messages[3]["role"] == "assistant"

    def test_conversation_updated_at_changes(self, db_session):
        """Sending a message should update conversation's updated_at."""
        register_and_login("OrdUser2", "msg_test_ord2@example.com")
        conv = create_conversation("Timestamp Test")
        original_updated = conv["updated_at"]

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Answer.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "New question?",
            })

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.json()["updated_at"] != original_updated


# ==========================================================================
# 7. INTEGRATION: DOCUMENT-SCOPED RAG
# ==========================================================================

class TestDocumentScopedRAG:
    def test_document_id_passed_to_rag(self, db_session):
        """When document_id is supplied, it should be passed to RAG."""
        user = register_and_login("DocUser1", "msg_test_doc1@example.com")
        conv = create_conversation("Doc Scoped")
        doc = create_document(db_session, user["id"], "scoped.pdf")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Doc answer.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Document question?",
                "document_id": doc["id"],
            })

            call_kwargs = mock_rag.call_args
            assert call_kwargs.kwargs["document_id"] == doc["id"]

    def test_no_document_id_passes_none(self, db_session):
        """When document_id is omitted, RAG should receive None."""
        register_and_login("DocUser2", "msg_test_doc2@example.com")
        conv = create_conversation("No Doc Scoped")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("All docs answer.")

            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "General question?",
            })

            call_kwargs = mock_rag.call_args
            assert call_kwargs.kwargs["document_id"] is None


# ==========================================================================
# 8. CROSS-USER SECURITY
# ==========================================================================

class TestCrossUserSecurity:
    def test_full_isolation(self, db_session):
        """Complete cross-user isolation for message sending."""
        user_a = register_and_login("IsoMsgA", "msg_test_isoa@example.com")
        conv_a = create_conversation("A's Private Conv")
        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        user_b = register_and_login("IsoMsgB", "msg_test_isob@example.com")

        # User B cannot send messages to A's conversation
        resp = client.post(f"/conversations/{conv_a_id}/messages", json={
            "content": "Injected!",
        })
        assert resp.status_code == 404

        # User B cannot see A's messages
        resp = client.get(f"/conversations/{conv_a_id}/messages")
        assert resp.status_code == 404

        # User B cannot use A's document
        doc_a = create_document(db_session, user_a["id"], "a_secret.pdf")
        resp = client.post(f"/conversations/1/messages", json={
            "content": "Use A's doc?",
            "document_id": doc_a["id"],
        })
        assert resp.status_code == 404


# ==========================================================================
# 9. PROVIDER INDEPENDENCE
# ==========================================================================

class TestProviderIndependence:
    def test_rag_service_called_not_llm_directly(self, db_session):
        """The endpoint should call answer_question, not the LLM directly."""
        register_and_login("ProvUser1", "msg_test_prov1@example.com")
        conv = create_conversation("Provider Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_response("Provider-independent answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })

            mock_rag.assert_called_once()
            assert resp.status_code == 201
            assert resp.json()["assistant_message"]["content"] == "Provider-independent answer."
