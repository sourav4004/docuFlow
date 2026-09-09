"""Phase 5.5 Step 1 tests: MessageSource persistence.

Tests cover:
- MessageSource model creation
- Relationship to Message
- Multiple sources per message
- Source persistence after RAG
- No-source behavior
- Cascade deletion
- History replay (sources available without re-calling RAG)
- Source ordering
- Security (ownership, cross-user isolation)
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
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM document_content WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM documents WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "src_test_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "src_test_%%@example.com"})
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


def create_document_with_chunks(db_session, user_id, filename="test.pdf", num_chunks=1):
    """Create a document and N chunks in the DB, return doc and list of chunks."""
    import uuid as _uuid
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"test/{user_id}/{filename}/{_uuid.uuid4().hex[:8]}",
        mime_type="application/pdf",
        file_size=1024,
        status="ready",
    )
    db_session.add(doc)
    db_session.flush()

    chunks = []
    for i in range(num_chunks):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            text=f"Test document content chunk {i}.",
            char_start=i * 40,
            char_end=(i + 1) * 40,
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.commit()
    db_session.refresh(doc)
    for c in chunks:
        db_session.refresh(c)
    return doc, chunks


def make_sources(doc, chunks, scores=None):
    """Create a list of SourceReference objects pointing to real doc/chunks.

    Each SourceReference points to a different chunk.
    chunks can be a single chunk or a list of chunks.
    """
    if not isinstance(chunks, list):
        chunks = [chunks]
    count = len(chunks)
    if scores is None:
        scores = [0.9 - i * 0.05 for i in range(count)]
    return [
        SourceReference(
            document_id=doc.id,
            filename=doc.original_filename,
            chunk_id=chunks[i].id,
            chunk_index=chunks[i].chunk_index,
            page_start=1,
            page_end=1,
            similarity_score=scores[i],
        )
        for i in range(count)
    ]


def mock_rag_with_sources(sources, answer="Test answer.", grounded=True):
    """Return a RAGResponse with specific sources."""
    return RAGResponse(
        answer=answer,
        sources=sources,
        grounded=grounded,
        retrieval_count=len(sources),
        model="fake-llm",
        provider="fake",
    )


# ==========================================================================
# 1. MODEL CREATION
# ==========================================================================

class TestMessageSourceModel:
    def test_create_source(self, db_session):
        """MessageSource can be created directly."""
        user = register_and_login("SrcUser1", "src_test_1@example.com")
        conv = create_conversation("Model Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        chunk = chunks[0]

        msg = Message(conversation_id=conv["id"], role="assistant", content="Answer")
        db_session.add(msg)
        db_session.flush()

        source = MessageSource(
            message_id=msg.id,
            document_id=doc.id,
            chunk_id=chunk.id,
            chunk_index=0,
            page_start=1,
            page_end=1,
            similarity_score=0.85,
        )
        db_session.add(source)
        db_session.commit()
        db_session.refresh(source)

        assert source.id is not None
        assert source.message_id == msg.id
        assert source.document_id == doc.id
        assert source.chunk_id == chunk.id
        assert source.similarity_score == 0.85

    def test_source_belongs_to_message(self, db_session):
        """MessageSource is accessible via Message.sources relationship."""
        user = register_and_login("SrcUser2", "src_test_2@example.com")
        conv = create_conversation("Rel Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        chunk = chunks[0]

        msg = Message(conversation_id=conv["id"], role="assistant", content="Answer")
        db_session.add(msg)
        db_session.flush()

        source = MessageSource(
            message_id=msg.id,
            document_id=doc.id,
            chunk_id=chunk.id,
            chunk_index=0,
            similarity_score=0.90,
        )
        db_session.add(source)
        db_session.commit()

        # Reload and verify relationship
        db_session.refresh(msg)
        assert len(msg.sources) == 1
        assert msg.sources[0].similarity_score == 0.90

    def test_multiple_sources(self, db_session):
        """Assistant message can have multiple sources."""
        user = register_and_login("SrcUser3", "src_test_3@example.com")
        conv = create_conversation("Multi Source")
        doc, chunks = create_document_with_chunks(db_session, user["id"], num_chunks=3)

        msg = Message(conversation_id=conv["id"], role="assistant", content="Answer")
        db_session.add(msg)
        db_session.flush()

        for i, chunk in enumerate(chunks):
            src = MessageSource(
                message_id=msg.id,
                document_id=doc.id,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                similarity_score=0.9 - i * 0.1,
            )
            db_session.add(src)
        db_session.commit()

        db_session.refresh(msg)
        assert len(msg.sources) == 3


# ==========================================================================
# 2. CASCADE DELETION
# ==========================================================================

class TestCascadeDeletion:
    def test_delete_message_deletes_sources(self, db_session):
        """Deleting a message deletes its MessageSource records."""
        user = register_and_login("SrcUser4", "src_test_4@example.com")
        conv = create_conversation("Cascade Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        chunk = chunks[0]

        msg = Message(conversation_id=conv["id"], role="assistant", content="Answer")
        db_session.add(msg)
        db_session.flush()

        src = MessageSource(
            message_id=msg.id, document_id=doc.id,
            chunk_id=chunk.id, chunk_index=0, similarity_score=0.8,
        )
        db_session.add(src)
        db_session.commit()
        msg_id = msg.id

        # Delete message
        db_session.delete(msg)
        db_session.commit()

        # Verify sources deleted
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg_id
        ).count()
        assert count == 0

    def test_delete_conversation_cleans_sources(self, db_session):
        """Deleting a conversation cascades to messages and their sources."""
        user = register_and_login("SrcUser5", "src_test_5@example.com")
        conv = create_conversation("Cascade Conv")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        chunk = chunks[0]

        msg = Message(conversation_id=conv["id"], role="assistant", content="Answer")
        db_session.add(msg)
        db_session.flush()

        src = MessageSource(
            message_id=msg.id, document_id=doc.id,
            chunk_id=chunk.id, chunk_index=0, similarity_score=0.8,
        )
        db_session.add(src)
        db_session.commit()

        # Delete conversation via API
        resp = client.delete(f"/conversations/{conv['id']}")
        assert resp.status_code == 200

        # Verify sources deleted
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == msg.id
        ).count()
        assert count == 0


# ==========================================================================
# 3. SOURCE PERSISTENCE VIA API
# ==========================================================================

class TestSourcePersistenceAPI:
    def test_sources_persisted_after_rag(self, db_session):
        """Sources from RAG are persisted as MessageSource records."""
        user = register_and_login("SrcUser6", "src_test_6@example.com")
        conv = create_conversation("Persist Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"], num_chunks=2)
        sources = make_sources(doc, chunks)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources, answer="Persisted answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "What is the policy?",
            })
            assert resp.status_code == 201
            data = resp.json()
            assistant_msg_id = data["assistant_message"]["id"]

        # Verify sources persisted in DB
        db_sources = db_session.query(MessageSource).filter(
            MessageSource.message_id == assistant_msg_id
        ).all()
        assert len(db_sources) == 2
        assert db_sources[0].similarity_score == 0.9
        assert db_sources[1].similarity_score == 0.85

    def test_no_sources_when_rag_returns_none(self, db_session):
        """When RAG returns no sources, no MessageSource records created."""
        user = register_and_login("SrcUser7", "src_test_7@example.com")
        conv = create_conversation("No Source Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = RAGResponse(
                answer="No info available.",
                sources=[],
                grounded=False,
                retrieval_count=0,
            )

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Unknown question?",
            })
            assert resp.status_code == 201
            data = resp.json()
            assistant_msg_id = data["assistant_message"]["id"]

        # Verify no sources persisted
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == assistant_msg_id
        ).count()
        assert count == 0
        assert data["sources"] == []

    def test_no_sources_when_rag_fails(self, db_session):
        """When RAG fails, no sources are persisted and no assistant message created."""
        from app.services.rag_service import RAGError
        user = register_and_login("SrcUser8", "src_test_8@example.com")
        conv = create_conversation("Fail Source Test")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.side_effect = RAGError("Provider failed")
            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            assert resp.status_code == 500

        # Verify no sources and only user message
        messages_resp = client.get(f"/conversations/{conv['id']}/messages")
        messages = messages_resp.json()["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_sources_in_api_response(self, db_session):
        """Sources are returned in the SendMessageResponse."""
        user = register_and_login("SrcUser9", "src_test_9@example.com")
        conv = create_conversation("Response Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        sources = make_sources(doc, chunks)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources)

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()
            assert len(data["sources"]) == 1
            assert data["sources"][0]["document_id"] == doc.id


# ==========================================================================
# 4. HISTORY REPLAY
# ==========================================================================

class TestHistoryReplay:
    def test_sources_available_on_reload(self, db_session):
        """After reloading conversation from DB, assistant message has sources."""
        user = register_and_login("SrcUser10", "src_test_10@example.com")
        conv = create_conversation("Replay Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"], num_chunks=2)
        sources = make_sources(doc, chunks)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources, answer="Replay answer.")

            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "What is the policy?",
            })

        # Reload conversation from DB — this does NOT call RAG
        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()

        # Find the assistant message
        assistant_msgs = [m for m in data["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

        # Verify sources are present on the reloaded message
        assert len(assistant_msgs[0]["sources"]) == 2

    def test_no_rag_called_on_reload(self, db_session):
        """Reloading a conversation does NOT re-call RAG."""
        user = register_and_login("SrcUser11", "src_test_11@example.com")
        conv = create_conversation("No RAG Reload")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        sources = make_sources(doc, chunks)

        # Send message (RAG called)
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources)
            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })

        # Reload conversation — RAG should NOT be called
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            resp = client.get(f"/conversations/{conv['id']}")
            assert resp.status_code == 200
            mock_rag.assert_not_called()

    def test_user_message_has_no_sources(self, db_session):
        """User messages should have empty sources list."""
        user = register_and_login("SrcUser12", "src_test_12@example.com")
        conv = create_conversation("User Msg No Source")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        sources = make_sources(doc, chunks)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources)
            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })

        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()
        user_msgs = [m for m in data["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["sources"] == []


# ==========================================================================
# 5. SOURCE ORDERING
# ==========================================================================

class TestSourceOrdering:
    def test_multiple_sources_preserve_order(self, db_session):
        """Multiple sources are returned in the same order as RAG results."""
        user = register_and_login("SrcUser13", "src_test_13@example.com")
        conv = create_conversation("Order Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"], num_chunks=3)
        sources = make_sources(doc, chunks, scores=[0.95, 0.80, 0.65])

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources)
            client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })

        resp = client.get(f"/conversations/{conv['id']}")
        data = resp.json()
        assistant_msgs = [m for m in data["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        src_list = assistant_msgs[0]["sources"]
        assert len(src_list) == 3
        # Sources ordered by creation (ID), which follows insertion order
        assert src_list[0]["similarity_score"] == 0.95
        assert src_list[1]["similarity_score"] == 0.80
        assert src_list[2]["similarity_score"] == 0.65


# ==========================================================================
# 6. SECURITY
# ==========================================================================

class TestSecurity:
    def test_client_cannot_inject_sources(self, db_session):
        """Client cannot submit source IDs for creating MessageSource records."""
        user = register_and_login("SrcUser14", "src_test_14@example.com")
        conv = create_conversation("Injection Test")

        # Client tries to send a message — sources are ONLY from RAG, never from client
        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = RAGResponse(
                answer="Answer.",
                sources=[],  # RAG returns no sources
                grounded=True,
                retrieval_count=0,
            )
            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()

        # No sources should be persisted despite client request
        assert data["sources"] == []
        assistant_msg_id = data["assistant_message"]["id"]
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == assistant_msg_id
        ).count()
        assert count == 0

    def test_cross_user_source_isolation(self, db_session):
        """User A's sources cannot be seen by User B."""
        user_a = register_and_login("SrcIsoA", "src_test_isoa@example.com")
        conv_a = create_conversation("A's Conv")
        doc_a, chunks_a = create_document_with_chunks(db_session, user_a["id"], "a_doc.pdf")
        sources_a = make_sources(doc_a, chunks_a)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources_a, answer="A's answer.")
            client.post(f"/conversations/{conv_a['id']}/messages", json={
                "content": "A's question?",
            })

        conv_a_id = conv_a["id"]

        # Switch to User B
        auth._sessions.clear()
        user_b = register_and_login("SrcIsoB", "src_test_isob@example.com")

        # User B cannot access A's conversation
        resp = client.get(f"/conversations/{conv_a_id}")
        assert resp.status_code == 404

    def test_skips_nonexistent_document(self, db_session):
        """Source referencing non-existent document is gracefully skipped in persistence."""
        user = register_and_login("SrcUser15", "src_test_15@example.com")
        conv = create_conversation("Skip Test")

        # RAG returns a source with a fake document_id
        fake_source = SourceReference(
            document_id=99999,
            filename="ghost.pdf",
            chunk_id=99999,
            chunk_index=0,
            similarity_score=0.8,
        )

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources([fake_source], answer="Answer.")
            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            assert resp.status_code == 201
            data = resp.json()
            assistant_msg_id = data["assistant_message"]["id"]

        # No sources persisted in DB (non-existent document skipped)
        count = db_session.query(MessageSource).filter(
            MessageSource.message_id == assistant_msg_id
        ).count()
        assert count == 0


# ==========================================================================
# 7. SCHEMA VALIDATION
# ==========================================================================

class TestSchemaValidation:
    def test_source_info_response_fields(self, db_session):
        """SourceInfoResponse contains correct fields."""
        from app.schemas.conversation import SourceInfoResponse
        from datetime import datetime

        source = SourceInfoResponse(
            id=1,
            document_id=10,
            chunk_id=20,
            chunk_index=3,
            page_start=5,
            page_end=6,
            similarity_score=0.88,
            created_at=datetime.utcnow(),
        )
        assert source.document_id == 10
        assert source.chunk_id == 20
        assert source.similarity_score == 0.88

    def test_message_response_includes_sources(self, db_session):
        """MessageResponse schema supports sources field."""
        user = register_and_login("SrcUser16", "src_test_16@example.com")
        conv = create_conversation("Schema Test")
        doc, chunks = create_document_with_chunks(db_session, user["id"])
        sources = make_sources(doc, chunks)

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            mock_rag.return_value = mock_rag_with_sources(sources)
            resp = client.post(f"/conversations/{conv['id']}/messages", json={
                "content": "Question?",
            })
            data = resp.json()

        # Both messages have sources field
        assert "sources" in data["user_message"]
        assert "sources" in data["assistant_message"]
        assert data["user_message"]["sources"] == []
        assert len(data["assistant_message"]["sources"]) == 1
