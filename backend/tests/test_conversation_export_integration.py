"""Phase 5.8 Step 3: Conversation export integration tests.

Tests cover what existing export tests do NOT:
- Deleted document (cascade removes MessageSource) → export still works
- Deleted chunk (cascade removes MessageSource) → export still works
- 100+ message large conversation → complete export
- Export does NOT call retrieval service
- Side-effect safety: zero DB writes during export
- Export uses complete conversation (not paginated endpoint)
- Rapid successive exports produce consistent results
- Content exactness: exported content matches DB content exactly
- NULL metadata combinations for source fields
- Frontend TypeScript/build validation (run externally)
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

    # Clean up test data with unique prefix
    try:
        db_session.execute(text(
            "DELETE FROM message_sources WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)))"
        ), {"p1": "eint_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1))"
        ), {"p1": "eint_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "eint_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM documents WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "eint_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM users WHERE email LIKE :p1"
        ), {"p1": "eint_%%@example.com"})
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


def add_message(db_session: Session, conversation_id: int, role: str, content: str) -> dict:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return {"id": msg.id, "role": msg.role, "content": msg.content}


def create_doc(db_session: Session, user_id: int, filename: str = "test.pdf") -> dict:
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"eint/{user_id}/{filename}",
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


def add_source(
    db_session: Session,
    message_id: int,
    doc_id: int,
    chunk_id: int,
    page_start=1,
    page_end=1,
    score=0.85,
    chunk_index=0,
) -> MessageSource:
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
# 1. DELETED DOCUMENT CASCADE → EXPORT STILL WORKS
# ==========================================================================

class TestExportDeletedDocumentCascade:
    """When a referenced document is deleted, the DB cascade removes
    MessageSource rows. Export must still work without crash."""

    def test_export_after_document_deleted(self, db_session):
        """Export works after the referenced document is deleted."""
        user = register_and_login("eint_DelDoc1", "eint_deldoc1@example.com")
        conv = create_conversation("Del Doc Conv")
        doc = create_doc(db_session, user["id"], "source.pdf")

        # Create assistant message with source
        asst = add_message(db_session, conv["id"], "assistant", "Answer with source")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"])

        # Verify source exists before deletion
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Document #" in resp.text

        # Delete the document (cascades to MessageSource via FK)
        db_session.execute(text("DELETE FROM documents WHERE id = :doc_id"), {"doc_id": doc["id"]})
        db_session.commit()

        # Export should still work — message remains, sources are gone
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Answer with source" in resp.text

    def test_export_after_document_deleted_no_sources_section(self, db_session):
        """After document cascade, assistant with no remaining sources
        should not show a Sources section."""
        user = register_and_login("eint_DelDoc2", "eint_deldoc2@example.com")
        conv = create_conversation("Del Doc No Src")
        doc = create_doc(db_session, user["id"], "source2.pdf")

        asst = add_message(db_session, conv["id"], "assistant", "Answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"])

        # Delete the document
        db_session.execute(text("DELETE FROM documents WHERE id = :doc_id"), {"doc_id": doc["id"]})
        db_session.commit()

        # Verify source was cascade-deleted
        remaining = db_session.query(MessageSource).filter(
            MessageSource.document_id == doc["id"]
        ).count()
        assert remaining == 0

        # Export: message exists, no Sources section
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Answer" in resp.text
        assert "Sources:" not in resp.text


# ==========================================================================
# 2. DELETED CHUNK CASCADE → EXPORT STILL WORKS
# ==========================================================================

class TestExportDeletedChunkCascade:
    """When a referenced chunk is deleted, the DB cascade removes
    MessageSource rows. Export must still work."""

    def test_export_after_chunk_deleted(self, db_session):
        """Export works after the referenced chunk is deleted."""
        user = register_and_login("eint_DelChk1", "eint_delchk1@example.com")
        conv = create_conversation("Del Chunk Conv")
        doc = create_doc(db_session, user["id"], "chunk_source.pdf")

        asst = add_message(db_session, conv["id"], "assistant", "Answer with chunk")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"])

        # Delete the chunk (cascades to MessageSource via FK)
        db_session.execute(text("DELETE FROM document_chunks WHERE id = :chunk_id"), {"chunk_id": doc["chunk_id"]})
        db_session.commit()

        # Export should still work
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Answer with chunk" in resp.text


# ==========================================================================
# 3. LARGE CONVERSATIONS (100+ messages)
# ==========================================================================

class TestExportLargeConversation:
    """Test conversations with 100+ messages to verify complete export
    (not paginated)."""

    def test_100_messages_all_exported(self, db_session):
        """Conversation with 100 messages (50 QA pairs) exports all."""
        register_and_login("eint_Lrg1", "eint_lrg1@example.com")
        conv = create_conversation("100 Msg Conv")
        num = 100
        for i in range(num):
            role = "user" if i % 2 == 0 else "assistant"
            add_message(db_session, conv["id"], role, f"Message_{i:04d}")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        text = resp.text

        # All 100 messages must be present
        for i in range(num):
            assert f"Message_{i:04d}" in text, f"Message_{i:04d} missing from export"

    def test_100_messages_ordering_correct(self, db_session):
        """Ordering is correct across 100 messages."""
        register_and_login("eint_Lrg2", "eint_lrg2@example.com")
        conv = create_conversation("100 Order Conv")
        for i in range(100):
            add_message(db_session, conv["id"], "user", f"Q_{i:04d}")
            add_message(db_session, conv["id"], "assistant", f"A_{i:04d}")

        resp = client.get(f"/conversations/{conv['id']}/export")
        text = resp.text

        # Verify first and last
        first_pos = text.find("Q_0000")
        last_pos = text.find("A_0099")
        assert first_pos >= 0
        assert last_pos >= 0
        assert first_pos < last_pos

    def test_150_messages_with_mixed_sources(self, db_session):
        """150 messages with sources on some assistant messages."""
        user = register_and_login("eint_Lrg3", "eint_lrg3@example.com")
        conv = create_conversation("150 Mix Conv")
        doc = create_doc(db_session, user["id"], "large_source.pdf")

        for i in range(75):
            add_message(db_session, conv["id"], "user", f"Q{i}")
            asst = add_message(db_session, conv["id"], "assistant", f"A{i}")
            # Add sources to every 5th assistant message
            if i % 5 == 0:
                add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                           page_start=i + 1, page_end=i + 1, score=0.8)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        text = resp.text

        # Verify all messages present
        for i in range(75):
            assert f"Q{i}" in text
            assert f"A{i}" in text

        # Should have 15 sources (0, 5, 10, ..., 70)
        source_count = text.count("Document #")
        assert source_count == 15


# ==========================================================================
# 4. SIDE-EFFECT SAFETY (zero DB writes during export)
# ==========================================================================

class TestExportSideEffectSafety:
    """Export must be completely read-only."""

    def test_export_creates_no_messages(self, db_session):
        """Exporting does not create any new Message records."""
        user = register_and_login("eint_Side1", "eint_side1@example.com")
        conv = create_conversation("Side Effect Conv")
        add_message(db_session, conv["id"], "user", "Q1")
        add_message(db_session, conv["id"], "assistant", "A1")

        # Count messages before
        count_before = db_session.query(Message).filter(
            Message.conversation_id == conv["id"]
        ).count()

        # Export 3 times
        for _ in range(3):
            resp = client.get(f"/conversations/{conv['id']}/export")
            assert resp.status_code == 200

        # Count messages after — must be the same
        count_after = db_session.query(Message).filter(
            Message.conversation_id == conv["id"]
        ).count()
        assert count_before == count_after

    def test_export_creates_no_conversations(self, db_session):
        """Exporting does not create any new Conversation records."""
        user = register_and_login("eint_Side2", "eint_side2@example.com")
        conv = create_conversation("No New Conv")

        count_before = db_session.query(Conversation).filter(
            Conversation.user_id == user["id"]
        ).count()

        client.get(f"/conversations/{conv['id']}/export")
        client.get(f"/conversations/{conv['id']}/export")

        count_after = db_session.query(Conversation).filter(
            Conversation.user_id == user["id"]
        ).count()
        assert count_before == count_after

    def test_export_creates_no_message_sources(self, db_session):
        """Exporting does not create any new MessageSource records."""
        user = register_and_login("eint_Side3", "eint_side3@example.com")
        conv = create_conversation("No New Sources")
        doc = create_doc(db_session, user["id"])

        asst = add_message(db_session, conv["id"], "assistant", "Answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"])

        count_before = db_session.query(MessageSource).count()

        client.get(f"/conversations/{conv['id']}/export")
        client.get(f"/conversations/{conv['id']}/export")
        client.get(f"/conversations/{conv['id']}/export")

        count_after = db_session.query(MessageSource).count()
        assert count_before == count_after

    def test_export_does_not_modify_documents(self, db_session):
        """Exporting does not modify any Document records."""
        user = register_and_login("eint_Side4", "eint_side4@example.com")
        conv = create_conversation("No Doc Modify")
        doc = create_doc(db_session, user["id"])

        # Get original document state
        original = db_session.query(Document).filter(Document.id == doc["id"]).first()
        original_status = original.status
        original_filename = original.original_filename

        client.get(f"/conversations/{conv['id']}/export")

        # Verify unchanged
        db_session.refresh(original)
        assert original.status == original_status
        assert original.original_filename == original_filename


# ==========================================================================
# 5. NO RETRIEVAL SERVICE DURING EXPORT
# ==========================================================================

class TestExportNoRetrieval:
    """Export must NOT call the retrieval/embedding/vector search pipeline."""

    def test_no_retrieval_called(self, db_session):
        """Export does not call the retrieval service."""
        user = register_and_login("eint_NoRet1", "eint_noret1@example.com")
        conv = create_conversation("No Retrieval")
        add_message(db_session, conv["id"], "user", "Question?")

        with patch("app.services.vector_search.search_similar_chunks") as mock_vs:
            resp = client.get(f"/conversations/{conv['id']}/export")
            assert resp.status_code == 200
            mock_vs.assert_not_called()

    def test_no_embedding_called(self, db_session):
        """Export does not call the embedding service."""
        user = register_and_login("eint_NoRet2", "eint_noret2@example.com")
        conv = create_conversation("No Embedding")
        add_message(db_session, conv["id"], "assistant", "Answer")

        with patch("app.services.embedding_service.generate_document_embeddings") as mock_emb:
            resp = client.get(f"/conversations/{conv['id']}/export")
            assert resp.status_code == 200
            mock_emb.assert_not_called()

    def test_export_does_not_use_paginated_endpoint(self, db_session):
        """The export endpoint queries all messages directly,
        not through the paginated messages endpoint."""
        user = register_and_login("eint_NoRet3", "eint_noret3@example.com")
        conv = create_conversation("No Pagination")
        for i in range(10):
            add_message(db_session, conv["id"], "user", f"Q{i}")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200

        # All 10 messages in the export
        for i in range(10):
            assert f"Q{i}" in resp.text

        # Compare with paginated endpoint (page_size=5)
        resp_page1 = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=5")
        resp_page2 = client.get(f"/conversations/{conv['id']}/messages?page=2&page_size=5")
        assert resp_page1.json()["total"] == 10
        assert resp_page2.json()["has_next"] is False

        # Export has all 10 — the paginated endpoint requires 2 pages
        assert resp.text.count("Q") >= 10


# ==========================================================================
# 6. RAPID SUCCESSIVE EXPORTS
# ==========================================================================

class TestExportRapidSuccessive:
    """Multiple rapid exports must all succeed and produce consistent content."""

    def test_rapid_successive_exports_all_succeed(self, db_session):
        """5 rapid successive exports all return 200."""
        user = register_and_login("eint_Rapid1", "eint_rapid1@example.com")
        conv = create_conversation("Rapid Export")
        add_message(db_session, conv["id"], "user", "Stable content")
        add_message(db_session, conv["id"], "assistant", "Stable response")

        results = []
        for _ in range(5):
            resp = client.get(f"/conversations/{conv['id']}/export")
            results.append(resp)

        for resp in results:
            assert resp.status_code == 200
            assert "Stable content" in resp.text
            assert "Stable response" in resp.text

    def test_rapid_exports_identical_content(self, db_session):
        """Rapid successive exports (ignoring timestamp) are content-identical."""
        user = register_and_login("eint_Rapid2", "eint_rapid2@example.com")
        conv = create_conversation("Identical Export")
        add_message(db_session, conv["id"], "user", "Q1")
        add_message(db_session, conv["id"], "assistant", "A1")

        resp1 = client.get(f"/conversations/{conv['id']}/export")
        resp2 = client.get(f"/conversations/{conv['id']}/export")
        resp3 = client.get(f"/conversations/{conv['id']}/export")

        # Remove timestamp line for comparison
        def strip_timestamp(text):
            return "\n".join(
                line for line in text.split("\n")
                if not line.startswith("*Exported on")
            )

        t1 = strip_timestamp(resp1.text)
        t2 = strip_timestamp(resp2.text)
        t3 = strip_timestamp(resp3.text)

        assert t1 == t2 == t3


# ==========================================================================
# 7. CONTENT EXACTNESS
# ==========================================================================

class TestExportContentExactness:
    """Exported content must exactly match what's in the database."""

    def test_exact_user_message_content(self, db_session):
        """User message content is reproduced exactly."""
        register_and_login("eint_Exact1", "eint_exact1@example.com")
        conv = create_conversation("Exact Content")
        exact = "This is a very specific message with <special> & \"quotes\"."
        add_message(db_session, conv["id"], "user", exact)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert exact in resp.text

    def test_exact_assistant_message_content(self, db_session):
        """Assistant message content is reproduced exactly."""
        register_and_login("eint_Exact2", "eint_exact2@example.com")
        conv = create_conversation("Exact Assistant")
        exact = "The answer is **42** and here is `code`."
        add_message(db_session, conv["id"], "assistant", exact)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert exact in resp.text

    def test_exact_multiline_content(self, db_session):
        """Multiline content is reproduced exactly."""
        register_and_login("eint_Exact3", "eint_exact3@example.com")
        conv = create_conversation("Exact Multi")
        exact = "Line 1\nLine 2\n\nLine 4 after blank"
        add_message(db_session, conv["id"], "user", exact)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Line 1\nLine 2" in resp.text
        assert "Line 4 after blank" in resp.text

    def test_all_messages_count_matches(self, db_session):
        """Number of message sections in export matches DB count."""
        register_and_login("eint_Exact4", "eint_exact4@example.com")
        conv = create_conversation("Count Match")
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            add_message(db_session, conv["id"], role, f"Msg{i}")

        resp = client.get(f"/conversations/{conv['id']}/export")
        text = resp.text

        # Count unique message markers
        user_count = db_session.query(Message).filter(
            Message.conversation_id == conv["id"],
            Message.role == "user"
        ).count()
        asst_count = db_session.query(Message).filter(
            Message.conversation_id == conv["id"],
            Message.role == "assistant"
        ).count()

        assert text.count("**You**") == user_count
        assert text.count("**DocuFlow**") == asst_count


# ==========================================================================
# 8. NULL METADATA COMBINATIONS
# ==========================================================================

class TestExportNullMetadata:
    """Sources with various NULL metadata fields must export without crash."""

    def test_all_metadata_null(self, db_session):
        """Source with all optional fields NULL."""
        user = register_and_login("eint_Null1", "eint_null1@example.com")
        conv = create_conversation("All Null")
        doc = create_doc(db_session, user["id"])
        asst = add_message(db_session, conv["id"], "assistant", "Answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=None, page_end=None, score=None)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Sources:" in resp.text

    def test_only_page_start_null(self, db_session):
        """Source with only page_start NULL (page_end not shown)."""
        user = register_and_login("eint_Null2", "eint_null2@example.com")
        conv = create_conversation("Partial Pages")
        doc = create_doc(db_session, user["id"])
        asst = add_message(db_session, conv["id"], "assistant", "Answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=None, page_end=5, score=0.7)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200

    def test_only_score_null(self, db_session):
        """Source with only similarity_score NULL."""
        user = register_and_login("eint_Null3", "eint_null3@example.com")
        conv = create_conversation("No Score")
        doc = create_doc(db_session, user["id"])
        asst = add_message(db_session, conv["id"], "assistant", "Answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=1, page_end=3, score=None)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Document #" in resp.text


# ==========================================================================
# 9. CROSS-CONVERSATION ISOLATION
# ==========================================================================

class TestExportCrossConversationIsolation:
    """Export of one conversation never leaks data from another."""

    def test_export_only_contains_own_messages(self, db_session):
        """Export only contains messages from the target conversation."""
        user = register_and_login("eint_Iso1", "eint_iso1@example.com")

        # Create two conversations
        conv1 = create_conversation("Conv One")
        conv2 = create_conversation("Conv Two")

        add_message(db_session, conv1["id"], "user", "Secret message for conv1")
        add_message(db_session, conv2["id"], "user", "Secret message for conv2")

        # Export conv1
        resp = client.get(f"/conversations/{conv1['id']}/export")
        assert "Secret message for conv1" in resp.text
        assert "Secret message for conv2" not in resp.text

        # Export conv2
        resp = client.get(f"/conversations/{conv2['id']}/export")
        assert "Secret message for conv2" in resp.text
        assert "Secret message for conv1" not in resp.text


# ==========================================================================
# 10. TITLE CHANGE REFLECTS IN EXPORT
# ==========================================================================

class TestExportTitleChange:
    """Title changes are reflected in export, including filename."""

    def test_title_change_reflected_in_content(self, db_session):
        """Updated title appears in export content."""
        register_and_login("eint_Tit1", "eint_tit1@example.com")
        conv = create_conversation("Old Title")

        client.patch(f"/conversations/{conv['id']}", json={"title": "Updated Title"})

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "# Updated Title" in resp.text
        assert "Old Title" not in resp.text

    def test_title_change_reflected_in_filename(self, db_session):
        """Updated title is used in the Content-Disposition filename."""
        register_and_login("eint_Tit2", "eint_tit2@example.com")
        conv = create_conversation("Old File Name")

        client.patch(f"/conversations/{conv['id']}", json={"title": "New File Name"})

        resp = client.get(f"/conversations/{conv['id']}/export")
        cd = resp.headers.get("Content-Disposition", "")
        assert "New_File_Name" in cd
        assert "Old_File_Name" not in cd
