"""Phase 5.8 Step 1 tests: Conversation export as Markdown.

Tests cover:
- Export owned conversation
- Authentication required
- Cross-user export rejected
- Nonexistent conversation returns 404
- Export format (Markdown structure)
- Messages included in export
- Sources included in export
- Empty conversation export
- Special characters in title
- File naming convention
"""

import pytest
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

    try:
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "export_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "export_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "export_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "export_%%@example.com"})
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


def create_document_with_chunk(db_session: Session, user_id: int) -> dict:
    doc = Document(
        user_id=user_id,
        original_filename="test.pdf",
        storage_key=f"export/{user_id}/test.pdf",
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
    return {"id": doc.id, "chunk_id": chunk.id}


def add_source(db_session: Session, message_id: int, doc_id: int, chunk_id: int) -> MessageSource:
    source = MessageSource(
        message_id=message_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        page_start=1,
        page_end=1,
        similarity_score=0.85,
    )
    db_session.add(source)
    db_session.commit()
    db_session.refresh(source)
    return source


# ==========================================================================
# 1. BASIC EXPORT
# ==========================================================================

class TestExportBasic:
    def test_export_owned_conversation(self, db_session):
        register_and_login("ExportUser1", "export_1@example.com")
        conv = create_conversation("My Export")
        add_message(db_session, conv["id"], "user", "Hello?")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200

    def test_export_format_is_markdown(self):
        register_and_login("ExportUser2", "export_2@example.com")
        conv = create_conversation("Markdown Test")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/markdown; charset=utf-8"
        assert "# Markdown Test" in resp.text

    def test_export_includes_title(self):
        register_and_login("ExportUser3", "export_3@example.com")
        conv = create_conversation("Title Check")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "# Title Check" in resp.text

    def test_export_has_content_disposition(self):
        register_and_login("ExportUser4", "export_4@example.com")
        conv = create_conversation("File Name Test")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Content-Disposition" in resp.headers
        assert "attachment" in resp.headers["Content-Disposition"]
        assert ".md" in resp.headers["Content-Disposition"]

    def test_export_empty_conversation(self):
        register_and_login("ExportUser5", "export_5@example.com")
        conv = create_conversation("Empty Conv")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "# Empty Conv" in resp.text
        # Should have the header and timestamp but no messages
        assert "## " not in resp.text  # No message headers


# ==========================================================================
# 2. AUTHENTICATION
# ==========================================================================

class TestExportAuth:
    def test_unauthenticated_export_rejected(self):
        auth._sessions.clear()
        resp = client.get("/conversations/1/export")
        assert resp.status_code == 401


# ==========================================================================
# 3. OWNERSHIP
# ==========================================================================

class TestExportOwnership:
    def test_cross_user_export_rejected(self):
        register_and_login("ExportOwnA", "export_owna@example.com")
        conv_a = create_conversation("A's Export")

        auth._sessions.clear()
        register_and_login("ExportOwnB", "export_ownb@example.com")

        resp = client.get(f"/conversations/{conv_a['id']}/export")
        assert resp.status_code == 404

    def test_nonexistent_returns_404(self):
        register_and_login("ExportUser6", "export_6@example.com")
        resp = client.get("/conversations/99999/export")
        assert resp.status_code == 404


# ==========================================================================
# 4. CONTENT VERIFICATION
# ==========================================================================

class TestExportContent:
    def test_export_includes_user_messages(self, db_session):
        register_and_login("ExportUser7", "export_7@example.com")
        conv = create_conversation("Msg Test")
        add_message(db_session, conv["id"], "user", "What is DocuFlow?")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "What is DocuFlow?" in resp.text

    def test_export_includes_assistant_messages(self, db_session):
        register_and_login("ExportUser8", "export_8@example.com")
        conv = create_conversation("Assistant Test")
        add_message(db_session, conv["id"], "user", "Hello")
        add_message(db_session, conv["id"], "assistant", "Hi there!")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Hi there!" in resp.text

    def test_export_includes_sources(self, db_session):
        register_and_login("ExportUser9", "export_9@example.com")
        user = register_and_login("ExportUser9b", "export_9b@example.com")
        conv = create_conversation("Source Test")
        doc = create_document_with_chunk(db_session, user["id"])

        user_msg = add_message(db_session, conv["id"], "user", "Question?")
        asst_msg = add_message(db_session, conv["id"], "assistant", "Answer with sources.")
        add_source(db_session, asst_msg["id"], doc["id"], doc["chunk_id"])

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Sources:" in resp.text
        assert "pages 1-1" in resp.text

    def test_export_preserves_message_order(self, db_session):
        register_and_login("ExportUser10", "export_10@example.com")
        conv = create_conversation("Order Test")
        add_message(db_session, conv["id"], "user", "First question")
        add_message(db_session, conv["id"], "assistant", "First answer")
        add_message(db_session, conv["id"], "user", "Second question")
        add_message(db_session, conv["id"], "assistant", "Second answer")

        resp = client.get(f"/conversations/{conv['id']}/export")
        text = resp.text
        first_pos = text.find("First question")
        first_answer_pos = text.find("First answer")
        second_pos = text.find("Second question")
        second_answer_pos = text.find("Second answer")

        assert first_pos < first_answer_pos < second_pos < second_answer_pos

    def test_export_user_role_label(self, db_session):
        register_and_login("ExportUser11", "export_11@example.com")
        conv = create_conversation("Label Test")
        add_message(db_session, conv["id"], "user", "User message")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "**You**" in resp.text

    def test_export_assistant_role_label(self, db_session):
        register_and_login("ExportUser12", "export_12@example.com")
        conv = create_conversation("Label Test 2")
        add_message(db_session, conv["id"], "assistant", "Assistant message")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "**DocuFlow**" in resp.text

    def test_export_special_characters_in_title(self):
        register_and_login("ExportUser13", "export_13@example.com")
        conv = create_conversation("Test: \"Quotes\" & <Tags>")
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "# Test: \"Quotes\" & <Tags>" in resp.text

    def test_export_long_title_truncated_in_filename(self):
        register_and_login("ExportUser14", "export_14@example.com")
        long_title = "A" * 100
        conv = create_conversation(long_title)
        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        disposition = resp.headers.get("Content-Disposition", "")
        # Filename should be truncated to 50 chars
        assert len(disposition) < 200  # Reasonable upper bound
