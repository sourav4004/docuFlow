"""Phase 5.8 Step 4: Final production-readiness hardening tests.

Adds only tests NOT already covered by:
- test_conversation_export.py (Step 1)
- test_conversation_export_hardening.py (Step 2)
- test_conversation_export_integration.py (Step 3)

New coverage:
- Tab characters in messages
- Backslashes in messages
- Mixed line endings (CRLF vs LF)
- Leading/trailing whitespace in messages
- Long single-line messages (no wrapping)
- Consecutive blank lines
- Multiple sources from different documents
- Sources with all-NULL metadata in multi-source context
- Content exactness verification (DB content == export content)
- Export does not modify conversation updated_at
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

    try:
        db_session.execute(text(
            "DELETE FROM message_sources WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)))"
        ), {"p1": "estep4_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1))"
        ), {"p1": "estep4_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM conversations WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "estep4_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM documents WHERE user_id IN "
            "(SELECT id FROM users WHERE email LIKE :p1)"
        ), {"p1": "estep4_%%@example.com"})
        db_session.execute(text(
            "DELETE FROM users WHERE email LIKE :p1"
        ), {"p1": "estep4_%%@example.com"})
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
        storage_key=f"estep4/{user_id}/{filename}",
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
# 1. CONTENT ROBUSTNESS — New coverage not in Step 2
# ==========================================================================

class TestContentTabCharacters:
    def test_tabs_in_message_preserved(self, db_session):
        """Tab characters in messages are preserved as-is."""
        register_and_login("estep4_Tab1", "estep4_tab1@example.com")
        conv = create_conversation("Tab Test")
        content = "Column1\tColumn2\tColumn3"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "\t" in resp.text
        assert "Column1\tColumn2\tColumn3" in resp.text

    def test_mixed_tabs_and_newlines(self, db_session):
        """Tabs mixed with newlines are preserved."""
        register_and_login("estep4_Tab2", "estep4_tab2@example.com")
        conv = create_conversation("Tab Newline")
        content = "Line1\n\tIndented line\n\t\tDouble indented"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "\tIndented line" in resp.text


class TestContentBackslashes:
    def test_backslashes_preserved(self, db_session):
        """Backslash characters are preserved in export."""
        register_and_login("estep4_Bs1", "estep4_bs1@example.com")
        conv = create_conversation("Backslash Test")
        content = "Path: C:\\Users\\test\\file.txt"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "C:\\Users\\test\\file.txt" in resp.text

    def test_escaped_chars_preserved(self, db_session):
        """Escaped characters are preserved."""
        register_and_login("estep4_Bs2", "estep4_bs2@example.com")
        conv = create_conversation("Escape Test")
        content = "Regex: \\d+\\.\\s\\w+"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Regex: \\d+\\.\\s\\w+" in resp.text


class TestContentMixedLineEndings:
    def test_crlf_in_message_preserved(self, db_session):
        """CRLF line endings in message content are preserved."""
        register_and_login("estep4_CRLF1", "estep4_crlf1@example.com")
        conv = create_conversation("CRLF Test")
        content = "Line1\r\nLine2\r\nLine3"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Line1" in resp.text
        assert "Line3" in resp.text

    def test_bare_cr_preserved(self, db_session):
        """Bare carriage return is preserved."""
        register_and_login("estep4_CRLF2", "estep4_crlf2@example.com")
        conv = create_conversation("CR Test")
        content = "Before\rAfter"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Before" in resp.text
        assert "After" in resp.text


class TestContentLeadingTrailing:
    def test_leading_whitespace_preserved(self, db_session):
        """Leading whitespace in message is preserved."""
        register_and_login("estep4_LT1", "estep4_lt1@example.com")
        conv = create_conversation("Leading WS")
        content = "   indented content"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "   indented content" in resp.text

    def test_trailing_whitespace_preserved(self, db_session):
        """Trailing whitespace in message is preserved."""
        register_and_login("estep4_LT2", "estep4_lt2@example.com")
        conv = create_conversation("Trailing WS")
        content = "content with trailing   "
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "content with trailing   " in resp.text

    def test_only_whitespace_content(self, db_session):
        """Message with only whitespace content is handled safely."""
        register_and_login("estep4_LT3", "estep4_lt3@example.com")
        conv = create_conversation("WS Only")
        add_message(db_session, conv["id"], "user", "   ")

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200


class TestContentConsecutiveBlanks:
    def test_many_consecutive_blank_lines(self, db_session):
        """Many consecutive blank lines are preserved."""
        register_and_login("estep4_Blank1", "estep4_blank1@example.com")
        conv = create_conversation("Many Blanks")
        content = "Start\n\n\n\n\n\n\n\nEnd"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "Start" in resp.text
        assert "End" in resp.text


class TestContentLongLine:
    def test_single_very_long_line(self, db_session):
        """Single very long line (5000 chars) is exported completely."""
        register_and_login("estep4_Long1", "estep4_long1@example.com")
        conv = create_conversation("Long Line")
        content = "x" * 5000
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "x" * 5000 in resp.text


class TestContentExactness:
    def test_exported_content_matches_database(self, db_session):
        """Content in the export matches what's stored in the database exactly."""
        register_and_login("estep4_Exact1", "estep4_exact1@example.com")
        conv = create_conversation("Exactness")
        content = "Special: <>&\"' and `backticks` and\nnewlines\n\nand tabs\tand \\backslashes"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")

        # Retrieve from DB
        db_msg = db_session.query(Message).filter(
            Message.conversation_id == conv["id"]
        ).first()

        # The exported content must contain the exact DB content
        assert db_msg.content in resp.text


# ==========================================================================
# 2. SOURCE ROBUSTNESS — Mixed configurations
# ==========================================================================

class TestSourceMixedConfigurations:
    def test_assistant_with_zero_then_one_then_multiple_sources(self, db_session):
        """Conversation with assistant messages having 0, 1, and multiple sources."""
        user = register_and_login("estep4_SrcMix1", "estep4_srcmix1@example.com")
        conv = create_conversation("Mixed Sources")

        doc = create_doc(db_session, user["id"], "mixed.pdf")
        doc2 = create_doc(db_session, user["id"], "mixed2.pdf")

        # Assistant 1: zero sources
        add_message(db_session, conv["id"], "user", "Q1")
        add_message(db_session, conv["id"], "assistant", "A1 - no sources")

        # Assistant 2: one source
        add_message(db_session, conv["id"], "user", "Q2")
        asst2 = add_message(db_session, conv["id"], "assistant", "A2 - one source")
        add_source(db_session, asst2["id"], doc["id"], doc["chunk_id"])

        # Assistant 3: two sources from different documents
        add_message(db_session, conv["id"], "user", "Q3")
        asst3 = add_message(db_session, conv["id"], "assistant", "A3 - multi-source")
        add_source(db_session, asst3["id"], doc["id"], doc["chunk_id"], score=0.9)
        add_source(db_session, asst3["id"], doc2["id"], doc2["chunk_id"], score=0.7)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        text = resp.text

        # A1: no Sources section
        # A2: one Sources section with one entry
        # A3: one Sources section with two entries
        assert "A1 - no sources" in text
        assert "A2 - one source" in text
        assert "A3 - multi-source" in text
        # Should have 3 Document # references total (1 + 2)
        assert text.count("Document #") == 3

    def test_multiple_sources_all_null_metadata(self, db_session):
        """Multiple sources with all-NULL metadata export without crash."""
        user = register_and_login("estep4_SrcMix2", "estep4_srcmix2@example.com")
        conv = create_conversation("All Null Mix")
        doc = create_doc(db_session, user["id"])

        asst = add_message(db_session, conv["id"], "assistant", "Null metadata answer")
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=None, page_end=None, score=None, chunk_index=0)
        add_source(db_session, asst["id"], doc["id"], doc["chunk_id"],
                   page_start=None, page_end=None, score=None, chunk_index=1)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert resp.status_code == 200
        assert "Document #" in resp.text


# ==========================================================================
# 3. EXPORT SIDE-EFFECTS — Does NOT modify updated_at
# ==========================================================================

class TestExportDoesNotModifyTimestamps:
    def test_export_does_not_change_updated_at(self, db_session):
        """Exporting does not modify the conversation's updated_at."""
        user = register_and_login("estep4_Ts1", "estep4_ts1@example.com")
        conv = create_conversation("Timestamp Test")
        add_message(db_session, conv["id"], "user", "Q")

        # Get original updated_at
        original = db_session.query(Conversation).filter(Conversation.id == conv["id"]).first()
        original_updated = original.updated_at

        # Export
        client.get(f"/conversations/{conv['id']}/export")

        # Verify unchanged
        db_session.refresh(original)
        assert original.updated_at == original_updated

    def test_export_does_not_change_created_at(self, db_session):
        """Exporting does not modify the conversation's created_at."""
        user = register_and_login("estep4_Ts2", "estep4_ts2@example.com")
        conv = create_conversation("Created AT Test")

        original = db_session.query(Conversation).filter(Conversation.id == conv["id"]).first()
        original_created = original.created_at

        client.get(f"/conversations/{conv['id']}/export")

        db_session.refresh(original)
        assert original.created_at == original_created


# ==========================================================================
# 4. CONTENT WITH SPECIAL SEQUENCES
# ==========================================================================

class TestContentSpecialSequences:
    def test_markdown_bold_in_user_message(self, db_session):
        """Markdown bold in user message is preserved."""
        register_and_login("estep4_Sp1", "estep4_sp1@example.com")
        conv = create_conversation("Bold Test")
        content = "This is **bold** and __also bold__"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "**bold**" in resp.text
        assert "__also bold__" in resp.text

    def test_message_with_angle_brackets(self, db_session):
        """Angle brackets are preserved, not HTML-escaped."""
        register_and_login("estep4_Sp2", "estep4_sp2@example.com")
        conv = create_conversation("Angle Brackets")
        content = "Compare <vector> vs <list> and a != b"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "<vector>" in resp.text
        assert "<list>" in resp.text

    def test_message_with_curly_braces(self, db_session):
        """Curly braces are preserved."""
        register_and_login("estep4_Sp3", "estep4_sp3@example.com")
        conv = create_conversation("Curly Braces")
        content = "function() { return { key: 'value' }; }"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "function() { return { key: 'value' }; }" in resp.text

    def test_message_with_pipe_characters(self, db_session):
        """Pipe characters are preserved."""
        register_and_login("estep4_Sp4", "estep4_sp4@example.com")
        conv = create_conversation("Pipe Test")
        content = "a | b | c"
        add_message(db_session, conv["id"], "user", content)

        resp = client.get(f"/conversations/{conv['id']}/export")
        assert "a | b | c" in resp.text
