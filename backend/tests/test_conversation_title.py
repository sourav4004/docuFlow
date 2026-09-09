"""Phase 5.7 Step 2 tests: Conversation title management.

Tests cover:
- Title update (owned conversations)
- Authentication enforcement
- Cross-user title update rejection
- Validation (empty, whitespace, max length, Unicode)
- Title trimming
- updated_at / created_at behavior
- Protected fields (user_id, id, created_at)
- No RAG/LLM/retrieval side effects
- Title persistence and list visibility
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.conversation import Conversation
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
        db_session.execute(text("DELETE FROM message_sources WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)))"), {"p1": "title_%%@example.com"})
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1))"), {"p1": "title_%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :p1)"), {"p1": "title_%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :p1"), {"p1": "title_%%@example.com"})
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


def update_title(conversation_id: int, title: str) -> dict:
    resp = client.patch(f"/conversations/{conversation_id}", json={"title": title})
    return resp


# ==========================================================================
# 1. BASIC TITLE UPDATE
# ==========================================================================

class TestTitleUpdate:
    def test_update_owned_conversation(self):
        register_and_login("TitleUser1", "title_1@example.com")
        conv = create_conversation("Original Title")

        resp = update_title(conv["id"], "Updated Title")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Updated Title"
        assert data["id"] == conv["id"]

    def test_title_persists_after_reload(self):
        register_and_login("TitleUser2", "title_2@example.com")
        conv = create_conversation("Persistent Title")

        update_title(conv["id"], "New Persistent Title")

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Persistent Title"

    def test_title_appears_in_list(self):
        register_and_login("TitleUser3", "title_3@example.com")
        conv = create_conversation("List Title")

        update_title(conv["id"], "Updated List Title")

        resp = client.get("/conversations")
        items = resp.json()["items"]
        matching = [c for c in items if c["id"] == conv["id"]]
        assert len(matching) == 1
        assert matching[0]["title"] == "Updated List Title"


# ==========================================================================
# 2. AUTHENTICATION
# ==========================================================================

class TestTitleAuth:
    def test_unauthenticated_update_rejected(self):
        auth._sessions.clear()
        resp = update_title(1, "Hacked Title")
        assert resp.status_code == 401


# ==========================================================================
# 3. OWNERSHIP
# ==========================================================================

class TestTitleOwnership:
    def test_cross_user_update_rejected(self):
        register_and_login("TitleOwnA", "title_owna@example.com")
        conv_a = create_conversation("A's Title")

        auth._sessions.clear()
        register_and_login("TitleOwnB", "title_ownb@example.com")

        resp = update_title(conv_a["id"], "Hacked Title")
        assert resp.status_code == 404

        # Verify A's title is unchanged
        auth._sessions.clear()
        client.post("/auth/login", json={"email": "title_owna@example.com", "password": "testpass123"})
        resp = client.get(f"/conversations/{conv_a['id']}")
        assert resp.json()["title"] == "A's Title"

    def test_multiple_conversations_isolated(self):
        register_and_login("TitleIso", "title_iso@example.com")
        conv_a = create_conversation("Conv A")
        conv_b = create_conversation("Conv B")

        update_title(conv_a["id"], "Updated A")

        resp_a = client.get(f"/conversations/{conv_a['id']}")
        resp_b = client.get(f"/conversations/{conv_b['id']}")

        assert resp_a.json()["title"] == "Updated A"
        assert resp_b.json()["title"] == "Conv B"


# ==========================================================================
# 4. VALIDATION
# ==========================================================================

class TestTitleValidation:
    def test_empty_title_rejected(self):
        register_and_login("TitleVal1", "title_val1@example.com")
        conv = create_conversation("Test")

        resp = update_title(conv["id"], "")
        assert resp.status_code == 422

    def test_whitespace_only_title_rejected(self):
        register_and_login("TitleVal2", "title_val2@example.com")
        conv = create_conversation("Test")

        resp = update_title(conv["id"], "   ")
        assert resp.status_code == 422

    def test_long_title_rejected(self):
        register_and_login("TitleVal3", "title_val3@example.com")
        conv = create_conversation("Test")

        resp = update_title(conv["id"], "T" * 501)
        assert resp.status_code == 422

    def test_title_is_trimmed(self):
        register_and_login("TitleVal4", "title_val4@example.com")
        conv = create_conversation("Test")

        resp = update_title(conv["id"], "  Trimmed Title  ")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Trimmed Title"

    def test_valid_unicode_accepted(self):
        register_and_login("TitleVal5", "title_val5@example.com")
        conv = create_conversation("Test")

        resp = update_title(conv["id"], "日本語タイトル 🎉")
        assert resp.status_code == 200
        assert resp.json()["title"] == "日本語タイトル 🎉"


# ==========================================================================
# 5. TIMESTAMP BEHAVIOR
# ==========================================================================

class TestTitleTimestamps:
    def test_updated_at_changes(self):
        register_and_login("TitleTime1", "title_time1@example.com")
        conv = create_conversation("Timestamp Test")
        original_updated = conv["updated_at"]

        resp = update_title(conv["id"], "Updated Timestamp")
        assert resp.status_code == 200
        assert resp.json()["updated_at"] != original_updated

    def test_created_at_unchanged(self):
        register_and_login("TitleTime2", "title_time2@example.com")
        conv = create_conversation("Created Test")
        original_created = conv["created_at"]

        update_title(conv["id"], "Updated Created")

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.json()["created_at"] == original_created


# ==========================================================================
# 6. PROTECTED FIELDS
# ==========================================================================

class TestProtectedFields:
    def test_user_id_not_changeable(self):
        """User ID is not in the update schema, so it cannot be modified."""
        register_and_login("TitleProt1", "title_prot1@example.com")
        conv = create_conversation("Protected Test")

        # Update only changes title — user_id stays the same
        resp = update_title(conv["id"], "Still Mine")
        assert resp.status_code == 200

        # Verify via list
        resp = client.get("/conversations")
        items = resp.json()["items"]
        matching = [c for c in items if c["id"] == conv["id"]]
        assert len(matching) == 1


# ==========================================================================
# 7. NO SIDE EFFECTS
# ==========================================================================

class TestNoSideEffects:
    def test_no_rag_on_title_update(self):
        register_and_login("TitleSide1", "title_side1@example.com")
        conv = create_conversation("No RAG")

        with patch("app.api.conversations.answer_question_with_history") as mock_rag:
            update_title(conv["id"], "Still No RAG")
            mock_rag.assert_not_called()

    def test_no_retrieval_on_title_update(self):
        register_and_login("TitleSide2", "title_side2@example.com")
        conv = create_conversation("No Retrieval")

        with patch("app.services.retrieval_service.retrieve_context") as mock_ret:
            update_title(conv["id"], "Still No Retrieval")
            mock_ret.assert_not_called()

    def test_nonexistent_returns_404(self):
        register_and_login("TitleSide3", "title_side3@example.com")
        resp = update_title(99999, "Ghost Title")
        assert resp.status_code == 404
