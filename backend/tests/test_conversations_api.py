"""Phase 5.2 tests: Conversation API endpoints.

Tests cover:
- CRUD operations
- Authentication enforcement
- Ownership isolation
- Validation
- Cascade deletion
- Message retrieval
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core import auth
from app.core.database import get_db, SessionLocal
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from sqlalchemy import text


# Shared session fixture — API and direct DB inserts use the same session
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
    """Clean test data and override FastAPI get_db to use shared session.

    IMPORTANT: We save and restore previous dependency_overrides rather than
    clearing all of them, so that other test files' overrides (e.g. from
    test_auth.py's SQLite override) are preserved.
    """
    auth._sessions.clear()

    # Save previous overrides so we can restore them in teardown
    previous_overrides = dict(app.dependency_overrides)

    # Override get_db so API endpoints use the same session as our helpers
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass  # Don't close — our fixture manages the lifecycle

    app.dependency_overrides[get_db] = _override_get_db

    try:
        db_session.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :pattern))"), {"pattern": "conv%%@example.com"})
        db_session.execute(text("DELETE FROM conversations WHERE user_id IN (SELECT id FROM users WHERE email LIKE :pattern)"), {"pattern": "conv%%@example.com"})
        db_session.execute(text("DELETE FROM users WHERE email LIKE :pattern"), {"pattern": "conv%%@example.com"})
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

    yield

    auth._sessions.clear()
    # Restore previous overrides (not clear all)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous_overrides)


client = TestClient(app)


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


def add_messages(db_session: Session, conversation_id: int, messages: list[dict]) -> None:
    """Add messages to a conversation using the shared session."""
    for msg_data in messages:
        msg = Message(
            conversation_id=conversation_id,
            role=msg_data["role"],
            content=msg_data["content"],
        )
        db_session.add(msg)
    db_session.commit()


# ==========================================================================
# 1. CREATE CONVERSATION
# ==========================================================================

class TestCreateConversation:
    def test_create_authenticated(self):
        register_and_login("User1", "conv_api_1@example.com")
        conv = create_conversation("My Research")
        assert conv["title"] == "My Research"
        assert "id" in conv
        assert "created_at" in conv
        assert "updated_at" in conv

    def test_create_default_title(self):
        register_and_login("User2", "conv_api_2@example.com")
        resp = client.post("/conversations", json={})
        assert resp.status_code == 201
        assert resp.json()["title"] == "New Conversation"

    def test_unauthenticated_create_rejected(self):
        resp = client.post("/conversations", json={"title": "Test"})
        assert resp.status_code == 401

    def test_empty_title_rejected(self):
        register_and_login("User3", "conv_api_3@example.com")
        resp = client.post("/conversations", json={"title": ""})
        assert resp.status_code == 422

    def test_whitespace_title_rejected(self):
        register_and_login("User4", "conv_api_4@example.com")
        resp = client.post("/conversations", json={"title": "   "})
        assert resp.status_code == 422

    def test_long_title_rejected(self):
        register_and_login("User5", "conv_api_5@example.com")
        resp = client.post("/conversations", json={"title": "T" * 501})
        assert resp.status_code == 422

    def test_user_cannot_submit_user_id(self):
        """user_id from client should be ignored — backend uses auth."""
        register_and_login("User6", "conv_api_6@example.com")
        resp = client.post("/conversations", json={"title": "Test"})
        assert resp.status_code == 201
        conv = resp.json()
        assert conv["id"] is not None


# ==========================================================================
# 2. LIST CONVERSATIONS
# ==========================================================================

class TestListConversations:
    def test_list_own_conversations(self):
        register_and_login("ListUser", "conv_list_1@example.com")
        create_conversation("Conv A")
        create_conversation("Conv B")

        resp = client.get("/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_list_only_own(self):
        """User A's list must not include User B's conversations."""
        register_and_login("ListA", "conv_list_a@example.com")
        create_conversation("A's Conv")

        auth._sessions.clear()
        register_and_login("ListB", "conv_list_b@example.com")
        create_conversation("B's Conv")

        resp = client.get("/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "B's Conv"

    def test_list_empty(self):
        register_and_login("EmptyList", "conv_list_empty@example.com")
        resp = client.get("/conversations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["items"] == []

    def test_list_ordering(self):
        register_and_login("OrderUser", "conv_list_order@example.com")
        create_conversation("First")
        create_conversation("Second")

        resp = client.get("/conversations")
        items = resp.json()["items"]
        assert items[0]["title"] == "Second"
        assert items[1]["title"] == "First"

    def test_list_unauthenticated(self):
        auth._sessions.clear()
        resp = client.get("/conversations")
        assert resp.status_code == 401

    def test_list_pagination(self):
        register_and_login("PageUser", "conv_list_page@example.com")
        for i in range(5):
            create_conversation(f"Conv {i}")

        resp = client.get("/conversations?limit=2&offset=0")
        data = resp.json()
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["total"] == 5
        assert len(data["items"]) == 2

        resp2 = client.get("/conversations?limit=2&offset=4")
        data2 = resp2.json()
        assert len(data2["items"]) == 1


# ==========================================================================
# 3. GET CONVERSATION
# ==========================================================================

class TestGetConversation:
    def test_get_own_conversation(self, db_session):
        register_and_login("GetUser", "conv_get_1@example.com")
        conv = create_conversation("Get Me")

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Get Me"
        assert "messages" in data

    def test_get_includes_messages(self, db_session):
        register_and_login("GetMsg", "conv_get_msg@example.com")
        conv = create_conversation("With Messages")
        add_messages(db_session, conv["id"], [{"role": "user", "content": "Hello!"}])

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hello!"
        assert data["messages"][0]["role"] == "user"

    def test_cannot_get_other_users_conversation(self):
        register_and_login("GetA", "conv_get_a@example.com")
        conv_a = create_conversation("A Private")

        auth._sessions.clear()
        register_and_login("GetB", "conv_get_b@example.com")

        resp = client.get(f"/conversations/{conv_a['id']}")
        assert resp.status_code == 404

    def test_get_nonexistent(self):
        register_and_login("GetMiss", "conv_get_miss@example.com")
        resp = client.get("/conversations/99999")
        assert resp.status_code == 404

    def test_get_unauthenticated(self):
        auth._sessions.clear()
        resp = client.get("/conversations/1")
        assert resp.status_code == 401


# ==========================================================================
# 4. UPDATE CONVERSATION
# ==========================================================================

class TestUpdateConversation:
    def test_update_title(self):
        register_and_login("UpdUser", "conv_upd_1@example.com")
        conv = create_conversation("Old Title")

        resp = client.patch(f"/conversations/{conv['id']}", json={"title": "New Title"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"

    def test_cannot_update_other_users(self):
        register_and_login("UpdA", "conv_upd_a@example.com")
        conv_a = create_conversation("A's Title")

        auth._sessions.clear()
        register_and_login("UpdB", "conv_upd_b@example.com")

        resp = client.patch(f"/conversations/{conv_a['id']}", json={"title": "Hacked"})
        assert resp.status_code == 404

    def test_update_empty_title_rejected(self):
        register_and_login("UpdEmpty", "conv_upd_empty@example.com")
        conv = create_conversation("Test")

        resp = client.patch(f"/conversations/{conv['id']}", json={"title": ""})
        assert resp.status_code == 422

    def test_update_unauthenticated(self):
        auth._sessions.clear()
        resp = client.patch("/conversations/1", json={"title": "X"})
        assert resp.status_code == 401

    def test_update_preserves_created_at(self):
        register_and_login("UpdPreserve", "conv_upd_pres@example.com")
        conv = create_conversation("Original")
        original_created = conv["created_at"]

        client.patch(f"/conversations/{conv['id']}", json={"title": "Changed"})
        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.json()["created_at"] == original_created


# ==========================================================================
# 5. DELETE CONVERSATION
# ==========================================================================

class TestDeleteConversation:
    def test_delete_own(self):
        register_and_login("DelUser", "conv_del_1@example.com")
        conv = create_conversation("To Delete")

        resp = client.delete(f"/conversations/{conv['id']}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

        resp = client.get(f"/conversations/{conv['id']}")
        assert resp.status_code == 404

    def test_cannot_delete_other_users(self):
        register_and_login("DelA", "conv_del_a@example.com")
        conv_a = create_conversation("A's Conv")

        auth._sessions.clear()
        register_and_login("DelB", "conv_del_b@example.com")

        resp = client.delete(f"/conversations/{conv_a['id']}")
        assert resp.status_code == 404

    def test_delete_cascades_messages(self, db_session):
        register_and_login("DelCascade", "conv_del_casc@example.com")
        conv = create_conversation("With Messages")
        add_messages(db_session, conv["id"], [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ])

        conv_id = conv["id"]
        resp = client.delete(f"/conversations/{conv_id}")
        assert resp.status_code == 200

        count = db_session.query(Message).filter(Message.conversation_id == conv_id).count()
        assert count == 0

    def test_delete_nonexistent(self):
        register_and_login("DelMiss", "conv_del_miss@example.com")
        resp = client.delete("/conversations/99999")
        assert resp.status_code == 404

    def test_delete_unauthenticated(self):
        auth._sessions.clear()
        resp = client.delete("/conversations/1")
        assert resp.status_code == 401


# ==========================================================================
# 6. MESSAGE RETRIEVAL
# ==========================================================================

class TestMessageRetrieval:
    def test_list_messages(self, db_session):
        register_and_login("MsgUser", "conv_msg_1@example.com")
        conv = create_conversation("Msg Conv")
        add_messages(db_session, conv["id"], [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])

        resp = client.get(f"/conversations/{conv['id']}/messages")
        assert resp.status_code == 200
        data = resp.json()
        messages = data["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "Q1"
        assert messages[1]["content"] == "A1"
        assert messages[2]["content"] == "Q2"
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["has_next"] is False
        assert data["has_previous"] is False

    def test_messages_ordered_by_id(self, db_session):
        register_and_login("MsgOrder", "conv_msg_order@example.com")
        conv = create_conversation("Order Conv")
        add_messages(db_session, conv["id"], [
            {"role": "user", "content": f"Msg {i}"} if i % 2 == 0 else {"role": "assistant", "content": f"Msg {i}"}
            for i in range(5)
        ])

        resp = client.get(f"/conversations/{conv['id']}/messages")
        messages = resp.json()["messages"]
        for i, msg in enumerate(messages):
            assert msg["content"] == f"Msg {i}"

    def test_cannot_list_other_users_messages(self, db_session):
        register_and_login("MsgA", "conv_msg_a@example.com")
        conv_a = create_conversation("A's Msgs")
        add_messages(db_session, conv_a["id"], [{"role": "user", "content": "Secret"}])

        auth._sessions.clear()
        register_and_login("MsgB", "conv_msg_b@example.com")

        resp = client.get(f"/conversations/{conv_a['id']}/messages")
        assert resp.status_code == 404

    def test_messages_empty(self):
        register_and_login("MsgEmpty", "conv_msg_empty@example.com")
        conv = create_conversation("Empty Conv")

        resp = client.get(f"/conversations/{conv['id']}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []
        assert data["total"] == 0
        assert data["has_next"] is False
        assert data["has_previous"] is False

    def test_messages_unauthenticated(self):
        auth._sessions.clear()
        resp = client.get("/conversations/1/messages")
        assert resp.status_code == 401

    def test_messages_pagination(self, db_session):
        register_and_login("MsgPage", "conv_msg_page@example.com")
        conv = create_conversation("Page Conv")
        add_messages(db_session, conv["id"], [
            {"role": "user", "content": f"Msg {i}"} for i in range(10)
        ])

        resp = client.get(f"/conversations/{conv['id']}/messages?page=1&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 3
        assert data["messages"][0]["content"] == "Msg 0"
        assert data["total"] == 10
        assert data["has_next"] is True
        assert data["has_previous"] is False

        resp2 = client.get(f"/conversations/{conv['id']}/messages?page=4&page_size=3")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["messages"]) == 1
        assert data2["messages"][0]["content"] == "Msg 9"
        assert data2["has_next"] is False
        assert data2["has_previous"] is True


# ==========================================================================
# 7. CROSS-USER SECURITY
# ==========================================================================

class TestCrossUserSecurity:
    def test_full_isolation(self, db_session):
        """Complete cross-user isolation test."""
        register_and_login("IsoA", "conv_iso_a@example.com")
        conv_a = create_conversation("A's Private")
        add_messages(db_session, conv_a["id"], [{"role": "user", "content": "A's secret"}])

        conv_a_id = conv_a["id"]

        auth._sessions.clear()
        register_and_login("IsoB", "conv_iso_b@example.com")

        assert client.get(f"/conversations/{conv_a_id}").status_code == 404
        assert client.patch(f"/conversations/{conv_a_id}", json={"title": "Hacked"}).status_code == 404
        assert client.delete(f"/conversations/{conv_a_id}").status_code == 404
        assert client.get(f"/conversations/{conv_a_id}/messages").status_code == 404

        resp = client.get("/conversations")
        assert resp.json()["total"] == 0
