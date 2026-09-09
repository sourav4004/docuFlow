"""Tests for message search within conversations (Phase 5.9 Step 5)."""

import pytest
from tests.test_auth import client, TestingSessionLocal
from app.core import auth
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message


@pytest.fixture(autouse=True)
def setup_and_teardown():
    auth._sessions.clear()
    yield
    auth._sessions.clear()


_email_counter = 0


def _register_and_login(email=None, name="Searcher"):
    global _email_counter
    _email_counter += 1
    if email is None:
        email = f"msgsearch_{_email_counter}@test.com"
    client.post("/auth/register", json={"name": name, "email": email, "password": "TestPass123!"})
    return client.post("/auth/login", json={"email": email, "password": "TestPass123!"}).status_code == 200


def _create_conversation(title):
    return client.post("/conversations", json={"title": title}).json()["id"]


def _send_message(conv_id, content, role="user"):
    """Insert a message directly for testing (bypasses RAG)."""
    db = TestingSessionLocal()
    try:
        msg = Message(conversation_id=conv_id, role=role, content=content)
        db.add(msg)
        db.commit()
    finally:
        db.close()


# ==========================================================================
# 1. BASIC SEARCH
# ==========================================================================

class TestBasicSearch:
    def test_exact_match(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "What is machine learning?")
        _send_message(conv_id, "Machine learning is a subset of AI")
        _send_message(conv_id, "The weather is nice today")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=machine")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_partial_match(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "Hello world")
        _send_message(conv_id, "Hello there")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=Hello")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_case_insensitive(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "Machine Learning")
        _send_message(conv_id, "machine learning")
        _send_message(conv_id, "MACHINE LEARNING")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=machine")
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_no_matches(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "Hello world")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["messages"] == []

    def test_search_both_user_and_assistant(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "What is Python?", role="user")
        _send_message(conv_id, "Python is a programming language", role="assistant")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=Python")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2


# ==========================================================================
# 2. EDGE CASES
# ==========================================================================

class TestSearchEdgeCases:
    def test_empty_query_rejected(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        resp = client.get(f"/conversations/{conv_id}/messages/search?q=")
        assert resp.status_code == 422  # FastAPI validation

    def test_whitespace_only_query_rejected(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        resp = client.get(f"/conversations/{conv_id}/messages/search?q=%20%20%20")
        assert resp.status_code == 400

    def test_unicode_search(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "日本語のメッセージです")
        _send_message(conv_id, "Hello world")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=日本")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_special_characters(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "Email: test@example.com")
        _send_message(conv_id, "URL: https://example.com/path?q=1")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=test@example.com")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_sql_injection_attempt(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        _send_message(conv_id, "Normal message")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=' OR 1=1 --")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ==========================================================================
# 3. PAGINATION
# ==========================================================================

class TestSearchPagination:
    def test_pagination(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        for i in range(10):
            _send_message(conv_id, f"Message {i} with keyword")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=keyword&page=1&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["messages"]) == 3
        assert data["has_next"] is True
        assert data["has_previous"] is False

    def test_last_page(self):
        _register_and_login()
        conv_id = _create_conversation("Test Conv")
        for i in range(10):
            _send_message(conv_id, f"Message {i} with keyword")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=keyword&page=4&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["has_next"] is False
        assert data["has_previous"] is True


# ==========================================================================
# 4. SECURITY
# ==========================================================================

class TestSearchSecurity:
    def test_cross_user_search(self):
        _register_and_login(email="userA@example.com")
        conv_resp = client.post("/conversations", json={"title": "Secret Conv"})
        conv_id = conv_resp.json()["id"]
        _send_message(conv_id, "Confidential message")

        # Logout userA
        client.post("/auth/logout")
        auth._sessions.clear()

        # Register and login as userB
        _register_and_login(email="userB@example.com")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=Confidential")
        assert resp.status_code == 404

    def test_nonexistent_conversation(self):
        _register_and_login()
        resp = client.get("/conversations/99999/messages/search?q=test")
        assert resp.status_code == 404

    def test_unauthenticated_search(self):
        client.post("/auth/logout")
        resp = client.get("/conversations/1/messages/search?q=test")
        assert resp.status_code == 401


# ==========================================================================
# 5. RAG ISOLATION
# ==========================================================================

class TestSearchRAGIsolation:
    def test_no_rag_during_search(self):
        _register_and_login()
        conv_id = _create_conversation("RAG Test")
        _send_message(conv_id, "What is AI?")

        resp = client.get(f"/conversations/{conv_id}/messages/search?q=AI")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
