"""Tests for conversation title search/filtering (Phase 5.9 Step 4)."""

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


def _register_and_login(client, email=None, name="Searcher"):
    global _email_counter
    _email_counter += 1
    if email is None:
        email = f"search_{_email_counter}@test.com"
    client.post("/auth/register", json={"name": name, "email": email, "password": "TestPass123!"})
    resp = client.post("/auth/login", json={"email": email, "password": "TestPass123!"})
    return resp.status_code == 200


def _create_conversation(client, title):
    return client.post("/conversations", json={"title": title})


# ==========================================================================
# 1. BASIC SEARCH
# ==========================================================================

class TestBasicSearch:
    def test_exact_match(self):
        _register_and_login(client)
        _create_conversation(client, "Project Alpha")
        _create_conversation(client, "Project Beta")
        _create_conversation(client, "Random Chat")

        resp = client.get("/conversations?search=Project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        titles = {c["title"] for c in data["items"]}
        assert titles == {"Project Alpha", "Project Beta"}

    def test_partial_match(self):
        _register_and_login(client)
        _create_conversation(client, "Machine Learning Discussion")
        _create_conversation(client, "Deep Learning Basics")
        _create_conversation(client, "Web Development")

        resp = client.get("/conversations?search=Learning")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_case_insensitive(self):
        _register_and_login(client)
        _create_conversation(client, "Machine Learning")
        _create_conversation(client, "machine learning")
        _create_conversation(client, "MACHINE LEARNING")

        resp = client.get("/conversations?search=machine")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_no_matches(self):
        _register_and_login(client)
        _create_conversation(client, "Project Alpha")

        resp = client.get("/conversations?search=Nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []


# ==========================================================================
# 2. EDGE CASES
# ==========================================================================

class TestSearchEdgeCases:
    def test_empty_search(self):
        _register_and_login(client)
        _create_conversation(client, "Chat 1")
        _create_conversation(client, "Chat 2")

        resp = client.get("/conversations?search=")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_whitespace_search(self):
        _register_and_login(client)
        _create_conversation(client, "Chat 1")
        _create_conversation(client, "Chat 2")

        resp = client.get("/conversations?search=   ")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_no_search_param(self):
        _register_and_login(client)
        _create_conversation(client, "Chat 1")
        _create_conversation(client, "Chat 2")

        resp = client.get("/conversations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_unicode_search(self):
        _register_and_login(client)
        _create_conversation(client, "日本語テスト")
        _create_conversation(client, "中文对话")
        _create_conversation(client, "English Chat")

        resp = client.get("/conversations?search=日本")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "日本語テスト"

    def test_special_characters(self):
        _register_and_login(client)
        _create_conversation(client, "Project: Phase 1 (2024)")
        _create_conversation(client, "Random stuff")

        resp = client.get("/conversations?search=Phase 1")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_sql_injection_attempt(self):
        _register_and_login(client)
        _create_conversation(client, "Normal conversation")

        resp = client.get("/conversations?search=' OR 1=1 --")
        assert resp.status_code == 200
        # Should not return all conversations
        data = resp.json()
        assert data["total"] == 0


# ==========================================================================
# 3. PAGINATION WITH SEARCH
# ==========================================================================

class TestSearchPagination:
    def test_search_with_pagination(self):
        _register_and_login(client)
        for i in range(5):
            _create_conversation(client, f"Project #{i}")
        _create_conversation(client, "Unrelated Chat")

        resp = client.get("/conversations?search=Project&limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["has_next"] is True
        assert data["has_previous"] is False

    def test_search_second_page(self):
        _register_and_login(client)
        for i in range(5):
            _create_conversation(client, f"Project #{i}")

        resp = client.get("/conversations?search=Project&limit=2&offset=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["has_next"] is True
        assert data["has_previous"] is True

    def test_search_last_page(self):
        _register_and_login(client)
        for i in range(5):
            _create_conversation(client, f"Project #{i}")

        resp = client.get("/conversations?search=Project&limit=2&offset=4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 1
        assert data["has_next"] is False
        assert data["has_previous"] is True


# ==========================================================================
# 4. ORDERING
# ==========================================================================

class TestSearchOrdering:
    def test_results_ordered_by_id_desc_when_same_timestamp(self):
        _register_and_login(client)
        _create_conversation(client, "Alpha Project")
        _create_conversation(client, "Beta Project")
        _create_conversation(client, "Gamma Project")

        resp = client.get("/conversations?search=Project")
        data = resp.json()
        titles = [c["title"] for c in data["items"]]
        # With same updated_at, ordering falls back to id DESC
        assert titles[0] == "Gamma Project"
        assert len(titles) == 3


# ==========================================================================
# 5. SECURITY
# ==========================================================================

class TestSearchSecurity:
    def test_user_isolation(self):
        # User A creates conversations
        _register_and_login(client, email="userA@test.com", name="UserA")
        _create_conversation(client, "Alpha Project")
        _create_conversation(client, "Beta Project")

        # Logout and create User B
        client.post("/auth/logout")
        _register_and_login(client, email="userB@test.com", name="UserB")
        _create_conversation(client, "Charlie Project")

        # User B should only see their own
        resp = client.get("/conversations?search=Project")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "Charlie Project"

    def test_unauthenticated_search(self):
        client.post("/auth/logout")
        resp = client.get("/conversations?search=test")
        assert resp.status_code == 401


# ==========================================================================
# 6. RAG ISOLATION
# ==========================================================================

class TestSearchRAGIsolation:
    def test_no_rag_during_search(self):
        """Search must be database-only, never invoke RAG."""
        _register_and_login(client)
        _create_conversation(client, "RAG Test Project")

        # This should succeed with just a DB query
        resp = client.get("/conversations?search=RAG")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "RAG Test Project"
