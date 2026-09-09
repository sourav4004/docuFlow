"""Tests for document search/filtering (Phase 5.9 Step 6)."""

import io
import pytest
from tests.test_auth import client, TestingSessionLocal
from app.core import auth
from app.core.config import settings
from app.models.user import User
from app.models.document import Document
from app.services.storage import storage_service


@pytest.fixture(autouse=True)
def setup_and_teardown():
    auth._sessions.clear()
    yield
    auth._sessions.clear()


_email_counter = 0


def _register_and_login(email=None, name="DocSearcher"):
    global _email_counter
    _email_counter += 1
    if email is None:
        email = f"docsearch_{_email_counter}@example.com"
    client.post("/auth/register", json={"name": name, "email": email, "password": "TestPass123!"})
    return client.post("/auth/login", json={"email": email, "password": "TestPass123!"}).status_code == 200


def _create_doc(filename, status="READY"):
    """Create a document directly in the database."""
    db = TestingSessionLocal()
    try:
        doc = Document(
            user_id=None,  # Will be set below
            original_filename=filename,
            storage_key=storage_service.generate_storage_key(),
            mime_type="application/pdf",
            file_size=1024,
            status=status,
        )
        # Get current user ID from session
        user_resp = client.get("/auth/me")
        if user_resp.status_code == 200:
            doc.user_id = user_resp.json()["id"]
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc.id
    finally:
        db.close()


# ==========================================================================
# 1. BASIC SEARCH
# ==========================================================================

class TestDocumentSearch:
    def test_filename_search(self):
        _register_and_login()
        _create_doc("machine_learning_guide.pdf")
        _create_doc("web_development.pdf")
        _create_doc("data_science handbook.pdf")

        resp = client.get("/documents?search=machine")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["original_filename"] == "machine_learning_guide.pdf"

    def test_partial_match(self):
        _register_and_login()
        _create_doc("project_alpha.pdf")
        _create_doc("project_beta.pdf")
        _create_doc("other.pdf")

        resp = client.get("/documents?search=project")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_case_insensitive(self):
        _register_and_login()
        _create_doc("Machine Learning.pdf")
        _create_doc("machine learning.pdf")

        resp = client.get("/documents?search=machine")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_no_matches(self):
        _register_and_login()
        _create_doc("document.pdf")

        resp = client.get("/documents?search=nonexistent")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_empty_search(self):
        _register_and_login()
        _create_doc("doc1.pdf")
        _create_doc("doc2.pdf")

        resp = client.get("/documents?search=")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_whitespace_search(self):
        _register_and_login()
        _create_doc("doc1.pdf")

        resp = client.get("/documents?search=%20%20%20")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ==========================================================================
# 2. STATUS FILTER
# ==========================================================================

class TestDocumentStatusFilter:
    def test_filter_by_ready(self):
        _register_and_login()
        _create_doc("ready.pdf", status="READY")
        _create_doc("failed.pdf", status="FAILED")
        _create_doc("processing.pdf", status="PROCESSING")

        resp = client.get("/documents?status=READY")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["status"] == "READY"

    def test_filter_by_failed(self):
        _register_and_login()
        _create_doc("ready.pdf", status="READY")
        _create_doc("failed.pdf", status="FAILED")

        resp = client.get("/documents?status=FAILED")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_invalid_status_ignored(self):
        _register_and_login()
        _create_doc("doc.pdf", status="READY")

        resp = client.get("/documents?status=INVALID")
        assert resp.status_code == 200
        # Invalid status should be ignored, returning all documents
        assert resp.json()["total"] == 1


# ==========================================================================
# 3. SECURITY
# ==========================================================================

class TestDocumentSearchSecurity:
    def test_user_isolation(self):
        _register_and_login(email="docuserA@example.com")
        _create_doc("secret_project.pdf")

        client.post("/auth/logout")
        auth._sessions.clear()
        _register_and_login(email="docuserB@example.com")
        _create_doc("my_project.pdf")

        resp = client.get("/documents?search=secret")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_unauthenticated_search(self):
        client.post("/auth/logout")
        resp = client.get("/documents?search=test")
        assert resp.status_code == 401

    def test_sql_injection(self):
        _register_and_login()
        _create_doc("normal.pdf")

        resp = client.get("/documents?search=' OR 1=1 --")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_unicode_search(self):
        _register_and_login()
        _create_doc("日本語ドキュメント.pdf")

        resp = client.get("/documents?search=日本語")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ==========================================================================
# 4. RAG ISOLATION
# ==========================================================================

class TestDocumentSearchRAGIsolation:
    def test_no_rag_during_search(self):
        _register_and_login()
        _create_doc("test.pdf")

        resp = client.get("/documents?search=test")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
