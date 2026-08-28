"""Tests for document upload, listing, retrieval, download, and deletion API endpoints."""

import io
from pathlib import Path
from unittest.mock import patch
import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User
from app.models.document import Document
from app.core import auth
from app.services.storage import storage_service
from tests.test_auth import client, TestingSessionLocal

# Minimal valid PDF binary header and structure
VALID_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF"
ANOTHER_VALID_PDF = b"%PDF-1.4\n%Another PDF payload for testing\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage directory and sessions before each test."""
    auth._sessions.clear()

    # Point storage service to an isolated temporary directory
    original_storage_dir = storage_service.base_dir
    test_storage_dir = tmp_path / "test_docs_storage"
    test_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_service.base_dir = test_storage_dir

    yield

    # Restore storage service base_dir
    storage_service.base_dir = original_storage_dir


def register_and_login(name: str = "Doc User", email: str = "docuser@example.com", password: str = "password123") -> dict:
    """Helper to register and login a user, returning the cookies."""
    client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password}
    )
    login_resp = client.post(
        "/auth/login",
        json={"email": email, "password": password}
    )
    return login_resp.cookies


def upload_sample_document(cookies: dict, filename: str = "sample.pdf", content: bytes = VALID_PDF_CONTENT) -> dict:
    """Helper to upload a document and return response data."""
    files = {"file": (filename, io.BytesIO(content), "application/pdf")}
    resp = client.post("/documents", files=files, cookies=cookies)
    assert resp.status_code == 201
    return resp.json()


# ==========================================================================
# 1. UPLOAD TESTS (Phase 2 Step 2 Regression)
# ==========================================================================

def test_upload_unauthenticated_fails_401():
    """Test that unauthenticated upload request is rejected with 401."""
    files = {
        "file": ("contract.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }
    response = client.post("/documents", files=files)
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_upload_authenticated_success():
    """Test successful document upload for authenticated user."""
    cookies = register_and_login(email="auth_success@example.com")
    files = {
        "file": ("invoice_2026.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["original_filename"] == "invoice_2026.pdf"
    assert data["mime_type"] == "application/pdf"
    assert data["file_size"] == len(VALID_PDF_CONTENT)
    assert data["status"] == "UPLOADED"
    assert "created_at" in data
    assert "updated_at" in data
    # Ensure sensitive/internal storage paths are not exposed
    assert "storage_key" not in data
    assert "storage_path" not in data


def test_upload_creates_database_record_and_file():
    """Test that document record exists in database and file exists on disk."""
    cookies = register_and_login(name="Alice", email="alice_verify@example.com")
    files = {
        "file": ("specs.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 201
    doc_id = response.json()["id"]

    # Verify database record
    db = TestingSessionLocal()
    try:
        user = db.query(User).filter(User.email == "alice_verify@example.com").first()
        assert user is not None

        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert doc.user_id == user.id
        assert doc.original_filename == "specs.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.file_size == len(VALID_PDF_CONTENT)
        assert doc.status == "UPLOADED"

        # Verify storage key is safe and not the original filename
        assert doc.storage_key != "specs.pdf"
        assert "specs" not in doc.storage_key

        # Verify physical file exists on disk and contents match
        assert storage_service.exists(doc.storage_key) is True
        saved_bytes = storage_service.retrieve(doc.storage_key)
        assert saved_bytes == VALID_PDF_CONTENT
    finally:
        db.close()


def test_upload_empty_file_rejected():
    """Test that an empty file (0 bytes) is rejected with 400."""
    cookies = register_and_login(email="empty_test@example.com")
    files = {
        "file": ("empty.pdf", io.BytesIO(b""), "application/pdf")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_non_pdf_type_rejected():
    """Test that non-PDF file types are rejected with 400."""
    cookies = register_and_login(email="txt_test@example.com")
    files = {
        "file": ("document.txt", io.BytesIO(b"Plain text content"), "text/plain")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 400
    assert "only pdf" in response.json()["detail"].lower()


def test_upload_fake_pdf_signature_rejected():
    """Test that file claiming to be PDF but without PDF magic bytes is rejected."""
    cookies = register_and_login(email="fake_pdf@example.com")
    fake_pdf_content = b"This is plain text pretending to be a PDF file."
    files = {
        "file": ("malicious.pdf", io.BytesIO(fake_pdf_content), "application/pdf")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower() or "invalid pdf" in response.json()["detail"].lower()


def test_upload_oversized_file_rejected(monkeypatch):
    """Test that files exceeding MAX_UPLOAD_SIZE are rejected with 400."""
    cookies = register_and_login(email="oversized@example.com")
    # Temporarily set max_upload_size to 100 bytes for testing
    monkeypatch.setattr(settings, "max_upload_size", 100)

    oversized_content = VALID_PDF_CONTENT + (b"A" * 200)
    files = {
        "file": ("big.pdf", io.BytesIO(oversized_content), "application/pdf")
    }
    response = client.post("/documents", files=files, cookies=cookies)
    assert response.status_code == 400
    assert "exceeds maximum allowed limit" in response.json()["detail"].lower()


def test_upload_ignores_client_provided_user_id():
    """Test that client-supplied user_id form/body parameter is ignored."""
    cookies = register_and_login(name="Actual User", email="actual_secure@example.com")
    files = {
        "file": ("doc.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }
    # Attacker tries to pass user_id = 999
    data = {"user_id": "999"}
    response = client.post("/documents", files=files, data=data, cookies=cookies)
    assert response.status_code == 201
    doc_id = response.json()["id"]

    db = TestingSessionLocal()
    try:
        actual_user = db.query(User).filter(User.email == "actual_secure@example.com").first()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc.user_id == actual_user.id
        assert doc.user_id != 999
    finally:
        db.close()


def test_cleanup_file_when_database_fails():
    """Test that saved file is deleted if the database commit fails."""
    cookies = register_and_login(email="fail_db_user@example.com")
    files = {
        "file": ("fail_db.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }

    original_save = storage_service.save
    saved_keys = []

    def mock_save(*args, **kwargs):
        key = original_save(*args, **kwargs)
        saved_keys.append(key)
        return key

    with patch.object(storage_service, "save", side_effect=mock_save):
        with patch.object(Session, "commit", side_effect=Exception("Simulated Database Error")):
            response = client.post("/documents", files=files, cookies=cookies)
            assert response.status_code == 500
            assert "Failed to save document metadata" in response.json()["detail"]

    # Verify that the physical file was cleaned up and does not remain orphaned on disk
    assert len(saved_keys) == 1
    storage_key = saved_keys[0]
    assert storage_service.exists(storage_key) is False


def test_no_database_record_when_storage_fails():
    """Test that if storage fails, no database record is created."""
    cookies = register_and_login(email="fail_storage_user@example.com")
    files = {
        "file": ("fail_storage.pdf", io.BytesIO(VALID_PDF_CONTENT), "application/pdf")
    }

    with patch.object(storage_service, "save", side_effect=IOError("Disk write failure")):
        response = client.post("/documents", files=files, cookies=cookies)
        assert response.status_code == 500
        assert "Failed to store document file" in response.json()["detail"]

    # Verify database has 0 documents with this filename
    db = TestingSessionLocal()
    try:
        count = db.query(Document).filter(Document.original_filename == "fail_storage.pdf").count()
        assert count == 0
    finally:
        db.close()


# ==========================================================================
# 2. LIST DOCUMENTS TESTS (GET /documents)
# ==========================================================================

def test_list_documents_unauthenticated_fails_401():
    """Test that unauthenticated GET /documents returns 401."""
    response = client.get("/documents")
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_list_documents_returns_authenticated_user_documents():
    """Test that authenticated user can list their own documents."""
    cookies = register_and_login(email="list_user@example.com")
    upload_sample_document(cookies, "doc1.pdf")
    upload_sample_document(cookies, "doc2.pdf")

    response = client.get("/documents", cookies=cookies)
    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert data["total"] == 2
    assert len(data["items"]) == 2
    filenames = [item["original_filename"] for item in data["items"]]
    assert "doc1.pdf" in filenames
    assert "doc2.pdf" in filenames


def test_list_documents_does_not_show_other_users_documents():
    """Test that listing documents never includes documents belonging to other users."""
    cookies_a = register_and_login(name="User A", email="user_a_list@example.com")
    cookies_b = register_and_login(name="User B", email="user_b_list@example.com")

    upload_sample_document(cookies_a, "user_a_private.pdf")
    upload_sample_document(cookies_b, "user_b_private.pdf")

    # User A lists documents
    resp_a = client.get("/documents", cookies=cookies_a)
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["total"] == 1
    assert data_a["items"][0]["original_filename"] == "user_a_private.pdf"

    # User B lists documents
    resp_b = client.get("/documents", cookies=cookies_b)
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["total"] == 1
    assert data_b["items"][0]["original_filename"] == "user_b_private.pdf"


def test_list_documents_pagination_and_ordering():
    """Test pagination limit/offset and newest-first ordering."""
    cookies = register_and_login(email="page_user@example.com")
    upload_sample_document(cookies, "first.pdf")
    upload_sample_document(cookies, "second.pdf")
    upload_sample_document(cookies, "third.pdf")

    # Page 1 (limit 2, offset 0) -> should have newest first (third, second)
    resp_page1 = client.get("/documents?limit=2&offset=0", cookies=cookies)
    assert resp_page1.status_code == 200
    data1 = resp_page1.json()
    assert data1["total"] == 3
    assert len(data1["items"]) == 2
    assert data1["items"][0]["original_filename"] == "third.pdf"
    assert data1["items"][1]["original_filename"] == "second.pdf"

    # Page 2 (limit 2, offset 2) -> should have (first)
    resp_page2 = client.get("/documents?limit=2&offset=2", cookies=cookies)
    assert resp_page2.status_code == 200
    data2 = resp_page2.json()
    assert len(data2["items"]) == 1
    assert data2["items"][0]["original_filename"] == "first.pdf"


# ==========================================================================
# 3. GET DOCUMENT METADATA (GET /documents/{id})
# ==========================================================================

def test_get_document_metadata_unauthenticated_fails_401():
    """Test that unauthenticated metadata request returns 401."""
    response = client.get("/documents/1")
    assert response.status_code == 401


def test_get_document_metadata_owner_success():
    """Test owner can retrieve document metadata."""
    cookies = register_and_login(email="meta_user@example.com")
    uploaded = upload_sample_document(cookies, "metadata_test.pdf")
    doc_id = uploaded["id"]

    response = client.get(f"/documents/{doc_id}", cookies=cookies)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == doc_id
    assert data["original_filename"] == "metadata_test.pdf"
    assert data["mime_type"] == "application/pdf"
    assert data["file_size"] == len(VALID_PDF_CONTENT)
    assert data["status"] in ("UPLOADED", "QUEUED", "PROCESSING", "READY")


def test_get_document_metadata_non_owner_receives_404():
    """Test non-owner receives 404 when querying another user's document metadata."""
    cookies_owner = register_and_login(email="owner_meta@example.com")
    cookies_other = register_and_login(email="other_meta@example.com")

    uploaded = upload_sample_document(cookies_owner, "secret_meta.pdf")
    doc_id = uploaded["id"]

    # Other user attempts to fetch metadata
    response = client.get(f"/documents/{doc_id}", cookies=cookies_other)
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]


def test_get_nonexistent_document_metadata_returns_404():
    """Test fetching nonexistent document ID returns 404."""
    cookies = register_and_login(email="nonexistent_meta@example.com")
    response = client.get("/documents/999999", cookies=cookies)
    assert response.status_code == 404


# ==========================================================================
# 4. GET DOCUMENT FILE (GET /documents/{id}/file)
# ==========================================================================

def test_get_document_file_unauthenticated_fails_401():
    """Test that unauthenticated file download returns 401."""
    response = client.get("/documents/1/file")
    assert response.status_code == 401


def test_get_document_file_owner_success():
    """Test that document owner can download the actual file."""
    cookies = register_and_login(email="download_owner@example.com")
    uploaded = upload_sample_document(cookies, "report_2026.pdf", ANOTHER_VALID_PDF)
    doc_id = uploaded["id"]

    response = client.get(f"/documents/{doc_id}/file", cookies=cookies)
    assert response.status_code == 200
    assert response.content == ANOTHER_VALID_PDF
    assert response.headers["content-type"] == "application/pdf"
    assert 'filename="report_2026.pdf"' in response.headers.get("content-disposition", "")


def test_get_document_file_non_owner_receives_404():
    """Test that non-owner cannot download another user's file (returns 404)."""
    cookies_owner = register_and_login(email="file_owner@example.com")
    cookies_attacker = register_and_login(email="file_attacker@example.com")

    uploaded = upload_sample_document(cookies_owner, "confidential.pdf")
    doc_id = uploaded["id"]

    response = client.get(f"/documents/{doc_id}/file", cookies=cookies_attacker)
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]


def test_get_document_file_missing_physical_file_handled():
    """Test graceful handling when DB record exists but physical file was removed."""
    cookies = register_and_login(email="missing_file@example.com")
    uploaded = upload_sample_document(cookies, "ghost.pdf")
    doc_id = uploaded["id"]

    # Delete physical file from storage directly
    db = TestingSessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        storage_service.delete(doc.storage_key)
    finally:
        db.close()

    response = client.get(f"/documents/{doc_id}/file", cookies=cookies)
    assert response.status_code == 404
    assert "file not found" in response.json()["detail"].lower()


# ==========================================================================
# 5. DELETE DOCUMENT (DELETE /documents/{id})
# ==========================================================================

def test_delete_document_unauthenticated_fails_401():
    """Test that unauthenticated delete returns 401."""
    response = client.delete("/documents/1")
    assert response.status_code == 401


def test_delete_document_owner_success():
    """Test that document owner can delete document and removes file and db record."""
    cookies = register_and_login(email="delete_owner@example.com")
    uploaded = upload_sample_document(cookies, "to_delete.pdf")
    doc_id = uploaded["id"]

    # Get storage key before deletion to verify file removal
    db = TestingSessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        storage_key = doc.storage_key
        assert storage_service.exists(storage_key) is True
    finally:
        db.close()

    # Delete document
    response = client.delete(f"/documents/{doc_id}", cookies=cookies)
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"].lower()

    # Verify database record is gone
    db = TestingSessionLocal()
    try:
        deleted_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert deleted_doc is None
    finally:
        db.close()

    # Verify physical file is gone
    assert storage_service.exists(storage_key) is False


def test_delete_document_non_owner_receives_404():
    """Test that non-owner cannot delete another user's document."""
    cookies_owner = register_and_login(email="del_owner@example.com")
    cookies_attacker = register_and_login(email="del_attacker@example.com")

    uploaded = upload_sample_document(cookies_owner, "protected_del.pdf")
    doc_id = uploaded["id"]

    # Attacker tries to delete
    response = client.delete(f"/documents/{doc_id}", cookies=cookies_attacker)
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]

    # Verify document still exists in DB and storage
    db = TestingSessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert storage_service.exists(doc.storage_key) is True
    finally:
        db.close()


def test_delete_document_when_physical_file_already_missing():
    """Test deleting document when physical file was already missing still cleans up DB record."""
    cookies = register_and_login(email="del_missing@example.com")
    uploaded = upload_sample_document(cookies, "missing_on_del.pdf")
    doc_id = uploaded["id"]

    # Remove physical file prior to DELETE request
    db = TestingSessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        storage_service.delete(doc.storage_key)
    finally:
        db.close()

    response = client.delete(f"/documents/{doc_id}", cookies=cookies)
    assert response.status_code == 200

    # DB record is cleaned up
    db = TestingSessionLocal()
    try:
        assert db.query(Document).filter(Document.id == doc_id).first() is None
    finally:
        db.close()


# ==========================================================================
# 6. MANDATORY CROSS-USER ISOLATION SECURITY SUITE (Step 7)
# ==========================================================================

def test_full_cross_user_isolation_matrix():
    """
    Mandatory security test verifying complete isolation between User A and User B:
    User A:
    - GET /documents -> sees A only
    - GET A metadata -> allowed (200)
    - GET A file -> allowed (200)
    - GET B metadata -> denied (404)
    - GET B file -> denied (404)
    - DELETE B -> denied (404)
    - DELETE A -> allowed (200)

    User B:
    - GET /documents -> sees B only
    - GET B metadata -> allowed (200)
    - GET B file -> allowed (200)
    - GET A metadata -> denied (404)
    - GET A file -> denied (404)
    - DELETE A -> denied (404)
    - DELETE B -> allowed (200)
    """
    user_a_cookies = register_and_login(name="User A", email="user_a_matrix@example.com")
    user_b_cookies = register_and_login(name="User B", email="user_b_matrix@example.com")

    # Upload Doc A for User A, Doc B for User B
    doc_a = upload_sample_document(user_a_cookies, "doc_A.pdf", VALID_PDF_CONTENT)
    doc_b = upload_sample_document(user_b_cookies, "doc_B.pdf", ANOTHER_VALID_PDF)

    doc_a_id = doc_a["id"]
    doc_b_id = doc_b["id"]

    # 1. User A checks
    # GET /documents -> sees A only
    resp_a_list = client.get("/documents", cookies=user_a_cookies)
    assert resp_a_list.status_code == 200
    items_a = resp_a_list.json()["items"]
    assert len(items_a) == 1
    assert items_a[0]["id"] == doc_a_id

    # GET A metadata -> allowed
    resp_a_meta = client.get(f"/documents/{doc_a_id}", cookies=user_a_cookies)
    assert resp_a_meta.status_code == 200
    assert resp_a_meta.json()["id"] == doc_a_id

    # GET A file -> allowed
    resp_a_file = client.get(f"/documents/{doc_a_id}/file", cookies=user_a_cookies)
    assert resp_a_file.status_code == 200
    assert resp_a_file.content == VALID_PDF_CONTENT

    # GET B metadata -> denied (404)
    resp_a_get_b = client.get(f"/documents/{doc_b_id}", cookies=user_a_cookies)
    assert resp_a_get_b.status_code == 404

    # GET B file -> denied (404)
    resp_a_get_b_file = client.get(f"/documents/{doc_b_id}/file", cookies=user_a_cookies)
    assert resp_a_get_b_file.status_code == 404

    # DELETE B -> denied (404)
    resp_a_del_b = client.delete(f"/documents/{doc_b_id}", cookies=user_a_cookies)
    assert resp_a_del_b.status_code == 404

    # 2. User B checks
    # GET /documents -> sees B only
    resp_b_list = client.get("/documents", cookies=user_b_cookies)
    assert resp_b_list.status_code == 200
    items_b = resp_b_list.json()["items"]
    assert len(items_b) == 1
    assert items_b[0]["id"] == doc_b_id

    # GET B metadata -> allowed
    resp_b_meta = client.get(f"/documents/{doc_b_id}", cookies=user_b_cookies)
    assert resp_b_meta.status_code == 200
    assert resp_b_meta.json()["id"] == doc_b_id

    # GET B file -> allowed
    resp_b_file = client.get(f"/documents/{doc_b_id}/file", cookies=user_b_cookies)
    assert resp_b_file.status_code == 200
    assert resp_b_file.content == ANOTHER_VALID_PDF

    # GET A metadata -> denied (404)
    resp_b_get_a = client.get(f"/documents/{doc_a_id}", cookies=user_b_cookies)
    assert resp_b_get_a.status_code == 404

    # GET A file -> denied (404)
    resp_b_get_a_file = client.get(f"/documents/{doc_a_id}/file", cookies=user_b_cookies)
    assert resp_b_get_a_file.status_code == 404

    # DELETE A -> denied (404)
    resp_b_del_a = client.delete(f"/documents/{doc_a_id}", cookies=user_b_cookies)
    assert resp_b_del_a.status_code == 404

    # 3. Deletion by owners
    # User A deletes A -> allowed
    resp_del_a = client.delete(f"/documents/{doc_a_id}", cookies=user_a_cookies)
    assert resp_del_a.status_code == 200

    # User B deletes B -> allowed
    resp_del_b = client.delete(f"/documents/{doc_b_id}", cookies=user_b_cookies)
    assert resp_del_b.status_code == 200
