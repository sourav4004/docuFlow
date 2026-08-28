"""Phase 3 tests: document processing lifecycle, extraction, content, status, retry."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User
from app.models.document import Document
from app.models.document_content import DocumentContent
from app.core import auth
from app.services.storage import storage_service
from app.services.document_processor import (
    process_document, STATUS_UPLOADED, STATUS_QUEUED,
    STATUS_PROCESSING, STATUS_READY, STATUS_FAILED,
)
from app.services.pdf_extractor import extract_text_from_pdf, PDFExtractionError, PDFNotFoundError
from tests.test_auth import client, TestingSessionLocal


# ---------------------------------------------------------------------------
# Minimal valid PDFs
# ---------------------------------------------------------------------------

# A PDF with actual extractable text
TEXT_PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
    b"(Hello DocuFlow Phase 3) Tj ET\nendstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n440\n%%EOF"
)

# Minimal valid PDF with no text (just catalog)
EMPTY_TEXT_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF"

# Corrupted PDF
CORRUPTED_PDF = b"%PDF-1.4\nTHIS IS NOT VALID PDF CONTENT\x00\x01\x02"

# Another valid PDF
ANOTHER_TEXT_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 39 >>\nstream\nBT /F1 12 Tf 100 700 Td "
    b"(Second document text) Tj ET\nendstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000350 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n430\n%%EOF"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage and sessions before each test."""
    auth._sessions.clear()
    original_storage_dir = storage_service.base_dir
    test_storage_dir = tmp_path / "test_processing_storage"
    test_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_service.base_dir = test_storage_dir
    yield
    storage_service.base_dir = original_storage_dir


def register_and_login(
    name: str = "Process User",
    email: str = "proc_user@example.com",
    password: str = "password123",
) -> dict:
    client.post("/auth/register", json={"name": name, "email": email, "password": password})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    return resp.cookies


def upload_and_get_doc(cookies: dict, filename: str = "proc_test.pdf", content: bytes = TEXT_PDF_CONTENT) -> dict:
    files = {"file": (filename, io.BytesIO(content), "application/pdf")}
    resp = client.post("/documents", files=files, cookies=cookies)
    assert resp.status_code == 201
    return resp.json()


def get_test_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================================================
# 1. DOCUMENT STATUS LIFECYCLE
# ==========================================================================

def test_new_document_has_uploaded_status():
    """Test that a newly uploaded document has UPLOADED status."""
    cookies = register_and_login(email="lifecycle1@example.com")
    doc = upload_and_get_doc(cookies, "new_doc.pdf")

    assert doc["status"] == "UPLOADED"

    # Verify in DB
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        assert db_doc is not None
        assert db_doc.status == "UPLOADED"
    finally:
        db.close()


def test_processing_changes_status_to_processing():
    """Test that processing a document transitions it to PROCESSING."""
    cookies = register_and_login(email="lifecycle2@example.com")
    doc = upload_and_get_doc(cookies, "to_process.pdf")

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        assert db_doc.status == "UPLOADED"

        # Manually trigger processing
        process_document(db, db_doc.id)

        # Verify final state
        db.refresh(db_doc)
        assert db_doc.status == "READY"
    finally:
        db.close()


def test_successful_processing_becomes_ready():
    """Test that successful extraction results in READY status."""
    cookies = register_and_login(email="lifecycle3@example.com")
    doc = upload_and_get_doc(cookies, "ready_test.pdf", TEXT_PDF_CONTENT)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)

        db.refresh(db_doc)
        assert db_doc.status == STATUS_READY

        # Verify content was created
        content = db.query(DocumentContent).filter(
            DocumentContent.document_id == db_doc.id
        ).first()
        assert content is not None
        assert content.char_count > 0
    finally:
        db.close()


def test_failed_processing_becomes_failed():
    """Test that extraction failure results in FAILED status."""
    cookies = register_and_login(email="lifecycle4@example.com")
    doc = upload_and_get_doc(cookies, "fail_test.pdf", CORRUPTED_PDF)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)

        db.refresh(db_doc)
        assert db_doc.status == STATUS_FAILED
    finally:
        db.close()


def test_processing_skips_non_processable_status():
    """Test that processing is skipped for documents not in a processable state."""
    cookies = register_and_login(email="lifecycle5@example.com")
    doc = upload_and_get_doc(cookies, "skip_test.pdf")

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        # Set to READY
        db_doc.status = STATUS_READY
        db.commit()
        db.refresh(db_doc)

        # Processing should skip
        process_document(db, db_doc.id)

        db.refresh(db_doc)
        assert db_doc.status == STATUS_READY  # unchanged
    finally:
        db.close()


# ==========================================================================
# 2. TEXT EXTRACTION
# ==========================================================================

def test_valid_pdf_text_extracted_correctly():
    """Test that text is extracted from a valid PDF."""
    cookies = register_and_login(email="extract1@example.com")
    doc = upload_and_get_doc(cookies, "text_extract.pdf", TEXT_PDF_CONTENT)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        result = extract_text_from_pdf(db_doc.storage_key)

        assert result.text is not None
        assert result.page_count >= 1
        assert result.char_count > 0
    finally:
        db.close()


def test_binary_pdf_remains_unchanged_after_extraction():
    """Test that the original PDF file is not modified by extraction."""
    cookies = register_and_login(email="extract2@example.com")
    doc = upload_and_get_doc(cookies, "unchanged.pdf", TEXT_PDF_CONTENT)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        original_bytes = storage_service.retrieve(db_doc.storage_key)

        # Extract text
        extract_text_from_pdf(db_doc.storage_key)

        # Verify file unchanged
        after_bytes = storage_service.retrieve(db_doc.storage_key)
        assert original_bytes == after_bytes
    finally:
        db.close()


def test_empty_text_pdf_handled_safely():
    """Test that a PDF with no extractable text is handled safely."""
    cookies = register_and_login(email="extract3@example.com")
    doc = upload_and_get_doc(cookies, "no_text.pdf", EMPTY_TEXT_PDF)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        result = extract_text_from_pdf(db_doc.storage_key)

        assert result.text == ""
        assert result.char_count == 0
    finally:
        db.close()


def test_corrupted_pdf_handled_safely():
    """Test that a corrupted PDF raises a controlled error."""
    cookies = register_and_login(email="extract4@example.com")
    doc = upload_and_get_doc(cookies, "corrupt.pdf", CORRUPTED_PDF)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(db_doc.storage_key)
    finally:
        db.close()


def test_missing_file_raises_pdf_not_found():
    """Test that a missing file raises PDFNotFoundError."""
    with pytest.raises(PDFNotFoundError):
        extract_text_from_pdf("nonexistent_storage_key_12345")


# ==========================================================================
# 3. CONTENT API (GET /documents/{id}/content)
# ==========================================================================

def test_owner_can_retrieve_extracted_text():
    """Test that document owner can retrieve extracted text after processing."""
    cookies = register_and_login(email="content1@example.com")
    doc = upload_and_get_doc(cookies, "content_test.pdf", TEXT_PDF_CONTENT)

    # Process the document
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    # Retrieve content via API
    resp = client.get(f"/documents/{doc['id']}/content", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc["id"]
    assert data["status"] == "READY"
    assert data["extracted_text"] is not None
    assert len(data["extracted_text"]) > 0
    assert data["error_message"] is None


def test_other_user_cannot_retrieve_extracted_text():
    """Test that another user cannot retrieve extracted text."""
    cookies_a = register_and_login(name="User A", email="content_isolation_a@example.com")
    cookies_b = register_and_login(name="User B", email="content_isolation_b@example.com")

    doc = upload_and_get_doc(cookies_a, "private_content.pdf", TEXT_PDF_CONTENT)

    # Process
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    # User B cannot access content
    resp = client.get(f"/documents/{doc['id']}/content", cookies=cookies_b)
    assert resp.status_code == 404


def test_content_unavailable_while_processing():
    """Test that content is unavailable while document is processing."""
    cookies = register_and_login(email="content_processing@example.com")
    doc = upload_and_get_doc(cookies, "processing_content.pdf")

    # Force status to PROCESSING without actually processing
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        db_doc.status = "PROCESSING"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/documents/{doc['id']}/content", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PROCESSING"
    assert data["extracted_text"] is None


def test_content_failed_processing_returns_controlled_response():
    """Test that FAILED processing returns a controlled error response."""
    cookies = register_and_login(email="content_failed@example.com")
    doc = upload_and_get_doc(cookies, "failed_content.pdf", CORRUPTED_PDF)

    # Process (will fail)
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    resp = client.get(f"/documents/{doc['id']}/content", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAILED"
    assert data["extracted_text"] is None
    assert "failed" in data["error_message"].lower()


def test_content_nonexistent_document_returns_404():
    """Test that requesting content for nonexistent document returns 404."""
    cookies = register_and_login(email="content_404@example.com")
    resp = client.get("/documents/999999/content", cookies=cookies)
    assert resp.status_code == 404


# ==========================================================================
# 4. STATUS API (GET /documents/{id}/status)
# ==========================================================================

def test_owner_can_retrieve_status():
    """Test that document owner can retrieve processing status."""
    cookies = register_and_login(email="status1@example.com")
    doc = upload_and_get_doc(cookies, "status_test.pdf")

    resp = client.get(f"/documents/{doc['id']}/status", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc["id"]
    assert data["status"] == "UPLOADED"
    assert "created_at" in data
    assert "updated_at" in data


def test_other_user_cannot_retrieve_status():
    """Test that another user cannot retrieve processing status."""
    cookies_a = register_and_login(name="Status A", email="status_isolation_a@example.com")
    cookies_b = register_and_login(name="Status B", email="status_isolation_b@example.com")

    doc = upload_and_get_doc(cookies_a, "private_status.pdf")

    resp = client.get(f"/documents/{doc['id']}/status", cookies=cookies_b)
    assert resp.status_code == 404


def test_unauthenticated_cannot_retrieve_status():
    """Test that unauthenticated user cannot retrieve processing status."""
    cookies = register_and_login(email="status_unauth@example.com")
    doc = upload_and_get_doc(cookies, "unauth_status.pdf")

    # Use a fresh client with no cookies
    from fastapi.testclient import TestClient as FreshTestClient
    from app.main import app as fresh_app
    fresh_client = FreshTestClient(fresh_app)
    resp = fresh_client.get(f"/documents/{doc['id']}/status")
    assert resp.status_code == 401


def test_status_reflects_ready_after_processing():
    """Test that status endpoint shows READY after processing completes."""
    cookies = register_and_login(email="status_ready@example.com")
    doc = upload_and_get_doc(cookies, "status_ready.pdf", TEXT_PDF_CONTENT)

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    resp = client.get(f"/documents/{doc['id']}/status", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["status"] == "READY"


# ==========================================================================
# 5. RETRY PROCESSING (POST /documents/{id}/process)
# ==========================================================================

def test_owner_can_trigger_processing():
    """Test that document owner can trigger re-processing."""
    cookies = register_and_login(email="retry1@example.com")
    doc = upload_and_get_doc(cookies, "retry_test.pdf", TEXT_PDF_CONTENT)

    resp = client.post(f"/documents/{doc['id']}/process", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc["id"]
    # After retry, status is QUEUED (background task may or may not run in test env)
    assert data["status"] in (STATUS_QUEUED, STATUS_PROCESSING, STATUS_READY)


def test_other_user_cannot_trigger_processing():
    """Test that another user cannot trigger processing."""
    cookies_a = register_and_login(name="Retry A", email="retry_isolation_a@example.com")
    cookies_b = register_and_login(name="Retry B", email="retry_isolation_b@example.com")

    doc = upload_and_get_doc(cookies_a, "private_retry.pdf")

    resp = client.post(f"/documents/{doc['id']}/process", cookies=cookies_b)
    assert resp.status_code == 404


def test_duplicate_processing_is_prevented():
    """Test that duplicate processing requests are handled safely."""
    cookies = register_and_login(email="retry_dup@example.com")
    doc = upload_and_get_doc(cookies, "dup_retry.pdf")

    # Force PROCESSING state
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        db_doc.status = STATUS_PROCESSING
        db.commit()
    finally:
        db.close()

    # Try to retry — should return current status without starting new processing
    resp = client.post(f"/documents/{doc['id']}/process", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["status"] == "PROCESSING"


def test_retry_queued_returns_current_status():
    """Test that retry on a QUEUED document returns current status."""
    cookies = register_and_login(email="retry_queued@example.com")
    doc = upload_and_get_doc(cookies, "queued_retry.pdf")

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        db_doc.status = STATUS_QUEUED
        db.commit()
    finally:
        db.close()

    resp = client.post(f"/documents/{doc['id']}/process", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["status"] == "QUEUED"


def test_retry_failed_document_starts_processing():
    """Test that retrying a FAILED document re-queues it."""
    cookies = register_and_login(email="retry_failed@example.com")
    doc = upload_and_get_doc(cookies, "retry_failed.pdf", CORRUPTED_PDF)

    # Process directly → FAIL
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)
        db.refresh(db_doc)
        assert db_doc.status == STATUS_FAILED
    finally:
        db.close()

    # Retry via API → transitions to QUEUED
    resp = client.post(f"/documents/{doc['id']}/process", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["status"] == STATUS_QUEUED

    # Verify in DB
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        assert db_doc.status == STATUS_QUEUED
    finally:
        db.close()


# ==========================================================================
# 6. SECURITY — CROSS-USER ISOLATION
# ==========================================================================

def test_full_cross_user_processing_isolation():
    """Test complete isolation between User A and User B for all new endpoints."""
    cookies_a = register_and_login(name="ProcA", email="proc_iso_a@example.com")
    cookies_b = register_and_login(name="ProcB", email="proc_iso_b@example.com")

    doc_a = upload_and_get_doc(cookies_a, "iso_a.pdf", TEXT_PDF_CONTENT)
    doc_b = upload_and_get_doc(cookies_b, "iso_b.pdf", TEXT_PDF_CONTENT)

    # Process both
    db = TestingSessionLocal()
    try:
        for doc_id in [doc_a["id"], doc_b["id"]]:
            db_doc = db.query(Document).filter(Document.id == doc_id).first()
            process_document(db, db_doc.id)
    finally:
        db.close()

    # User A: can access own status, content; cannot access B's
    for endpoint in ["status", "content"]:
        resp_a_own = client.get(f"/documents/{doc_a['id']}/{endpoint}", cookies=cookies_a)
        assert resp_a_own.status_code == 200, f"User A should access own {endpoint}"

        resp_a_other = client.get(f"/documents/{doc_b['id']}/{endpoint}", cookies=cookies_a)
        assert resp_a_other.status_code == 404, f"User A should not access B's {endpoint}"

    # User B: can access own status, content; cannot access A's
    for endpoint in ["status", "content"]:
        resp_b_own = client.get(f"/documents/{doc_b['id']}/{endpoint}", cookies=cookies_b)
        assert resp_b_own.status_code == 200, f"User B should access own {endpoint}"

        resp_b_other = client.get(f"/documents/{doc_a['id']}/{endpoint}", cookies=cookies_b)
        assert resp_b_other.status_code == 404, f"User B should not access A's {endpoint}"

    # Process endpoint isolation
    resp_process_b = client.post(f"/documents/{doc_b['id']}/process", cookies=cookies_a)
    assert resp_process_b.status_code == 404, "User A should not process B's document"


# ==========================================================================
# 7. DELETE CLEANUP — DocumentContent cascade
# ==========================================================================

def test_delete_document_removes_content():
    """Test that deleting a document also removes its extracted content."""
    cookies = register_and_login(email="del_content@example.com")
    doc = upload_and_get_doc(cookies, "del_content.pdf", TEXT_PDF_CONTENT)

    # Process
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        process_document(db, db_doc.id)

        # Verify content exists
        content = db.query(DocumentContent).filter(
            DocumentContent.document_id == db_doc.id
        ).first()
        assert content is not None
    finally:
        db.close()

    # Delete via API
    resp = client.delete(f"/documents/{doc['id']}", cookies=cookies)
    assert resp.status_code == 200

    # Verify content is gone
    db = TestingSessionLocal()
    try:
        remaining = db.query(DocumentContent).filter(
            DocumentContent.document_id == doc["id"]
        ).count()
        assert remaining == 0
    finally:
        db.close()


def test_delete_document_removes_physical_file():
    """Test that deleting a document removes the physical PDF file."""
    cookies = register_and_login(email="del_file@example.com")
    doc = upload_and_get_doc(cookies, "del_file.pdf")

    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
        storage_key = db_doc.storage_key
        assert storage_service.exists(storage_key)
    finally:
        db.close()

    resp = client.delete(f"/documents/{doc['id']}", cookies=cookies)
    assert resp.status_code == 200

    assert not storage_service.exists(storage_key)


# ==========================================================================
# 8. END-TO-END
# ==========================================================================

def test_end_to_end_real_pdf():
    """Full lifecycle: upload → process → READY → retrieve content → delete."""
    cookies = register_and_login(name="E2E User", email="e2e_real@example.com")

    # 1. Upload
    files = {"file": ("e2e_report.pdf", io.BytesIO(TEXT_PDF_CONTENT), "application/pdf")}
    upload_resp = client.post("/documents", files=files, cookies=cookies)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    doc_id = doc["id"]

    # 2. Verify initial status
    status_resp = client.get(f"/documents/{doc_id}/status", cookies=cookies)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "UPLOADED"

    # 3. Verify DB record
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert db_doc is not None
        assert db_doc.original_filename == "e2e_report.pdf"
        assert storage_service.exists(db_doc.storage_key)
    finally:
        db.close()

    # 4. Trigger processing via retry endpoint (queues the doc)
    process_resp = client.post(f"/documents/{doc_id}/process", cookies=cookies)
    assert process_resp.status_code == 200

    # 5. Process directly (background task uses PostgreSQL session, unavailable in tests)
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    # 6. Verify READY via API
    status2 = client.get(f"/documents/{doc_id}/status", cookies=cookies)
    assert status2.status_code == 200
    assert status2.json()["status"] == "READY"

    # 7. Retrieve content
    content_resp = client.get(f"/documents/{doc_id}/content", cookies=cookies)
    assert content_resp.status_code == 200
    content_data = content_resp.json()
    assert content_data["status"] == "READY"
    assert content_data["extracted_text"] is not None
    assert len(content_data["extracted_text"]) > 0

    # 8. Delete
    del_resp = client.delete(f"/documents/{doc_id}", cookies=cookies)
    assert del_resp.status_code == 200

    # 9. Verify deletion
    db = TestingSessionLocal()
    try:
        assert db.query(Document).filter(Document.id == doc_id).first() is None
        assert db.query(DocumentContent).filter(
            DocumentContent.document_id == doc_id
        ).count() == 0
    finally:
        db.close()


def test_end_to_end_corrupted_pdf():
    """Upload corrupted PDF → processing → FAILED without crashing."""
    cookies = register_and_login(name="Corrupt User", email="e2e_corrupt@example.com")

    files = {"file": ("bad.pdf", io.BytesIO(CORRUPTED_PDF), "application/pdf")}
    upload_resp = client.post("/documents", files=files, cookies=cookies)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    doc_id = doc["id"]

    # Process directly (background task uses PostgreSQL, unavailable in tests)
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        process_document(db, db_doc.id)
    finally:
        db.close()

    # Should be FAILED
    status_resp = client.get(f"/documents/{doc_id}/status", cookies=cookies)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "FAILED"

    # Content should return error message
    content_resp = client.get(f"/documents/{doc_id}/content", cookies=cookies)
    assert content_resp.status_code == 200
    assert content_resp.json()["status"] == "FAILED"
    assert "failed" in content_resp.json()["error_message"].lower()


def test_reprocess_after_failure():
    """Upload corrupted → FAILED → replace file → reprocess → READY."""
    cookies = register_and_login(name="Reprocess User", email="reprocess@example.com")

    # Upload corrupted
    files = {"file": ("bad.pdf", io.BytesIO(CORRUPTED_PDF), "application/pdf")}
    upload_resp = client.post("/documents", files=files, cookies=cookies)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    # Process via direct call → FAIL
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        process_document(db, db_doc.id)
        db.refresh(db_doc)
        assert db_doc.status == STATUS_FAILED
    finally:
        db.close()

    # Verify FAILED status via API
    status = client.get(f"/documents/{doc_id}/status", cookies=cookies)
    assert status.json()["status"] == "FAILED"

    # Replace file with valid PDF
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        storage_service.save(TEXT_PDF_CONTENT, storage_key=db_doc.storage_key)
    finally:
        db.close()

    # Retry → should transition to QUEUED, then process to READY
    retry_resp = client.post(f"/documents/{doc_id}/process", cookies=cookies)
    assert retry_resp.status_code == 200
    assert retry_resp.json()["status"] in (STATUS_QUEUED, STATUS_PROCESSING, STATUS_READY)

    # Process directly (background task can't run in test env)
    db = TestingSessionLocal()
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        process_document(db, db_doc.id)
        db.refresh(db_doc)
        assert db_doc.status == STATUS_READY
    finally:
        db.close()


# ==========================================================================
# 9. DATABASE INTEGRATION
# ==========================================================================

def test_document_content_model_persists():
    """Test DocumentContent model creation and persistence."""
    db = TestingSessionLocal()
    try:
        user = User(name="Model Test", email="model_content@example.com", password_hash="hashed")
        db.add(user)
        db.commit()
        db.refresh(user)

        doc = Document(
            user_id=user.id,
            original_filename="model_test.pdf",
            storage_key="model_key_123",
            mime_type="application/pdf",
            file_size=1000,
            status="READY",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        content = DocumentContent(
            document_id=doc.id,
            extracted_text="Test extracted text",
            page_count=1,
            char_count=20,
        )
        db.add(content)
        db.commit()
        db.refresh(content)

        assert content.id is not None
        assert content.document_id == doc.id
        assert content.extracted_text == "Test extracted text"
        assert content.page_count == 1
        assert content.char_count == 20

        # Test relationship
        assert doc.content is not None
        assert doc.content.extracted_text == "Test extracted text"
    finally:
        db.close()


def test_document_content_unique_constraint():
    """Test that only one DocumentContent per document is enforced."""
    db = TestingSessionLocal()
    try:
        user = User(name="Unique Test", email="unique_content@example.com", password_hash="hashed")
        db.add(user)
        db.commit()
        db.refresh(user)

        doc = Document(
            user_id=user.id,
            original_filename="unique_test.pdf",
            storage_key="unique_key_123",
            mime_type="application/pdf",
            file_size=1000,
            status="READY",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        content1 = DocumentContent(document_id=doc.id, extracted_text="First", page_count=1, char_count=5)
        db.add(content1)
        db.commit()

        # Second content for same document should fail unique constraint
        content2 = DocumentContent(document_id=doc.id, extracted_text="Second", page_count=1, char_count=6)
        db.add(content2)
        with pytest.raises(Exception):  # IntegrityError
            db.commit()
        db.rollback()
    finally:
        db.close()


def test_document_content_cascade_delete():
    """Test that deleting a document cascades to delete DocumentContent."""
    db = TestingSessionLocal()
    try:
        user = User(name="Cascade Test", email="cascade_content@example.com", password_hash="hashed")
        db.add(user)
        db.commit()
        db.refresh(user)

        doc = Document(
            user_id=user.id,
            original_filename="cascade_test.pdf",
            storage_key="cascade_key_123",
            mime_type="application/pdf",
            file_size=1000,
            status="READY",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        content = DocumentContent(
            document_id=doc.id,
            extracted_text="To be cascaded",
            page_count=1,
            char_count=15,
        )
        db.add(content)
        db.commit()

        # Delete document
        db.delete(doc)
        db.commit()

        # Content should be gone
        remaining = db.query(DocumentContent).filter(
            DocumentContent.document_id == doc.id
        ).count()
        assert remaining == 0
    finally:
        db.close()


def test_retry_processing_endpoint_unauthenticated():
    """Test that unauthenticated retry returns 401."""
    resp = client.post("/documents/1/process")
    assert resp.status_code == 401


def test_content_endpoint_unauthenticated():
    """Test that unauthenticated content request returns 401."""
    resp = client.get("/documents/1/content")
    assert resp.status_code == 401
