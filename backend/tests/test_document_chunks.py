"""Phase 4.3 tests: document chunk persistence, relationships, API, and pipeline integration."""

import io
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import auth
from app.models.user import User
from app.models.document import Document
from app.models.document_content import DocumentContent
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.chunker import TextChunk
from app.services.chunk_service import (
    persist_chunks,
    get_chunks_for_document,
    delete_chunks_for_document,
)
from app.services.document_processor import process_document
from tests.test_auth import client, TestingSessionLocal


# ---------------------------------------------------------------------------
# Minimal valid PDFs
# ---------------------------------------------------------------------------

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

CORRUPTED_PDF = b"%PDF-1.4\nTHIS IS NOT VALID PDF CONTENT\x00\x01\x02"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage and sessions before each test."""
    auth._sessions.clear()
    original_storage_dir = storage_service.base_dir
    test_storage_dir = tmp_path / "test_chunks_storage"
    test_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_service.base_dir = test_storage_dir
    yield
    storage_service.base_dir = original_storage_dir


def register_and_login(
    name: str = "Chunk User",
    email: str = "chunk_user@example.com",
    password: str = "password123",
) -> dict:
    client.post("/auth/register", json={"name": name, "email": email, "password": password})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    return resp.cookies


def upload_and_get_doc(cookies: dict, filename: str = "chunk_test.pdf", content: bytes = TEXT_PDF_CONTENT) -> dict:
    files = {"file": (filename, io.BytesIO(content), "application/pdf")}
    resp = client.post("/documents", files=files, cookies=cookies)
    assert resp.status_code == 201
    return resp.json()


# ==========================================================================
# 1. DocumentChunk MODEL CREATION
# ==========================================================================

class TestDocumentChunkModel:
    def test_create_chunk(self):
        """Test basic DocumentChunk creation."""
        db = TestingSessionLocal()
        try:
            user = User(name="ChunkModel", email="chunk_model@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="test.pdf",
                storage_key="key_001", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            chunk = DocumentChunk(
                document_id=doc.id, chunk_index=0, text="Hello world",
                char_start=0, char_end=11,
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)

            assert chunk.id is not None
            assert chunk.document_id == doc.id
            assert chunk.chunk_index == 0
            assert chunk.text == "Hello world"
            assert chunk.char_start == 0
            assert chunk.char_end == 11
            assert chunk.page_start is None
            assert chunk.page_end is None
            assert chunk.created_at is not None
        finally:
            db.close()

    def test_page_fields_nullable(self):
        """Test that page fields are optional."""
        db = TestingSessionLocal()
        try:
            user = User(name="PageTest", email="page_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="test.pdf",
                storage_key="key_page", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            chunk = DocumentChunk(
                document_id=doc.id, chunk_index=0, text="Page text",
                char_start=0, char_end=9, page_start=1, page_end=2,
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)

            assert chunk.page_start == 1
            assert chunk.page_end == 2
        finally:
            db.close()


# ==========================================================================
# 2. RELATIONSHIPS
# ==========================================================================

class TestRelationships:
    def test_document_to_chunks_relationship(self):
        """Test Document → chunks relationship."""
        db = TestingSessionLocal()
        try:
            user = User(name="RelTest", email="rel_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="rel.pdf",
                storage_key="key_rel", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            for i in range(3):
                chunk = DocumentChunk(
                    document_id=doc.id, chunk_index=i,
                    text=f"Chunk {i}", char_start=i * 10, char_end=(i + 1) * 10,
                )
                db.add(chunk)
            db.commit()

            db.refresh(doc)
            assert len(doc.chunks) == 3
            # Verify ordering
            assert [c.chunk_index for c in doc.chunks] == [0, 1, 2]
        finally:
            db.close()

    def test_chunk_to_document_relationship(self):
        """Test DocumentChunk → Document relationship."""
        db = TestingSessionLocal()
        try:
            user = User(name="RevRel", email="rev_rel@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="rev.pdf",
                storage_key="key_rev", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            chunk = DocumentChunk(
                document_id=doc.id, chunk_index=0,
                text="Rev chunk", char_start=0, char_end=9,
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)

            assert chunk.document is not None
            assert chunk.document.id == doc.id
        finally:
            db.close()


# ==========================================================================
# 3. UNIQUE CONSTRAINT
# ==========================================================================

class TestUniqueConstraint:
    def test_duplicate_chunk_index_rejected(self):
        """Test that duplicate (document_id, chunk_index) is rejected."""
        db = TestingSessionLocal()
        try:
            user = User(name="UniqTest", email="uniq_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="uniq.pdf",
                storage_key="key_uniq", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            chunk1 = DocumentChunk(
                document_id=doc.id, chunk_index=0,
                text="First", char_start=0, char_end=5,
            )
            db.add(chunk1)
            db.commit()

            chunk2 = DocumentChunk(
                document_id=doc.id, chunk_index=0,
                text="Second", char_start=0, char_end=6,
            )
            db.add(chunk2)
            with pytest.raises(IntegrityError):
                db.commit()
            db.rollback()
        finally:
            db.close()

    def test_same_index_different_documents_allowed(self):
        """Test that same chunk_index for different documents is allowed."""
        db = TestingSessionLocal()
        try:
            user = User(name="DiffDoc", email="diff_doc@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc1 = Document(
                user_id=user.id, original_filename="d1.pdf",
                storage_key="key_d1", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            doc2 = Document(
                user_id=user.id, original_filename="d2.pdf",
                storage_key="key_d2", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add_all([doc1, doc2])
            db.commit()
            db.refresh(doc1)
            db.refresh(doc2)

            chunk1 = DocumentChunk(
                document_id=doc1.id, chunk_index=0,
                text="Doc1 chunk", char_start=0, char_end=10,
            )
            chunk2 = DocumentChunk(
                document_id=doc2.id, chunk_index=0,
                text="Doc2 chunk", char_start=0, char_end=10,
            )
            db.add_all([chunk1, chunk2])
            db.commit()

            assert chunk1.id != chunk2.id
        finally:
            db.close()


# ==========================================================================
# 4. CASCADE DELETION
# ==========================================================================

class TestCascadeDeletion:
    def test_deleting_document_deletes_chunks(self):
        """Test that deleting a document cascades to delete its chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="CascadeTest", email="cascade_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="cascade.pdf",
                storage_key="key_cascade", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            for i in range(5):
                chunk = DocumentChunk(
                    document_id=doc.id, chunk_index=i,
                    text=f"Cascade chunk {i}", char_start=i * 20, char_end=(i + 1) * 20,
                )
                db.add(chunk)
            db.commit()

            doc_id = doc.id
            db.delete(doc)
            db.commit()

            remaining = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc_id
            ).count()
            assert remaining == 0
        finally:
            db.close()


# ==========================================================================
# 5. PERSISTENCE SERVICE
# ==========================================================================

class TestChunkPersistenceService:
    def test_persist_chunks_creates_records(self):
        """Test persist_chunks creates DocumentChunk records."""
        db = TestingSessionLocal()
        try:
            user = User(name="PersistTest", email="persist_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="persist.pdf",
                storage_key="key_persist", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            text_chunks = [
                TextChunk(chunk_index=0, text="First chunk", char_start=0, char_end=11),
                TextChunk(chunk_index=1, text="Second chunk", char_start=11, char_end=23),
                TextChunk(chunk_index=2, text="Third chunk", char_start=23, char_end=34),
            ]

            persist_chunks(db, doc.id, text_chunks)
            db.commit()

            chunks = get_chunks_for_document(db, doc.id)
            assert len(chunks) == 3
            assert [c.chunk_index for c in chunks] == [0, 1, 2]
            assert chunks[0].text == "First chunk"
            assert chunks[2].text == "Third chunk"
        finally:
            db.close()

    def test_persist_chunks_replaces_existing(self):
        """Test that persist_chunks replaces existing chunks (not duplicates)."""
        db = TestingSessionLocal()
        try:
            user = User(name="ReplaceTest", email="replace_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="replace.pdf",
                storage_key="key_replace", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # First persist
            chunks_v1 = [
                TextChunk(chunk_index=0, text="V1 chunk 0", char_start=0, char_end=10),
                TextChunk(chunk_index=1, text="V1 chunk 1", char_start=10, char_end=20),
            ]
            persist_chunks(db, doc.id, chunks_v1)
            db.commit()

            assert len(get_chunks_for_document(db, doc.id)) == 2

            # Second persist — should replace
            chunks_v2 = [
                TextChunk(chunk_index=0, text="V2 chunk 0", char_start=0, char_end=10),
            ]
            persist_chunks(db, doc.id, chunks_v2)
            db.commit()

            chunks = get_chunks_for_document(db, doc.id)
            assert len(chunks) == 1
            assert chunks[0].text == "V2 chunk 0"
        finally:
            db.close()

    def test_persist_empty_chunks(self):
        """Test persisting empty chunk list removes existing chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="EmptyTest", email="chunk_empty_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="empty.pdf",
                storage_key="key_empty", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Add some chunks
            chunks = [TextChunk(chunk_index=0, text="X", char_start=0, char_end=1)]
            persist_chunks(db, doc.id, chunks)
            db.commit()
            assert len(get_chunks_for_document(db, doc.id)) == 1

            # Replace with empty
            persist_chunks(db, doc.id, [])
            db.commit()
            assert len(get_chunks_for_document(db, doc.id)) == 0
        finally:
            db.close()

    def test_delete_chunks_for_document(self):
        """Test delete_chunks_for_document removes all chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="DelTest", email="del_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="del.pdf",
                storage_key="key_del", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            chunks = [
                TextChunk(chunk_index=i, text=f"C{i}", char_start=i, char_end=i + 1)
                for i in range(4)
            ]
            persist_chunks(db, doc.id, chunks)
            db.commit()

            deleted = delete_chunks_for_document(db, doc.id)
            assert deleted == 4
            assert len(get_chunks_for_document(db, doc.id)) == 0
        finally:
            db.close()


# ==========================================================================
# 6. INTEGRATION WITH PROCESSING PIPELINE
# ==========================================================================

class TestPipelineIntegration:
    def _create_doc_with_pdf(self, db, user, filename="pipe.pdf"):
        """Helper: save a real PDF to storage and create a Document record."""
        storage_key = storage_service.save(TEXT_PDF_CONTENT)
        doc = Document(
            user_id=user.id, original_filename=filename,
            storage_key=storage_key, mime_type="application/pdf",
            file_size=len(TEXT_PDF_CONTENT), status="UPLOADED",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    def test_process_document_creates_chunks(self):
        """Test that process_document creates chunks after extraction."""
        db = TestingSessionLocal()
        try:
            user = User(name="PipeTest", email="pipe_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = self._create_doc_with_pdf(db, user)

            process_document(db, doc.id)

            db.refresh(doc)
            assert doc.status == "READY"
            assert doc.content is not None
            assert len(doc.chunks) > 0
            assert doc.chunks[0].chunk_index == 0
        finally:
            db.close()

    def test_reprocessing_replaces_chunks(self):
        """Test that reprocessing a document replaces rather than duplicates chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="ReprocTest", email="reproc_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = self._create_doc_with_pdf(db, user)

            # First processing
            process_document(db, doc.id)
            db.refresh(doc)
            first_count = len(doc.chunks)
            assert first_count > 0

            # Set status back to FAILED to allow reprocessing
            doc.status = "FAILED"
            db.commit()

            # Second processing
            process_document(db, doc.id)
            db.refresh(doc)
            assert doc.status == "READY"
            assert len(doc.chunks) == first_count  # Same count, not doubled
        finally:
            db.close()

    def test_chunk_failure_marks_failed(self):
        """Test that if chunking fails, document is marked FAILED."""
        from unittest.mock import patch, MagicMock

        db = TestingSessionLocal()
        try:
            user = User(name="FailTest", email="fail_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            storage_key = storage_service.save(TEXT_PDF_CONTENT)
            doc = Document(
                user_id=user.id, original_filename="fail.pdf",
                storage_key=storage_key, mime_type="application/pdf",
                file_size=len(TEXT_PDF_CONTENT), status="UPLOADED",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            with patch("app.services.document_processor.chunk_text", side_effect=Exception("Chunk error")):
                process_document(db, doc.id)

            db.refresh(doc)
            assert doc.status == "FAILED"
            assert len(doc.chunks) == 0
        finally:
            db.close()


# ==========================================================================
# 7. API — GET /documents/{id}/chunks
# ==========================================================================

class TestChunksAPI:
    def test_owner_can_retrieve_chunks(self):
        """Test that document owner can retrieve chunks via API."""
        cookies = register_and_login(email="api_chunks@example.com")
        doc = upload_and_get_doc(cookies, "api_chunks.pdf")

        # Process directly
        db = TestingSessionLocal()
        try:
            db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
            process_document(db, db_doc.id)
        finally:
            db.close()

        resp = client.get(f"/documents/{doc['id']}/chunks", cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == doc["id"]
        assert data["total_chunks"] > 0
        assert len(data["chunks"]) > 0
        # Verify chunk ordering
        indices = [c["chunk_index"] for c in data["chunks"]]
        assert indices == sorted(indices)

    def test_other_user_cannot_retrieve_chunks(self):
        """Test that another user cannot retrieve chunks."""
        cookies_a = register_and_login(name="UserA", email="chunks_iso_a@example.com")
        cookies_b = register_and_login(name="UserB", email="chunks_iso_b@example.com")
        doc = upload_and_get_doc(cookies_a, "private_chunks.pdf")

        resp = client.get(f"/documents/{doc['id']}/chunks", cookies=cookies_b)
        assert resp.status_code == 404

    def test_unauthenticated_cannot_retrieve_chunks(self):
        """Test that unauthenticated user cannot retrieve chunks."""
        from fastapi.testclient import TestClient as FreshTestClient
        from app.main import app as fresh_app
        fresh_client = FreshTestClient(fresh_app)

        resp = fresh_client.get("/documents/1/chunks")
        assert resp.status_code == 401

    def test_nonexistent_document_returns_404(self):
        """Test that requesting chunks for nonexistent document returns 404."""
        cookies = register_and_login(email="chunks_404@example.com")
        resp = client.get("/documents/999999/chunks", cookies=cookies)
        assert resp.status_code == 404

    def test_chunks_empty_for_unprocessed_doc(self):
        """Test that unprocessed document returns empty chunks."""
        cookies = register_and_login(email="chunks_empty@example.com")
        doc = upload_and_get_doc(cookies, "empty_chunks.pdf")

        resp = client.get(f"/documents/{doc['id']}/chunks", cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 0
        assert data["chunks"] == []


# ==========================================================================
# 8. DELETE CLEANUP — chunks cascade
# ==========================================================================

class TestDeleteCleanup:
    def test_delete_document_removes_chunks_via_api(self):
        """Test that deleting a document via API removes its chunks."""
        cookies = register_and_login(email="del_chunks@example.com")
        doc = upload_and_get_doc(cookies, "del_chunks.pdf")

        # Process
        db = TestingSessionLocal()
        try:
            db_doc = db.query(Document).filter(Document.id == doc["id"]).first()
            process_document(db, db_doc.id)
        finally:
            db.close()

        # Verify chunks exist
        resp = client.get(f"/documents/{doc['id']}/chunks", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] > 0

        # Delete document
        del_resp = client.delete(f"/documents/{doc['id']}", cookies=cookies)
        assert del_resp.status_code == 200

        # Verify chunks are gone
        db = TestingSessionLocal()
        try:
            remaining = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc["id"]
            ).count()
            assert remaining == 0
        finally:
            db.close()


# ==========================================================================
# 9. LARGE DOCUMENT
# ==========================================================================

class TestLargeDocument:
    def test_many_chunks_persisted_correctly(self):
        """Test persisting a document that generates many chunks."""
        db = TestingSessionLocal()
        try:
            user = User(name="LargeTest", email="large_test@example.com", password_hash="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)

            doc = Document(
                user_id=user.id, original_filename="large.pdf",
                storage_key="key_large", mime_type="application/pdf",
                file_size=100, status="READY",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Create 50 chunks
            text_chunks = [
                TextChunk(
                    chunk_index=i,
                    text=f"Chunk {i} content. " * 10,
                    char_start=i * 200,
                    char_end=(i + 1) * 200,
                )
                for i in range(50)
            ]

            persist_chunks(db, doc.id, text_chunks)
            db.commit()

            chunks = get_chunks_for_document(db, doc.id)
            assert len(chunks) == 50
            assert [c.chunk_index for c in chunks] == list(range(50))
        finally:
            db.close()
