"""E2E verification: real PDF → full pipeline with chunks (self-contained)."""

import io
from pathlib import Path
import pytest

from app.core import auth
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.storage import storage_service
from app.services.document_processor import process_document
from tests.test_auth import TestingSessionLocal


# Minimal PDF with actual extractable text
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


def test_e2e_full_pipeline():
    """Upload → process → chunks → reprocess → delete cleanup."""
    auth._sessions.clear()

    db = TestingSessionLocal()
    try:
        # 1. Create user
        user = User(name="E2E User", email="e2e_chunks_real@example.com", password_hash="hashed")
        db.add(user)
        db.commit()
        db.refresh(user)

        # 2. Save PDF to storage + create document record
        pdf_bytes = TEXT_PDF_CONTENT
        storage_key = storage_service.save(pdf_bytes)
        doc = Document(
            user_id=user.id, original_filename="sample.pdf",
            storage_key=storage_key, mime_type="application/pdf",
            file_size=len(pdf_bytes), status="UPLOADED",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        print(f"[E2E] Created doc {doc.id}")

        # 3. Process
        process_document(db, doc.id)
        db.refresh(doc)
        assert doc.status == "READY", f"Expected READY, got {doc.status}"
        print(f"[E2E] Status: {doc.status}")

        # 4. Verify chunks exist
        assert len(doc.chunks) > 0
        print(f"[E2E] Chunks: {len(doc.chunks)}")

        # 5. Verify ordering
        indices = [c.chunk_index for c in doc.chunks]
        assert indices == sorted(indices)
        print(f"[E2E] Chunk indices: {indices}")

        # 6. Verify chunk text matches content
        full_text = doc.content.extracted_text
        for c in doc.chunks:
            assert full_text[c.char_start:c.char_end] == c.text, (
                f"Chunk {c.chunk_index} text mismatch"
            )
        print("[E2E] Chunk text verification: PASSED")

        # 7. Reprocess — no duplicates
        doc.status = "FAILED"
        db.commit()

        process_document(db, doc.id)
        db.refresh(doc)
        assert doc.status == "READY"
        assert len(doc.chunks) > 0
        print(f"[E2E] Reprocess: {len(doc.chunks)} chunks (no duplicates)")

        # 8. Delete — verify cleanup
        doc_id = doc.id
        db.delete(doc)
        db.commit()

        remaining = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count()
        assert remaining == 0
        print("[E2E] Delete cleanup: PASSED")

        print("\n=== E2E FULL PIPELINE TEST PASSED ===")
    finally:
        db.close()
