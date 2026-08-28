"""PDF text extraction service using PyMuPDF (fitz)."""

import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

from .storage import storage_service

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""


class PDFNotFoundError(Exception):
    """Raised when the physical PDF file is missing from storage."""


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    text: str
    page_count: int
    char_count: int


def extract_text_from_pdf(storage_key: str) -> ExtractionResult:
    """Extract text content from a stored PDF file.

    Args:
        storage_key: The storage key identifying the PDF file.

    Returns:
        ExtractionResult with extracted text, page count, and character count.

    Raises:
        PDFNotFoundError: If the physical file is missing from storage.
        PDFExtractionError: If extraction fails for any other reason.
    """
    # Retrieve the file bytes through StorageService (never raw filesystem access)
    try:
        pdf_bytes = storage_service.retrieve(storage_key)
    except FileNotFoundError as exc:
        raise PDFNotFoundError(f"PDF file not found in storage: {storage_key}") from exc
    except ValueError as exc:
        raise PDFExtractionError(f"Invalid storage key: {storage_key}") from exc

    if not pdf_bytes:
        raise PDFExtractionError("PDF file is empty (0 bytes)")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF: {exc}") from exc

    try:
        page_count = doc.page_count

        if page_count == 0:
            return ExtractionResult(text="", page_count=0, char_count=0)

        pages_text = []
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            pages_text.append(text)

        extracted_text = "\n".join(pages_text)
        char_count = len(extracted_text)

        return ExtractionResult(
            text=extracted_text,
            page_count=page_count,
            char_count=char_count,
        )
    except Exception as exc:
        raise PDFExtractionError(f"Text extraction failed: {exc}") from exc
    finally:
        doc.close()
