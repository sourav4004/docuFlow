"""Multimodal document intelligence — pages, layout, tables, OCR.

- ``DocumentPage`` rows capture per-page text plus layout/tables/image JSON so
  downstream consumers get structure without re-parsing raw text.
- ``OCRProvider`` is an interface; ``LocalTextOCRProvider`` is a deterministic
  local implementation (no third-party credentials required). A production
  adapter can be configured behind the same interface.
- Table intelligence normalizes rows/headers/units and never treats tables as
  plain paragraphs.
- Multimodal retrieval returns page/table regions with full provenance.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase16 import DocumentPage

logger = logging.getLogger(__name__)

LAYOUT_KINDS = ("heading", "paragraph", "list", "table", "figure",
                "caption", "header", "footer", "other")

_NUMBER_RE = re.compile(r"^[-+]?\d[\d,]*\.?\d*$")


@dataclass
class OcrResult:
    """OCR output for one page region set."""
    provider: str
    pages: list = field(default_factory=list)  # [{page_number, text, confidence}]
    blocks: list = field(default_factory=list)  # [{page, kind, text, confidence}]
    low_confidence_regions: list = field(default_factory=list)


class OCRProvider(ABC):
    """OCR abstraction — local or production adapters behind one interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (safe for storage/logs)."""

    @abstractmethod
    def extract(self, db: Session, document, version: Optional[int] = None) -> OcrResult:
        """Extract text + per-region confidence for a document.

        Never requires credentials; production adapters must enforce MIME,
        size, timeout and provider policy server-side.
        """


class LocalTextOCRProvider(OCRProvider):
    """Deterministic local OCR using already-extracted document text.

    Splits content into pages on form feeds (or a single page), classifies
    layout blocks, and estimates a conservative per-block confidence from
    block readability heuristics. Intended for development/tests and as the
    fallback when no third-party OCR is configured.
    """

    @property
    def name(self) -> str:
        return "local_text"

    def extract(self, db: Session, document, version: Optional[int] = None) -> OcrResult:
        text = self._document_text(db, document)
        raw_pages = self._split_pages(text)
        pages = []
        blocks = []
        low = []
        for i, page_text in enumerate(raw_pages, start=1):
            confidence = self._page_confidence(page_text)
            page_blocks = classify_layout(page_text)
            for kind, block_text in page_blocks:
                block_conf = self._block_confidence(block_text, kind)
                blocks.append({
                    "page": i, "kind": kind, "text": block_text,
                    "confidence": block_conf,
                })
                if block_conf < 0.5:
                    low.append({"page": i, "kind": kind,
                                "confidence": block_conf})
            pages.append({
                "page_number": i, "text": page_text,
                "confidence": round(confidence, 3),
            })
        return OcrResult(
            provider=self.name,
            pages=pages,
            blocks=blocks,
            low_confidence_regions=low,
        )

    def _document_text(self, db: Session, document) -> str:
        from ..models.document_content import DocumentContent
        content = (
            db.query(DocumentContent)
            .filter(DocumentContent.document_id == document.id)
            .order_by(DocumentContent.created_at.desc())
            .first()
        )
        if content is not None:
            return content.extracted_text or ""
        return ""

    @staticmethod
    def _split_pages(text: str) -> list[str]:
        if "\f" in text:
            return [p.strip() for p in text.split("\f") if p.strip()]
        if not text.strip():
            return []
        return [text.strip()]

    @staticmethod
    def _page_confidence(text: str) -> float:
        if not text.strip():
            return 0.0
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        readable = sum(
            1 for ln in lines
            if len(ln.strip()) >= 2 and not _looks_garbled(ln)
        )
        return round(readable / len(lines), 3)

    @staticmethod
    def _block_confidence(text: str, kind: str) -> float:
        if not text.strip():
            return 0.0
        score = 0.9
        if _looks_garbled(text):
            score -= 0.5
        if kind == "table":
            score -= 0.2  # tables need structural checks; never overstate OCR
        if len(text) < 4:
            score -= 0.2
        return round(max(0.0, min(1.0, score)), 2)


def _looks_garbled(text: str) -> bool:
    """Cheap determinism guard — mojibake/garbage detection."""
    if not text:
        return False
    sample = text[:200]
    weird = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\t\n\r")
    return weird > max(2, len(sample) // 50)


# ---------------------------------------------------------------------------
# Layout intelligence
# ---------------------------------------------------------------------------

def classify_layout(text: str) -> list[tuple[str, str]]:
    """Split page text into reading-order blocks with layout kinds.

    Pure and deterministic. Paragraph groups are delimited by blank lines and
    each group is classified as table / caption / heading / list / paragraph.
    """
    if not text:
        return []
    groups = re.split(r"\n\s*\n", text)
    blocks: list[tuple[str, str]] = []
    for group in groups:
        lines = [ln.strip() for ln in group.splitlines() if ln.strip()]
        if not lines:
            continue
        joined = "\n".join(lines)
        if is_table_block(joined):
            blocks.append(("table", joined))
        elif lines[0].startswith(("Figure ", "Fig. ")):
            blocks.append(("caption", joined))
        elif len(lines) == 1 and _is_heading(lines[0]):
            blocks.append(("heading", lines[0]))
        elif any(_is_list_item(ln) for ln in lines):
            blocks.append(("list", joined))
        else:
            blocks.append(("paragraph", joined))
    return blocks


def _is_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    if line.endswith((".", ",", ";", ":", "!", "?")):
        return False
    words = line.split()
    if not words:
        return False
    # Section-numbered headings: "3.2 Budget" or "SECTION 4".
    if re.match(r"^(\d+(\.\d+)*[\.\)]?\s|[A-Z]+\s\d+|SECTION\s)", line,
                re.IGNORECASE):
        return True
    # Short, mostly-capitalized or title-case lines read as headings.
    return len(words) <= 8 and (
        line.isupper() or all(w[0].isupper() for w in words if w[0].isalpha())
    )


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*([-*•]|\d+[\.\)]|[a-zA-Z][\.\)])\s", line))


def is_table_block(text: str) -> bool:
    """Detect tabular text: pipes, tabs, or consistent multi-column spacing."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if any("\t" in ln or "|" in ln for ln in lines):
        return True
    # Heuristic: ≥2 lines with two or more runs of digits separated by spaces.
    digit_runs = [len(re.findall(r"\d", ln)) for ln in lines]
    return sum(1 for n in digit_runs if n >= 4) >= 2


# ---------------------------------------------------------------------------
# Table intelligence 2.0
# ---------------------------------------------------------------------------

def normalize_number(value: str) -> Optional[float]:
    """Normalize numeric cells: strip $/currency, thousands separators."""
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    if cleaned.endswith("k") or cleaned.endswith("K"):
        try:
            return float(cleaned[:-1]) * 1000
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_table_block(text: str) -> Optional[dict]:
    """Parse a table block into headers + normalized rows.

    Returns None when the block is not a well-formed table (never fabricates).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    if "|" in text:
        # Pipe tables may carry a caption/intro line before the header row;
        # only lines containing the delimiter are table structure.
        pipe_lines = [ln.strip("|") for ln in lines if "|" in ln]
        if len(pipe_lines) < 2:
            return None
        rows = [re.split(r"\s*\|\s*", ln) for ln in pipe_lines]
    else:
        rows = [ln.split() for ln in lines]
    headers = rows[0]
    data_rows = rows[1:]
    if not headers or not data_rows:
        return None
    width = len(headers)
    normalized_rows = []
    numeric_count = 0
    for row in data_rows:
        row = row[:width]
        if len(row) < width:
            row = row + [""] * (width - len(row))
        normalized = []
        for cell in row:
            num = normalize_number(cell) if _NUMBER_RE.match(
                cell.replace(",", "").replace("$", "").replace("%", "")
            ) else None
            if num is not None:
                numeric_count += 1
            normalized.append({"value": cell, "number": num,
                               "unit": _unit_of(cell)})
        normalized_rows.append(normalized)
    return {
        "headers": headers,
        "rows": normalized_rows,
        "shape": [len(normalized_rows), width],
        "numeric_cells": numeric_count,
        "has_totals": any(
            h.lower() in ("total", "sum", "grand total") for h in headers
        ) or _is_total_row(headers, data_rows),
    }


def _unit_of(cell: str) -> Optional[str]:
    text = cell.strip()
    if text.startswith(("$", "€", "£")):
        return text[0]
    if text.endswith("%"):
        return "%"
    match = re.match(r"^(USD|EUR|GBP)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z]{2,4})$", text)
    return match.group(1) if match else None


def _is_total_row(headers: list, data_rows: list) -> bool:
    first_col = (headers[0] if headers else "").lower()
    if first_col not in ("total", "sum", "grand total"):
        return False
    return any(row and row[0].lower() in ("total", "sum", "grand total")
               for row in data_rows)


# ---------------------------------------------------------------------------
# Page storage + region retrieval
# ---------------------------------------------------------------------------

def ingest_pages(
    db: Session,
    document_id: int,
    version: int,
    pages: list,
    ocr_provider_name: Optional[str] = None,
) -> list[DocumentPage]:
    """Store page-level structure (upsert per document/version/page)."""
    created = []
    for page in pages:
        number = int(page["page_number"])
        text = page.get("text") or ""
        layout = classify_layout(text)
        table_blocks = [
            parse_table_block(block_text)
            for kind, block_text in layout if kind == "table"
        ]
        row = (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document_id,
                    DocumentPage.version == version,
                    DocumentPage.page_number == number)
            .first()
        )
        if row is None:
            row = DocumentPage(document_id=document_id, version=version,
                               page_number=number)
            db.add(row)
        row.text = text
        row.layout_json = json.dumps(layout)
        row.tables_json = json.dumps([t for t in table_blocks if t])
        row.ocr_confidence = page.get("confidence")
        row.ocr_provider = page.get("ocr_provider") or ocr_provider_name
        created.append(row)
    db.flush()
    return created


def list_pages(db: Session, document_id: int,
               version: Optional[int] = None,
               limit: int = 500) -> list[DocumentPage]:
    query = db.query(DocumentPage).filter(
        DocumentPage.document_id == document_id)
    if version is not None:
        query = query.filter(DocumentPage.version == version)
    return query.order_by(DocumentPage.page_number).limit(min(limit, 2000)).all()


def page_region_search(
    db: Session,
    document_ids: list[int],
    kind: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
) -> list[dict]:
    """Multimodal region retrieval — text/table/page blocks with provenance."""
    query = db.query(DocumentPage).filter(
        DocumentPage.document_id.in_(document_ids))
    pages = query.order_by(DocumentPage.document_id,
                           DocumentPage.page_number).limit(2000).all()
    results = []
    for page in pages:
        if min_confidence is not None and (
                page.ocr_confidence is None or page.ocr_confidence < min_confidence):
            continue
        try:
            layout = json.loads(page.layout_json or "[]")
        except ValueError:
            layout = []
        for entry in layout:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            k, text = entry[0], entry[1]
            if kind and k != kind:
                continue
            results.append({
                "document_id": page.document_id,
                "page_number": page.page_number,
                "version": page.version,
                "kind": k,
                "text": text,
                "ocr_confidence": page.ocr_confidence,
            })
    return results[:min(limit, 200)]
