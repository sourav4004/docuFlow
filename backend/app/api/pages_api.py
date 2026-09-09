"""Multimodal document intelligence API — page structure, tables, OCR."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..services.permission_service import require_workspace_membership
from ..services.multimodal import (
    ingest_pages, list_pages, page_region_search,
)
from ..models.document import Document

router = APIRouter(prefix="/documents/{document_id}/pages", tags=["pages"])


def _doc_and_membership(db, document_id, principal) -> Document:
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    require_workspace_membership(db, document.workspace_id, principal.user.id)
    return document


class PageInput(BaseModel):
    page_number: int
    text: Optional[str] = None
    confidence: Optional[float] = None
    ocr_provider: Optional[str] = None


class PagesIngest(BaseModel):
    version: int = 1
    pages: list[PageInput]


def _page_dict(p) -> dict:
    return {
        "id": p.id,
        "document_id": p.document_id,
        "version": p.version,
        "page_number": p.page_number,
        "text": p.text,
        "layout_json": p.layout_json,
        "tables_json": p.tables_json,
        "ocr_confidence": p.ocr_confidence,
        "ocr_provider": p.ocr_provider,
    }


@router.post("", status_code=201)
def ingest(
    document_id: int,
    data: PagesIngest,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _doc_and_membership(db, document_id, principal)
    rows = ingest_pages(
        db, document_id, data.version,
        [
            {"page_number": p.page_number, "text": p.text,
             "confidence": p.confidence, "ocr_provider": p.ocr_provider}
            for p in data.pages
        ],
    )
    db.commit()
    return {"items": [_page_dict(r) for r in rows]}


@router.get("")
def list_pages_endpoint(
    document_id: int,
    version: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _doc_and_membership(db, document_id, principal)
    rows = list_pages(db, document_id, version=version)
    return {"items": [_page_dict(r) for r in rows], "count": len(rows)}


@router.get("/regions")
def regions_endpoint(
    document_id: int,
    kind: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _doc_and_membership(db, document_id, principal)
    return {"items": page_region_search(
        db, [document_id], kind=kind, min_confidence=min_confidence,
        limit=limit)}
