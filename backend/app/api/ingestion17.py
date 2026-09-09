"""Phase 17 ingestion + knowledge federation API — durable ingestion runs,
fingerprint duplicate intelligence, and connector sources/syncs. Everything
is workspace-scoped through membership checks."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.workspace import Workspace
from ..models.document import Document
from ..services.permission_service import require_workspace_membership
from ..services import ingestion2 as ing
from ..services import fingerprint as fp
from ..services import federation as fed
from ..services.worker_platform import JobNotFoundError

router = APIRouter(tags=["ingestion17"])


def _workspace(db: Session, workspace_id: int, user_id: int) -> Workspace:
    return require_workspace_membership(db, workspace_id, user_id)


def _owned_document(db: Session, workspace_id: int,
                    document_id: int) -> Document:
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == workspace_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class StartIngestionBody(BaseModel):
    document_id: int
    batch_document_ids: Optional[list[int]] = None
    idempotency_key: Optional[str] = None


@router.post("/ingestion")
def start_ingestion(workspace_id: int, body: StartIngestionBody,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    _owned_document(db, workspace_id, body.document_id)
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    run = ing.start_ingestion(
        db, workspace_id=workspace_id, document_id=body.document_id,
        user_id=principal.user.id,
        organization_id=ws.organization_id if ws else None,
        batch_document_ids=body.batch_document_ids,
        idempotency_key=body.idempotency_key)
    db.commit()
    return {"run_id": run.id, "status": run.status,
            "current_stage": run.current_stage}


@router.post("/ingestion/{run_id}/advance")
def advance_ingestion(run_id: int, workspace_id: int,
                      principal: AuthPrincipal = Depends(get_current_principal),
                      db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    run = db.query(ing.IngestionRun).filter(
        ing.IngestionRun.id == run_id,
        ing.IngestionRun.workspace_id == workspace_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    result = ing.advance_ingestion_run(db, run_id)
    db.commit()
    return result


@router.post("/ingestion/{run_id}/retry")
def retry_ingestion(run_id: int, workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    try:
        result = ing.retry_ingestion(db, run_id, principal.user.id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    db.commit()
    return result


@router.get("/ingestion")
def list_ingestion(workspace_id: int, status: Optional[str] = None,
                   limit: int = 50, offset: int = 0,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    return ing.list_runs(db, workspace_id, status=status, limit=limit,
                         offset=offset)


@router.get("/ingestion/{run_id}")
def ingestion_detail(run_id: int, workspace_id: int,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    run = db.query(ing.IngestionRun).filter(
        ing.IngestionRun.id == run_id,
        ing.IngestionRun.workspace_id == workspace_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return ing.ingestion_progress(db, run_id)


# ---------------------------------------------------------------------------
# Fingerprints + duplicates
# ---------------------------------------------------------------------------

@router.post("/documents/{document_id}/fingerprint")
def compute_fingerprint(document_id: int, workspace_id: int,
                        principal: AuthPrincipal = Depends(get_current_principal),
                        db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    _owned_document(db, workspace_id, document_id)
    result = fp.compute_fingerprint(db, document_id)
    db.commit()
    return {"document_id": document_id,
            "content_hash": result.content_hash,
            "structural_hash": result.structural_hash}


@router.post("/documents/{document_id}/duplicates")
def scan_duplicates(document_id: int, workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    _owned_document(db, workspace_id, document_id)
    results = fp.classify_duplicates(db, workspace_id, document_id)
    db.commit()
    return {"document_id": document_id, "candidates": results}


@router.get("/duplicates")
def list_duplicates(workspace_id: int, limit: int = 50,
                    reviewed: Optional[bool] = None,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    return fp.list_candidates(db, workspace_id, limit=limit,
                              reviewed=reviewed)


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

class ConnectorBody(BaseModel):
    name: str
    kind: str = "connector_stub"
    scopes: Optional[list[str]] = None
    permissions: Optional[list[str]] = None
    allowed_domains: Optional[list[str]] = None
    credential_ref: Optional[str] = None
    retention_days: Optional[int] = None


@router.post("/connectors")
def create_connector(workspace_id: int, body: ConnectorBody,
                     principal: AuthPrincipal = Depends(get_current_principal),
                     db: Session = Depends(get_db)):
    ws = _workspace(db, workspace_id, principal.user.id)
    try:
        src = fed.create_source(
            db, workspace_id=workspace_id,
            organization_id=ws.organization_id,
            user_id=principal.user.id, name=body.name, kind=body.kind,
            scopes=body.scopes, permissions=body.permissions,
            allowed_domains=body.allowed_domains,
            credential_ref=body.credential_ref,
            retention_days=body.retention_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db.commit()
    return {"source_id": src.id, "name": src.name, "kind": src.kind}


@router.get("/connectors")
def list_connectors(workspace_id: int, limit: int = 50,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    return fed.list_sources(db, workspace_id, limit=limit)


@router.post("/connectors/{source_id}/sync")
def sync_connector(source_id: int, workspace_id: int,
                   principal: AuthPrincipal = Depends(get_current_principal),
                   db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    try:
        result = fed.run_connector_sync(db, source_id,
                                        workspace_id=workspace_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    db.commit()
    return result


@router.get("/connectors/{source_id}/syncs")
def connector_syncs(source_id: int, workspace_id: int,
                    principal: AuthPrincipal = Depends(get_current_principal),
                    db: Session = Depends(get_db)):
    _workspace(db, workspace_id, principal.user.id)
    try:
        return fed.sync_state(db, source_id, workspace_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
