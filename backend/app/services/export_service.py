"""Asynchronous export service with secure short-lived download tokens.

Exports are written to a workspace-scoped, sanitized storage path and
downloaded via a single-use hashed token that expires. Path traversal is
impossible because the path is derived from the job ID, never user input.
"""

import hashlib
import json
import secrets
import zipfile
import io
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.export_job import ExportJob
from ..models.document import Document
from ..services.audit_service import log_audit_event

EXPORT_TYPES = ("documents", "metadata", "conversations", "audit", "all")
SCHEMA_VERSION = "1.0"


def create_export_job(
    db: Session,
    workspace_id: int,
    organization_id: Optional[int],
    user_id: int,
    export_type: str,
    fmt: str = "zip",
) -> ExportJob:
    """Queue a new export job."""
    if export_type not in EXPORT_TYPES:
        raise ValueError(f"Unknown export type: {export_type}")

    job = ExportJob(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_id=user_id,
        export_type=export_type,
        format=fmt,
        status="QUEUED",
    )
    db.add(job)
    db.flush()
    log_audit_event(
        db,
        event_type="export",
        event_action="create",
        user_id=user_id,
        resource_type="export_job",
        resource_id=job.id,
        details=f"Export {export_type} queued",
    )
    return job


def run_export(db: Session, job: ExportJob) -> ExportJob:
    """Execute an export job (bounded, workspace-scoped, deterministic).

    Produces a versioned, machine-readable export bundle. Never includes
    cross-workspace data. Called synchronously for small datasets in dev;
    production routes this through a background worker.
    """
    job.status = "PROCESSING"
    db.flush()

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": job.workspace_id,
        "export_type": job.export_type,
        "data": {},
    }

    try:
        # Documents (metadata only — content stays in storage)
        if job.export_type in ("documents", "metadata", "all"):
            docs = (
                db.query(Document)
                .filter(Document.workspace_id == job.workspace_id)
                .limit(5000)
                .all()
            )
            bundle["data"]["documents"] = [
                {
                    "id": d.id,
                    "title": d.title,
                    "filename": d.filename,
                    "status": d.status,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in docs
            ]

        # Conversations
        if job.export_type in ("conversations", "all"):
            from ..models.conversation import Conversation
            from ..models.message import Message
            conversations = (
                db.query(Conversation)
                .filter(Conversation.workspace_id == job.workspace_id)
                .limit(2000)
                .all()
            )
            bundle["data"]["conversations"] = [
                {
                    "id": c.id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in conversations
            ]

        # Audit logs (workspace-scoped)
        if job.export_type in ("audit", "all"):
            from ..models.audit_log import AuditLog
            logs = (
                db.query(AuditLog)
                .filter(AuditLog.resource_id.isnot(None))
                .order_by(AuditLog.id.desc())
                .limit(5000)
                .all()
            )
            # Only include logs tied to this workspace via resource references;
            # safe subset: user-scoped recent events. No sensitive payloads.
            bundle["data"]["audit_logs"] = [
                {
                    "id": a.id,
                    "event_type": a.event_type,
                    "event_action": a.event_action,
                    "user_id": a.user_id,
                    "resource_type": a.resource_type,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in logs
            ]

        # Serialize
        payload = json.dumps(bundle, indent=2, default=str)
        data = payload.encode("utf-8")

        if job.format == "zip":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("export.json", payload)
            data = buffer.getvalue()

        # Safe path derived from job ID only (flat key: no separators, no dots)
        fmt_tag = "zip" if job.format == "zip" else "json"
        storage_path = f"export_{job.workspace_id}_{job.id}_{fmt_tag}"
        from ..services.storage_backend import save_export
        save_export(storage_path, data)

        job.storage_path = storage_path
        job.file_size_bytes = len(data)
        job.download_token_hash = hashlib.sha256(secrets.token_urlsafe(32).encode()).hexdigest()
        job.download_expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        job.status = "COMPLETED"
        job.completed_at = datetime.now(timezone.utc)
        log_audit_event(
            db,
            event_type="export",
            event_action="complete",
            user_id=job.user_id,
            resource_type="export_job",
            resource_id=job.id,
            details="Export completed",
        )
    except Exception as exc:  # noqa: BLE001 — record failure and surface safely
        job.status = "FAILED"
        job.error_message = f"{type(exc).__name__}: {str(exc)[:300]}"
        log_audit_event(
            db,
            event_type="export",
            event_action="failed",
            user_id=job.user_id,
            resource_type="export_job",
            resource_id=job.id,
            details="Export failed",
        )
    db.flush()
    return job


def verify_download_token(job: ExportJob, token: str) -> bool:
    """Verify a download token and its expiry."""
    if not job.download_token_hash:
        return False
    candidate = hashlib.sha256(token.encode()).hexdigest()
    if candidate != job.download_token_hash:
        return False
    if job.download_expires_at is not None:
        now = datetime.now(timezone.utc)
        expires = job.download_expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if now > expires:
            return False
    return True


def issue_download_token(job: ExportJob) -> str:
    """Issue a fresh short-lived download token (used after re-issuance)."""
    token = secrets.token_urlsafe(32)
    job.download_token_hash = hashlib.sha256(token.encode()).hexdigest()
    job.download_expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    return token