"""Phase 16 operations API — worker jobs, queue metrics, traces, provider
health, capabilities, and vector status. Admin endpoints are gated by
organization-admin (or workspace-owner when no organization exists).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.phase16 import WorkerJob, WorkerHeartbeat, ProviderCapability
from ..models.phase15 import ProviderHealth
from ..services.worker_platform import (
    queue_metrics, cancel_job, JobNotFoundError, enqueue_job,
)
from ..services.trace_service import list_traces, trace_summary
from ..services.provider_platform import (
    list_capabilities, upsert_capability, provider_status, gateway_status,
)
from ..services.vector_platform import detect_pgvector
from ..services.permission_service import (
    require_permission, require_workspace_membership, get_organization_role,
)
from ..models.workspace import Workspace

jobs_router = APIRouter(prefix="/worker-jobs", tags=["worker-jobs"])
ops_router = APIRouter(prefix="/ops", tags=["ops"])


def _admin_gate(db: Session, principal: AuthPrincipal,
                organization_id: Optional[int] = None,
                workspace_id: Optional[int] = None):
    """Org-admin gate; falls back to workspace OWNER without an org."""
    if organization_id is not None:
        role = get_organization_role(db, organization_id, principal.user.id)
        if role in ("OWNER", "ADMIN"):
            return
        raise HTTPException(status_code=403, detail="Organization admin required")
    if workspace_id is not None:
        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        if ws.organization_id is not None:
            role = get_organization_role(db, ws.organization_id, principal.user.id)
            if role in ("OWNER", "ADMIN"):
                return
            raise HTTPException(status_code=403,
                                detail="Organization admin required")
        if ws.owner_id == principal.user.id:
            return
        raise HTTPException(status_code=403, detail="Workspace owner required")
    raise HTTPException(status_code=403, detail="Scope required for admin access")


def _job_dict(job: WorkerJob) -> dict:
    return {
        "id": job.id,
        "queue_name": job.queue_name,
        "job_type": job.job_type,
        "status": job.status,
        "priority": job.priority,
        "workspace_id": job.workspace_id,
        "organization_id": job.organization_id,
        "user_id": job.user_id,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "dedupe_key": job.dedupe_key,
        "claimed_by": job.claimed_by,
        "run_after": job.run_after,
        "next_retry_at": job.next_retry_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "trace_id": job.trace_id,
        "created_at": job.created_at,
    }


# ---------------------------------------------------------------------------
# Worker jobs (workspace members)
# ---------------------------------------------------------------------------

@jobs_router.get("")
def list_jobs(
    workspace_id: int,
    queue: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    query = db.query(WorkerJob).filter(WorkerJob.workspace_id == workspace_id)
    if queue:
        query = query.filter(WorkerJob.queue_name == queue)
    if status:
        query = query.filter(WorkerJob.status == status)
    items = query.order_by(WorkerJob.created_at.desc()).limit(
        min(limit, 200)).all()
    return {"items": [_job_dict(j) for j in items], "count": len(items)}


@jobs_router.get("/{job_id}")
def get_job(
    job_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    require_workspace_membership(db, job.workspace_id, principal.user.id)
    return _job_dict(job)


@jobs_router.post("/{job_id}/cancel")
def cancel_job_endpoint(
    job_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    require_workspace_membership(db, job.workspace_id, principal.user.id)
    try:
        job = cancel_job(db, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return _job_dict(job)


@ops_router.get("/queue-metrics")
def ops_queue_metrics(
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    queue: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _admin_gate(db, principal, organization_id=organization_id,
                workspace_id=workspace_id)
    return queue_metrics(db, queue_name=queue)


@ops_router.get("/vector-status")
def ops_vector_status(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Backend detection — reports pgvector vs JSON fallback honestly."""
    return detect_pgvector(db)


@ops_router.get("/traces")
def ops_traces(
    workspace_id: int,
    trace_id: Optional[str] = None,
    span_type: Optional[str] = None,
    limit: int = 50,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    spans = list_traces(db, workspace_id, trace_id=trace_id,
                        span_type=span_type, limit=limit)
    return {
        "items": [
            {
                "span_id": s.span_id,
                "trace_id": s.trace_id,
                "execution_id": s.execution_id,
                "span_type": s.span_type,
                "status": s.status,
                "latency_ms": s.latency_ms,
                "model": s.model,
                "provider": s.provider,
                "error_class": s.error_class,
                "input_summary": s.input_summary,
                "started_at": s.started_at,
            }
            for s in spans
        ],
        "count": len(spans),
    }


@ops_router.get("/trace-summary")
def ops_trace_summary(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    require_workspace_membership(db, workspace_id, principal.user.id)
    return trace_summary(db, workspace_id=workspace_id)


@ops_router.get("/provider-status")
def ops_provider_status(
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    provider: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _admin_gate(db, principal, organization_id=organization_id,
                workspace_id=workspace_id)
    return {
        "items": [
            {
                "id": h.id,
                "provider": h.provider,
                "model": h.model,
                "status": h.status,
                "circuit_state": h.circuit_state,
                "consecutive_failures": h.consecutive_failures,
                "success_count": h.success_count,
                "failure_count": h.failure_count,
                "avg_latency_ms": h.avg_latency_ms,
                "last_error": h.last_error,
                "last_checked_at": h.last_checked_at,
            }
            for h in provider_status(db, provider=provider)
        ]
    }


@ops_router.get("/capabilities")
def ops_capabilities(
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _admin_gate(db, principal, organization_id=organization_id,
                workspace_id=workspace_id)
    return {
        "items": [
            {
                "provider": c.provider,
                "model": c.model,
                "supports_text": c.supports_text,
                "supports_vision": c.supports_vision,
                "supports_tools": c.supports_tools,
                "supports_structured": c.supports_structured,
                "supports_streaming": c.supports_streaming,
                "context_window": c.context_window,
                "max_output": c.max_output,
                "embedding_dimensions": c.embedding_dimensions,
                "cost_per_1k_input": c.cost_per_1k_input,
                "cost_per_1k_output": c.cost_per_1k_output,
                "latency_class": c.latency_class,
            }
            for c in list_capabilities(db)
        ]
    }


class CapabilityCreate(BaseModel):
    provider: str
    model: str
    supports_text: bool = True
    supports_vision: bool = False
    supports_tools: bool = False
    supports_structured: bool = False
    supports_streaming: bool = False
    context_window: Optional[int] = None
    max_output: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    latency_class: Optional[str] = None


@ops_router.post("/capabilities")
def ops_capability_create(
    data: CapabilityCreate,
    organization_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _admin_gate(db, principal, organization_id=organization_id,
                workspace_id=workspace_id)
    cap = upsert_capability(db, data.provider, data.model,
                            supports_text=data.supports_text,
                            supports_vision=data.supports_vision,
                            supports_tools=data.supports_tools,
                            supports_structured=data.supports_structured,
                            supports_streaming=data.supports_streaming,
                            context_window=data.context_window,
                            max_output=data.max_output,
                            embedding_dimensions=data.embedding_dimensions,
                            cost_per_1k_input=data.cost_per_1k_input,
                            cost_per_1k_output=data.cost_per_1k_output,
                            latency_class=data.latency_class)
    db.commit()
    return {"provider": cap.provider, "model": cap.model}


@ops_router.get("/gateway-status")
def ops_gateway_status(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Safe gateway configuration summary — no secrets, no keys."""
    return gateway_status()
