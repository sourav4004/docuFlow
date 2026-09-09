"""Phase 19 — ingestion platform 3.0.

Resource governor (file/page/extraction/memory/OCR/embedding limits), stage
priorities, stage-DAG validation (cycles rejected), durable stage
checkpoints, crash/timeout resume (only unfinished stages restart), document
quality scoring with human-readable explanations, and safe failure
explanations that never leak stack traces or secrets.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

STAGES = ("UPLOAD", "VALIDATE", "EXTRACT", "OCR", "LAYOUT", "CHUNK",
          "EMBED", "INDEX", "INTELLIGENCE", "READY")

STAGE_PRIORITIES = {
    "UPLOAD": "HIGH", "VALIDATE": "HIGH", "EXTRACT": "NORMAL",
    "OCR": "LOW", "LAYOUT": "NORMAL", "CHUNK": "NORMAL", "EMBED": "LOW",
    "INDEX": "NORMAL", "INTELLIGENCE": "LOW", "READY": "HIGH",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def resource_governor() -> dict:
    """Safe ingestion limits from environment (bounded, never unbounded)."""
    return {
        "max_file_size_mb": int(os.getenv("INGESTION_MAX_FILE_MB", "100")),
        "max_page_count": int(os.getenv("INGESTION_MAX_PAGES", "1000")),
        "max_extraction_seconds": int(
            os.getenv("INGESTION_MAX_EXTRACTION_S", "600")),
        "max_memory_mb": int(os.getenv("INGESTION_MAX_MEMORY_MB", "1024")),
        "max_ocr_pages": int(os.getenv("INGESTION_MAX_OCR_PAGES", "500")),
        "max_embedding_batch": int(
            os.getenv("INGESTION_MAX_EMBED_BATCH", "2000")),
    }


def governor_decision(*, file_size_mb: Optional[float] = None,
                      page_count: Optional[int] = None,
                      extraction_seconds: Optional[float] = None,
                      memory_mb: Optional[int] = None,
                      ocr_pages: Optional[int] = None,
                      limits: Optional[dict] = None) -> dict:
    """Deterministic resource-governor admission check."""
    limits = limits or resource_governor()
    violations = []
    checks = (
        ("file_size_mb", file_size_mb, limits["max_file_size_mb"], "file"),
        ("page_count", page_count, limits["max_page_count"], "pages"),
        ("extraction_seconds", extraction_seconds,
         limits["max_extraction_seconds"], "extraction time"),
        ("memory_mb", memory_mb, limits["max_memory_mb"], "memory"),
        ("ocr_pages", ocr_pages, limits["max_ocr_pages"], "OCR work"),
    )
    for name, value, limit, label in checks:
        if value is None:
            continue
        if value > limit:
            violations.append({"field": name, "value": value, "limit":
                               limit, "label": f"{label} exceeds limit"})
    if violations:
        return {"allowed": False, "code": "INGESTION_RESOURCE_LIMIT",
                "violations": violations}
    return {"allowed": True, "code": None, "violations": []}


def stage_priority(stage: str) -> str:
    return STAGE_PRIORITIES.get(stage.upper(), "NORMAL")


def validate_stage_dag(nodes: list[dict]) -> dict:
    """Validate an ingestion stage DAG: known stages, dependencies exist,
    no cycles. Returns the topologically ordered stages on success."""
    by_id = {node.get("id"): node for node in nodes}
    known = set(STAGES)
    for node in nodes:
        if node.get("id") not in known:
            return {"valid": False,
                    "reason": f"unknown stage {node.get('id')}"}
        for dep in node.get("depends_on", []):
            if dep not in by_id and dep not in known:
                return {"valid": False,
                        "reason": f"stage {node.get('id')} depends on "
                                  f"unknown {dep}"}
    # cycle detection (DFS)
    visiting, visited = set(), set()
    order = []

    def visit(node_id):
        if node_id in visiting:
            raise _CycleError(node_id)
        if node_id in visited:
            return
        visiting.add(node_id)
        node = by_id.get(node_id)
        if node:
            for dep in node.get("depends_on", []):
                if dep in by_id:
                    visit(dep)
        visiting.discard(node_id)
        visited.add(node_id)
        order.append(node_id)

    try:
        for node in nodes:
            visit(node["id"])
    except _CycleError as exc:
        return {"valid": False, "reason": f"cycle detected at {exc.stage}"}
    return {"valid": True, "order": order}


class _CycleError(Exception):
    def __init__(self, stage):
        super().__init__(f"cycle at {stage}")
        self.stage = stage


# ---------------------------------------------------------------------------
# Stage checkpoints + resume
# ---------------------------------------------------------------------------

def checkpoint_stage(db: Session, *, run_id: int, stage: str,
                     status: str = "COMPLETED",
                     output_reference: Optional[str] = None,
                     error: Optional[str] = None,
                     idempotency_key: Optional[str] = None) -> dict:
    """Persist a stage checkpoint. Restart only restarts unfinished stages."""
    from ..models.phase17 import IngestionRun, IngestionStage
    run = db.query(IngestionRun).get(run_id)
    if run is None:
        raise ValueError("ingestion run not found")
    stage = stage.upper()
    if stage not in STAGES:
        raise ValueError("unknown stage")
    row = (db.query(IngestionStage)
           .filter(IngestionStage.run_id == run_id,
                   IngestionStage.stage == stage).first())
    if row is None:
        row = IngestionStage(run_id=run_id, stage=stage)
        db.add(row)
    row.status = status
    row.attempts = (row.attempts or 0) + (1 if status == "RUNNING" else 0)
    row.output_reference = output_reference or row.output_reference
    row.error = error
    if idempotency_key:
        row.idempotency_key = idempotency_key
    if status in ("COMPLETED", "FAILED"):
        row.completed_at = _utcnow()
    elif status == "RUNNING":
        row.started_at = row.started_at or _utcnow()
    db.flush()
    # advance run progress
    index = STAGES.index(stage)
    run.current_stage = stage
    run.progress_pct = int(((index + 1) / len(STAGES)) * 100)
    if status == "FAILED":
        run.error = error
    db.flush()
    return {"run_id": run_id, "stage": stage, "status": status,
            "progress_pct": run.progress_pct}


def unfinished_stages(db: Session, run_id: int) -> list[str]:
    """Stages that have not durably completed (canonical order).

    A stage is unfinished when its checkpoint row is absent or not
    COMPLETED — restart only re-runs these stages.
    """
    from ..models.phase17 import IngestionStage
    completed = {
        row.stage for row in db.query(IngestionStage)
        .filter(IngestionStage.run_id == run_id,
                IngestionStage.status == "COMPLETED").all()
    }
    return [stage for stage in STAGES if stage not in completed]


def resume_run(db: Session, run_id: int) -> dict:
    """Resume after crash/timeout/provider outage: only unfinished stages."""
    from ..models.phase17 import IngestionRun
    run = db.query(IngestionRun).get(run_id)
    if run is None:
        raise ValueError("ingestion run not found")
    if run.status in ("COMPLETED", "CANCELLED"):
        return {"run_id": run_id, "status": "NOT_RESUMABLE",
                "detail": run.status}
    pending = unfinished_stages(db, run_id)
    if not pending:
        run.status = "COMPLETED"
        run.completed_at = _utcnow()
        run.progress_pct = 100
        db.flush()
        return {"run_id": run_id, "status": "COMPLETED",
                "restarted_stages": []}
    run.status = "RUNNING"
    run.retry_count = (run.retry_count or 0) + 1
    db.flush()
    return {"run_id": run_id, "status": "RUNNING",
            "restarted_stages": pending}


# ---------------------------------------------------------------------------
# Quality score + failure explanations
# ---------------------------------------------------------------------------

def record_quality(db: Session, *, workspace_id: int,
                   document_id: Optional[int] = None,
                   run_id: Optional[int] = None,
                   extraction_completeness: float = 1.0,
                   ocr_quality: Optional[float] = None,
                   metadata_completeness: float = 1.0,
                   chunk_quality: float = 1.0,
                   embedding_coverage: float = 1.0) -> dict:
    from ..models.phase19 import IngestionQualityReport
    quality = quality_score(extraction_completeness=extraction_completeness,
                            ocr_quality=ocr_quality,
                            metadata_completeness=metadata_completeness,
                            chunk_quality=chunk_quality,
                            embedding_coverage=embedding_coverage)
    row = IngestionQualityReport(
        workspace_id=workspace_id, document_id=document_id, run_id=run_id,
        extraction_completeness=round(extraction_completeness, 4),
        ocr_quality=round(ocr_quality, 4) if ocr_quality is not None else None,
        metadata_completeness=round(metadata_completeness, 4),
        chunk_quality=round(chunk_quality, 4),
        embedding_coverage=round(embedding_coverage, 4),
        quality_score=quality["score"],
        detail_json=json.dumps(quality["factors"]))
    db.add(row)
    db.flush()
    return {"report_id": row.id, **quality}


def quality_score(*, extraction_completeness: float,
                  ocr_quality: Optional[float],
                  metadata_completeness: float, chunk_quality: float,
                  embedding_coverage: float) -> dict:
    """Weighted quality indicators with per-factor explanations."""
    factors = {
        "extraction_completeness": round(extraction_completeness, 4),
        "ocr_quality": round(ocr_quality, 4) if ocr_quality is not None
        else None,
        "metadata_completeness": round(metadata_completeness, 4),
        "chunk_quality": round(chunk_quality, 4),
        "embedding_coverage": round(embedding_coverage, 4),
    }
    components = [("extraction_completeness", 0.25),
                  ("ocr_quality", 0.2), ("metadata_completeness", 0.1),
                  ("chunk_quality", 0.25), ("embedding_coverage", 0.2)]
    score = 0.0
    weights = 0.0
    for key, weight in components:
        value = factors[key]
        if value is None:
            continue
        score += value * weight
        weights += weight
    score = round(score / weights, 4) if weights else 0.0
    explanation = []
    if factors["ocr_quality"] is not None and factors["ocr_quality"] < 0.6:
        explanation.append("OCR quality is low; scanned text may be "
                           "unreliable")
    if factors["embedding_coverage"] < 0.9:
        explanation.append("some chunks are not embedded yet")
    if factors["metadata_completeness"] < 0.8:
        explanation.append("document metadata is incomplete")
    return {"score": score, "factors": factors,
            "explanation": explanation,
            "grade": "EXCELLENT" if score >= 0.9 else (
                "GOOD" if score >= 0.75 else (
                    "FAIR" if score >= 0.6 else "POOR"))}


def explain_failure(error_type: str, stage: Optional[str] = None,
                    detail: Optional[str] = None) -> dict:
    """Safe, human-readable failure explanation — never stack traces."""
    messages = {
        "timeout": "the stage exceeded its time budget and was retried "
                   "within bounds",
        "provider_unavailable": "the AI provider was unavailable; the stage "
                                "will retry with backoff",
        "file_too_large": "the file exceeds the configured size limit",
        "corrupt_document": "the document could not be parsed",
        "embedding_failed": "embedding generation failed for some chunks; "
                            "they will be retried without repeating OCR",
        "quarantined": "repeated failures moved the document to quarantine "
                       "for operator review",
        "unknown": "an unexpected error occurred; contact an administrator "
                   "with the request/correlation id",
    }
    message = messages.get(error_type, messages["unknown"])
    if stage:
        message = f"[{stage}] {message}"
    return {"error_type": error_type, "stage": stage, "message": message,
            "detail": (str(detail)[:200] if detail else None),
            "retryable": error_type in ("timeout", "provider_unavailable",
                                        "embedding_failed")}
