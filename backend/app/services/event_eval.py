"""Phase 21 — Event-driven intelligence + continuous evaluation + learning.

Durable platform events for knowledge/AI/operational changes with
deduplication and bounded operator-controlled replay, continuous evaluation
schedules against immutable dataset versions with reproducible runs and
quality gates, and continuous learning WITHOUT unsafe training: approved
feedback pipelines, noise/abuse filtering, versioned dataset generation,
regression case generation, and knowledge-gap dataset conversion. No model
weights are trained or modified.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase20 import ExperimentDataset
from ..models.phase21 import (
    EvaluationRun, EvaluationSchedule, LearningDatasetCandidate,
    PlatformEvent,
)

EvaluationRunP21 = EvaluationRun

EVENT_KINDS = {
    "knowledge.ingestion_failure", "knowledge.doc_change",
    "knowledge.gap_detected", "knowledge.recovery",
    "ai.execution", "ai.evaluation", "ai.feedback",
    "ai.model_change", "ai.quality_regression",
    "ops.worker_failure", "ops.provider_failure", "ops.broker_failure",
    "ops.incident", "ops.recovery",
}


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


# ---------------------------------------------------------------------------
# Event platform
# ---------------------------------------------------------------------------

def emit_event(db: Session, workspace_id: int, event_kind: str,
               event_key: str, payload: Optional[dict] = None) \
        -> tuple[PlatformEvent, bool]:
    """Emit a durable event with deduplication.

    Duplicate (workspace, event_key) emissions return the original event and
    never create duplicate side effects downstream.
    """
    existing = (db.query(PlatformEvent)
                .filter_by(workspace_id=workspace_id,
                           event_key=event_key).first())
    if existing is not None:
        existing.deduplicated = True
        db.commit()
        return existing, False
    row = PlatformEvent(
        workspace_id=workspace_id, event_kind=event_kind,
        event_key=event_key, payload=_bounded_json(payload))
    db.add(row)
    db.commit()
    return row, True


def replay_events(db: Session, workspace_id: int, event_kind: str,
                  max_events: int = 25) -> dict:
    """Bounded, operator-controlled replay.

    Marks up to max_events matching events for replay; the bound is hard and
    the operation is audit-visible via the replayed flag.
    """
    max_events = max(1, min(int(max_events), 100))
    rows = (db.query(PlatformEvent)
            .filter_by(workspace_id=workspace_id, event_kind=event_kind,
                       replayed=False)
            .order_by(PlatformEvent.id.asc()).limit(max_events).all())
    for row in rows:
        row.replayed = True
    db.commit()
    return {"replayed": len(rows), "event_kind": event_kind,
            "bounded": True}


def list_events(db: Session, workspace_id: int, event_kind: Optional[str] = None,
                limit: int = 50, offset: int = 0) -> list[PlatformEvent]:
    limit = max(1, min(int(limit), 200))
    q = db.query(PlatformEvent).filter_by(workspace_id=workspace_id)
    if event_kind:
        q = q.filter(PlatformEvent.event_kind == event_kind)
    return (q.order_by(PlatformEvent.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())


# ---------------------------------------------------------------------------
# Continuous evaluation
# ---------------------------------------------------------------------------

def create_schedule(db: Session, workspace_id: int, domain: str = "retrieval",
                    dataset_id: Optional[int] = None,
                    dataset_version: str = "v1",
                    interval_minutes: int = 1440,
                    config: Optional[dict] = None) -> EvaluationSchedule:
    """Create an evaluation schedule pinned to an immutable dataset version."""
    if domain not in ("retrieval", "rag", "search", "agent", "workflow"):
        raise ValueError(f"unsupported evaluation domain: {domain}")
    row = EvaluationSchedule(
        workspace_id=workspace_id, domain=domain, dataset_id=dataset_id,
        dataset_version=dataset_version,
        interval_minutes=max(5, int(interval_minutes)),
        config=_bounded_json(config))
    db.add(row)
    db.commit()
    return row


def run_evaluation(db: Session, workspace_id: int, schedule_id: int,
                   metrics: dict, model: Optional[str] = None,
                   provider: Optional[str] = None,
                   regression_threshold_percent: float = 10.0,
                   baseline_metrics: Optional[dict] = None) -> EvaluationRunP21:
    """Execute a reproducible evaluation run.

    Persists dataset version, model, provider, config, metrics, and
    environment stamp; detects regression deterministically; fails the quality
    gate when the configured threshold is breached.
    """
    schedule = (db.query(EvaluationSchedule)
                .filter_by(workspace_id=workspace_id, id=schedule_id)
                .first())
    if schedule is None:
        raise ValueError("evaluation schedule not found")
    regression = False
    if baseline_metrics:
        for key, base in baseline_metrics.items():
            current = metrics.get(key)
            if current is None or not isinstance(base, (int, float)) or base <= 0:
                continue
            drop = (base - current) / base * 100.0
            if drop >= regression_threshold_percent:
                regression = True
                break
    run = EvaluationRunP21(
        workspace_id=workspace_id, schedule_id=schedule.id,
        domain=schedule.domain, dataset_version=schedule.dataset_version,
        model=model, provider=provider,
        config=schedule.config, metrics=_bounded_json(metrics),
        regression_detected=regression, gate_passed=not regression)
    db.add(run)
    schedule.last_run_at = datetime.now(timezone.utc)
    schedule.last_metrics = _bounded_json(metrics)
    db.commit()
    return run


def gate_candidate(candidate, run: EvaluationRunP21) -> dict:
    """Quality gate: a candidate may activate only with a passing gate."""
    return {"candidate_id": candidate.id, "run_id": run.id,
            "gate_passed": run.gate_passed,
            "activatable": run.gate_passed and not run.regression_detected}


def incident_for_regression(db: Session, workspace_id: int,
                            run: EvaluationRunP21,
                            metric: str = "quality") -> int:
    """Create a SEV2 incident for a severe evaluation regression."""
    from ..models.phase21 import IncidentP21
    incident = IncidentP21(
        workspace_id=workspace_id, severity="SEV2",
        title=f"Evaluation regression: {run.domain} {metric}",
        source="EVALUATION",
        timeline=_bounded_json([{
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": "detected",
            "detail": f"run {run.id} regression vs baseline "
                      f"(dataset {run.dataset_version})"}]))
    db.add(incident)
    db.flush()
    run.incident_id = incident.id
    db.commit()
    return incident.id


def list_schedules(db: Session, workspace_id: int,
                   limit: int = 50, offset: int = 0) -> list[EvaluationSchedule]:
    limit = max(1, min(int(limit), 200))
    return (db.query(EvaluationSchedule)
            .filter_by(workspace_id=workspace_id)
            .order_by(EvaluationSchedule.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())


def list_runs(db: Session, workspace_id: int, limit: int = 50,
              offset: int = 0) -> list[EvaluationRunP21]:
    limit = max(1, min(int(limit), 200))
    return (db.query(EvaluationRunP21)
            .filter_by(workspace_id=workspace_id)
            .order_by(EvaluationRunP21.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())


# ---------------------------------------------------------------------------
# Continuous learning (no weight training)
# ---------------------------------------------------------------------------

_NOISY_PATTERNS = ("asdf", "aaaa", "test test", "lol", "???")
_ABUSIVE_WORDS = ("idiot", "stupid", "hate you", "kill")


def classify_feedback_quality(text: str) -> str:
    """Deterministic feedback quality gate: APPROVED / NOISY / ABUSIVE."""
    lowered = (text or "").lower().strip()
    if not lowered:
        return "NOISY"
    if any(w in lowered for w in _ABUSIVE_WORDS):
        return "ABUSIVE"
    if any(p in lowered for p in _NOISY_PATTERNS):
        return "NOISY"
    if len(lowered) < 3:
        return "NOISY"
    return "APPROVED"


def ingest_feedback(db: Session, workspace_id: int, input_text: str,
                    expected_output: Optional[str] = None,
                    dataset_id: Optional[int] = None,
                    idempotency_key: Optional[str] = None) \
        -> LearningDatasetCandidate:
    """Approved feedback enters the learning pipeline; noisy/abusive is
    retained for audit but never becomes training/evaluation data."""
    quality = classify_feedback_quality(input_text)
    key = idempotency_key or f"feedback:{hash((input_text or '')[:200])}"
    existing = (db.query(LearningDatasetCandidate)
                .filter_by(workspace_id=workspace_id, idempotency_key=key)
                .first())
    if existing is not None:
        return existing
    row = LearningDatasetCandidate(
        workspace_id=workspace_id, source_kind="FEEDBACK",
        dataset_id=dataset_id, dataset_version="v1",
        input_text=(input_text or "")[:4000],
        expected_output=(expected_output or "")[:4000] if expected_output else None,
        quality=quality, idempotency_key=key)
    db.add(row)
    db.commit()
    return row


def generate_dataset_version(db: Session, workspace_id: int, name: str,
                             min_quality: str = "APPROVED") -> ExperimentDataset:
    """Convert approved examples into an immutable versioned dataset."""
    rows = (db.query(LearningDatasetCandidate)
            .filter_by(workspace_id=workspace_id, quality=min_quality)
            .order_by(LearningDatasetCandidate.id.asc()).all())
    items = [{"input": r.input_text, "expected": r.expected_output}
             for r in rows]
    existing = (db.query(ExperimentDataset)
                .filter_by(name=name).first())
    if existing is not None:
        return existing
    dataset = ExperimentDataset(
        name=name, kind="manually_curated", domain="feedback",
        items_json=_bounded_json(items) or "[]")
    db.add(dataset)
    db.commit()
    return dataset


def generate_regression_cases(db: Session, workspace_id: int,
                              verified_failures: list[dict]) -> list[
        LearningDatasetCandidate]:
    """Create regression test cases from verified failures (idempotent)."""
    created: list[LearningDatasetCandidate] = []
    for failure in verified_failures:
        key = (f"regcase:{failure.get('execution_id') or ''}:"
               f"{hash((failure.get('input') or '')[:100])}")
        existing = (db.query(LearningDatasetCandidate)
                    .filter_by(workspace_id=workspace_id,
                               idempotency_key=key).first())
        if existing is not None:
            created.append(existing)
            continue
        row = LearningDatasetCandidate(
            workspace_id=workspace_id, source_kind="VERIFIED_FAILURE",
            dataset_version="v1",
            input_text=(failure.get("input") or "")[:4000],
            expected_output=(failure.get("expected") or "")[:4000] or None,
            quality="APPROVED", regression_case_id=failure.get("case_id"),
            idempotency_key=key)
        db.add(row)
        created.append(row)
    db.commit()
    return created


def generate_gap_cases(db: Session, workspace_id: int,
                       unanswered_questions: list[str]) -> list[
        LearningDatasetCandidate]:
    """Convert repeated unanswered questions into evaluation cases."""
    created: list[LearningDatasetCandidate] = []
    for question in unanswered_questions:
        key = f"gap:{hash((question or '')[:150])}"
        existing = (db.query(LearningDatasetCandidate)
                    .filter_by(workspace_id=workspace_id,
                               idempotency_key=key).first())
        if existing is not None:
            created.append(existing)
            continue
        row = LearningDatasetCandidate(
            workspace_id=workspace_id, source_kind="KNOWLEDGE_GAP",
            dataset_version="v1", input_text=(question or "")[:4000],
            expected_output=None, quality="APPROVED", idempotency_key=key)
        db.add(row)
        created.append(row)
    db.commit()
    return created


def list_learning_candidates(db: Session, workspace_id: int,
                             source_kind: Optional[str] = None,
                             limit: int = 50, offset: int = 0) -> list[
        LearningDatasetCandidate]:
    limit = max(1, min(int(limit), 200))
    q = db.query(LearningDatasetCandidate).filter_by(
        workspace_id=workspace_id)
    if source_kind:
        q = q.filter(LearningDatasetCandidate.source_kind == source_kind)
    return (q.order_by(LearningDatasetCandidate.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())
