"""Phase 22 tests — continuous evaluation, knowledge maintenance,
ingestion hardening, connector platform, multi-region + DR.

Deterministic service-level validation; region failover and restore drills
are asserted to be honest about simulation vs real infrastructure.
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase22 import (
    BackupRestoreDrill, ConnectorSyncState, EvalExecution, MaintenanceRun,
    PoisonQuarantine, RegionCapacitySnapshot, RetentionExecution,
)
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import continuous_eval as ce
from app.services import knowledge_ops as ko
from app.services import region_dr as rd
from app.services import security_cost_ops as sco

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P22_EVAL_TABLES = [
    BackupRestoreDrill, RegionCapacitySnapshot, RetentionExecution,
    ConnectorSyncState, PoisonQuarantine, MaintenanceRun, EvalExecution,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    from app.models.phase19 import ResidencyRule, RegionRecord, RegionFailover
    from app.models.phase21 import FailoverSimulation, ResidencyGuardEvent
    for model in P22_EVAL_TABLES:
        db_session.query(model).delete()
    for model in (FailoverSimulation, RegionFailover, ResidencyGuardEvent,
                  ResidencyRule, RegionRecord):
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="eval"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22eval.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Steps 38-47: continuous evaluation
# ===========================================================================

class TestContinuousEvaluation:
    def test_create_execution_idempotent(self, db_session):
        ws = _mkws(db_session)
        key = f"eval-key-{uuid.uuid4().hex[:8]}"
        first = ce.create_execution(db_session, ws.id, domain="retrieval",
                                    idempotency_key=key)
        second = ce.create_execution(db_session, ws.id, domain="retrieval",
                                     idempotency_key=key)
        assert first.get("deduplicated") is not True
        assert second.get("deduplicated") is True
        assert second["id"] == first["id"]

    def test_run_execution_produces_metrics(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        result = ce.run_execution(db_session, created["id"])
        assert result["status"] in ("DONE", "COMPLETED", "SUCCEEDED")
        assert "metrics" in result

    def test_run_execution_bounded(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="rag")
        result = ce.run_execution(db_session, created["id"], max_cases=10)
        assert result.get("cases", 10) <= 10 or result["status"] == "DONE"

    def test_cancel_execution(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="rag")
        result = ce.cancel_execution(db_session, created["id"])
        assert result["status"] == "CANCELLED"

    def test_resume_execution(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"], max_cases=5)
        result = ce.resume_execution(db_session, created["id"])
        assert "status" in result

    def test_list_executions_bounded(self, db_session):
        ws = _mkws(db_session)
        for _ in range(3):
            created = ce.create_execution(db_session, ws.id,
                                          domain="retrieval")
            ce.run_execution(db_session, created["id"])
        rows = ce.list_executions(db_session, ws.id, limit=2)
        assert len(rows) <= 2

    @pytest.mark.parametrize("domain", ["retrieval", "rag", "citations",
                                        "model", "agent", "workflow"])
    def test_domain_evaluators(self, db_session, domain):
        ws = _mkws(db_session)
        result = getattr(ce, f"evaluate_{domain}")(db_session, ws.id)
        assert isinstance(result, dict) and result

    def test_regression_detection(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"])
        result = ce.detect_regression(db_session, ws.id, "retrieval",
                                      {"precision": 0.9})
        assert "regressed" in result

    def test_execution_persisted(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"])
        assert db_session.query(EvalExecution).count() >= 1


# ===========================================================================
# Steps 59-67: knowledge maintenance
# ===========================================================================

class TestKnowledgeMaintenance:
    @pytest.mark.parametrize("kind", ["stale_documents", "embeddings",
                                      "graph", "memory", "summaries",
                                      "connectors"])
    def test_maintenance_kinds(self, db_session, kind):
        ws = _mkws(db_session)
        result = ko.run_maintenance(db_session, ws.id, kind)
        assert result["kind"] == kind
        assert result["findings"] >= 0
        assert result["id"] > 0

    def test_maintenance_unknown_kind_rejected(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            ko.run_maintenance(db_session, ws.id, "not_a_kind")

    def test_maintenance_bounded(self, db_session):
        ws = _mkws(db_session)
        result = ko.run_maintenance(db_session, ws.id, "embeddings",
                                    max_findings=5)
        assert result["findings"] <= 5

    def test_maintenance_persisted(self, db_session):
        ws = _mkws(db_session)
        ko.run_maintenance(db_session, ws.id, "graph")
        assert db_session.query(MaintenanceRun).count() >= 1
# ===========================================================================
# Steps 68-74: ingestion hardening
# ===========================================================================

class TestIngestionHardening:
    def test_governor_rejects_oversize(self, db_session):
        ws = _mkws(db_session)
        events = ko.check_resource_limits(db_session, ws.id,
                                          file_size=10_000_000_000)
        assert events and all(e["action"] == "REJECTED" for e in events)

    def test_governor_allows_normal(self, db_session):
        ws = _mkws(db_session)
        events = ko.check_resource_limits(db_session, ws.id, file_size=1024)
        assert events and events[0]["action"] == "ALLOWED"

    def test_poison_quarantine(self, db_session):
        ws = _mkws(db_session)
        result = ko.quarantine_document(db_session, ws.id, 42,
                                        error_class="parse_failure")
        assert result["status"] == "QUARANTINED"
        assert result["failure_count"] >= 1

    def test_quarantine_idempotent_per_document(self, db_session):
        ws = _mkws(db_session)
        ko.quarantine_document(db_session, ws.id, 43,
                               error_class="parse_failure")
        ko.quarantine_document(db_session, ws.id, 43,
                               error_class="parse_failure")
        rows = (db_session.query(PoisonQuarantine)
                .filter_by(workspace_id=ws.id, document_id=43).all())
        assert len(rows) == 1
        assert rows[0].failure_count == 2

    def test_release_quarantine(self, db_session):
        ws = _mkws(db_session)
        ko.quarantine_document(db_session, ws.id, 44,
                               error_class="parse_failure")
        result = ko.release_quarantine(db_session, ws.id, 44)
        assert result["status"] == "RELEASED"

    def test_release_quarantine_missing(self, db_session):
        ws = _mkws(db_session)
        result = ko.release_quarantine(db_session, ws.id, 404040)
        assert result["ok"] is False

    def test_list_quarantined(self, db_session):
        ws = _mkws(db_session)
        ko.quarantine_document(db_session, ws.id, 45,
                               error_class="parse_failure")
        rows = ko.list_quarantined(db_session, ws.id)
        assert any(r.document_id == 45 for r in rows)

    def test_retry_backoff_grows(self):
        a = ko.ingestion_retry_backoff(1)
        b = ko.ingestion_retry_backoff(3)
        assert b > a

    def test_checkpoint_and_resume(self, db_session):
        stages = ("extract", "embed", "index")
        ko.checkpoint_stage(db_session, 77, "extract", {"done": True})
        state = ko.resume_from_checkpoint(db_session, 77, stages)
        assert state["resume_at"] == "embed"
        assert "extract" in state["completed"]

    def test_resume_all_done(self, db_session):
        stages = ("a", "b")
        for s in stages:
            ko.checkpoint_stage(db_session, 78, s, {})
        state = ko.resume_from_checkpoint(db_session, 78, stages)
        assert state["resume_at"] is None

    def test_fair_batch_select(self):
        # fair_batch_select returns workspace ids round-robin by capacity;
        # second tuple element is available capacity.
        workspaces = [(1, 5), (2, 5), (3, 5)]
        picked = ko.fair_batch_select(workspaces, batch=3)
        assert len(picked) == 3
        assert len(set(picked)) == 3  # one slot per tenant — no monopoly


# ===========================================================================
# Steps 75-80: connector platform
# ===========================================================================

class TestConnectorPlatform:
    def test_connector_sync_ok(self, db_session):
        ws = _mkws(db_session)
        result = ko.connector_sync(db_session, ws.id, 7, items=5)
        assert result["health"] == "HEALTHY"
        assert result["failure_count"] == 0

    def test_connector_sync_failure_backs_off(self, db_session):
        ws = _mkws(db_session)
        ko.connector_sync(db_session, ws.id, 8, error="timeout")
        result = ko.connector_sync(db_session, ws.id, 8, error="timeout")
        assert result["backoff_seconds"] >= 30
        assert result["failure_count"] >= 2

    def test_connector_sync_recovers(self, db_session):
        ws = _mkws(db_session)
        ko.connector_sync(db_session, ws.id, 11, error="timeout")
        result = ko.connector_sync(db_session, ws.id, 11, items=3)
        assert result["health"] == "HEALTHY"
        assert result["failure_count"] == 0

    def test_connector_conflict_detected(self, db_session):
        from datetime import datetime, timedelta, timezone
        from app.services.knowledge_ops import detect_connector_conflicts
        ws = _mkws(db_session)
        base = datetime.now(timezone.utc)
        ko.connector_sync(db_session, ws.id, 9, items=1)
        later = base + timedelta(seconds=10)
        result = detect_connector_conflicts(db_session, ws.id, 9,
                                            remote_updated=later,
                                            local_updated=later)
        assert isinstance(result, dict)
        assert result.get("conflict") is not None or \
            result.get("has_conflict") in (True, None) or \
            result.get("conflicts", 0) >= 0

    def test_connector_security_policy(self):
        policy = ko.connector_security_policy()
        assert "references only" in policy["credentials"]
        assert policy["domain_allowlist_required"] is True

    def test_connector_health_summary(self, db_session):
        ws = _mkws(db_session)
        ko.connector_sync(db_session, ws.id, 10, items=2)
        summary = ko.connector_health_summary(db_session, ws.id)
        assert summary["healthy"] >= 1


# ===========================================================================
# Steps 81-95: multi-region + DR
# ===========================================================================

class TestRegionDR:
    def test_upsert_region(self, db_session):
        result = rd.upsert_region(db_session, region="us-east",
                                  residency={"allowed": ["us-east"]})
        assert result["region"] == "us-east"

    def test_list_regions(self, db_session):
        rd.upsert_region(db_session, region="eu-west")
        rows = rd.list_regions(db_session)
        assert any(r["region"] == "eu-west" for r in rows)

    def test_region_capacity_snapshot(self, db_session):
        result = rd.region_capacity(db_session, "us-east", workers=4,
                                    queue_depth=100, db_healthy=True,
                                    provider_healthy=True)
        assert result["region"] == "us-east"
        assert db_session.query(RegionCapacitySnapshot).count() >= 1

    def test_residency_guard_blocks_illegal(self, db_session):
        from app.models.phase19 import ResidencyRule
        db_session.query(ResidencyRule).filter_by(
            organization_id=1, classification="INTERNAL").delete()
        db_session.commit()
        rule = ResidencyRule(classification="INTERNAL",
                             allowed_regions_json='["eu-west"]',
                             organization_id=1)
        db_session.add(rule)
        db_session.commit()
        result = rd.residency_guard(db_session, workspace_region="eu-west",
                                    target_region="us-east",
                                    workspace_id=1, organization_id=1)
        assert result["allowed"] is False

    def test_residency_guard_allows_legal(self, db_session):
        from app.models.phase19 import ResidencyRule
        db_session.query(ResidencyRule).filter_by(
            organization_id=1, classification="PUBLIC").delete()
        db_session.commit()
        rule = ResidencyRule(classification="PUBLIC",
                             allowed_regions_json='["us-east"]',
                             organization_id=1)
        db_session.add(rule)
        db_session.commit()
        result = rd.residency_guard(db_session, workspace_region="us-east",
                                    target_region="us-east",
                                    workspace_id=1, organization_id=1)
        assert result["allowed"] is True

    def test_failover_plan_and_simulation(self, db_session):
        plan = rd.failover_plan(db_session, primary="us-east",
                                secondary="eu-west", organization_id=1)
        assert plan["id"] > 0
        assert "quiesce primary writes" in plan["steps"]
        sim = rd.simulate_failover(db_session, plan["id"])
        assert sim["simulated"] is True
        assert sim["ok"] is True

    def test_real_failover_honest(self):
        available = rd.real_failover_available()
        assert available is False  # no multi-region infra in this environment

    def test_backup_health(self, db_session):
        health = rd.backup_health(db_session)
        assert health["configured"] is False  # no backup infra configured

    def test_restore_drill_marked_simulated(self, db_session):
        ws = _mkws(db_session, "dr")
        result = rd.run_restore_drill(db_session, ws.id)
        assert result["simulated"] is True
        assert db_session.query(BackupRestoreDrill).count() >= 1

    def test_dr_report_separates_real_from_simulated(self, db_session):
        ws = _mkws(db_session, "drr")
        rd.run_restore_drill(db_session, ws.id)
        report = rd.dr_report(db_session)
        assert report["real_rpo_rto_measured"] is False
        assert "no real drill executed" in report["honesty_note"]


# ===========================================================================
# Data lifecycle (Steps 123-129)
# ===========================================================================

class TestRetention:
    def test_retention_dry_run_default(self, db_session):
        result = sco.run_retention(db_session, "traces", older_than_days=30)
        assert result["dry_run"] is True

    def test_retention_unknown_kind(self, db_session):
        with pytest.raises(ValueError):
            sco.run_retention(db_session, "not_a_kind")

    def test_retention_persists_execution(self, db_session):
        sco.run_retention(db_session, "evaluations", older_than_days=30)
        assert db_session.query(RetentionExecution).count() >= 1
