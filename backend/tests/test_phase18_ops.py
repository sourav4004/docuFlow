"""Phase 18 tests — observability, cost control, search platform, data
operations, cleanup, disaster recovery.

Covers: sensitive redaction, SLO snapshots/status, correlation chains;
cost attribution/forecast/budget enforcement; unified search planner and
authorized analytics; import validation/dry-run/commit and export
isolation; bounded hold-aware cleanup; backup inventory and restore
validation.
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.organization import Organization  # noqa: E402
from app.models.ai_execution import AIExecution  # noqa: E402
from app.models.phase18 import (  # noqa: E402
    SloSnapshot, SearchAnalyticsEvent, ImportJob,
)
from app.services import (  # noqa: E402
    observability2 as obs, cost3, search4, dataops, cleanup_ops, dr,
)

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(SloSnapshot).delete()
    db_session.query(SearchAnalyticsEvent).delete()
    db_session.query(ImportJob).delete()
    db_session.query(AIExecution).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p18ops"):
    _counter[0] += 1
    user = User(name=f"P18 OPS {_counter[0]}",
                email=f"{tag}{_counter[0]}@p18-ops.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_org(db, user):
    org = Organization(name=f"p18 ops org {_counter[0]}",
                       slug=f"p18ops-{_counter[0]}-{uuid.uuid4().hex[:6]}",
                       owner_id=user.id)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def fresh_workspace(db, user, org=None):
    _counter[0] += 1
    ws = Workspace(name=f"p18 ops ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    if org is not None:
        ws.organization_id = org.id
        db.commit()
    return ws


def fresh_execution(db, ws, user, model="gpt-4o", cost=1.0, tokens=100,
                    execution_type="rag"):
    execution = AIExecution(
        id=str(uuid.uuid4()), workspace_id=ws.id, user_id=user.id,
        organization_id=ws.organization_id,
        execution_type=execution_type, task_type="test", status="COMPLETED",
        model=model, estimated_cost=cost, total_tokens=tokens,
        created_at=datetime.now(timezone.utc))
    db.add(execution)
    db.commit()
    return execution


# ============================================================
# Observability
# ============================================================

class TestObservability:
    def test_redact_api_key(self):
        out = obs.redact("key=sk-abcdefghijklmnop123456")
        assert "sk-" not in out
        assert "[REDACTED]" in out

    def test_redact_authorization_header(self):
        out = obs.redact("Authorization: Bearer abc.def.ghi12345")
        assert "Bearer" not in out

    def test_redact_password(self):
        out = obs.redact("password=hunter2secretvalue")
        assert "hunter2" not in out

    def test_redact_aws_key(self):
        out = obs.redact("AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in out

    def test_redact_leaves_plain_text(self):
        text = "Revenue grew 20% this quarter."
        assert obs.redact(text) == text

    def test_redact_empty(self):
        assert obs.redact("") == ""
        assert obs.redact(None) == ""

    def test_structured_record_redacts(self):
        record = obs.structured_record("info", "provider_call",
                                       api_key="sk-secret1234567890123456",
                                       latency_ms=42)
        assert record["event"] == "provider_call"
        assert "sk-secret1234567890123456" not in record["api_key"]

    def test_propagate_ids_chain(self):
        chain = obs.propagate_ids({"request_id": "r1", "execution_id": "e1",
                                   "job_id": "j1"})
        assert chain["correlation_id"] == "r1"
        assert chain["chain"]["execution_id"] == "e1"

    def test_propagate_ids_order(self):
        chain = obs.propagate_ids({"workflow_id": "w1"})
        assert chain["correlation_id"] == "w1"

    def test_record_slo(self, db_session):
        snap = obs.record_slo(db_session, availability=0.995,
                              latency_p95_ms=120.0, error_rate=0.01)
        db_session.commit()
        assert snap.id is not None
        assert snap.availability == 0.995

    def test_slo_status_healthy(self, db_session):
        for _ in range(3):
            obs.record_slo(db_session, availability=0.999, error_rate=0.001)
        db_session.commit()
        status = obs.slo_status(db_session)
        assert status["status"] == "HEALTHY"
        assert status["availability"] == 0.999

    def test_slo_status_breached(self, db_session):
        for _ in range(3):
            obs.record_slo(db_session, availability=0.50, error_rate=0.9)
        db_session.commit()
        status = obs.slo_status(db_session)
        assert status["status"] == "BREACHED"

    def test_slo_status_no_data(self, db_session):
        status = obs.slo_status(db_session)
        assert status["status"] == "NO_DATA"


# ============================================================
# Cost control
# ============================================================

class TestCost3:
    def test_attribution_empty(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        result = cost3.attribution(db_session, organization_id=org.id)
        assert result["total_cost"] == 0.0

    def test_attribution_totals(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        fresh_execution(db_session, ws, user, model="gpt-4o", cost=2.5,
                        execution_type="rag")
        fresh_execution(db_session, ws, user, model="claude-3", cost=1.5,
                        execution_type="agent")
        result = cost3.attribution(db_session, organization_id=org.id)
        assert result["total_cost"] == 4.0
        assert result["total_tokens"] == 200
        assert len(result["by_model"]) == 2

    def test_attribution_by_feature(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        fresh_execution(db_session, ws, user, execution_type="workflow")
        result = cost3.attribution(db_session, organization_id=org.id)
        assert result["by_feature"][0]["key"] == "workflow"

    def test_forecast_labels_estimate(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        for _ in range(3):
            fresh_execution(db_session, ws, user, cost=1.0)
        result = cost3.forecast(db_session, organization_id=org.id)
        assert result["is_estimate"] is True
        assert result["assumptions"]
        assert result["projected_30d_cost_est"] >= 0

    def test_budget_allow(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        result = cost3.enforce_budget(
            db_session, organization_id=org.id, workspace_id=ws.id,
            feature="rag", estimated_cost=1.0, hard_limit=100.0)
        assert result["action"] == "ALLOW"

    def test_budget_hard_block(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        fresh_execution(db_session, ws, user, cost=50.0)
        result = cost3.enforce_budget(
            db_session, organization_id=org.id, workspace_id=ws.id,
            feature="rag", estimated_cost=60.0, hard_limit=100.0)
        assert result["action"] == "BLOCK"

    def test_budget_soft_requires_approval(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        fresh_execution(db_session, ws, user, cost=50.0)
        result = cost3.enforce_budget(
            db_session, organization_id=org.id, workspace_id=ws.id,
            feature="rag", estimated_cost=20.0, soft_limit=60.0)
        assert result["action"] == "REQUIRE_APPROVAL"

    def test_budget_org_isolation(self, db_session):
        user = fresh_user(db_session)
        org1 = fresh_org(db_session, user)
        org2 = fresh_org(db_session, user)
        ws1 = fresh_workspace(db_session, user, org1)
        ws2 = fresh_workspace(db_session, user, org2)
        fresh_execution(db_session, ws1, user, cost=90.0)
        # Org2's usage is independent of org1's spend.
        result = cost3.enforce_budget(
            db_session, organization_id=org2.id, workspace_id=ws2.id,
            feature="rag", estimated_cost=1.0, hard_limit=50.0)
        assert result["action"] == "ALLOW"


# ============================================================
# Search platform
# ============================================================

class TestSearch4:
    def test_unified_plan_hybrid(self):
        plan = search4.unified_plan("quarterly revenue summary")
        assert plan["mode"] == "hybrid"

    def test_unified_plan_graph(self):
        plan = search4.unified_plan("acme", {"entity_id": 3})
        assert plan["mode"] == "graph"

    def test_unified_plan_memory(self):
        plan = search4.unified_plan("policy", {"memory": True})
        assert plan["mode"] == "memory"

    def test_unified_plan_metadata(self):
        plan = search4.unified_plan("documents", {"metadata": {"kind": "pdf"}})
        assert plan["mode"] == "metadata"

    def test_unified_plan_no_cot(self):
        plan = search4.unified_plan("who signed the contract?")
        explanation = json_dumps(plan["explanation"])
        # Never reveals reasoning/chain-of-thought.
        assert "because" not in explanation.lower()

    def test_explain_results(self):
        plan = search4.unified_plan("query")
        results = [{"document_id": 1, "chunk_id": 2, "score": 0.8,
                    "source": "document"}]
        explanation = search4.explain_results(plan, results)
        assert explanation["total"] == 1
        assert explanation["top_results"][0]["document_id"] == 1
        assert "chain-of-thought" not in json_dumps(explanation).lower()

    def test_query_hash_deterministic(self):
        assert search4.query_hash("Query Text") == \
            search4.query_hash("  query text  ")
        assert search4.query_hash("a") != search4.query_hash("b")

    def test_record_search(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = search4.record_search(db_session, workspace_id=ws.id,
                                    organization_id=None,
                                    query="revenue report",
                                    mode="hybrid", latency_ms=120,
                                    result_count=3)
        db_session.commit()
        assert row.zero_results is False
        assert row.query_hash == search4.query_hash("revenue report")

    def test_record_zero_result_search(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search4.record_search(db_session, workspace_id=ws.id,
                              organization_id=None, query="zzz",
                              mode="keyword", latency_ms=5, result_count=0)
        db_session.commit()
        analytics = search4.search_analytics(db_session, ws.id)
        assert analytics["searches"] == 1
        assert analytics["zero_result_rate"] == 1.0

    def test_search_analytics_aggregation(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for i in range(5):
            search4.record_search(db_session, workspace_id=ws.id,
                                  organization_id=None,
                                  query=f"query number {i}",
                                  mode="hybrid", latency_ms=100 + i,
                                  result_count=2)
        db_session.commit()
        analytics = search4.search_analytics(db_session, ws.id)
        assert analytics["searches"] == 5
        assert analytics["zero_result_rate"] == 0.0
        assert analytics["p95_latency_ms"] is not None
        assert "hybrid" in analytics["by_mode"]

    def test_search_analytics_workspace_isolated(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        search4.record_search(db_session, workspace_id=ws1.id,
                              organization_id=None, query="q",
                              mode="hybrid", latency_ms=10, result_count=1)
        db_session.commit()
        assert search4.search_analytics(db_session, ws2.id)["searches"] == 0


# ============================================================
# Data operations
# ============================================================

class TestDataOps:
    def test_import_valid_documents(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "Doc A", "workspace_id": ws.id},
                     {"title": "Doc B"}])
        db_session.commit()
        assert job.status == "READY"
        assert job.records_valid == 2
        assert job.records_invalid == 0

    def test_import_rejects_invalid_type(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            dataops.create_import_job(
                db_session, workspace_id=ws.id, organization_id=None,
                user_id=user.id, import_type="executables", records=[])

    def test_import_rejects_missing_title(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"no_title": True}])
        db_session.commit()
        assert job.status == "VALIDATION_FAILED"
        assert job.records_invalid == 1

    def test_import_rejects_cross_workspace(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws1.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "Bad", "workspace_id": ws2.id}])
        db_session.commit()
        assert job.records_invalid == 1
        errors = dataops._errors(job)
        assert "cross-workspace" in errors[0]["error"]

    def test_import_oversized_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            dataops.create_import_job(
                db_session, workspace_id=ws.id, organization_id=None,
                user_id=user.id, import_type="documents",
                records=[{"title": f"x{i}"}
                         for i in range(dataops.MAX_RECORDS + 1)])

    def test_import_preview(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="metadata",
            records=[{"document_id": 1}])
        db_session.commit()
        preview = dataops.import_preview(db_session, workspace_id=ws.id,
                                         job_id=job.id)
        assert preview["status"] == "READY"
        assert preview["records_valid"] == 1

    def test_import_preview_cross_workspace_denied(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws1.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "A"}])
        db_session.commit()
        with pytest.raises(KeyError):
            dataops.import_preview(db_session, workspace_id=ws2.id,
                                   job_id=job.id)

    def test_commit_import_not_ready_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"bad": True}])
        db_session.commit()
        assert job.status == "VALIDATION_FAILED"
        with pytest.raises(ValueError):
            dataops.commit_import(db_session, workspace_id=ws.id,
                                  job_id=job.id, user_id=user.id)

    def test_commit_import_dry_run(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "A"}], dry_run=True)
        db_session.commit()
        result = dataops.commit_import(db_session, workspace_id=ws.id,
                                       job_id=job.id, user_id=user.id)
        db_session.commit()
        assert result["committed"] is False

    def test_rollback_import(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "A"}])
        db_session.commit()
        result = dataops.rollback_import(db_session, workspace_id=ws.id,
                                         job_id=job.id, user_id=user.id)
        db_session.commit()
        assert result["status"] == "ROLLED_BACK"

    def test_rollback_committed_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = dataops.create_import_job(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, import_type="documents",
            records=[{"title": "A"}], dry_run=False)
        db_session.commit()
        job.status = "COMMITTED"
        db_session.commit()
        with pytest.raises(ValueError):
            dataops.rollback_import(db_session, workspace_id=ws.id,
                                    job_id=job.id, user_id=user.id)

    def test_export_manifest_workspace_scoped(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        manifest = dataops.export_manifest(
            db_session, workspace_id=ws.id, organization_id=None,
            include=["documents", "usage"])
        assert manifest["workspace_id"] == ws.id
        assert "documents" in manifest["counts"]
        assert manifest["isolation"] == "workspace-scoped"


# ============================================================
# Cleanup
# ============================================================

class TestCleanup:
    def test_cleanup_candidates_traces(self, db_session):
        from app.models.phase16 import TraceSpan
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        db_session.add(TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws.id, span_type="provider",
            started_at=datetime.now(timezone.utc) - timedelta(days=400),
            created_at=datetime.now(timezone.utc) - timedelta(days=400)))
        db_session.commit()
        candidates = cleanup_ops.cleanup_candidates(
            db_session, entity_type="traces", workspace_id=ws.id,
            older_than_days=365)
        assert len(candidates) == 1

    def test_cleanup_candidates_excludes_recent(self, db_session):
        from app.models.phase16 import TraceSpan
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        db_session.add(TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws.id, span_type="provider",
            started_at=datetime.now(timezone.utc)))
        db_session.commit()
        candidates = cleanup_ops.cleanup_candidates(
            db_session, entity_type="traces", workspace_id=ws.id,
            older_than_days=30)
        assert candidates == []

    def test_cleanup_unknown_type_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            cleanup_ops.cleanup_candidates(
                db_session, entity_type="everything", workspace_id=ws.id)

    def test_run_cleanup_dry_run_default(self, db_session):
        from app.models.phase16 import TraceSpan
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws.id, span_type="provider",
            started_at=datetime.now(timezone.utc) - timedelta(days=400),
            created_at=datetime.now(timezone.utc) - timedelta(days=400))
        db_session.add(span)
        db_session.commit()
        result = cleanup_ops.run_bounded_cleanup(
            db_session, entity_type="traces", workspace_id=ws.id,
            older_than_days=365)
        db_session.commit()
        assert result["candidates"] == 1
        assert result["deleted"] == 0  # dry run by default
        assert cleanup_ops._under_hold(db_session, ws.id, "traces", span.id) \
            is False

    def test_run_cleanup_actual_delete(self, db_session):
        from app.models.phase16 import TraceSpan
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        db_session.add(TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws.id, span_type="provider",
            started_at=datetime.now(timezone.utc) - timedelta(days=400),
            created_at=datetime.now(timezone.utc) - timedelta(days=400)))
        db_session.commit()
        result = cleanup_ops.run_bounded_cleanup(
            db_session, entity_type="traces", workspace_id=ws.id,
            older_than_days=365, dry_run=False)
        db_session.commit()
        assert result["deleted"] == 1

    def test_run_cleanup_respects_hold(self, db_session):
        from app.models.phase16 import TraceSpan
        from app.models.phase17 import LegalHold, HoldEntity
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws.id, span_type="provider",
            started_at=datetime.now(timezone.utc) - timedelta(days=400),
            created_at=datetime.now(timezone.utc) - timedelta(days=400))
        db_session.add(span)
        db_session.commit()
        hold = LegalHold(workspace_id=ws.id, name="litigation",
                         status="ACTIVE")
        db_session.add(hold)
        db_session.commit()
        db_session.add(HoldEntity(hold_id=hold.id, entity_type="traces",
                                  entity_id=span.id))
        db_session.commit()
        result = cleanup_ops.run_bounded_cleanup(
            db_session, entity_type="traces", workspace_id=ws.id,
            older_than_days=365, dry_run=False)
        db_session.commit()
        assert result["held_skipped"] == 1
        assert result["deleted"] == 0

    def test_enqueue_cleanup(self, db_session):
        from app.models.phase16 import WorkerJob
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cleanup_ops.enqueue_cleanup(
            db_session, entity_type="traces", workspace_id=ws.id)
        db_session.commit()
        assert result["enqueued"] is True
        job = db_session.query(WorkerJob).filter(
            WorkerJob.job_type == "RETENTION_CLEANUP").first()
        assert job is not None

    def test_cleanup_dead_letters(self, db_session):
        from app.models.phase16 import WorkerJob
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        db_session.add(WorkerJob(
            queue_name="AI_TASKS", job_type="test", status="DEAD_LETTERED",
            workspace_id=ws.id,
            created_at=datetime.now(timezone.utc) - timedelta(days=400),
            completed_at=datetime.now(timezone.utc) - timedelta(days=400)))
        db_session.commit()
        candidates = cleanup_ops.cleanup_candidates(
            db_session, entity_type="dead_letters", workspace_id=ws.id,
            older_than_days=365)
        assert len(candidates) == 1


# ============================================================
# Disaster recovery
# ============================================================

class TestDR:
    def test_backup_inventory_shape(self, db_session):
        inventory = dr.backup_inventory(db_session)
        assert inventory["database"]["engine"] == "postgresql"
        assert "migrations" in inventory
        assert "secrets are referenced" in inventory["note"]

    def test_validate_restore(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = dr.validate_restore(db_session)
        assert "migration_in_sync" in result
        assert "authorization_orphans" in result
        assert result["tenant_data"]["users"] >= 1

    def test_tenant_isolation_after_restore(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = dr.tenant_isolation_after_restore(db_session)
        assert result["cross_tenant_anomalies"] == 0


def json_dumps(value) -> str:
    import json
    return json.dumps(value, default=str)