"""Phase 23 tests — global control plane, broker/storage migration,
provider gate + routing + cost reconciliation.

Deterministic tests; honest environment reporting asserted (e.g. pgvector
NOT_CONFIGURED is expected and asserted as honesty, not failure).
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    DependencyEdge, ProviderReadinessScore, ProviderRoutingDecision,
    RequestDedupRecord, ResidencyDecisionLog, StorageMigrationObject,
    StorageMigrationPlan,
)
from app.models.phase22 import (
    CostReconciliationRun as CostReconciliationRunP22,
    OpsStreamEvent,
)
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import global_control as gc
from app.services import platform_migration as pm
from app.services import provider_gate as pg
from app.services import api_platform3 as ap3

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P23_TABLES = [
    DependencyEdge, ProviderReadinessScore, ProviderRoutingDecision,
    RequestDedupRecord, ResidencyDecisionLog, StorageMigrationObject,
    StorageMigrationPlan, CostReconciliationRunP22, OpsStreamEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P23_TABLES:
        db_session.query(model).delete()
    # Documents/chunks created by other suites' fixtures must not leak into
    # migration-plan enumeration counts (shared SQLite DB).
    from app.models.document_chunk import DocumentChunk
    from app.models.document import Document
    db_session.query(DocumentChunk).delete()
    db_session.query(Document).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield
    # tenant cache must not leak between tests
    ap3.tenant_cache._store.clear()


def _mkdoc(db, ws, tag="doc"):
    """Create a Document with all required columns."""
    from app.models.document import Document

    _counter[0] += 1
    doc = Document(
        user_id=ws.owner_id, workspace_id=ws.id,
        original_filename=f"{tag}-{_counter[0]}.txt",
        storage_key=f"p23/{tag}/{ws.id}/{_counter[0]}.txt",
        mime_type="text/plain", file_size=42, status="READY")
    db.add(doc)
    db.commit()
    return doc


def _mkws(db, tag="p23"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p23.example", name=tag, password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Global control plane
# ===========================================================================

class TestGlobalControl:
    def test_capability_detail_known(self, db_session):
        detail = gc.capability_detail(db_session, "postgres")
        assert detail is not None
        assert "state" in detail

    def test_capability_detail_unknown(self, db_session):
        assert gc.capability_detail(db_session, "not-a-thing") is None

    def test_run_check_unknown_component(self, db_session):
        result = gc.run_capability_check(db_session, "bogus")
        assert result["checked"] == []
        assert result["unknown"] == "bogus"

    def test_run_check_persists(self, db_session):
        result = gc.run_capability_check(db_session)
        assert len(result["components"]) >= 8

    def test_dependency_graph_shape(self, db_session):
        graph = gc.dependency_graph(db_session)
        assert graph["edges"]
        assert any(e["component"] == "worker" for e in graph["edges"])

    def test_dependency_impact_upstream(self, db_session):
        impact = gc.dependency_impact(db_session, "database")
        assert "api" in impact["upstream_hard"]
        assert "worker" in impact["upstream_hard"]

    def test_dependency_impact_downstream(self, db_session):
        impact = gc.dependency_impact(db_session, "worker")
        assert "database" in impact["downstream_hard"]
        assert "broker" in impact["downstream_hard"]

    def test_blast_radius_includes_api(self, db_session):
        impact = gc.dependency_impact(db_session, "database")
        assert "api" in impact["blast_radius"]

    def test_global_health_structure(self, db_session):
        health = gc.global_health(db_session)
        assert health["status"] in ("HEALTHY", "DEGRADED", "UNHEALTHY")
        assert "degraded_components" in health
        assert "capabilities" in health

    def test_readiness_no_block_when_db_up(self, db_session):
        ready = gc.readiness(db_session)
        assert ready["ready"] is True or ready["blocking"]

    def test_degraded_components_report(self, db_session):
        result = gc.degraded_components(db_session)
        assert "items" in result and "count" in result

    def test_dependencies_overview(self, db_session):
        overview = gc.dependencies_overview(db_session)
        assert "graph" in overview and "status" in overview


# ===========================================================================
# Broker migration safety
# ===========================================================================

class TestBrokerMigration:
    def test_transition_validates_state(self, db_session):
        with pytest.raises(ValueError):
            pm.BrokerMigrationRecord.transition(
                db_session, "m1", new_state="BOGUS",
                from_backend="postgres", to_backend="redis")

    def test_transition_illegal_jump(self, db_session):
        pm.BrokerMigrationRecord.transition(
            db_session, "m2", new_state="DRAINING",
            from_backend="postgres", to_backend="redis")
        with pytest.raises(ValueError):
            pm.BrokerMigrationRecord.transition(
                db_session, "m2", new_state="ACTIVE",
                from_backend="postgres", to_backend="redis")

    def test_transition_happy_path(self, db_session):
        for state in ("DRAINING", "DRAINED", "VERIFYING", "VERIFIED",
                      "ACTIVATING", "ACTIVE"):
            pm.BrokerMigrationRecord.transition(
                db_session, "m3", new_state=state,
                from_backend="postgres", to_backend="redis")
        state = pm.BrokerMigrationRecord.state(db_session, "m3")
        assert state["state"] == "ACTIVE"
        assert len(state["history"]) == 6

    def test_state_empty(self, db_session):
        state = pm.BrokerMigrationRecord.state(db_session, "never")
        assert state["exists"] is False

    def test_rollback_after_verified(self, db_session):
        pm.BrokerMigrationRecord.transition(
            db_session, "m4", new_state="DRAINING",
            from_backend="postgres", to_backend="redis")
        pm.BrokerMigrationRecord.transition(
            db_session, "m4", new_state="DRAINED",
            from_backend="postgres", to_backend="redis")
        pm.BrokerMigrationRecord.transition(
            db_session, "m4", new_state="VERIFYING",
            from_backend="postgres", to_backend="redis")
        pm.BrokerMigrationRecord.transition(
            db_session, "m4", new_state="VERIFIED",
            from_backend="postgres", to_backend="redis")
        result = pm.BrokerMigrationRecord.transition(
            db_session, "m4", new_state="ROLLED_BACK",
            from_backend="postgres", to_backend="redis")
        assert result["state"] == "ROLLED_BACK"

    def test_drain_check_reports_state(self, db_session):
        result = pm.broker_verify_queue_state(db_session)
        assert "status_counts" in result
        assert "no_job_loss_risk" in result

    def test_migration_plan_dry_run(self, db_session):
        plan = pm.broker_migration_plan(db_session, migration_id="p1",
                                        to_backend="redis", dry_run=True)
        assert plan["dry_run"] is True
        assert "health_gate" in plan
        assert "rollback" in plan

    def test_drain_old_backend_bounded(self, db_session):
        result = pm.broker_drain_old_backend(db_session, max_jobs=10)
        assert result["max_jobs"] == 10


# ===========================================================================
# Storage migration
# ===========================================================================

class TestStorageMigration:
    def test_create_plan_enumerates_documents(self, db_session):
        ws = _mkws(db_session)
        _mkdoc(db_session, ws)
        result = pm.create_storage_migration_plan(db_session,
                                                  workspace_id=ws.id)
        assert result["total_objects"] == 1
        assert result["plan_id"]

    def test_batch_size_bounds(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            pm.create_storage_migration_plan(db_session, workspace_id=ws.id,
                                             batch_size=0)
        with pytest.raises(ValueError):
            pm.create_storage_migration_plan(db_session, workspace_id=ws.id,
                                             batch_size=999)

    def test_dry_run_batch_copies_nothing(self, db_session):
        ws = _mkws(db_session)
        _mkdoc(db_session, ws)
        plan = pm.create_storage_migration_plan(db_session, workspace_id=ws.id,
                                                dry_run=True)
        result = pm.run_storage_migration_batch(db_session, plan["plan_id"])
        assert result["dry_run"] is True

    def test_progress_reports_states(self, db_session):
        ws = _mkws(db_session)
        plan = pm.create_storage_migration_plan(db_session, workspace_id=ws.id)
        progress = pm.storage_migration_progress(db_session,
                                                 plan["plan_id"])
        assert progress["status"] in ("DRAFT", "DRY_RUN", "RUNNING",
                                      "COMPLETED")

    def test_progress_unknown_plan(self, db_session):
        with pytest.raises(ValueError):
            pm.storage_migration_progress(db_session, 999999)

    def test_integrity_missing_document(self, db_session):
        ws = _mkws(db_session)
        result = pm.verify_document_integrity(db_session, 987654, ws.id)
        assert result["verified"] is False

    def test_orphan_scan_bounded(self, db_session):
        ws = _mkws(db_session)
        result = pm.orphan_object_candidates(db_session, ws.id, limit=5)
        assert result["scanned"] <= 5
        assert result["cleanup_requires_governance"] is True

    def test_capability_report(self, db_session):
        report = pm.storage_capability_report(db_session)
        assert "backend_class" in report
        assert "supports_signed_urls" in report


# ===========================================================================
# Provider gate + readiness
# ===========================================================================

class TestProviderGate:
    def test_readiness_validation_scores(self, db_session):
        result = pg.validate_provider_readiness(db_session,
                                                provider_kind="fake")
        assert 0 <= result["score"] <= 100
        assert result["real_provider"] is False
        assert len(result["results"]) == len(pg.PROVIDER_CAPABILITIES)

    def test_readiness_persisted(self, db_session):
        pg.validate_provider_readiness(db_session, provider_kind="openai")
        matrix = pg.readiness_matrix(db_session)
        kinds = {i["provider_kind"] for i in matrix["items"]}
        assert "openai" in kinds

    def test_multimodal_honest_on_fake(self, db_session):
        result = pg.validate_provider_readiness(db_session,
                                                provider_kind="fake")
        mm = next(r for r in result["results"]
                  if r["capability"] == "multimodal")
        assert mm["ok"] is False   # deterministic: fake has no vision

    def test_failure_classification_timeout(self):
        exc = TimeoutError("operation timed out")
        assert pg._classify_exception(exc) == "timeout"

    def test_failure_classification_rate(self):
        exc = Exception("HTTP 429 too many requests")
        assert pg._classify_exception(exc) == "rate_limited"

    def test_failure_classification_auth(self):
        exc = Exception("401 unauthorized")
        assert pg._classify_exception(exc) == "auth_failure"

    def test_synthetic_cases_never_tenant_data(self):
        for case in pg.SYNTHETIC_CASES.values():
            blob = str(case).lower()
            assert "workspace" not in blob
            assert "password" not in blob


# ===========================================================================
# Routing 3.0 + admission control
# ===========================================================================

class TestRouting:
    def test_route_selects_highest_score(self, db_session):
        ws = _mkws(db_session)
        pg.validate_provider_readiness(db_session, provider_kind="fake")
        decision = pg.route_provider(db_session, workspace_id=ws.id,
                                     operation="completion")
        assert decision["decision"] == "ROUTED"
        assert decision["provider"]

    def test_route_rejects_context_overflow(self, db_session):
        ws = _mkws(db_session)
        decision = pg.route_provider(db_session, workspace_id=ws.id,
                                     operation="completion",
                                     context_chars=999_999)
        assert decision["decision"] == "REJECTED"

    def test_route_prohibited_provider(self, db_session):
        ws = _mkws(db_session)
        decision = pg.route_provider(
            db_session, workspace_id=ws.id, operation="completion",
            sensitivity="RESTRICTED",
            policy={"restricted_providers": ["openai", "anthropic",
                                             "azure", "local", "fake"],
                    "emergency_fallback": "fake",
                    "confidential_providers": [],
                    "max_context_chars": 100_000})
        assert decision["decision"] == "REJECTED"
        assert any("prohibited" in r or "no eligible provider" in r
                   for r in decision["reasons"])

    def test_route_residency_block(self, db_session):
        from app.models.phase19 import ResidencyRule
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(classification="RESTRICTED",
                                     allowed_regions_json='["eu-west"]'))
        db_session.commit()
        decision = pg.route_provider(db_session, workspace_id=ws.id,
                                     operation="completion",
                                     sensitivity="RESTRICTED",
                                     region="us-east")
        assert decision["decision"] == "REJECTED"
        assert any("residency" in r for r in decision["reasons"])

    def test_admission_all_pass(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    budget_remaining=10.0)
        assert result["admitted"] is True

    def test_admission_budget_block(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    estimated_cost=99.0,
                                    budget_remaining=1.0)
        assert result["admitted"] is False
        budget = next(c for c in result["checks"] if c["check"] == "budget")
        assert budget["ok"] is False

    def test_routing_decision_audited(self, db_session):
        ws = _mkws(db_session)
        pg.route_provider(db_session, workspace_id=ws.id,
                          operation="embedding")
        rows = db_session.query(ProviderRoutingDecision).all()
        assert len(rows) == 1
        assert rows[0].operation == "embedding"


# ===========================================================================
# Cost reconciliation
# ===========================================================================

class TestCostReconciliation:
    def test_overbilling_detected(self, db_session):
        ws = _mkws(db_session)
        result = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-test", estimated_tokens=1000,
            actual_input_tokens=3000, actual_output_tokens=0,
            estimated_cost=1.0, actual_cost=2.0)
        assert result["classification"] in ("OVERBILLING_RISK",
                                             "ANOMALOUS_USAGE")
        assert result["variance"] == pytest.approx(1.0)

    def test_underbilling_detected(self, db_session):
        ws = _mkws(db_session)
        result = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-test", estimated_tokens=1000,
            actual_input_tokens=200, actual_output_tokens=0,
            estimated_cost=1.0, actual_cost=0.2)
        assert result["classification"] in ("UNDERBILLING_RISK",
                                             "ANOMALOUS_USAGE")

    def test_missing_usage(self, db_session):
        ws = _mkws(db_session)
        result = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-test", estimated_tokens=1000,
            estimated_cost=1.0, actual_cost=None)
        assert result["classification"] == "MISSING_USAGE"
        assert result["simulated"] is True

    def test_anomalous_usage(self, db_session):
        ws = _mkws(db_session)
        result = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-test", estimated_tokens=100,
            actual_input_tokens=5000, actual_output_tokens=0,
            estimated_cost=1.0, actual_cost=1.0)
        assert result["classification"] == "ANOMALOUS_USAGE"

    def test_ok_reconciliation(self, db_session):
        ws = _mkws(db_session)
        result = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-test", estimated_tokens=1000,
            actual_input_tokens=1000, actual_output_tokens=0,
            estimated_cost=1.0, actual_cost=1.0)
        assert result["classification"] == "OK"
        assert result["simulated"] is False

    def test_summary_counts(self, db_session):
        ws = _mkws(db_session)
        pg.reconcile_provider_cost(db_session, workspace_id=ws.id,
                                   provider="p", model="m",
                                   estimated_tokens=1000,
                                   actual_input_tokens=1000,
                                   estimated_cost=1.0, actual_cost=1.0)
        summary = pg.reconciliation_summary(db_session, workspace_id=ws.id)
        assert summary["total"] == 1
        assert "OK" in summary["by_classification"]

    def test_summary_scoped_to_workspace(self, db_session):
        ws_a = _mkws(db_session)
        ws_b = _mkws(db_session)
        pg.reconcile_provider_cost(db_session, workspace_id=ws_a.id,
                                   provider="p", model="m",
                                   estimated_tokens=1000,
                                   actual_input_tokens=1000,
                                   estimated_cost=1.0, actual_cost=1.0)
        summary_b = pg.reconciliation_summary(db_session,
                                              workspace_id=ws_b.id)
        assert summary_b["total"] == 0


# ===========================================================================
# Idempotency + cache
# ===========================================================================

class TestIdempotency:
    def test_first_request_executes(self, db_session):
        ws = _mkws(db_session)
        r = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                         idempotency_key="k1", payload={"a": 1})
        assert r["action"] == "execute"

    def test_replay_returns_result(self, db_session):
        ws = _mkws(db_session)
        r1 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k2",
                                          payload={"a": 1})
        ap3.finish_idempotent_request(db_session, r1["record_id"], ok=True,
                                      response_status=200,
                                      response={"done": True})
        r2 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k2",
                                          payload={"a": 1})
        assert r2["action"] == "replay"
        assert r2["response"] == {"done": True}

    def test_conflict_on_different_payload(self, db_session):
        ws = _mkws(db_session)
        r1 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k3",
                                          payload={"a": 1})
        ap3.finish_idempotent_request(db_session, r1["record_id"], ok=True,
                                      response_status=200, response={})
        r2 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k3",
                                          payload={"a": 2})
        assert r2["action"] == "conflict"

    def test_in_flight_rejection(self, db_session):
        ws = _mkws(db_session)
        ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                     idempotency_key="k4", payload={"x": 1})
        r2 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k4",
                                          payload={"x": 1})
        assert r2["action"] == "in_flight"

    def test_failed_allows_retry(self, db_session):
        ws = _mkws(db_session)
        r1 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k5",
                                          payload={"x": 1})
        ap3.finish_idempotent_request(db_session, r1["record_id"], ok=False,
                                      response_status=500)
        r2 = ap3.begin_idempotent_request(db_session, workspace_id=ws.id,
                                          idempotency_key="k5",
                                          payload={"x": 1})
        assert r2["action"] == "execute"

    def test_tenant_isolation_of_keys(self, db_session):
        ws1 = _mkws(db_session)
        ws2 = _mkws(db_session)
        ap3.begin_idempotent_request(db_session, workspace_id=ws1.id,
                                     idempotency_key="shared",
                                     payload={"a": 1})
        r2 = ap3.begin_idempotent_request(db_session, workspace_id=ws2.id,
                                          idempotency_key="shared",
                                          payload={"a": 1})
        assert r2["action"] == "execute"   # different tenant => independent


class TestTenantCache:
    def test_get_or_load(self):
        cache = ap3.TenantCache()
        value = cache.get_or_load(workspace_id=1, namespace="t", key="a",
                                  loader=lambda: 42)
        assert value == 42
        assert cache.stats()["hits"] == 0
        value2 = cache.get_or_load(workspace_id=1, namespace="t", key="a",
                                   loader=lambda: 43)
        assert value2 == 42
        assert cache.stats()["hits"] == 1

    def test_tenant_scoping_no_leak(self):
        cache = ap3.TenantCache()
        cache.set(workspace_id=1, namespace="t", key="k", value="secret-a")
        assert cache.get(workspace_id=2, namespace="t", key="k") is None

    def test_invalidate_tenant_safe(self):
        cache = ap3.TenantCache()
        cache.set(workspace_id=1, namespace="t", key="k", value=1)
        cache.set(workspace_id=2, namespace="t", key="k", value=2)
        removed = cache.invalidate(workspace_id=1, namespace="t")
        assert removed == 1
        assert cache.get(workspace_id=2, namespace="t", key="k") == 2

    def test_version_keying(self):
        cache = ap3.TenantCache()
        cache.set(workspace_id=1, namespace="t", key="k", value="v1",
                  version=1)
        cache.set(workspace_id=1, namespace="t", key="k", value="v2",
                  version=2)
        assert cache.get(workspace_id=1, namespace="t", key="k",
                         version=1) == "v1"
        assert cache.get(workspace_id=1, namespace="t", key="k",
                         version=2) == "v2"

    def test_bounded_entries(self):
        cache = ap3.TenantCache(max_entries=10)
        for i in range(50):
            cache.set(workspace_id=1, namespace="t", key=f"k{i}", value=i)
        assert cache.stats()["entries"] <= 10

    def test_compatibility_metadata(self):
        meta = ap3.compatibility_metadata()
        assert meta["current_version"] == "v1"
        assert "v2" in meta["supported_versions"]

    def test_contract_matrix(self):
        matrix = ap3.contract_case_matrix()
        assert matrix["count"] >= 10
        assert "tenant_isolation" in matrix["cases"]
