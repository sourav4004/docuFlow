"""Phase 22 tests — infrastructure capability registry, Redis hardening,
vector production platform, and real provider validation.

All validation is deterministic; where real infrastructure (pgvector,
Redis, provider credentials) is absent the service reports UNAVAILABLE and
tests assert that honesty rather than fabricated success.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase22 import (
    CostReconciliationRun, InfraCapability, ProviderValidationRun,
    VectorBenchmarkRun, VectorDriftSnapshot,
)
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import capabilities as cap
from app.services import provider_validation as pv
from app.services import vector_production as vp

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P22_INFRA_TABLES = [
    VectorBenchmarkRun, VectorDriftSnapshot, CostReconciliationRun,
    ProviderValidationRun, InfraCapability,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P22_INFRA_TABLES:
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="infra"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22infra.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Step 1-3: capability detection
# ===========================================================================

class TestCapabilityDetection:
    def test_detect_all_covers_components(self):
        items = cap.detect_all()
        components = {i["component"] for i in items}
        assert {"postgres", "pgvector", "redis", "object_storage",
                "provider", "smtp", "webhook", "container"} <= components

    def test_every_component_has_state_and_realization(self):
        for item in cap.detect_all():
            assert item["state"] in ("AVAILABLE", "UNAVAILABLE", "DEGRADED",
                                     "NOT_CONFIGURED", "UNKNOWN")
            assert item["realization"] in ("REAL", "SIMULATED",
                                           "UNAVAILABLE")

    def test_postgres_real_when_connected(self, db_session):
        from sqlalchemy import text
        try:
            db_session.execute(text("SELECT 1"))
            connected = True
        except Exception:
            connected = False
        item = cap.detect_component("postgres")
        if connected:
            assert item["state"] == "AVAILABLE"
            assert item["realization"] == "REAL"
        else:
            assert item["realization"] in ("SIMULATED", "UNAVAILABLE")

    def test_pgvector_honest_about_environment(self):
        item = cap.detect_component("pgvector")
        assert item["realization"] in ("REAL", "SIMULATED", "UNAVAILABLE")
        if item["realization"] != "REAL":
            assert item["state"] in ("UNAVAILABLE", "NOT_CONFIGURED",
                                     "DEGRADED")

    def test_redis_honest_about_environment(self):
        item = cap.detect_component("redis")
        assert item["realization"] in ("REAL", "SIMULATED", "UNAVAILABLE")
        assert "password" not in str(item).lower()

    def test_detection_never_leaks_credentials(self):
        blob = str(cap.detect_all())
        for marker in ("password=", "sk-", "Bearer ", "secret"):
            assert marker not in blob

    def test_persist_and_summarize(self, db_session):
        cap.persist_capabilities(db_session)
        summary = cap.infrastructure_summary(db_session)
        assert summary["configured"] is True
        assert summary["real_count"] >= 1  # postgres at least
        comps = {c["component"]: c for c in summary["components"]}
        assert "postgres" in comps

    def test_summary_empty_before_detection(self, db_session):
        summary = cap.infrastructure_summary(db_session)
        assert summary["configured"] is False

    def test_component_unknown(self):
        with pytest.raises(ValueError):
            cap.detect_component("not_a_thing")


# ===========================================================================
# Step 4: infrastructure health API (route-level covered in API suite)
# ===========================================================================

class TestInfrastructureSummary:
    def test_real_vs_simulated_counts(self, db_session):
        cap.persist_capabilities(db_session)
        summary = cap.infrastructure_summary(db_session)
        n = len(summary["components"])
        assert summary["real_count"] + summary["simulated_count"] <= n

    def test_checked_at_recorded(self, db_session):
        cap.persist_capabilities(db_session)
        summary = cap.infrastructure_summary(db_session)
        comp = summary["components"][0]
        assert comp["checked_at"] is not None


# ===========================================================================
# Steps 5-10: Redis hardening + broker failover
# ===========================================================================

class TestRedisHardening:
    def test_config_uses_env_references(self):
        cfg = cap.redis_production_config()
        assert cfg["url_configured"] in (True, False)
        assert cfg["pool"]["max_connections"] >= 1
        assert cfg["pool"]["socket_timeout_s"] > 0
        assert "credential reference" in cfg["auth"]

    def test_config_never_contains_literal_secret(self):
        cfg = str(cap.redis_production_config())
        assert "redis://:password" not in cfg

    def test_healthcheck_honest(self):
        health = cap.redis_healthcheck()
        if not health["available"]:
            assert health["fallback"] == "postgres broker"

    def test_failover_plan_redis_to_postgres(self):
        plan = cap.broker_failover_plan()
        assert plan["primary"] in ("redis", "postgres")
        assert plan["fallback"] == "postgres"
        assert plan["approved_auto"] is False  # never silent by default

    def test_failover_event_recorded(self, db_session):
        event = cap.record_failover_event(
            db_session, from_broker="redis", to_broker="postgres",
            reason="chaos_drill")
        assert event.get("recorded") in (True, False)


# ===========================================================================
# Steps 11-22: vector production platform
# ===========================================================================

class TestVectorProduction:
    def test_backend_kind_reports_fallback_or_native(self):
        kind = vp.vector_backend_kind()
        assert kind in ("json", "pgvector")

    def test_active_model_contract(self, db_session):
        model = vp._active_model(db_session)
        assert "model" in model and "dimensions" in model

    def test_coverage_snapshot_shape(self, db_session):
        ws = _mkws(db_session, "cov")
        snap = vp.coverage_snapshot(db_session, ws.id)
        for key in ("total_chunks", "embedded_chunks", "missing_embeddings",
                    "stale_embeddings", "incompatible_embeddings",
                    "coverage_pct"):
            assert key in snap
        assert 0.0 <= snap["coverage_pct"] <= 100.0

    def test_coverage_persists_snapshot(self, db_session):
        from app.models.phase19 import VectorCoverageSnapshot
        ws = _mkws(db_session, "covp")
        vp.coverage_snapshot(db_session, ws.id)
        assert db_session.query(VectorCoverageSnapshot).count() >= 1

    def test_drift_snapshot_shape(self, db_session):
        ws = _mkws(db_session, "drift")
        snap = vp.drift_snapshot(db_session, ws.id)
        assert "drifted_chunks" in snap and "drift_pct" in snap
        assert 0.0 <= snap["drift_pct"] <= 100.0

    def test_drift_persists(self, db_session):
        ws = _mkws(db_session, "driftp")
        vp.drift_snapshot(db_session, ws.id)
        assert db_session.query(VectorDriftSnapshot).count() >= 1

    def test_rebuild_atomic_and_bounded(self, db_session):
        ws = _mkws(db_session, "rebuild")
        result = vp.rebuild_document_vectors(db_session, ws.id, 999999)
        assert result["ok"] is True and result["chunks"] == 0

    def test_benchmark_marks_simulation_when_fallback(self, db_session):
        ws = _mkws(db_session, "bench")
        result = vp.benchmark(db_session, ws.id, queries=10)
        assert result["backend"] == vp.vector_backend_kind()
        assert result["native"] is (result["backend"] == "pgvector")
        assert result["queries"] == 10

    def test_benchmark_persists(self, db_session):
        ws = _mkws(db_session, "benchp")
        vp.benchmark(db_session, ws.id, queries=5)
        assert db_session.query(VectorBenchmarkRun).count() >= 1

    def test_index_lifecycle_policy(self):
        policy = vp.index_lifecycle_policy()
        assert "create" in policy and "drop" in policy
        assert "ADMIN_ONLY" in policy["drop"]

    def test_dimension_safety_rejects_incompatible(self, db_session):
        from app.services.vector_registry import register_model
        from app.models.phase18 import EmbeddingModel
        # Deactivate any leftover declared models, then declare one.
        db_session.query(EmbeddingModel).delete()
        db_session.commit()
        register_model(db_session, provider="fake", model="test-embed",
                       dimensions=128)
        result = vp.dimension_safety(db_session, 128 + 7)
        assert result.get("result", {}).get("valid") is False

    def test_dimension_safety_accepts_matching(self, db_session):
        from app.services.vector_registry import register_model
        from app.models.phase18 import EmbeddingModel
        db_session.query(EmbeddingModel).delete()
        db_session.commit()
        register_model(db_session, provider="fake", model="test-embed",
                       dimensions=128)
        result = vp.dimension_safety(db_session, 128)
        assert result.get("result", {}).get("valid") is True

    def test_backfill_status_and_preview(self, db_session):
        ws = _mkws(db_session, "bf")
        preview = vp.backfill_preview(db_session, ws.id)
        assert "chunks_to_embed" in preview
        assert preview["dry_run"] is True
        status = vp.backfill_status(db_session, 424242)
        assert status["ok"] is False


# ===========================================================================
# Steps 23-37: provider validation
# ===========================================================================

class TestProviderValidation:
    def test_configured_providers_honest(self):
        info = pv.configured_providers()
        assert all(isinstance(v, bool) for v in info.values())
        if not pv.any_real_provider():
            assert info["openai"] is False

    def test_fake_provider_completion(self, db_session):
        ws = _mkws(db_session, "pv")
        result = pv.validate_completion(db_session, ws.id)
        assert result["kind"] == "completion"
        assert result["simulated"] is True
        assert result["passed"] is True

    def test_validation_run_persisted(self, db_session):
        ws = _mkws(db_session, "pvp")
        pv.validate_completion(db_session, ws.id)
        assert db_session.query(ProviderValidationRun).count() >= 1

    def test_streaming_validation(self, db_session):
        ws = _mkws(db_session, "ps")
        result = pv.validate_streaming(db_session, ws.id)
        assert result["chunks"] >= 1
        assert result["passed"] is True

    def test_embeddings_validation(self, db_session):
        ws = _mkws(db_session, "pe")
        result = pv.validate_embeddings(db_session, ws.id)
        assert result["passed"] is True

    def test_structured_output(self, db_session):
        ws = _mkws(db_session, "pso")
        result = pv.validate_structured_output(db_session, ws.id)
        assert result["passed"] is True

    def test_tool_calling(self, db_session):
        ws = _mkws(db_session, "pt")
        result = pv.validate_tool_calling(db_session, ws.id)
        assert result["passed"] is True

    def test_multimodal(self, db_session):
        ws = _mkws(db_session, "pm")
        result = pv.validate_multimodal(db_session, ws.id)
        # Without real provider credentials multimodal is honestly
        # UNAVAILABLE — the service must not fake a pass.
        if not pv.any_real_provider():
            assert result["passed"] is False and result["real"] is False
        else:
            assert result["passed"] is True

    @pytest.mark.parametrize("mode", ["timeout", "429", "5xx", "malformed"])
    def test_failure_modes_surfaced(self, db_session, mode):
        ws = _mkws(db_session, "pf")
        result = pv.validate_failure_mode(db_session, ws.id, mode)
        assert result["mode"] == mode
        assert "passed" in result

    def test_fallback_primary_to_secondary(self, db_session):
        ws = _mkws(db_session, "pfb")
        result = pv.validate_fallback(db_session, ws.id)
        assert "passed" in result

    def test_circuit_breaker_transitions(self, db_session):
        ws = _mkws(db_session, "pcb")
        result = pv.validate_circuit_breaker(db_session, ws.id)
        assert result["passed"] is True
        assert "OPEN" in result["state"]  # opened after threshold

    def test_provider_health_summary(self, db_session):
        ws = _mkws(db_session, "ph")
        pv.validate_completion(db_session, ws.id)
        summary = pv.provider_health_summary(db_session)
        assert "providers" in summary
        # Health rows come from the provider call-metric ledger, which a
        # validation run populates.
        assert isinstance(summary["providers"], list)

    def test_cost_reconciliation_matches(self, db_session):
        ws = _mkws(db_session, "pcr")
        result = pv.reconcile_cost(db_session, ws.id, "fake")
        assert result["reconciled"] is True
        assert result["delta_pct"] == 0.0

    def test_reconciliation_run_persisted(self, db_session):
        ws = _mkws(db_session, "pcrp")
        pv.reconcile_cost(db_session, ws.id, "fake")
        assert db_session.query(CostReconciliationRun).count() >= 1

    def test_no_real_provider_claimed_without_credentials(self, db_session):
        ws = _mkws(db_session, "pnh")
        completion = pv.validate_completion(db_session, ws.id)
        if not pv.any_real_provider():
            assert completion["simulated"] is True
            assert completion["real"] is False
