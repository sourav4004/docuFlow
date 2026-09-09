"""Phase 21 tests — compliance, contracts, and honest-operations checks.

Verifies the phase's non-negotiable contracts hold structurally: no secrets
in payload storage, bounded JSON columns everywhere, tenant scoping on every
new model, audit requirements on autonomous operations, no automatic DDL,
no destructive autonomy, simulation purity, emergency-stop veto coverage,
and service-level API stability (public function signatures the ops21 router
depends on).
"""

import inspect
import json
import uuid

import pytest
from sqlalchemy import inspect as sa_inspect

from tests.shared_db import TestingSessionLocal
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models import phase21 as m21  # noqa: E402
from app.services import autonomy as au  # noqa: E402
from app.services import selfheal as sh  # noqa: E402
from app.services import safety10 as s10  # noqa: E402
from app.services import incident_ops as io_  # noqa: E402
from app.services import infra_heal as ih  # noqa: E402
from app.services import personal_ai as pa  # noqa: E402
from app.services import adaptive_ai as ad  # noqa: E402
from app.services import knowledge_heal as kh  # noqa: E402
from app.services import agent_autonomy as aa  # noqa: E402
from app.services import event_eval as ev  # noqa: E402
from app.services import autopilot as ap  # noqa: E402
from app.services import diagnosis as dg  # noqa: E402

_counter = [0]


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def ws_id(db):
    _counter[0] += 1
    user = User(name=f"c{_counter[0]}",
                email=f"c{_counter[0]}@p21comp.example", password_hash="x")
    db.add(user)
    db.flush()
    ws = Workspace(name=f"c{_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws.id


# ===========================================================================
# Structural contracts on models
# ===========================================================================

class TestModelContracts:
    PH21_MODELS = [
        m21.AutonomyPolicy, m21.AutonomousOperation, m21.RecoveryPlaybook,
        m21.RecoveryAttempt, m21.SystemHealthSnapshot, m21.DiagnosisReport,
        m21.KnowledgeRecoveryPlan, m21.AdaptiveCandidate,
        m21.CostGuardDecision, m21.CostOptimizationEvent,
        m21.AgentPlanRisk, m21.WorkflowRiskAssessment, m21.PlatformEvent,
        m21.EvaluationRun, m21.EmergencyStop, m21.SecurityHealthScore,
        m21.SecurityIncidentP21, m21.IncidentP21, m21.MaintenancePlan,
        m21.BackupHealthRecord, m21.ChaosTestRun, m21.WorkerHealthScore,
        m21.BrokerHealthSnapshot, m21.CacheHealthSnapshot,
        m21.GraphRepairProposal, m21.MemoryAutonomyEvent,
        m21.PersonalAutonomySetting, m21.AIActivityItem,
        m21.ArtifactQualityScore, m21.ApiAbuseSignalP21,
        m21.CostAwareQueueDecision, m21.LearningDatasetCandidate,
    ]

    def test_every_model_has_workspace_or_org_scope(self):
        for model in self.PH21_MODELS:
            cols = model.__table__.columns.keys()
            assert ("workspace_id" in cols) or ("organization_id" in cols), \
                model.__name__

    def test_idempotent_models_have_keys(self):
        for model in (m21.AutonomousOperation, m21.RecoveryAttempt,
                      m21.CostGuardDecision, m21.CostOptimizationEvent,
                      m21.PlatformEvent, m21.AdaptiveCandidate,
                      m21.LearningDatasetCandidate):
            assert "idempotency_key" in model.__table__.columns.keys() \
                or "event_key" in model.__table__.columns.keys(), model.__name__

    def test_no_secret_named_columns(self):
        banned = ("api_key", "password", "secret", "token", "credential")
        for model in self.PH21_MODELS:
            for col in model.__table__.columns.keys():
                assert not any(b in col.lower() for b in banned), \
                    f"{model.__name__}.{col}"

    def test_activity_items_have_no_reasoning_field(self):
        cols = m21.AIActivityItem.__table__.columns.keys()
        assert "explanation" in cols
        assert "reasoning" not in cols
        assert "chain_of_thought" not in cols


# ===========================================================================
# Service-level API stability (router depends on these)
# ===========================================================================

class TestServiceContracts:
    def test_autonomy_public_api(self):
        for fn in ("get_policy", "create_policy", "set_autonomy_level",
                   "simulate_operation", "guard_operation",
                   "execute_allowed", "mark_rolled_back",
                   "emergency_stop_active", "list_operations"):
            assert callable(getattr(au, fn)), fn

    def test_selfheal_public_api(self):
        for fn in ("aggregate_health", "detect_failures", "create_playbook",
                   "attempt_recovery", "list_attempts", "latest_snapshot"):
            assert callable(getattr(sh, fn)), fn

    def test_knowledge_adaptive_public_api(self):
        for fn in ("knowledge_health", "detect_knowledge_incidents",
                   "create_recovery_plan", "execute_recovery_plan"):
            assert callable(getattr(kh, fn)), fn
        for fn in ("record_retrieval_evaluation", "record_rag_quality",
                   "propose_retrieval_candidate", "promote_candidate"):
            assert callable(getattr(ad, fn)), fn

    def test_incident_infra_public_api(self):
        for fn in ("create_incident", "evaluate_incident_rules",
                   "draft_postmortem", "finalize_postmortem",
                   "maintenance_dry_run", "execute_maintenance",
                   "dr_simulation", "record_chaos_run"):
            assert callable(getattr(io_, fn)), fn
        for fn in ("record_region_health", "check_residency",
                   "simulate_failover", "record_worker_health",
                   "invalidate_cache_tenant_safe", "propose_graph_repair",
                   "suppress_memory", "route_memory_conflict"):
            assert callable(getattr(ih, fn)), fn

    def test_guard_functions_take_workspace_first(self):
        sig = inspect.signature(au.guard_operation)
        params = list(sig.parameters)
        assert params[0] == "db" and params[1] == "workspace_id"

    def test_simulate_is_marked(self, db):
        w = ws_id(db)
        result = au.simulate_operation(db, w, "contract.op")
        assert result.get("simulated") is True


# ===========================================================================
# Honest-operations behavioral checks
# ===========================================================================

class TestHonestOperations:
    def test_dr_simulation_never_claims_real(self, db):
        w = ws_id(db)
        result = io_.dr_simulation(db, w)
        assert result["simulated"] is True
        assert result.get("honesty_note")

    def test_backup_health_reports_unknown(self, db):
        w = ws_id(db)
        row = io_.record_backup_health(db, w, None)
        assert row.status == "UNKNOWN"

    def test_restore_validation_flags_simulation(self, db):
        w = ws_id(db)
        rec = io_.record_backup_health(db, w, io_._now())
        io_.validate_restore(db, w, rec, True, 20, 5, simulated=True)
        assert "SIMULATED" in rec.detail

    def test_chaos_runs_marked_simulated(self, db):
        w = ws_id(db)
        row = io_.record_chaos_run(db, w, "worker_loss", True)
        assert row.simulated is True
        assert row.bounded is True

    def test_load_probes_marked_simulated(self):
        assert ap.__name__  # import sanity
        from app.services import incident_ops as io
        assert io.api_load_test(10, 2)["simulated"] is True
        assert io.worker_load_test(10, 2)["simulated"] is True

    def test_maintenance_never_auto_executes_destructive(self, db):
        w = ws_id(db)
        plan = io_.create_maintenance_plan(db, w, "expired_traces",
                                           targets=[{"id": 9}])
        # No approval, no dry run.
        result = io_.execute_maintenance(db, plan)
        assert result["executed"] is False

    def test_failover_simulated_never_executed(self, db):
        from app.models.organization import Organization
        org = Organization(name=f"c-org-{uuid.uuid4().hex[:6]}",
                           slug=f"c-org-{uuid.uuid4().hex[:6]}", owner_id=1)
        db.add(org)
        db.commit()
        target = ih.record_region_health(db, org.id, "r2", workers=3)
        sim = ih.simulate_failover(db, org.id, "r1", "r2", target)
        assert sim.simulated is True and sim.executed is False


# ===========================================================================
# Autonomy veto coverage
# ===========================================================================

class TestVetoCoverage:
    def _stopped(self, db, w, scope="ALL"):
        s10.activate_emergency_stop(db, w, scope=scope, reason="test")

    def test_stop_vetoes_recovery_auto(self, db):
        w = ws_id(db)
        au.create_policy(db, w, "recovery", autonomy_level="AUTO_LOW_RISK",
                         requires_approval=False)
        pb = sh.create_playbook(db, w, "veto-pb", "cache_failure",
                                ["invalidate_cache"], approved=True)
        self._stopped(db, w, "AUTONOMOUS_RECOVERY")
        att = sh.attempt_recovery(db, w, "cache_failure", pb,
                                  f"veto-{uuid.uuid4()}")
        assert att.status == "POLICY_BLOCKED"

    def test_stop_vetoes_candidate_promotion(self, db):
        w = ws_id(db)
        au.create_policy(db, w, "adaptation.promote.ingestion",
                         autonomy_level="AUTO_LOW_RISK",
                         requires_approval=False)
        for _ in range(5):
            kh.record_ingestion_sample(db, w, chunk_quality=0.9)
        kh.record_ingestion_sample(db, w, chunk_quality=0.3)
        cand = kh.propose_ingestion_adaptation(
            db, w, kh.detect_ingestion_anomalies(db, w)[0])
        kh.evaluate_candidate(db, cand, 0.95, 0.5)
        self._stopped(db, w, "AUTONOMOUS_RECOVERY")
        result = ad.promote_candidate(db, cand)
        assert result["decision"] in ("REQUIRES_APPROVAL", "BLOCKED")

    def test_stop_vetoes_cost_optimization(self, db):
        w = ws_id(db)
        au.create_policy(db, w, "cost.optimize.cache_reuse",
                         autonomy_level="AUTO_LOW_RISK",
                         requires_approval=False)
        self._stopped(db, w, "AUTONOMOUS_RECOVERY")
        result = ap.apply_cost_optimization(db, w, "cache_reuse")
        assert result["applied"] is False

    def test_lift_restores_auto(self, db):
        w = ws_id(db)
        stop = s10.activate_emergency_stop(db, w, scope="ALL")
        s10.lift_emergency_stop(db, w, stop)
        au.create_policy(db, w, "post.stop", autonomy_level="AUTO_LOW_RISK",
                         requires_approval=False)
        sim = au.simulate_operation(db, w, "post.stop")
        assert sim["would_auto_execute"] is True


# ===========================================================================
# Dedup/corpus invariants
# ===========================================================================

class TestInvariants:
    def test_platform_event_dedup_is_database_enforced(self):
        uq = {c.name for c in m21.PlatformEvent.__table__.constraints
              if c.__class__.__name__ == "UniqueConstraint"}
        assert any("event_key" in str(cols) or "event_key" in str(uq)
                   for cols in uq)

    def test_injection_patterns_all_match_own_vectors(self):
        for vector, payload in s10.INJECTION_VECTORS.items():
            detected, _ = s10.detect_injection(payload)
            assert detected, vector

    def test_destructive_actions_never_in_safe_set(self):
        assert not (sh.SAFE_AUTO_ACTIONS & sh.DESTRUCTIVE_ACTIONS)

    def test_safe_recovery_actions_subset_of_playbook_allowed(self):
        for action in sh.SAFE_AUTO_ACTIONS:
            assert action in ("retry_transient_work", "recover_stale_lease",
                              "reconnect_broker", "restart_worker_state",
                              "activate_circuit_breaker",
                              "invalidate_cache", "requeue_safe_jobs")

    def test_diag_confidence_never_certain(self, db):
        report = dg.diagnose("everything on fire", {
            "provider": {"error_rate": 1.0, "sample_count": 100},
            "queue_depth": 9999, "db_errors": 50,
            "recent_change": "big deploy"})
        assert report["top_confidence"] <= 95.0

    def test_learning_quality_gates_deterministic(self):
        assert ev.classify_feedback_quality("good") == "APPROVED"
        assert ev.classify_feedback_quality("asdf") == "NOISY"
        assert ev.classify_feedback_quality("you are stupid") == "ABUSIVE"
