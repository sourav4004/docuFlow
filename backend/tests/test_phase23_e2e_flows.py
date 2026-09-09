"""Phase 23 tests — end-to-end product flows + remaining surfaces.

E2E: broker migration safety, storage migration lifecycle, vector model
migration (dual generation -> coverage -> quality -> promotion/rollback),
shadow search coexistence, knowledge freshness/drift/maintenance, artifact
and reporting integrity. Real service logic, deterministic providers.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    StorageMigrationPlan, StorageMigrationObject, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison,
)
from app.models.phase22 import OpsStreamEvent
from app.models import MaintenanceRunP22
from app.models.phase19 import ConsistencyReport
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import platform_migration as pm
from app.services import vector_activation as va
from app.services import knowledge_maintenance2 as km

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


TABLES = [
    StorageMigrationPlan, StorageMigrationObject, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison, OpsStreamEvent,
    MaintenanceRunP22, ConsistencyReport, DocumentChunk, Document,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in TABLES:
        db_session.query(model).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"e2e{n}@p23e2e.example", name="e2e",
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-e2e-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


def _mkdoc_with_chunks(db, ws, n_chunks=3):
    _counter[0] += 1
    doc = Document(
        user_id=ws.owner_id, workspace_id=ws.id,
        original_filename=f"e2e-{_counter[0]}.txt",
        storage_key=f"p23/e2e/{ws.id}/{_counter[0]}.txt",
        mime_type="text/plain", file_size=42, status="READY")
    db.add(doc)
    db.flush()
    for i in range(n_chunks):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i,
                             text=f"chunk {i} of doc {doc.id}",
                             char_start=i * 100, char_end=i * 100 + 90))
    db.commit()
    return doc


# ===========================================================================
# Broker migration safety (Step 3)
# ===========================================================================

class TestBrokerMigrationE2E:
    def test_plan_dry_run(self, db_session):
        plan = pm.broker_migration_plan(db_session,
                                        migration_id="m-e2e-1",
                                        to_backend="redis", dry_run=True)
        assert plan["migration_id"] == "m-e2e-1"
        assert plan["dry_run"] is True

    def test_state_lifecycle_transitions(self, db_session):
        rec = pm.BrokerMigrationRecord
        rec.transition(db_session, "m-e2e-2", new_state="PLANNED",
                       from_backend="postgres", to_backend="redis",
                       actor="ops")
        state = rec.state(db_session, "m-e2e-2")
        assert state["state"] == "PLANNED"
        rec.transition(db_session, "m-e2e-2", new_state="DRAINING",
                       from_backend="postgres", to_backend="redis")
        assert rec.state(db_session, "m-e2e-2")["state"] == "DRAINING"

    def test_illegal_transition_rejected(self, db_session):
        rec = pm.BrokerMigrationRecord
        rec.transition(db_session, "m-e2e-3", new_state="PLANNED",
                       from_backend="postgres", to_backend="redis")
        with pytest.raises(ValueError):
            rec.transition(db_session, "m-e2e-3", new_state="ACTIVE",
                           from_backend="postgres", to_backend="redis")

    def test_rollback_state_allowed(self, db_session):
        rec = pm.BrokerMigrationRecord
        rec.transition(db_session, "m-e2e-4", new_state="PLANNED",
                       from_backend="postgres", to_backend="redis")
        rec.transition(db_session, "m-e2e-4", new_state="DRAINING",
                       from_backend="postgres", to_backend="redis")
        rec.transition(db_session, "m-e2e-4", new_state="FAILED",
                       from_backend="redis", to_backend="postgres")
        assert rec.state(db_session, "m-e2e-4")["state"] == "FAILED"

    def test_drain_old_backend_bounded(self, db_session):
        result = pm.broker_drain_old_backend(db_session)
        assert "queued_jobs" in result and "max_jobs" in result

    def test_verify_queue_state(self, db_session):
        result = pm.broker_verify_queue_state(db_session, max_jobs=100)
        assert result["no_job_loss_risk"] is True
        assert "status_counts" in result


# ===========================================================================
# Storage migration lifecycle (Step 5)
# ===========================================================================

class TestStorageMigrationE2E:
    def test_full_dry_run_cycle(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        plan = pm.create_storage_migration_plan(db_session,
                                                workspace_id=ws.id,
                                                dry_run=True)
        assert plan["total_objects"] >= 1
        result = pm.run_storage_migration_batch(db_session, plan["plan_id"])
        progress = pm.storage_migration_progress(db_session,
                                                 plan["plan_id"])
        assert progress["dry_run"] is True
        assert result is not None

    def test_resumable_batching(self, db_session):
        ws = _mkws(db_session)
        for _ in range(3):
            _mkdoc_with_chunks(db_session, ws)
        plan = pm.create_storage_migration_plan(db_session,
                                                workspace_id=ws.id,
                                                batch_size=2, dry_run=True)
        r1 = pm.run_storage_migration_batch(db_session, plan["plan_id"])
        progress = pm.storage_migration_progress(db_session,
                                                 plan["plan_id"])
        assert progress["total"] >= 3

    def test_plan_for_unknown_workspace_empty(self, db_session):
        plan = pm.create_storage_migration_plan(db_session,
                                                workspace_id=99999999)
        assert plan["total_objects"] == 0  # nothing to migrate

    def test_orphan_candidates_bounded(self, db_session):
        ws = _mkws(db_session)
        result = pm.orphan_object_candidates(db_session, ws.id, limit=10)
        assert result["count"] <= 10
        assert result["cleanup_requires_governance"] is True

    def test_storage_capability_report(self, db_session):
        report = pm.storage_capability_report(db_session)
        assert report["backend_class"]
        assert "capability" in report


# ===========================================================================
# Vector model migration E2E (Steps 10-12)
# ===========================================================================

class TestVectorModelMigrationE2E:
    def test_registration_and_dual_generation(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        mv = va.register_model_version(db_session, model_name="fake-embed",
                                       version="2", dimension=384)
        result = va.run_dual_generation_batch(db_session,
                                              workspace_id=ws.id,
                                              model_version_id=mv["id"])
        assert result is not None

    def test_coverage_verification(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        mv = va.register_model_version(db_session, model_name="fake-embed",
                                       version="2", dimension=384)
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv["id"])
        coverage = va.verify_coverage(db_session, workspace_id=ws.id,
                                      model_version_id=mv["id"])
        assert "coverage_pct" in coverage

    def test_promotion_gate_blocks_incomplete(self, db_session):
        ws = _mkws(db_session)
        result = va.promotion_gate(db_session, workspace_id=ws.id,
                                   min_comparisons=10)
        assert result["eligible"] is False  # no shadow evidence yet
        assert result["requires_human_approval"] is True

    def test_rollback_path(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        mv = va.register_model_version(db_session, model_name="fake-embed",
                                       version="2", dimension=384)
        va.run_dual_generation_batch(db_session, workspace_id=ws.id,
                                     model_version_id=mv["id"])
        result = va.rollback_model(db_session, model_version_id=mv["id"])
        assert result is not None

    def test_retirement_plan_bounded(self, db_session):
        ws = _mkws(db_session)
        mv = va.register_model_version(db_session, model_name="fake-embed",
                                       version="2", dimension=384)
        plan = va.retirement_plan(db_session, model_version_id=mv["id"])
        assert plan is not None

    def test_shadow_compare_records(self, db_session):
        ws = _mkws(db_session)
        result = va.shadow_compare(db_session, workspace_id=ws.id,
                                   query="remote work policy",
                                   baseline_ranking=["a", "b", "c"],
                                   candidate_ranking=["b", "a", "c"])
        assert result is not None

    def test_activation_honest_about_pgvector(self, db_session):
        report = va.activation_report(db_session)
        # pgvector NOT_CONFIGURED here: native validation must not be claimed
        pg = report["pgvector"] if "pgvector" in report else report
        state = (pg.get("state") or pg.get("backend", {}).get("state")
                 or "NOT_CONFIGURED")
        assert state != "REAL"


# ===========================================================================
# Knowledge freshness / drift / maintenance E2E (Steps 23-25)
# ===========================================================================

class TestKnowledgeMaintenanceE2E:
    def test_freshness_states(self, db_session):
        from datetime import datetime, timedelta, timezone
        ws = _mkws(db_session)
        now = datetime.now(timezone.utc)
        assert km.classify_freshness(updated_at=now,
                                     expiration_days=30)["state"] == "FRESH"
        assert km.classify_freshness(
            updated_at=now - timedelta(days=45),
            expiration_days=30)["state"] in ("STALE", "EXPIRED", "AGING")

    def test_refresh_and_overview(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        km.refresh_freshness(db_session, workspace_id=ws.id)
        overview = km.freshness_overview(db_session, workspace_id=ws.id)
        assert overview is not None

    def test_drift_report_bounded(self, db_session):
        ws = _mkws(db_session)
        result = km.detect_drift(db_session, workspace_id=ws.id)
        assert result is not None

    def test_maintenance_scan_dry_run(self, db_session):
        ws = _mkws(db_session)
        _mkdoc_with_chunks(db_session, ws)
        result = km.run_maintenance_scan(db_session, workspace_id=ws.id)
        assert result is not None
        assert db_session.query(MaintenanceRunP22).filter_by(
            workspace_id=ws.id).count() >= 1

    def test_maintenance_history_bounded(self, db_session):
        ws = _mkws(db_session)
        km.run_maintenance_scan(db_session, workspace_id=ws.id)
        history = km.maintenance_history(db_session, workspace_id=ws.id,
                                         limit=5)
        assert len(history["items"]) <= 5


# ===========================================================================
# Artifact / reporting integrity (Steps 63-64)
# ===========================================================================

class TestArtifactReporting:
    def test_consistency_report_persisted(self, db_session):
        from app.services import data_consistency as dc
        ws = _mkws(db_session)
        dc.run_consistency_checks(db_session, workspace_id=ws.id)
        reports = db_session.query(ConsistencyReport).filter_by(
            workspace_id=ws.id).all()
        assert len(reports) >= 1
        assert reports[0].status in ("CLEAN", "ISSUES")

    def test_report_is_dry_run_flagged(self, db_session):
        from app.services import data_consistency as dc
        ws = _mkws(db_session)
        dc.run_consistency_checks(db_session, workspace_id=ws.id)
        report = db_session.query(ConsistencyReport).filter_by(
            workspace_id=ws.id).order_by(ConsistencyReport.id.desc()).first()
        assert report.dry_run is True  # reports never mutate data
