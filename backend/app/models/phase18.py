"""Phase 18 models — Distributed AI Cloud + Enterprise Production Scale.

Tables cover: the vector model registry (provider/model/dimensions/version),
vector index operations, entity merge requests (approval-gated), poisoned
document quarantine, import jobs (validation/dry-run/commit), search
analytics, per-call provider latency metrics (p50/p95/p99), and SLO
snapshots.

All tenant scoping mirrors prior-phase conventions (workspace_id required
where workspace-scoped; organization_id optional but never bypasses a
workspace boundary).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float,
    ForeignKey, Index, UniqueConstraint,
)

from ..core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Vector model registry + index operations
# ---------------------------------------------------------------------------

class EmbeddingModel(Base):
    """Declared embedding model (provider, model, dimensions, version).

    Prevents silently mixing incompatible embeddings: chunks are tagged with
    the model version that produced them and retrieval validates the match.
    """

    __tablename__ = "embedding_models"
    __table_args__ = (
        UniqueConstraint("provider", "model", "dimensions", "version",
                         name="uq_embedding_model_version"),
        Index("ix_embedding_models_active", "active"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False)
    model = Column(String(120), nullable=False)
    dimensions = Column(Integer, nullable=False)
    version = Column(String(40), nullable=False, default="v1")
    active = Column(Boolean, nullable=False, default=True)
    notes = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class VectorIndexOp(Base):
    """Audit log of vector index operations (create/rebuild/validate)."""

    __tablename__ = "vector_index_ops"
    __table_args__ = (
        Index("ix_vector_index_ops_model", "embedding_model_id", "op_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model_id = Column(
        Integer, ForeignKey("embedding_models.id", ondelete="SET NULL"),
        nullable=True)
    op_type = Column(String(20), nullable=False)  # create/rebuild/validate
    index_name = Column(String(120), nullable=True)
    status = Column(String(20), nullable=False, default="RUNNING")
    detail = Column(String(1000), nullable=True)
    operator_user_id = Column(Integer, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Entity merge workflow (approval-gated)
# ---------------------------------------------------------------------------

class EntityMergeRequest(Base):
    """Human-approval-gated entity merge. Ambiguous merges are NEVER automatic."""

    __tablename__ = "entity_merge_requests"
    __table_args__ = (
        Index("ix_entity_merge_workspace_status", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    source_entity_id = Column(Integer,
                              ForeignKey("entities.id", ondelete="CASCADE"),
                              nullable=False)
    target_entity_id = Column(Integer,
                              ForeignKey("entities.id", ondelete="CASCADE"),
                              nullable=False)
    reason = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    # PENDING -> APPROVED / REJECTED / EXPIRED
    evidence_json = Column(Text, nullable=True)
    requested_by = Column(Integer, nullable=True)
    reviewed_by = Column(Integer, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Poisoned document quarantine
# ---------------------------------------------------------------------------

class PoisonDocument(Base):
    """Document quarantined after repeated stage failures.

    Quarantine is explicit and reviewable — the document is never silently
    deleted and an operator must resolve (retry/release/abandon) it.
    """

    __tablename__ = "poison_documents"
    __table_args__ = (
        Index("ix_poison_documents_workspace_status", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id",
                                             ondelete="CASCADE"),
                         nullable=False)
    stage = Column(String(20), nullable=False)
    failure_count = Column(Integer, nullable=False, default=0)
    last_error = Column(String(2000), nullable=True)
    status = Column(String(20), nullable=False, default="QUARANTINED")
    # QUARANTINED -> RELEASED / ABANDONED
    resolved_by = Column(Integer, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Import jobs (validation / dry-run / commit with rollback)
# ---------------------------------------------------------------------------

class ImportJob(Base):
    """Bounded import operation with validation-first semantics."""

    __tablename__ = "import_jobs"
    __table_args__ = (
        Index("ix_import_jobs_workspace_status", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, nullable=True)
    import_type = Column(String(40), nullable=False)  # documents/metadata/knowledge
    filename = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="VALIDATING")
    # VALIDATING -> READY / VALIDATION_FAILED / RUNNING -> COMMITTED / ROLLED_BACK / FAILED
    records_total = Column(Integer, nullable=False, default=0)
    records_valid = Column(Integer, nullable=False, default=0)
    records_invalid = Column(Integer, nullable=False, default=0)
    errors_json = Column(Text, nullable=True)
    staged_ref = Column(String(255), nullable=True)
    dry_run = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Search analytics (authorized, non-sensitive)
# ---------------------------------------------------------------------------

class SearchAnalyticsEvent(Base):
    """Aggregatable search metrics — never stores query text or results."""

    __tablename__ = "search_analytics"
    __table_args__ = (
        Index("ix_search_analytics_workspace_time", "workspace_id",
              "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    query_hash = Column(String(64), nullable=True)
    retrieval_mode = Column(String(20), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    result_count = Column(Integer, nullable=True)
    zero_results = Column(Boolean, nullable=False, default=False)
    useful_signal = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Provider call metrics (latency percentiles)
# ---------------------------------------------------------------------------

class ProviderCallMetric(Base):
    """Per-call provider latency samples for p50/p95/p99 computation."""

    __tablename__ = "provider_call_metrics"
    __table_args__ = (
        Index("ix_provider_call_metrics_provider_time", "provider",
              "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False)
    model = Column(String(120), nullable=True)
    ok = Column(Boolean, nullable=False, default=True)
    latency_ms = Column(Float, nullable=True)
    error_class = Column(String(40), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# SLO snapshots
# ---------------------------------------------------------------------------

class SloSnapshot(Base):
    """Periodic service-level objective snapshot (availability/latency/error)."""

    __tablename__ = "slo_snapshots"
    __table_args__ = (
        Index("ix_slo_snapshots_window", "window_start"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    availability = Column(Float, nullable=True)
    latency_p50_ms = Column(Float, nullable=True)
    latency_p95_ms = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    queue_age_max_s = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)