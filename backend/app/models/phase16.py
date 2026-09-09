"""Phase 16 models — production worker platform, provider infrastructure,
multimodal intelligence, and operational observability.

All timestamps are UTC. Every table carrying tenant data is workspace/org
scoped. No secrets are stored in these tables (credentials live in config /
secret stores only).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from ..core.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Durable worker platform
# ---------------------------------------------------------------------------

JOB_STATUSES = (
    "QUEUED", "CLAIMED", "RUNNING", "COMPLETED", "FAILED",
    "DEAD_LETTERED", "CANCELLED",
)

JOB_QUEUES = (
    "AI_EXECUTIONS", "DOCUMENTS", "WORKFLOWS", "NOTIFICATIONS",
    "WEBHOOKS", "AGENTS", "REPORTS",
)

WORKER_STATUSES = ("STARTING", "RUNNING", "IDLE", "STOPPING", "STOPPED", "DEAD")


class WorkerJob(Base):
    """Durable logical job queue record (provider-agnostic).

    A job is claimed atomically (single UPDATE guarded by status) so two
    workers can never process the same job. Retries are bounded by
    ``max_attempts``; exhausted jobs move to DEAD_LETTERED.
    """

    __tablename__ = "worker_jobs"
    __table_args__ = (
        Index("ix_worker_jobs_queue_status", "queue_name", "status"),
        Index("ix_worker_jobs_workspace_status", "workspace_id", "status"),
        Index("ix_worker_jobs_next_retry", "next_retry_at"),
        Index("ix_worker_jobs_claimed", "claimed_by"),
        UniqueConstraint("workspace_id", "queue_name", "dedupe_key",
                         name="uq_worker_job_dedupe"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    queue_name = Column(String(40), nullable=False)
    job_type = Column(String(60), nullable=False)
    payload_json = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="QUEUED")
    priority = Column(String(20), nullable=False, default="NORMAL")

    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    attempt = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    dedupe_key = Column(String(128), nullable=True)
    run_after = Column(DateTime(timezone=True), nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)

    claimed_by = Column(String(64), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    lease_token = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    trace_id = Column(String(64), nullable=True)
    correlation_id = Column(String(64), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)

    workspace = relationship("Workspace")
    organization = relationship("Organization")
    user = relationship("User")

    def __repr__(self):
        return f"<WorkerJob id={self.id} q={self.queue_name} {self.job_type} status={self.status}>"


class WorkerHeartbeat(Base):
    """Heartbeat record for a running worker (never exposes secrets)."""

    __tablename__ = "worker_heartbeats"
    __table_args__ = (
        Index("ix_worker_heartbeats_status", "status"),
        Index("ix_worker_heartbeats_last", "last_heartbeat"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(64), nullable=False, unique=True)
    queue_name = Column(String(40), nullable=True)
    status = Column(String(20), nullable=False, default="STARTING")
    current_job_type = Column(String(60), nullable=True)
    current_job_id = Column(Integer, nullable=True)
    hostname = Column(String(255), nullable=True)
    pid = Column(Integer, nullable=True)
    version = Column(String(50), nullable=True)
    started_at = Column(DateTime(timezone=True), default=_utcnow)
    last_heartbeat = Column(DateTime(timezone=True), default=_utcnow)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    # Phase 18 — capacity + load signals (never machine secrets)
    active_jobs = Column(Integer, nullable=False, default=0)
    load = Column(Float, nullable=True)
    queue_assignments = Column(String(500), nullable=True)


# ---------------------------------------------------------------------------
# Provider capability + health infrastructure
# ---------------------------------------------------------------------------

class ProviderCapability(Base):
    """Declared capabilities of a (provider, model) pair.

    The router checks capabilities before selecting a model — a vision-only
    task never reaches a text-only model and vice versa.
    """

    __tablename__ = "provider_capabilities"
    __table_args__ = (
        UniqueConstraint("provider", "model", name="uq_provider_capability"),
        Index("ix_provider_caps_provider", "provider"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    supports_text = Column(Boolean, nullable=False, default=True)
    supports_vision = Column(Boolean, nullable=False, default=False)
    supports_tools = Column(Boolean, nullable=False, default=False)
    supports_structured = Column(Boolean, nullable=False, default=False)
    supports_streaming = Column(Boolean, nullable=False, default=False)
    context_window = Column(Integer, nullable=True)
    max_output = Column(Integer, nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    cost_per_1k_input = Column(Float, nullable=True)
    cost_per_1k_output = Column(Float, nullable=True)
    latency_class = Column(String(10), nullable=True)  # fast / medium / slow
    notes = Column(String(500), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class EmbeddingCache(Base):
    """Deterministic embedding cache.

    The key is a hash of provider|model|dimensions|content-hash. When the
    deployment marks embeddings tenant-sensitive, the key additionally binds
    the workspace id so results are never reused across tenants.
    """

    __tablename__ = "embedding_cache"
    __table_args__ = (
        Index("ix_embedding_cache_hash", "content_hash"),
        Index("ix_embedding_cache_expires", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(64), nullable=False, unique=True)
    provider = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    dimensions = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    embedding_json = Column(Text, nullable=False)
    hits = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Observability — trace spans
# ---------------------------------------------------------------------------

SPAN_TYPES = (
    "request", "planning", "retrieval", "rerank", "provider", "tool",
    "validation", "output", "workflow", "agent",
)

SPAN_STATUSES = ("OK", "ERROR", "TIMEOUT", "RATE_LIMITED", "CANCELLED")


class TraceSpan(Base):
    """One span of an AI trace (never stores raw prompts by default).

    ``input_summary`` holds an optional truncated/redacted summary only.
    """

    __tablename__ = "trace_spans"
    __table_args__ = (
        Index("ix_trace_spans_trace_id", "trace_id"),
        Index("ix_trace_spans_workspace", "workspace_id"),
        Index("ix_trace_spans_type", "span_type"),
        Index("ix_trace_spans_execution", "execution_id"),
        Index("ix_trace_spans_started", "started_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    span_id = Column(String(36), nullable=False, unique=True)
    trace_id = Column(String(64), nullable=False)
    parent_span_id = Column(String(36), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    execution_id = Column(String(36), nullable=True)
    workflow_execution_id = Column(String(64), nullable=True)
    node_execution_id = Column(String(64), nullable=True)
    span_type = Column(String(30), nullable=False)
    status = Column(String(20), nullable=False, default="OK")
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    latency_ms = Column(Float, nullable=True)
    model = Column(String(100), nullable=True)
    provider = Column(String(100), nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    input_summary = Column(String(500), nullable=True)
    error_class = Column(String(40), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Multimodal document intelligence — pages
# ---------------------------------------------------------------------------

class DocumentPage(Base):
    """Page-level abstraction of an ingested document.

    Layout and table structure are stored as JSON so downstream consumers
    (retrieval, table intelligence, OCR quality) can use them without
    re-parsing raw text.
    """

    __tablename__ = "document_pages"
    __table_args__ = (
        Index("ix_document_pages_document", "document_id"),
        UniqueConstraint("document_id", "version", "page_number",
                         name="uq_document_page"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=True)
    layout_json = Column(Text, nullable=True)   # reading-order blocks with kinds
    tables_json = Column(Text, nullable=True)   # normalized table regions
    image_regions_json = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    ocr_provider = Column(String(60), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    document = relationship("Document")


# ---------------------------------------------------------------------------
# Memory 2.0 — conflicts
# ---------------------------------------------------------------------------

class MemoryConflict(Base):
    """Conflicting AI memories are preserved, never silently overwritten."""

    __tablename__ = "memory_conflicts"
    __table_args__ = (
        Index("ix_memory_conflicts_workspace", "workspace_id"),
        Index("ix_memory_conflicts_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    memory_a_id = Column(Integer, ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False)
    memory_b_id = Column(Integer, ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False)
    conflict_type = Column(String(30), nullable=False, default="CONTRADICTION")
    description = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN/RESOLVED
    resolved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    workspace = relationship("Workspace")
    memory_a = relationship("AIMemory", foreign_keys=[memory_a_id])
    memory_b = relationship("AIMemory", foreign_keys=[memory_b_id])


# ---------------------------------------------------------------------------
# Entity 3.0 — changes
# ---------------------------------------------------------------------------

ENTITY_CHANGE_TYPES = (
    "NEW_ENTITY", "RENAMED", "ATTRIBUTE_CHANGED",
    "RELATIONSHIP_CHANGED", "RELATIONSHIP_REMOVED",
)


class EntityChange(Base):
    """Evidence-backed entity change event across document versions."""

    __tablename__ = "entity_changes"
    __table_args__ = (
        Index("ix_entity_changes_workspace", "workspace_id"),
        Index("ix_entity_changes_entity", "entity_id"),
        Index("ix_entity_changes_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    change_type = Column(String(40), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    evidence = Column(Text, nullable=True)
    confidence = Column(String(10), nullable=False, default="MEDIUM")
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    entity = relationship("Entity")


# ---------------------------------------------------------------------------
# Backfill audit
# ---------------------------------------------------------------------------

BACKFILL_KINDS = ("document_workspace", "collection_workspace")
BACKFILL_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "DRY_RUN")


class BackfillRun(Base):
    """Auditable, resumable, batch-based legacy workspace backfill run."""

    __tablename__ = "backfill_runs"
    __table_args__ = (
        Index("ix_backfill_runs_kind_status", "kind", "status"),
        Index("ix_backfill_runs_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(30), nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")
    dry_run = Column(Boolean, nullable=False, default=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=False, default=100)
    cursor_id = Column(Integer, nullable=True)
    total = Column(Integer, nullable=False, default=0)
    processed = Column(Integer, nullable=False, default=0)
    assigned = Column(Integer, nullable=False, default=0)
    skipped = Column(Integer, nullable=False, default=0)
    ambiguous = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    error_summary = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Notification orchestration
# ---------------------------------------------------------------------------

class NotificationDelivery(Base):
    """Per-channel delivery record for a notification (dedupe per channel)."""

    __tablename__ = "notification_deliveries"
    __table_args__ = (
        Index("ix_notif_delivery_notification", "notification_id"),
        Index("ix_notif_delivery_status", "status"),
        UniqueConstraint("notification_id", "channel", name="uq_notif_delivery_channel"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    notification_id = Column(Integer, ForeignKey("notifications.id", ondelete="CASCADE"), nullable=False)
    channel = Column(String(20), nullable=False)  # IN_APP / EMAIL / WEBHOOK
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING/SENT/SKIPPED/FAILED
    provider = Column(String(60), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    notification = relationship("Notification")
