"""add phase16 platform tables

Revision ID: 023
Revises: 022
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa

revision = "023_add_phase16_platform"
down_revision = "022_add_artifact_family_id"
branch_labels = None
depends_on = None


def _unique_name(prefix, *parts):
    # Alembic constraint names must be <= 63 chars on PostgreSQL.
    raw = "_".join([prefix] + [p for p in parts if p])
    return raw[:63]


def upgrade():
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # ------------------------------------------------------------------
    # Worker platform
    # ------------------------------------------------------------------
    op.create_table(
        "worker_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("queue_name", sa.String(40), nullable=False),
        sa.Column("job_type", sa.String(60), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="QUEUED"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="NORMAL"),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("dedupe_key", sa.String(128), nullable=True),
        sa.Column("run_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_by", sa.String(64), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("correlation_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", "queue_name", "dedupe_key", name="uq_worker_job_dedupe"),
    )
    op.create_index("ix_worker_jobs_queue_status", "worker_jobs", ["queue_name", "status"])
    op.create_index("ix_worker_jobs_workspace_status", "worker_jobs", ["workspace_id", "status"])
    op.create_index("ix_worker_jobs_next_retry", "worker_jobs", ["next_retry_at"])
    op.create_index("ix_worker_jobs_claimed", "worker_jobs", ["claimed_by"])

    op.create_table(
        "worker_heartbeats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("worker_id", sa.String(64), nullable=False, unique=True),
        sa.Column("queue_name", sa.String(40), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="STARTING"),
        sa.Column("current_job_type", sa.String(60), nullable=True),
        sa.Column("current_job_id", sa.Integer(), nullable=True),
        sa.Column("hostname", sa.String(255), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("version", sa.String(50), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_worker_heartbeats_status", "worker_heartbeats", ["status"])
    op.create_index("ix_worker_heartbeats_last", "worker_heartbeats", ["last_heartbeat"])

    # ------------------------------------------------------------------
    # Provider capabilities + embedding cache
    # ------------------------------------------------------------------
    op.create_table(
        "provider_capabilities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("supports_text", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("supports_vision", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("supports_tools", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("supports_structured", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("supports_streaming", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("max_output", sa.Integer(), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column("cost_per_1k_input", sa.Float(), nullable=True),
        sa.Column("cost_per_1k_output", sa.Float(), nullable=True),
        sa.Column("latency_class", sa.String(10), nullable=True),
        sa.Column("notes", sa.String(500), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("provider", "model", name="uq_provider_capability"),
    )
    op.create_index("ix_provider_caps_provider", "provider_capabilities", ["provider"])

    op.create_table(
        "embedding_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cache_key", sa.String(64), nullable=False, unique=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_embedding_cache_hash", "embedding_cache", ["content_hash"])
    op.create_index("ix_embedding_cache_expires", "embedding_cache", ["expires_at"])

    # ------------------------------------------------------------------
    # Trace spans
    # ------------------------------------------------------------------
    op.create_table(
        "trace_spans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("span_id", sa.String(36), nullable=False, unique=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("parent_span_id", sa.String(36), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("execution_id", sa.String(36), nullable=True),
        sa.Column("workflow_execution_id", sa.String(64), nullable=True),
        sa.Column("node_execution_id", sa.String(64), nullable=True),
        sa.Column("span_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="OK"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("provider", sa.String(100), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("input_summary", sa.String(500), nullable=True),
        sa.Column("error_class", sa.String(40), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_trace_spans_trace_id", "trace_spans", ["trace_id"])
    op.create_index("ix_trace_spans_workspace", "trace_spans", ["workspace_id"])
    op.create_index("ix_trace_spans_type", "trace_spans", ["span_type"])
    op.create_index("ix_trace_spans_execution", "trace_spans", ["execution_id"])
    op.create_index("ix_trace_spans_started", "trace_spans", ["started_at"])

    # ------------------------------------------------------------------
    # Document pages (multimodal)
    # ------------------------------------------------------------------
    op.create_table(
        "document_pages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("layout_json", sa.Text(), nullable=True),
        sa.Column("tables_json", sa.Text(), nullable=True),
        sa.Column("image_regions_json", sa.Text(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("ocr_provider", sa.String(60), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("document_id", "version", "page_number", name="uq_document_page"),
    )
    op.create_index("ix_document_pages_document", "document_pages", ["document_id"])

    # ------------------------------------------------------------------
    # Memory conflicts
    # ------------------------------------------------------------------
    op.create_table(
        "memory_conflicts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_a_id", sa.Integer(), sa.ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_b_id", sa.Integer(), sa.ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("conflict_type", sa.String(30), nullable=False, server_default="CONTRADICTION"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="OPEN"),
        sa.Column("resolved_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_memory_conflicts_workspace", "memory_conflicts", ["workspace_id"])
    op.create_index("ix_memory_conflicts_status", "memory_conflicts", ["status"])

    # ------------------------------------------------------------------
    # Entity changes
    # ------------------------------------------------------------------
    op.create_table(
        "entity_changes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_id", sa.Integer(), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("change_type", sa.String(40), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("old_value", sa.Text(), nullable=True),
        sa.Column("new_value", sa.Text(), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("confidence", sa.String(10), nullable=False, server_default="MEDIUM"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_entity_changes_workspace", "entity_changes", ["workspace_id"])
    op.create_index("ix_entity_changes_entity", "entity_changes", ["entity_id"])
    op.create_index("ix_entity_changes_created", "entity_changes", ["created_at"])

    # ------------------------------------------------------------------
    # Backfill runs
    # ------------------------------------------------------------------
    op.create_table(
        "backfill_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("batch_size", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("cursor_id", sa.Integer(), nullable=True),
        sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assigned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ambiguous", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_backfill_runs_kind_status", "backfill_runs", ["kind", "status"])
    op.create_index("ix_backfill_runs_created", "backfill_runs", ["created_at"])

    # ------------------------------------------------------------------
    # Notification deliveries
    # ------------------------------------------------------------------
    op.create_table(
        "notification_deliveries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("notification_id", sa.Integer(), sa.ForeignKey("notifications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("provider", sa.String(60), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("notification_id", "channel", name="uq_notif_delivery_channel"),
    )
    op.create_index("ix_notif_delivery_notification", "notification_deliveries", ["notification_id"])
    op.create_index("ix_notif_delivery_status", "notification_deliveries", ["status"])

    # ------------------------------------------------------------------
    # Column extensions
    # ------------------------------------------------------------------
    op.add_column("ai_memories", sa.Column("lifecycle_status", sa.String(20), nullable=False, server_default="ACTIVE"))
    op.add_column("ai_memories", sa.Column("supersedes_id", sa.Integer(), nullable=True))
    if is_postgres:
        op.create_foreign_key("fk_ai_memories_supersedes", "ai_memories", "ai_memories", ["supersedes_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_ai_memories_lifecycle", "ai_memories", ["lifecycle_status"])

    for col, typ in [
        ("subject", sa.String(255)),
        ("action", sa.String(100)),
        ("threshold_unit", sa.String(50)),
        ("timeframe_text", sa.String(255)),
        ("scope_text", sa.String(255)),
    ]:
        op.add_column("policy_statements", sa.Column(col, typ, nullable=True))
    op.add_column("policy_statements", sa.Column("threshold_value", sa.Float(), nullable=True))
    op.add_column("policy_statements", sa.Column("condition_text", sa.Text(), nullable=True))
    op.add_column("policy_statements", sa.Column("exception_text", sa.Text(), nullable=True))

    op.add_column("entities", sa.Column("normalized_name", sa.String(255), nullable=True))
    op.add_column("entities", sa.Column("source_count", sa.Integer(), nullable=False, server_default="1"))
    op.create_index("ix_entities_normalized", "entities", ["normalized_name"])

    op.add_column("entity_relationships", sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True))
    op.add_column("entity_relationships", sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("entity_relationships", sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("entity_relationships", sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.create_index("ix_entity_rels_current", "entity_relationships", ["is_current"])


def downgrade():
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    op.drop_index("ix_entity_rels_current", table_name="entity_relationships")
    op.drop_column("entity_relationships", "is_current")
    op.drop_column("entity_relationships", "observed_at")
    op.drop_column("entity_relationships", "valid_until")
    op.drop_column("entity_relationships", "valid_from")
    op.drop_index("ix_entities_normalized", table_name="entities")
    op.drop_column("entities", "source_count")
    op.drop_column("entities", "normalized_name")
    op.drop_column("policy_statements", "exception_text")
    op.drop_column("policy_statements", "condition_text")
    op.drop_column("policy_statements", "threshold_value")
    op.drop_column("policy_statements", "scope_text")
    op.drop_column("policy_statements", "timeframe_text")
    op.drop_column("policy_statements", "threshold_unit")
    op.drop_column("policy_statements", "action")
    op.drop_column("policy_statements", "subject")
    op.drop_index("ix_ai_memories_lifecycle", table_name="ai_memories")
    if is_postgres:
        op.drop_constraint("fk_ai_memories_supersedes", "ai_memories", type_="foreignkey")
    op.drop_column("ai_memories", "supersedes_id")
    op.drop_column("ai_memories", "lifecycle_status")

    op.drop_index("ix_notif_delivery_status", table_name="notification_deliveries")
    op.drop_index("ix_notif_delivery_notification", table_name="notification_deliveries")
    op.drop_table("notification_deliveries")
    op.drop_index("ix_backfill_runs_created", table_name="backfill_runs")
    op.drop_index("ix_backfill_runs_kind_status", table_name="backfill_runs")
    op.drop_table("backfill_runs")
    op.drop_index("ix_entity_changes_created", table_name="entity_changes")
    op.drop_index("ix_entity_changes_entity", table_name="entity_changes")
    op.drop_index("ix_entity_changes_workspace", table_name="entity_changes")
    op.drop_table("entity_changes")
    op.drop_index("ix_memory_conflicts_status", table_name="memory_conflicts")
    op.drop_index("ix_memory_conflicts_workspace", table_name="memory_conflicts")
    op.drop_table("memory_conflicts")
    op.drop_index("ix_document_pages_document", table_name="document_pages")
    op.drop_table("document_pages")
    op.drop_index("ix_trace_spans_started", table_name="trace_spans")
    op.drop_index("ix_trace_spans_execution", table_name="trace_spans")
    op.drop_index("ix_trace_spans_type", table_name="trace_spans")
    op.drop_index("ix_trace_spans_workspace", table_name="trace_spans")
    op.drop_index("ix_trace_spans_trace_id", table_name="trace_spans")
    op.drop_table("trace_spans")
    op.drop_index("ix_embedding_cache_expires", table_name="embedding_cache")
    op.drop_index("ix_embedding_cache_hash", table_name="embedding_cache")
    op.drop_table("embedding_cache")
    op.drop_index("ix_provider_caps_provider", table_name="provider_capabilities")
    op.drop_table("provider_capabilities")
    op.drop_index("ix_worker_heartbeats_last", table_name="worker_heartbeats")
    op.drop_index("ix_worker_heartbeats_status", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")
    op.drop_index("ix_worker_jobs_claimed", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_next_retry", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_workspace_status", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_queue_status", table_name="worker_jobs")
    op.drop_table("worker_jobs")
