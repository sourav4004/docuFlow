"""Phase 18 migration — distributed AI cloud + enterprise production scale.

Adds: vector model registry + index ops, entity merge requests, poisoned
document quarantine, import jobs, search analytics, provider call metrics,
SLO snapshots, and worker capacity/load columns on worker_heartbeats.
"""

from alembic import op
import sqlalchemy as sa

revision = "025_phase18_cloud"
down_revision = "024_phase17_cloud"
branch_labels = None
depends_on = None


def _now() -> sa.DateTime:
    return sa.text("(now() at time zone 'utc')")


def upgrade() -> None:
    # --- worker capacity/load columns -------------------------------------
    op.add_column("worker_heartbeats",
                  sa.Column("active_jobs", sa.Integer(),
                            nullable=False, server_default="0"))
    op.add_column("worker_heartbeats",
                  sa.Column("load", sa.Float(), nullable=True))
    op.add_column("worker_heartbeats",
                  sa.Column("queue_assignments", sa.String(length=500),
                            nullable=True))

    # --- vector model registry --------------------------------------------
    op.create_table(
        "embedding_models",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=40), nullable=False,
                  server_default="v1"),
        sa.Column("active", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.UniqueConstraint("provider", "model", "dimensions", "version",
                            name="uq_embedding_model_version"),
    )
    op.create_index("ix_embedding_models_active", "embedding_models",
                    ["active"])

    # --- vector index operations ------------------------------------------
    op.create_table(
        "vector_index_ops",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("embedding_model_id", sa.Integer(), nullable=True),
        sa.Column("op_type", sa.String(length=20), nullable=False),
        sa.Column("index_name", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="RUNNING"),
        sa.Column("detail", sa.String(length=1000), nullable=True),
        sa.Column("operator_user_id", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["embedding_model_id"],
                                ["embedding_models.id"],
                                ondelete="SET NULL"),
    )
    op.create_index("ix_vector_index_ops_model", "vector_index_ops",
                    ["embedding_model_id", "op_type"])

    # --- entity merge requests --------------------------------------------
    op.create_table(
        "entity_merge_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("source_entity_id", sa.Integer(), nullable=False),
        sa.Column("target_entity_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="PENDING"),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("requested_by", sa.Integer(), nullable=True),
        sa.Column("reviewed_by", sa.Integer(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_entity_id"], ["entities.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_entity_id"], ["entities.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_entity_merge_workspace_status",
                    "entity_merge_requests", ["workspace_id", "status"])

    # --- poisoned document quarantine -------------------------------------
    op.create_table(
        "poison_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("last_error", sa.String(length=2000), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="QUARANTINED"),
        sa.Column("resolved_by", sa.Integer(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_poison_documents_workspace_status",
                    "poison_documents", ["workspace_id", "status"])

    # --- import jobs -------------------------------------------------------
    op.create_table(
        "import_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("import_type", sa.String(length=40), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="VALIDATING"),
        sa.Column("records_total", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("records_valid", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("records_invalid", sa.Integer(), nullable=False,
                  server_default="0"),
        sa.Column("errors_json", sa.Text(), nullable=True),
        sa.Column("staged_ref", sa.String(length=255), nullable=True),
        sa.Column("dry_run", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_import_jobs_workspace_status", "import_jobs",
                    ["workspace_id", "status"])

    # --- search analytics ---------------------------------------------------
    op.create_table(
        "search_analytics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("query_hash", sa.String(length=64), nullable=True),
        sa.Column("retrieval_mode", sa.String(length=20), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=True),
        sa.Column("zero_results", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),
        sa.Column("useful_signal", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_search_analytics_workspace_time", "search_analytics",
                    ["workspace_id", "created_at"])

    # --- provider call metrics ----------------------------------------------
    op.create_table(
        "provider_call_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=True),
        sa.Column("ok", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("error_class", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
    )
    op.create_index("ix_provider_call_metrics_provider_time",
                    "provider_call_metrics", ["provider", "created_at"])

    # --- SLO snapshots ------------------------------------------------------
    op.create_table(
        "slo_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("availability", sa.Float(), nullable=True),
        sa.Column("latency_p50_ms", sa.Float(), nullable=True),
        sa.Column("latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("error_rate", sa.Float(), nullable=True),
        sa.Column("queue_age_max_s", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=_now()),
    )
    op.create_index("ix_slo_snapshots_window", "slo_snapshots",
                    ["window_start"])


def downgrade() -> None:
    op.drop_table("slo_snapshots")
    op.drop_table("provider_call_metrics")
    op.drop_table("search_analytics")
    op.drop_table("import_jobs")
    op.drop_table("poison_documents")
    op.drop_table("entity_merge_requests")
    op.drop_table("vector_index_ops")
    op.drop_table("embedding_models")
    op.drop_column("worker_heartbeats", "queue_assignments")
    op.drop_column("worker_heartbeats", "load")
    op.drop_column("worker_heartbeats", "active_jobs")