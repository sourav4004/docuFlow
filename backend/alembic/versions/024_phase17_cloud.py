"""add phase17 distributed cloud tables

Revision ID: 024_add_phase17_distributed_cloud
Revises: 023_add_phase16_platform
Create Date: 2026-09-03

Phase 17 — Distributed AI + Enterprise Knowledge Cloud.
"""

import sqlalchemy as sa
from alembic import op

revision = "024_phase17_cloud"
down_revision = "023_add_phase16_platform"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # --- worker_jobs: explicit lease columns (additive) ---
    op.add_column("worker_jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("worker_jobs", sa.Column("lease_token", sa.String(length=64), nullable=True))

    # --- job_leases ---
    op.create_table(
        "job_leases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(), sa.ForeignKey("worker_jobs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("worker_id", sa.String(length=64), nullable=False),
        sa.Column("lease_token", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_leases_expiry", "job_leases", ["expires_at"])
    op.create_index("ix_job_leases_worker", "job_leases", ["worker_id"])

    # --- ingestion_runs / ingestion_stages ---
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("batch_json", sa.Text(), nullable=True),
        sa.Column("current_stage", sa.String(length=20), nullable=False, server_default="UPLOAD"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="RUNNING"),
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_page", sa.Integer(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("idempotency_key", sa.String(length=120), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_ingestion_run_idem"),
    )
    op.create_index("ix_ingestion_runs_workspace_status", "ingestion_runs", ["workspace_id", "status"])
    op.create_index("ix_ingestion_runs_document", "ingestion_runs", ["document_id"])
    op.create_index("ix_ingestion_runs_idem", "ingestion_runs", ["workspace_id", "idempotency_key"])

    op.create_table(
        "ingestion_stages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("ingestion_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(length=120), nullable=True),
        sa.Column("output_reference", sa.String(length=255), nullable=True),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("run_id", "stage", name="uq_ingestion_stage"),
    )
    op.create_index("ix_ingestion_stages_run", "ingestion_stages", ["run_id", "stage"])

    # --- document_fingerprints / duplicate_candidates ---
    op.create_table(
        "document_fingerprints",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("version_id", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("structural_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata_hash", sa.String(length=64), nullable=False),
        sa.Column("shingles_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fingerprints_content", "document_fingerprints", ["content_hash"])

    op.create_table(
        "duplicate_candidates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("other_document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("classification", sa.String(length=30), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("method", sa.String(length=40), nullable=False, server_default="fingerprint"),
        sa.Column("reviewed", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("document_id", "other_document_id", name="uq_duplicate_pair"),
    )
    op.create_index("ix_duplicate_candidates_doc", "duplicate_candidates", ["document_id"])

    # --- connector federation ---
    op.create_table(
        "connector_sources",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("config_ref", sa.String(length=255), nullable=True),
        sa.Column("scopes_json", sa.Text(), nullable=True),
        sa.Column("permissions_json", sa.Text(), nullable=True),
        sa.Column("allowed_domains_json", sa.Text(), nullable=True),
        sa.Column("credential_ref", sa.String(length=255), nullable=True),
        sa.Column("retention_days", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_connector_sources_workspace", "connector_sources", ["workspace_id"])

    op.create_table(
        "connector_syncs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("connector_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="RUNNING"),
        sa.Column("cursor_json", sa.Text(), nullable=True),
        sa.Column("items_added", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_changed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_connector_syncs_source", "connector_syncs", ["source_id", "status"])

    op.create_table(
        "connector_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("connector_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("external_parent_id", sa.String(length=255), nullable=True),
        sa.Column("item_type", sa.String(length=40), nullable=False, server_default="document"),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("payload_ref", sa.String(length=255), nullable=True),
        sa.Column("deleted", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_id", "external_id", name="uq_connector_item_ext"),
    )
    op.create_index("ix_connector_items_source", "connector_items", ["source_id", "external_id"])

    # --- knowledge graph 4.0 ---
    op.create_table(
        "entity_candidates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_id", sa.Integer(), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_id", sa.Integer(), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("method", sa.String(length=40), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_entity_candidates_entity", "entity_candidates", ["entity_id", "status"])

    op.create_table(
        "relationship_suggestions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_a_id", sa.Integer(), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_b_id", sa.Integer(), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(length=60), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("evidence", sa.String(length=1000), nullable=True),
        sa.Column("method", sa.String(length=40), nullable=False, server_default="shared_evidence"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rel_suggestions_entity_a", "relationship_suggestions", ["entity_a_id", "status"])

    # --- memory supersession ---
    op.create_table(
        "memory_supersessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("old_memory_id", sa.Integer(), sa.ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("new_memory_id", sa.Integer(), sa.ForeignKey("ai_memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_supersessions_old", "memory_supersessions", ["old_memory_id"])

    # --- agent plans + human handoffs ---
    op.create_table(
        "agent_plans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("objective", sa.String(length=1000), nullable=False),
        sa.Column("plan_json", sa.Text(), nullable=False),
        sa.Column("risk", sa.String(length=20), nullable=False, server_default="LOW"),
        sa.Column("budgets_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="VALIDATED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_plans_execution", "agent_plans", ["execution_id"])

    op.create_table(
        "human_handoffs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.String(length=1000), nullable=False),
        sa.Column("context_ref", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("answered_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_human_handoffs_execution", "human_handoffs", ["execution_id", "status"])

    # --- workflow runs ---
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("workflow_version_id", sa.Integer(), nullable=True),
        sa.Column("definition_hash", sa.String(length=64), nullable=False),
        sa.Column("definition_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="RUNNING"),
        sa.Column("control_state", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("node_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("node_state_json", sa.Text(), nullable=True),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("idle_timeout_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pause_reason", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(length=1000), nullable=True),
    )
    op.create_index("ix_workflow_runs_workspace_status", "workflow_runs", ["workspace_id", "status"])

    # --- governance: AI policy rules, retention assignments, legal holds ---
    op.create_table(
        "ai_policy_rules",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("feature", sa.String(length=60), nullable=True),
        sa.Column("rule_type", sa.String(length=30), nullable=False),
        sa.Column("allowlist_json", sa.Text(), nullable=True),
        sa.Column("deny_json", sa.Text(), nullable=True),
        sa.Column("sensitivity_max", sa.String(length=20), nullable=True),
        sa.Column("budget_max_usd", sa.Float(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ai_policy_rules_org", "ai_policy_rules", ["organization_id"])
    op.create_index("ix_ai_policy_rules_workspace", "ai_policy_rules", ["workspace_id"])

    op.create_table(
        "retention_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("entity_type", sa.String(length=40), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False, server_default="90"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retention_assignments_scope", "retention_assignments", ["workspace_id", "entity_type"])

    op.create_table(
        "legal_holds",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("started_by", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_by", sa.Integer(), nullable=True),
    )
    op.create_index("ix_legal_holds_workspace", "legal_holds", ["workspace_id", "status"])

    op.create_table(
        "hold_entities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("hold_id", sa.Integer(), sa.ForeignKey("legal_holds.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_type", sa.String(length=40), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
    )
    op.create_index("ix_hold_entities_hold", "hold_entities", ["hold_id"])
    op.create_index("ix_hold_entities_target", "hold_entities", ["entity_type", "entity_id"])

    # --- vector backfill, cost anomalies, alerts, api key events ---
    op.create_table(
        "vector_backfill_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PREVIEW"),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errors_json", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_vector_backfill_runs_status", "vector_backfill_runs", ["status"])

    op.create_table(
        "cost_anomalies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("workspace_id", sa.Integer(), nullable=True),
        sa.Column("period", sa.String(length=20), nullable=False),
        sa.Column("metric", sa.String(length=40), nullable=False),
        sa.Column("expected", sa.Float(), nullable=False),
        sa.Column("actual", sa.Float(), nullable=False),
        sa.Column("deviation", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False, server_default="MEDIUM"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="OPEN"),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cost_anomalies_scope", "cost_anomalies", ["organization_id", "workspace_id"])

    op.create_table(
        "alert_rules",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("metric", sa.String(length=60), nullable=False),
        sa.Column("operator", sa.String(length=10), nullable=False, server_default=">"),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False, server_default="WARNING"),
        sa.Column("cooldown_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_rules_workspace", "alert_rules", ["workspace_id", "enabled"])

    op.create_table(
        "alert_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("rule_id", sa.Integer(), sa.ForeignKey("alert_rules.id", ondelete="CASCADE"), nullable=True),
        sa.Column("workspace_id", sa.Integer(), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=False, server_default="WARNING"),
        sa.Column("message", sa.String(length=1000), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_events_rule", "alert_events", ["rule_id", "fired_at"])

    op.create_table(
        "api_key_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("detail", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_api_key_events_key", "api_key_events", ["api_key_id", "created_at"])

    if bind.dialect.name == "postgresql":
        op.execute("""CREATE INDEX IF NOT EXISTS ix_worker_jobs_lease_expiry
                      ON worker_jobs (lease_expires_at)
                      WHERE lease_expires_at IS NOT NULL""")
    else:
        op.create_index("ix_worker_jobs_lease_expiry", "worker_jobs", ["lease_expires_at"])


def downgrade() -> None:
    for table in (
        "api_key_events", "alert_events", "alert_rules", "cost_anomalies",
        "vector_backfill_runs", "hold_entities", "legal_holds",
        "retention_assignments", "ai_policy_rules", "workflow_runs",
        "human_handoffs", "agent_plans", "memory_supersessions",
        "relationship_suggestions", "entity_candidates", "connector_items",
        "connector_syncs", "connector_sources", "duplicate_candidates",
        "document_fingerprints", "ingestion_stages", "ingestion_runs",
        "job_leases",
    ):
        op.drop_table(table)
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_worker_jobs_lease_expiry")
    else:
        op.drop_index("ix_worker_jobs_lease_expiry", table_name="worker_jobs")
    op.drop_column("worker_jobs", "lease_token")
    op.drop_column("worker_jobs", "lease_expires_at")
