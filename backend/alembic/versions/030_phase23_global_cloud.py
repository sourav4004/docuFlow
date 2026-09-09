"""Phase 23 migration — Global AI Cloud Platform.

Adds 17 tables on top of the Phase 22 substrate:

- zero-downtime storage migration (storage_migration_plans,
  storage_migration_objects) — resumable, checksum-verified, idempotent
- provider production gate (provider_readiness_scores,
  provider_routing_decisions) — readiness + deterministic routing audit
- real data residency enforcement (residency_decision_logs) — per-operation
  evidence extending the Phase 19 ResidencyRule policy tables
- vector model migration + search coexistence (embedding_model_versions,
  document_embedding_status, search_shadow_comparisons) — immutable model
  versions, per-document resumable status, dual-index shadow comparisons
- region drain (region_drain_operations) — bounded audited drain
- dependency graph (dependency_edges) — blast-radius computation
- knowledge freshness (knowledge_freshness_states)
- data consistency engine (consistency_check_runs) — report + repair plan,
  never silent mutation
- request deduplication (request_dedup_records) — idempotency key identity
- scheduler dedup (scheduler_task_runs) — no duplicate scheduled side effects
- webhook reliability (webhook_delivery_attempts, webhook_endpoint_health)
- review decisions (review_decisions) — audits Phase 15 ReviewItem lifecycle

Forward compatible: additive only, no destructive operations, every new
table has tenant-scoped indexes; downgrade drops leaf tables only.
"""

from alembic import op
import sqlalchemy as sa

revision = "030_phase23_global_cloud"
down_revision = "029_phase22_production_cloud"
branch_labels = None
depends_on = None


def _now() -> sa.TextClause:
    return sa.text("(now() at time zone 'utc')")


def upgrade() -> None:
    bind = op.get_bind()

    op.create_table(
        "storage_migration_plans",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("organization_id", sa.Integer, nullable=True),
        sa.Column("source_backend", sa.String(30), nullable=False, server_default="local"),
        sa.Column("target_backend", sa.String(30), nullable=False, server_default="s3"),
        sa.Column("status", sa.String(20), nullable=False, server_default="DRAFT"),
        sa.Column("batch_size", sa.Integer, nullable=False, server_default="25"),
        sa.Column("dry_run", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("total_objects", sa.Integer, nullable=False, server_default="0"),
        sa.Column("migrated_objects", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed_objects", sa.Integer, nullable=False, server_default="0"),
        sa.Column("verified_objects", sa.Integer, nullable=False, server_default="0"),
        sa.Column("progress_json", sa.Text, nullable=True),
        sa.Column("audit_json", sa.Text, nullable=True),
        sa.Column("created_by", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ix_smp_workspace_status", "storage_migration_plans",
                    ["workspace_id", "status"])

    op.create_table(
        "storage_migration_objects",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("plan_id", sa.Integer, nullable=False, index=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("document_id", sa.Integer, nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", sa.String(1000), nullable=True),
        sa.Column("source_checksum", sa.String(128), nullable=True),
        sa.Column("target_checksum", sa.String(128), nullable=True),
        sa.Column("migrated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("plan_id", "document_id", name="uq_smo_plan_doc"),
    )
    op.create_index("ix_smo_plan_state", "storage_migration_objects",
                    ["plan_id", "state"])

    op.create_table(
        "provider_readiness_scores",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("provider_kind", sa.String(40), nullable=False, index=True),
        sa.Column("environment", sa.String(40), nullable=False, server_default="default"),
        sa.Column("score", sa.Float, nullable=False, server_default="0"),
        sa.Column("validated_capabilities_json", sa.Text, nullable=True),
        sa.Column("failure_summary_json", sa.Text, nullable=True),
        sa.Column("last_validation_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("provider_kind", "environment", name="uq_prs_kind_env"),
    )

    op.create_table(
        "provider_routing_decisions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("organization_id", sa.Integer, nullable=True),
        sa.Column("operation", sa.String(60), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("selected_provider", sa.String(60), nullable=True),
        sa.Column("selected_model", sa.String(120), nullable=True),
        sa.Column("fallback_provider", sa.String(60), nullable=True),
        sa.Column("reasons_json", sa.Text, nullable=True),
        sa.Column("candidate_ranking_json", sa.Text, nullable=True),
        sa.Column("estimated_cost", sa.Float, nullable=True),
        sa.Column("sensitivity", sa.String(20), nullable=True),
        sa.Column("region", sa.String(60), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ix_prd_workspace", "provider_routing_decisions",
                    ["workspace_id", "created_at"])

    op.create_table(
        "residency_decision_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("organization_id", sa.Integer, nullable=True),
        sa.Column("operation", sa.String(60), nullable=False),
        sa.Column("source_region", sa.String(60), nullable=True),
        sa.Column("destination_region", sa.String(60), nullable=True),
        sa.Column("classification", sa.String(20), nullable=True),
        sa.Column("policy_id", sa.Integer, nullable=True),
        sa.Column("allowed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.String(600), nullable=True),
        sa.Column("actor", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ix_rdl_workspace", "residency_decision_logs",
                    ["workspace_id", "created_at"])

    op.create_table(
        "embedding_model_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(60), nullable=False),
        sa.Column("dimension", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("model_name", "version", name="uq_emv_name_ver"),
    )

    op.create_table(
        "document_embedding_status",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("document_id", sa.Integer, nullable=False),
        sa.Column("model_version_id", sa.Integer, nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("chunks_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunks_done", sa.Integer, nullable=False, server_default="0"),
        sa.Column("quality_delta", sa.Float, nullable=True),
        sa.Column("last_error", sa.String(1000), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("workspace_id", "document_id", "model_version_id",
                            name="uq_des_doc_model"),
    )
    op.create_index("des_workspace_state", "document_embedding_status",
                    ["workspace_id", "state"])

    op.create_table(
        "search_shadow_comparisons",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("query", sa.String(500), nullable=False),
        sa.Column("baseline_ranking_json", sa.Text, nullable=True),
        sa.Column("candidate_ranking_json", sa.Text, nullable=True),
        sa.Column("overlap_at_5", sa.Float, nullable=True),
        sa.Column("mrr_delta", sa.Float, nullable=True),
        sa.Column("verdict", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ssc_workspace", "search_shadow_comparisons",
                    ["workspace_id", "created_at"])

    op.create_table(
        "region_drain_operations",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("region", sa.String(60), nullable=False, index=True),
        sa.Column("organization_id", sa.Integer, nullable=True),
        sa.Column("state", sa.String(20), nullable=False, server_default="DRAINING"),
        sa.Column("jobs_retried", sa.Integer, nullable=False, server_default="0"),
        sa.Column("jobs_marked", sa.Integer, nullable=False, server_default="0"),
        sa.Column("jobs_remaining", sa.Integer, nullable=False, server_default="0"),
        sa.Column("progress_json", sa.Text, nullable=True),
        sa.Column("actor", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("rdo_region_state", "region_drain_operations",
                    ["region", "state"])

    op.create_table(
        "dependency_edges",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("component", sa.String(60), nullable=False, index=True),
        sa.Column("depends_on", sa.String(60), nullable=False, index=True),
        sa.Column("criticality", sa.String(20), nullable=False, server_default="HARD"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("component", "depends_on", name="uq_dep_edge"),
    )

    op.create_table(
        "knowledge_freshness_states",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("document_id", sa.Integer, nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="UNKNOWN"),
        sa.Column("age_days", sa.Float, nullable=True),
        sa.Column("source_reliability", sa.Float, nullable=True),
        sa.Column("expiration_days", sa.Integer, nullable=True),
        sa.Column("reasons_json", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("workspace_id", "document_id", name="uq_kfs_doc"),
    )
    op.create_index("kfs_workspace_state", "knowledge_freshness_states",
                    ["workspace_id", "state"])

    op.create_table(
        "consistency_check_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=True, index=True),
        sa.Column("domain", sa.String(40), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="COMPLETED"),
        sa.Column("items_checked", sa.Integer, nullable=False, server_default="0"),
        sa.Column("issues_found", sa.Integer, nullable=False, server_default="0"),
        sa.Column("repaired", sa.Integer, nullable=False, server_default="0"),
        sa.Column("findings_json", sa.Text, nullable=True),
        sa.Column("repair_plan_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ccr_domain", "consistency_check_runs",
                    ["domain", "created_at"])

    op.create_table(
        "request_dedup_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_hash", sa.String(128), nullable=False),
        sa.Column("response_status", sa.Integer, nullable=True),
        sa.Column("response_json", sa.Text, nullable=True),
        sa.Column("state", sa.String(20), nullable=False, server_default="IN_FLIGHT"),
        sa.Column("conflict", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_rdr_key"),
    )
    op.create_index("ix_rdr_expires", "request_dedup_records", ["expires_at"])

    op.create_table(
        "scheduler_task_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task_key", sa.String(120), nullable=False),
        sa.Column("due_bucket", sa.String(40), nullable=False),
        sa.Column("leader_id", sa.String(120), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="RUNNING"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detail_json", sa.Text, nullable=True),
        sa.UniqueConstraint("task_key", "due_bucket", name="uq_str_task_bucket"),
    )
    op.create_index("ix_str_started", "scheduler_task_runs", ["started_at"])

    op.create_table(
        "webhook_delivery_attempts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("endpoint_id", sa.Integer, nullable=False, index=True),
        sa.Column("delivery_id", sa.String(80), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("response_status", sa.Integer, nullable=True),
        sa.Column("last_error", sa.String(600), nullable=True),
        sa.Column("signature_ts", sa.Integer, nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("wda_endpoint_state", "webhook_delivery_attempts",
                    ["endpoint_id", "state"])
    op.create_index("ix_wda_created", "webhook_delivery_attempts", ["created_at"])

    op.create_table(
        "webhook_endpoint_health",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("endpoint_id", sa.Integer, nullable=False),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("consecutive_failures", sa.Integer, nullable=False, server_default="0"),
        sa.Column("disabled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("disabled_reason", sa.String(600), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
        sa.UniqueConstraint("endpoint_id", name="uq_weh_endpoint"),
    )

    op.create_table(
        "review_decisions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("item_id", sa.Integer, nullable=False, index=True),
        sa.Column("workspace_id", sa.Integer, nullable=False, index=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("actor_user_id", sa.Integer, nullable=True),
        sa.Column("delegate_to", sa.Integer, nullable=True),
        sa.Column("reason", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now()),
    )
    op.create_index("ix_revdec_item", "review_decisions", ["item_id", "created_at"])

    # Seed the static dependency graph (Step 19) — idempotent
    edges = [
        ("api", "database", "HARD"), ("api", "broker", "SOFT"),
        ("worker", "database", "HARD"), ("worker", "broker", "HARD"),
        ("worker", "provider", "SOFT"), ("worker", "vector", "SOFT"),
        ("worker", "storage", "HARD"), ("scheduler", "database", "HARD"),
        ("event_processor", "database", "HARD"),
        ("event_processor", "broker", "HARD"),
        ("rag", "vector", "HARD"), ("rag", "provider", "HARD"),
        ("ingestion", "storage", "HARD"), ("ingestion", "database", "HARD"),
        ("connectors", "network", "SOFT"), ("notification", "database", "HARD"),
        ("notification", "network", "SOFT"),
    ]
    for comp, dep, crit in edges:
        bind.execute(sa.text(
            "INSERT INTO dependency_edges (component, depends_on, criticality, created_at) "
            "SELECT :c, :d, :k, (now() at time zone 'utc') "
            "WHERE NOT EXISTS (SELECT 1 FROM dependency_edges WHERE component = :c AND depends_on = :d)"
        ), {"c": comp, "d": dep, "k": crit})


def downgrade() -> None:
    tables = [
        "review_decisions",
        "webhook_endpoint_health",
        "webhook_delivery_attempts",
        "scheduler_task_runs",
        "request_dedup_records",
        "consistency_check_runs",
        "knowledge_freshness_states",
        "dependency_edges",
        "region_drain_operations",
        "search_shadow_comparisons",
        "document_embedding_status",
        "embedding_model_versions",
        "residency_decision_logs",
        "provider_routing_decisions",
        "provider_readiness_scores",
        "storage_migration_objects",
        "storage_migration_plans",
    ]
    for t in tables:
        op.drop_index(f"ix_smp_workspace_status", table_name="storage_migration_plans") \
            if t == "storage_migration_plans" else None
        op.drop_table(t)
