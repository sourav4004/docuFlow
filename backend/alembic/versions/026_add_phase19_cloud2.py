"""Phase 19 migration — Enterprise AI Cloud 2.0.

Adds: versioned control-plane snapshots with rollback audit, region/deployment
records + residency rules + explicit region failover, scheduler leadership
leases, worker quarantine + job migration audit, ingestion quality reports,
connector conflicts, memory suppressions, agent dead letters, cost
reservations + recommendations, embedding lifecycle events, vector coverage
snapshots, SLO definitions + budget windows, API abuse events, consistency
reports, backup records, quality gates, evaluation runs, and search
personalization profiles.

Single head; downgrade drops the new tables only (no destructive change to
pre-existing data).
"""

from alembic import op
import sqlalchemy as sa

revision = "026_phase19_cloud2"
down_revision = "025_phase18_cloud"
branch_labels = None
depends_on = None


def _now() -> sa.TextClause:
    return sa.text("(now() at time zone 'utc')")


def _c(name, typ, **kw):
    return sa.Column(name, typ, **kw)


def _t(name, cols, *, fks=(), uqs=(), indexes=()):
    """Create one table plus its constraints/indexes."""
    table = op.create_table(name, *cols)
    for local_cols, ref_spec, kw in fks:
        if isinstance(ref_spec, str):
            ref_table, ref_col = ref_spec.split(".")
            ref_cols = [ref_col]
        else:
            ref_table = ref_spec[0].split(".")[0]
            ref_cols = [c.split(".")[1] for c in ref_spec]
        op.create_foreign_key(None, name, ref_table, local_cols, ref_cols,
                              **kw)
    for uq in uqs:
        op.create_unique_constraint(uq[0], name, uq[1])
    for idx in indexes:
        op.create_index(idx[0], name, idx[1])
    return table


def upgrade() -> None:
    # --- control plane snapshots ------------------------------------------
    _t("control_plane_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("scope_type", sa.String(20), nullable=False),
        _c("scope_id", sa.Integer(), nullable=True),
        _c("version", sa.Integer(), nullable=False),
        _c("config_json", sa.Text(), nullable=False),
        _c("diff_json", sa.Text(), nullable=True),
        _c("actor_user_id", sa.Integer(), nullable=True),
        _c("reason", sa.String(1000), nullable=True),
        _c("is_active", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_cp_snapshot_version", ["scope_type", "scope_id", "version"])],
       indexes=[("ix_cp_snapshots_scope", ["scope_type", "scope_id"])])

    # --- regions + residency + failover ------------------------------------
    _t("region_records",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("region_id", sa.String(60), nullable=False),
        _c("name", sa.String(255), nullable=True),
        _c("deployment", sa.String(120), nullable=True),
        _c("status", sa.String(20), nullable=False,
           server_default="HEALTHY"),
        _c("health_score", sa.Float(), nullable=False, server_default="1.0"),
        _c("capabilities_json", sa.Text(), nullable=True),
        _c("capacity_json", sa.Text(), nullable=True),
        _c("provider_availability_json", sa.Text(), nullable=True),
        _c("vector_availability_json", sa.Text(), nullable=True),
        _c("residency_policy_json", sa.Text(), nullable=True),
        _c("failover_to", sa.String(60), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"})],
       uqs=[("uq_region_record", ["organization_id", "region_id"])],
       indexes=[("ix_region_records_org", ["organization_id", "region_id"])])

    _t("residency_rules",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("classification", sa.String(20), nullable=False),
        _c("allowed_regions_json", sa.Text(), nullable=True),
        _c("prohibited_regions_json", sa.Text(), nullable=True),
        _c("default_region", sa.String(60), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"})],
       uqs=[("uq_residency_class", ["organization_id", "classification"])],
       indexes=[("ix_residency_rules_org_class", ["organization_id",
                                                  "classification"])])

    _t("region_failovers",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("region_from", sa.String(60), nullable=False),
        _c("region_to", sa.String(60), nullable=False),
        _c("status", sa.String(20), nullable=False,
           server_default="INITIATED"),
        _c("reason", sa.String(1000), nullable=True),
        _c("actor_user_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("completed_at", sa.DateTime(timezone=True), nullable=True)],
       fks=[(["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"})],
       indexes=[("ix_region_failovers_org", ["organization_id",
                                             "created_at"])])

    # --- scheduler leadership + worker quarantine/migration -----------------
    _t("scheduler_leaders",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("leader_id", sa.String(64), nullable=False),
        _c("hostname", sa.String(255), nullable=True),
        _c("pid", sa.Integer(), nullable=True),
        _c("version", sa.String(50), nullable=True),
        _c("acquired_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("lease_until", sa.DateTime(timezone=True), nullable=False),
        _c("last_heartbeat", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("status", sa.String(20), nullable=False, server_default="LEADER")],
       uqs=[("uq_scheduler_leader_id", ["leader_id"])],
       indexes=[("ix_scheduler_leaders_status", ["status"])])

    _t("worker_quarantines",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("worker_id", sa.String(64), nullable=False),
        _c("reason", sa.String(1000), nullable=True),
        _c("failure_count", sa.Integer(), nullable=False, server_default="0"),
        _c("status", sa.String(20), nullable=False,
           server_default="QUARANTINED"),
        _c("auto_recover_after", sa.DateTime(timezone=True), nullable=True),
        _c("recovery_criteria_json", sa.Text(), nullable=True),
        _c("operator_user_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_worker_quarantine_worker", ["worker_id"])],
       indexes=[("ix_worker_quarantines_status", ["status"])])

    _t("job_migrations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("job_id", sa.Integer(), nullable=False),
        _c("from_worker", sa.String(64), nullable=True),
        _c("to_worker", sa.String(64), nullable=True),
        _c("reason", sa.String(500), nullable=True),
        _c("status", sa.String(20), nullable=False,
           server_default="MIGRATED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["job_id"], ["worker_jobs.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_job_migrations_job", ["job_id"])])

    # --- ingestion quality reports -----------------------------------------
    _t("ingestion_quality_reports",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("document_id", sa.Integer(), nullable=True),
        _c("run_id", sa.Integer(), nullable=True),
        _c("extraction_completeness", sa.Float(), nullable=False,
           server_default="0.0"),
        _c("ocr_quality", sa.Float(), nullable=True),
        _c("metadata_completeness", sa.Float(), nullable=False,
           server_default="0.0"),
        _c("chunk_quality", sa.Float(), nullable=False, server_default="0.0"),
        _c("embedding_coverage", sa.Float(), nullable=False,
           server_default="0.0"),
        _c("quality_score", sa.Float(), nullable=False, server_default="0.0"),
        _c("detail_json", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["workspace_id"], ["workspaces.id"], {"ondelete": "CASCADE"}),
            (["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"}),
            (["document_id"], ["documents.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_ingestion_quality_doc", ["document_id"])])

    # --- connector conflicts ------------------------------------------------
    _t("connector_conflicts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("source_id", sa.Integer(), nullable=False),
        _c("external_id", sa.String(255), nullable=False),
        _c("conflict_type", sa.String(40), nullable=False),
        _c("local_ref", sa.String(500), nullable=True),
        _c("external_ref", sa.String(500), nullable=True),
        _c("detail", sa.String(1000), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="OPEN"),
        _c("resolved_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["workspace_id"], ["workspaces.id"], {"ondelete": "CASCADE"}),
            (["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"}),
            (["source_id"], ["connector_sources.id"], {"ondelete":
                                                       "CASCADE"})],
       indexes=[("ix_connector_conflicts_source", ["source_id", "status"])])

    # --- memory suppressions ------------------------------------------------
    _t("memory_suppressions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("memory_id", sa.Integer(), nullable=False),
        _c("scope_type", sa.String(20), nullable=False),
        _c("suppressed_by_user_id", sa.Integer(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("reason", sa.String(500), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["memory_id"], ["ai_memories.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_memory_suppressions_memory", ["memory_id"])])

    # --- agent dead letters -------------------------------------------------
    _t("agent_dead_letters",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("execution_id", sa.String(36), nullable=False),
        _c("reason", sa.String(1000), nullable=True),
        _c("retry_count", sa.Integer(), nullable=False, server_default="0"),
        _c("last_checkpoint_json", sa.Text(), nullable=True),
        _c("safe_replay", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("status", sa.String(20), nullable=False, server_default="OPEN"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("resolved_at", sa.DateTime(timezone=True), nullable=True)],
       fks=[(["workspace_id"], ["workspaces.id"], {"ondelete": "CASCADE"}),
            (["execution_id"], ["ai_executions.id"], {"ondelete":
                                                      "CASCADE"})],
       indexes=[("ix_agent_dead_letters_execution", ["execution_id",
                                                     "status"])])

    # --- cost reservations + recommendations -------------------------------
    _t("cost_reservations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("execution_ref", sa.String(64), nullable=True),
        _c("feature", sa.String(60), nullable=False, server_default="ai"),
        _c("reserved_amount", sa.Float(), nullable=False,
           server_default="0.0"),
        _c("released_amount", sa.Float(), nullable=False,
           server_default="0.0"),
        _c("actual_cost", sa.Float(), nullable=True),
        _c("currency", sa.String(8), nullable=False, server_default="usd"),
        _c("status", sa.String(20), nullable=False, server_default="RESERVED"),
        _c("reserved_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("settled_at", sa.DateTime(timezone=True), nullable=True)],
       fks=[(["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"}),
            (["workspace_id"], ["workspaces.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_cost_reservations_scope", ["organization_id",
                                                "status"])])

    _t("cost_recommendations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("kind", sa.String(40), nullable=False),
        _c("reason", sa.String(1000), nullable=False),
        _c("savings_estimate", sa.Float(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="SUGGESTED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["organization_id"], ["organizations.id"], {"ondelete":
                                                         "CASCADE"})],
       indexes=[("ix_cost_recommendations_scope", ["organization_id",
                                                   "status"])])

    # --- embedding lifecycle + coverage -------------------------------------
    _t("embedding_lifecycle_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("embedding_model_id", sa.Integer(), nullable=False),
        _c("lifecycle", sa.String(30), nullable=False),
        _c("reason", sa.String(500), nullable=True),
        _c("decided_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["embedding_model_id"], ["embedding_models.id"], {"ondelete":
                                                               "CASCADE"})],
       indexes=[("ix_embedding_lifecycle_model", ["embedding_model_id"])])

    _t("vector_coverage_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("total_chunks", sa.Integer(), nullable=False, server_default="0"),
        _c("embedded_chunks", sa.Integer(), nullable=False,
           server_default="0"),
        _c("stale_chunks", sa.Integer(), nullable=False, server_default="0"),
        _c("failed_chunks", sa.Integer(), nullable=False, server_default="0"),
        _c("model_distribution_json", sa.Text(), nullable=True),
        _c("computed_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_vector_coverage_computed", ["computed_at"])])

    # --- SLO definitions + budget windows -----------------------------------
    _t("slo_definitions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("name", sa.String(120), nullable=False),
        _c("metric", sa.String(60), nullable=False),
        _c("operator", sa.String(10), nullable=False, server_default="<="),
        _c("target_value", sa.Float(), nullable=False),
        _c("window_minutes", sa.Integer(), nullable=False, server_default="60"),
        _c("burn_rate_threshold", sa.Float(), nullable=False,
           server_default="2.0"),
        _c("enabled", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_slo_definitions_enabled", ["enabled"])])

    _t("slo_budget_windows",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("definition_id", sa.Integer(), nullable=False),
        _c("window_start", sa.DateTime(timezone=True), nullable=False),
        _c("window_end", sa.DateTime(timezone=True), nullable=False),
        _c("budget_ratio", sa.Float(), nullable=False, server_default="0.0"),
        _c("burn_rate", sa.Float(), nullable=False, server_default="0.0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["definition_id"], ["slo_definitions.id"], {"ondelete":
                                                         "CASCADE"})],
       indexes=[("ix_slo_budget_windows_def", ["definition_id",
                                               "window_start"])])

    # --- API abuse events ----------------------------------------------------
    _t("api_abuse_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("api_key_id", sa.Integer(), nullable=True),
        _c("user_id", sa.Integer(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("kind", sa.String(40), nullable=False),
        _c("detail", sa.String(500), nullable=True),
        _c("window_start", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_api_abuse_events_window", ["kind", "window_start"])])

    # --- consistency reports -------------------------------------------------
    _t("consistency_reports",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("check_kind", sa.String(40), nullable=False),
        _c("status", sa.String(20), nullable=False, server_default="CLEAN"),
        _c("issue_count", sa.Integer(), nullable=False, server_default="0"),
        _c("issues_json", sa.Text(), nullable=True),
        _c("dry_run", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_consistency_reports_scope", ["workspace_id",
                                                  "check_kind"])])

    # --- backup records -------------------------------------------------------
    _t("backup_records",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("scope", sa.String(40), nullable=False),
        _c("backup_ref", sa.String(255), nullable=False),
        _c("database_version", sa.String(120), nullable=True),
        _c("migration_head", sa.String(120), nullable=True),
        _c("checksum", sa.String(128), nullable=True),
        _c("size_bytes", sa.Integer(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="PENDING"),
        _c("note", sa.String(500), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("validated_at", sa.DateTime(timezone=True), nullable=True)],
       indexes=[("ix_backup_records_status", ["status", "created_at"])])

    # --- quality gates + evaluation runs -------------------------------------
    _t("quality_gates",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("name", sa.String(120), nullable=False),
        _c("metric", sa.String(60), nullable=False),
        _c("operator", sa.String(10), nullable=False, server_default=">="),
        _c("threshold", sa.Float(), nullable=False),
        _c("enabled", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_quality_gates_org", ["organization_id", "name"])])

    _t("evaluation_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("dataset_name", sa.String(120), nullable=False),
        _c("mode", sa.String(20), nullable=False, server_default="offline"),
        _c("sample_count", sa.Integer(), nullable=False, server_default="0"),
        _c("metric_json", sa.Text(), nullable=True),
        _c("passed", sa.Boolean(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_evaluation_runs_dataset", ["dataset_name",
                                                "created_at"])])

    # --- search personalization ----------------------------------------------
    _t("search_personalization",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("user_id", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("preferences_json", sa.Text(), nullable=True),
        _c("recent_intents_json", sa.Text(), nullable=True),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["user_id"], ["users.id"], {"ondelete": "CASCADE"}),
            (["workspace_id"], ["workspaces.id"], {"ondelete": "CASCADE"})],
       uqs=[("uq_search_personalization", ["user_id", "workspace_id"])],
       indexes=[("ix_search_personalization_user", ["user_id",
                                                    "workspace_id"])])


def downgrade() -> None:
    for table in ("search_personalization", "evaluation_runs", "quality_gates",
                  "backup_records", "consistency_reports", "api_abuse_events",
                  "slo_budget_windows", "slo_definitions",
                  "vector_coverage_snapshots", "embedding_lifecycle_events",
                  "cost_recommendations", "cost_reservations",
                  "agent_dead_letters", "memory_suppressions",
                  "connector_conflicts", "ingestion_quality_reports",
                  "job_migrations", "worker_quarantines", "scheduler_leaders",
                  "region_failovers", "residency_rules", "region_records",
                  "control_plane_snapshots"):
        op.drop_table(table)
