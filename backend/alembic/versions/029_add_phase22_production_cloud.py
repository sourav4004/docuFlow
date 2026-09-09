"""Phase 22 migration — Real-World Production AI Cloud + Continuous Operations.

Adds 21 production-operations tables on top of the Phase 21 substrate:

- infrastructure capability registry (infra_capabilities)
- real provider validation runs + cost reconciliation (provider_validation_runs,
  cost_reconciliation_runs)
- vector production: drift snapshots + benchmarks (vector_drift_snapshots,
  vector_benchmark_runs). Coverage snapshots reuse Phase 19's
  vector_coverage_snapshots — no duplicate table.
- durable evaluation executions (eval_executions) with idempotency + checkpoint
- improvement gates (improvement_gate_runs)
- knowledge maintenance + ingestion hardening (maintenance_runs_p22,
  ingestion_governor_events, poison_quarantines)
- connector production platform (connector_sync_states)
- region capacity + DR drills (region_capacity_snapshots,
  backup_restore_drills)
- SLO burn events (slo_burn_events)
- retention executions (retention_executions)
- worker runtime events (worker_runtime_events)
- DB growth estimates + API platform audits (db_growth_estimates,
  api_platform_audits)
- security scan runs (security_scan_runs)
- operating loops (selfheal_loop_runs, autonomy_loop_runs)
- operations streams (ops_stream_events)

Single head; downgrade drops the new tables leaf-first only.
"""

from alembic import op
import sqlalchemy as sa

revision = "029_phase22_production_cloud"
down_revision = "028_phase21_autonomous_ai"
branch_labels = None
depends_on = None


def _now() -> sa.TextClause:
    return sa.text("(now() at time zone 'utc')")


def _c(name, typ, **kw):
    return sa.Column(name, typ, **kw)


def _t(name, cols, *, uqs=(), indexes=()):
    table = op.create_table(name, *cols)
    for uq in uqs:
        op.create_unique_constraint(uq[0], name, uq[1])
    for idx in indexes:
        op.create_index(idx[0], name, idx[1])
    return table


def upgrade() -> None:
    # --- infrastructure capability registry ---------------------------------
    _t("infra_capabilities",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("component", sa.String(48), nullable=False),
        _c("state", sa.String(20), nullable=False,
           server_default="UNKNOWN"),
        _c("realization", sa.String(20), nullable=False,
           server_default="SIMULATED"),
        _c("detail", sa.Text(), nullable=True),
        _c("version", sa.String(64), nullable=True),
        _c("checked_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_infra_capability_component", ["component"])],
       indexes=[("ix_infra_cap_state", ["state", "component"])])

    # --- real provider validation + cost reconciliation ---------------------
    _t("provider_validation_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("provider", sa.String(48), nullable=False),
        _c("kind", sa.String(40), nullable=False),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("latency_ms", sa.Float(), nullable=True),
        _c("detail", sa.Text(), nullable=True),
        _c("metrics", sa.Text(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_prov_val_provider", ["provider", "kind", "created_at"]),
                ("ix_provider_validation_runs_workspace_id",
                 ["workspace_id"]),
                ("ix_provider_validation_runs_created_at", ["created_at"])])

    _t("cost_reconciliation_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("provider", sa.String(48), nullable=False, server_default="fake"),
        _c("local_usage", sa.Float(), nullable=False, server_default="0"),
        _c("provider_usage", sa.Float(), nullable=False, server_default="0"),
        _c("delta", sa.Float(), nullable=False, server_default="0"),
        _c("delta_pct", sa.Float(), nullable=False, server_default="0"),
        _c("reconciled", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_recon_ws", ["workspace_id", "created_at"]),
                ("ix_cost_reconciliation_runs_created_at", ["created_at"])])

    # --- vector production --------------------------------------------------
    _t("vector_drift_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("active_model", sa.String(120), nullable=False,
           server_default=""),
        _c("active_dimensions", sa.Integer(), nullable=False,
           server_default="0"),
        _c("versions", sa.Text(), nullable=True),
        _c("drifted_chunks", sa.Integer(), nullable=False,
           server_default="0"),
        _c("drift_pct", sa.Float(), nullable=False, server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_vec_drift_ws", ["workspace_id", "created_at"]),
                ("ix_vector_drift_snapshots_created_at", ["created_at"])])

    _t("vector_benchmark_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("backend", sa.String(20), nullable=False, server_default="json"),
        _c("native", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("queries", sa.Integer(), nullable=False, server_default="0"),
        _c("p50_ms", sa.Float(), nullable=True),
        _c("p95_ms", sa.Float(), nullable=True),
        _c("recall_at_10", sa.Float(), nullable=True),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_vec_bench_ws", ["workspace_id", "created_at"]),
                ("ix_vector_benchmark_runs_created_at", ["created_at"])])

    # --- durable evaluation -------------------------------------------------
    _t("eval_executions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("schedule_id", sa.Integer(), nullable=True),
        _c("domain", sa.String(32), nullable=False, server_default="retrieval"),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("dataset_version", sa.String(32), nullable=False,
           server_default="v1"),
        _c("idempotency_key", sa.String(120), nullable=False),
        _c("status", sa.String(20), nullable=False, server_default="QUEUED"),
        _c("attempt", sa.Integer(), nullable=False, server_default="0"),
        _c("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        _c("checkpoint", sa.Text(), nullable=True),
        _c("metrics", sa.Text(), nullable=True),
        _c("cancelled", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("regression_detected", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("job_id", sa.Integer(), nullable=True),
        _c("started_at", sa.DateTime(timezone=True), nullable=True),
        _c("finished_at", sa.DateTime(timezone=True), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_eval_execution_idem", ["idempotency_key"])],
       indexes=[("ix_eval_exec_sched", ["schedule_id", "created_at"]),
                ("ix_eval_executions_workspace_id", ["workspace_id"]),
                ("ix_eval_executions_job_id", ["job_id"]),
                ("ix_eval_executions_created_at", ["created_at"])])

    # --- improvement gates --------------------------------------------------
    _t("improvement_gate_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("proposal_id", sa.Integer(), nullable=False),
        _c("gate", sa.String(24), nullable=False),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("decision", sa.String(20), nullable=False,
           server_default="PENDING"),
        _c("evidence", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_improv_gates_proposal", ["proposal_id", "created_at"]),
                ("ix_improvement_gate_runs_workspace_id", ["workspace_id"]),
                ("ix_improvement_gate_runs_created_at", ["created_at"])])

    # --- knowledge maintenance + ingestion hardening ------------------------
    _t("maintenance_runs_p22",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("kind", sa.String(32), nullable=False),
        _c("findings", sa.Text(), nullable=True),
        _c("auto_repaired", sa.Integer(), nullable=False, server_default="0"),
        _c("proposals_created", sa.Integer(), nullable=False,
           server_default="0"),
        _c("status", sa.String(20), nullable=False,
           server_default="COMPLETED"),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_maint_runs_p22", ["workspace_id", "kind", "created_at"]),
                ("ix_maintenance_runs_p22_created_at", ["created_at"])])

    _t("ingestion_governor_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("document_id", sa.Integer(), nullable=True),
        _c("limit_kind", sa.String(32), nullable=False),
        _c("limit_value", sa.Float(), nullable=True),
        _c("observed_value", sa.Float(), nullable=True),
        _c("action", sa.String(20), nullable=False, server_default="REJECTED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_ing_gov_ws", ["workspace_id", "created_at"]),
                ("ix_ingestion_governor_events_document_id", ["document_id"]),
                ("ix_ingestion_governor_events_created_at", ["created_at"])])

    _t("poison_quarantines",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("document_id", sa.Integer(), nullable=False),
        _c("failure_count", sa.Integer(), nullable=False, server_default="0"),
        _c("last_error_class", sa.String(48), nullable=True),
        _c("status", sa.String(20), nullable=False,
           server_default="QUARANTINED"),
        _c("released_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_poison_doc", ["workspace_id", "document_id"])],
       indexes=[("ix_poison_ws", ["workspace_id", "created_at"])])

    # --- connector production platform --------------------------------------
    _t("connector_sync_states",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("connector_id", sa.Integer(), nullable=False),
        _c("checkpoint", sa.Text(), nullable=True),
        _c("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        _c("sync_age_s", sa.Integer(), nullable=True),
        _c("item_count", sa.Integer(), nullable=False, server_default="0"),
        _c("failure_count", sa.Integer(), nullable=False, server_default="0"),
        _c("latency_ms", sa.Float(), nullable=True),
        _c("health", sa.String(20), nullable=False, server_default="UNKNOWN"),
        _c("backoff_seconds", sa.Integer(), nullable=False, server_default="0"),
        _c("conflicts", sa.Integer(), nullable=False, server_default="0"),
        _c("last_error_class", sa.String(48), nullable=True),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_connector_sync_ws", ["workspace_id", "connector_id"])],
       indexes=[("ix_conn_sync_health", ["health", "workspace_id"]),
                ("ix_connector_sync_states_connector_id", ["connector_id"])])

    # --- regions + DR -------------------------------------------------------
    _t("region_capacity_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("region", sa.String(32), nullable=False),
        _c("workers", sa.Integer(), nullable=False, server_default="0"),
        _c("queue_depth", sa.Integer(), nullable=False, server_default="0"),
        _c("db_healthy", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("provider_healthy", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_region_cap_region", ["region", "created_at"]),
                ("ix_region_capacity_snapshots_created_at", ["created_at"])])

    _t("backup_restore_drills",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("tenant_isolation_ok", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("measured_rpo_s", sa.Integer(), nullable=True),
        _c("measured_rto_s", sa.Integer(), nullable=True),
        _c("target_rpo_s", sa.Integer(), nullable=True),
        _c("target_rto_s", sa.Integer(), nullable=True),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_drill_ws", ["workspace_id", "created_at"]),
                ("ix_backup_restore_drills_created_at", ["created_at"])])

    # --- SLO 2.0 ------------------------------------------------------------
    _t("slo_burn_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("slo_id", sa.Integer(), nullable=False),
        _c("domain", sa.String(32), nullable=False, server_default="api"),
        _c("burn_rate", sa.Float(), nullable=False, server_default="0"),
        _c("threshold", sa.Float(), nullable=False, server_default="1"),
        _c("error_budget_remaining", sa.Float(), nullable=False,
           server_default="1"),
        _c("incident_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_slo_burn_slo", ["slo_id", "created_at"]),
                ("ix_slo_burn_events_incident_id", ["incident_id"]),
                ("ix_slo_burn_events_created_at", ["created_at"])])

    # --- data lifecycle -----------------------------------------------------
    _t("retention_executions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("kind", sa.String(32), nullable=False),
        _c("dry_run", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("candidates", sa.Integer(), nullable=False, server_default="0"),
        _c("deleted", sa.Integer(), nullable=False, server_default="0"),
        _c("held", sa.Integer(), nullable=False, server_default="0"),
        _c("status", sa.String(20), nullable=False,
           server_default="COMPLETED"),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_retention_exec", ["kind", "created_at"]),
                ("ix_retention_executions_workspace_id", ["workspace_id"]),
                ("ix_retention_executions_created_at", ["created_at"])])

    # --- worker production runtime ------------------------------------------
    _t("worker_runtime_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("worker_id", sa.String(64), nullable=True),
        _c("kind", sa.String(32), nullable=False),
        _c("detail", sa.Text(), nullable=True),
        _c("depth", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_worker_rt_kind", ["kind", "created_at"]),
                ("ix_worker_runtime_events_worker_id", ["worker_id"]),
                ("ix_worker_runtime_events_workspace_id", ["workspace_id"])])

    # --- DB + API platform audits ------------------------------------------
    _t("db_growth_estimates",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("table_name", sa.String(64), nullable=False),
        _c("estimated_rows", sa.Integer(), nullable=False, server_default="0"),
        _c("est_bytes", sa.Float(), nullable=False, server_default="0"),
        _c("growth_per_day", sa.Float(), nullable=False, server_default="0"),
        _c("partition_recommended", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("recommendation", sa.Text(), nullable=True),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_db_growth_table", ["table_name"])],
       indexes=[("ix_db_growth_estimates_created_at", ["created_at"])])

    _t("api_platform_audits",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("kind", sa.String(32), nullable=False),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("checked", sa.Integer(), nullable=False, server_default="0"),
        _c("violations", sa.Integer(), nullable=False, server_default="0"),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_api_audit_kind", ["kind", "created_at"])])

    # --- security operations ------------------------------------------------
    _t("security_scan_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("corpus", sa.String(40), nullable=False),
        _c("cases", sa.Integer(), nullable=False, server_default="0"),
        _c("blocked", sa.Integer(), nullable=False, server_default="0"),
        _c("leaked", sa.Integer(), nullable=False, server_default="0"),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("findings", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_sec_scan_corpus", ["corpus", "created_at"]),
                ("ix_security_scan_runs_workspace_id", ["workspace_id"])])

    # --- operating loops ----------------------------------------------------
    _t("selfheal_loop_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("trigger", sa.String(64), nullable=False),
        _c("stage", sa.String(24), nullable=False, server_default="DETECTION"),
        _c("detected", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("diagnosis_id", sa.Integer(), nullable=True),
        _c("decision", sa.String(24), nullable=True),
        _c("recovered", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("validated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("escalated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("learned", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("timeline", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_selfheal_loop_ws", ["workspace_id", "created_at"])])

    _t("autonomy_loop_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("domain", sa.String(32), nullable=False, server_default="retrieval"),
        _c("stage", sa.String(24), nullable=False, server_default="OBSERVE"),
        _c("deviation_detected", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("proposal_id", sa.Integer(), nullable=True),
        _c("evaluated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("governance", sa.String(24), nullable=True),
        _c("approved", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("activated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("monitored", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("rolled_back", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("timeline", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_autonomy_loop_ws", ["workspace_id", "created_at"]),
                ("ix_autonomy_loop_runs_proposal_id", ["proposal_id"])])

    # --- operations streams -------------------------------------------------
    _t("ops_stream_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("stream", sa.String(32), nullable=False),
        _c("seq", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("kind", sa.String(32), nullable=False),
        _c("payload", sa.Text(), nullable=True),
        _c("dedup_key", sa.String(120), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_ops_stream_seq", ["stream", "seq"]),
            ("uq_ops_stream_dedup", ["dedup_key"])],
       indexes=[("ix_ops_stream_stream", ["stream", "seq"]),
                ("ix_ops_stream_events_workspace_id", ["workspace_id"]),
                ("ix_ops_stream_events_created_at", ["created_at"])])


def downgrade() -> None:
    tables = [
        "ops_stream_events",
        "autonomy_loop_runs",
        "selfheal_loop_runs",
        "security_scan_runs",
        "api_platform_audits",
        "db_growth_estimates",
        "worker_runtime_events",
        "retention_executions",
        "slo_burn_events",
        "backup_restore_drills",
        "region_capacity_snapshots",
        "connector_sync_states",
        "poison_quarantines",
        "ingestion_governor_events",
        "maintenance_runs_p22",
        "improvement_gate_runs",
        "eval_executions",
        "vector_benchmark_runs",
        "vector_drift_snapshots",
        "cost_reconciliation_runs",
        "provider_validation_runs",
        "infra_capabilities",
    ]
    for t in tables:
        op.drop_table(t)
