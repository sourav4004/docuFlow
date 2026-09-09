"""Phase 21 migration — Autonomous Enterprise AI + Self-Healing Knowledge Cloud.

Adds the governed-autonomy substrate on top of Phase 20: autonomy policies +
transitions, autonomous operations with audited decisions and idempotency,
system health snapshots, recovery playbooks + attempts (idempotent,
cooldown-aware, escalating), root-cause diagnosis reports, knowledge recovery
plans, ingestion quality/anomaly tracking, adaptive candidates with promotion
gates, retrieval/RAG drift snapshots + failure clusters, model performance +
drift events + routing simulations, cost forecasts/anomalies/guards/
optimizations, agent plan risk + recovery + workflow risk assessments,
platform events with dedup, evaluation schedules + runs, emergency stops +
autonomy abuse + tool safety violations, security health + incidents, data
governance snapshots/impacts, region health + failover simulations + residency
guard, worker health/quarantine + capacity recommendations, broker/scheduler
health, slow queries + cache health, graph repair proposals + memory autonomy
events, personal autonomy settings + AI activity feed, incidents 2.0 with
postmortems, learning dataset candidates, artifact quality, API abuse signals,
cost-aware queue decisions, maintenance plans, backup health, and chaos run
records.

Single head; downgrade drops the new tables only (leaf-first, no destructive
change to pre-existing data).
"""

from alembic import op
import sqlalchemy as sa

revision = "028_phase21_autonomous_ai"
down_revision = "027_phase20_knowledge_os"
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
    # --- autonomous operations control plane --------------------------------
    _t("autonomy_policies",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("operation_type", sa.String(64), nullable=False),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("autonomy_level", sa.String(20), nullable=False,
           server_default="RECOMMEND"),
        _c("requires_approval", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("budget_limit_usd", sa.Float(), nullable=True),
        _c("execution_limit_per_hour", sa.Integer(), nullable=True),
        _c("cooldown_seconds", sa.Integer(), nullable=False,
           server_default="300"),
        _c("requires_audit", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("created_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_autonomy_policies_scope",
                 ["workspace_id", "operation_type"])])
    op.create_unique_constraint(
        "uq_autonomy_policy_scope_op_risk", "autonomy_policies",
        ["workspace_id", "operation_type", "risk_level"])

    _t("autonomy_transitions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("policy_id", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("previous_level", sa.String(20), nullable=True),
        _c("new_level", sa.String(20), nullable=False),
        _c("actor", sa.String(120), nullable=False,
           server_default="system"),
        _c("reason", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_autonomy_transitions_policy",
                 ["policy_id", "created_at"])])

    _t("autonomous_operations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("operation_type", sa.String(64), nullable=False),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("actor", sa.String(120), nullable=False, server_default="system"),
        _c("source", sa.String(32), nullable=False, server_default="SYSTEM"),
        _c("policy_id", sa.Integer(), nullable=True),
        _c("policy_level", sa.String(20), nullable=True),
        _c("input_payload", sa.Text(), nullable=True),
        _c("decision", sa.String(24), nullable=False),
        _c("decision_reason", sa.Text(), nullable=True),
        _c("status", sa.String(24), nullable=False, server_default="DECIDED"),
        _c("result", sa.Text(), nullable=True),
        _c("rollback_info", sa.Text(), nullable=True),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_autonomous_operations_ws",
                 ["workspace_id", "created_at"]),
                ("ix_autonomous_operations_op",
                 ["workspace_id", "operation_type"])])
    op.create_unique_constraint(
        "uq_autonomous_op_idem", "autonomous_operations",
        ["workspace_id", "idempotency_key"])

    # --- self-healing platform ----------------------------------------------
    _t("system_health_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("overall_state", sa.String(16), nullable=False,
           server_default="UNKNOWN"),
        _c("components", sa.Text(), nullable=False),
        _c("unhealthy_count", sa.Integer(), nullable=False,
           server_default="0"),
        _c("degraded_count", sa.Integer(), nullable=False,
           server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_system_health_ws", ["workspace_id", "created_at"])])

    _t("recovery_playbooks",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("name", sa.String(120), nullable=False),
        _c("trigger", sa.String(64), nullable=False),
        _c("detection_criteria", sa.Text(), nullable=True),
        _c("actions", sa.Text(), nullable=False),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("cooldown_seconds", sa.Integer(), nullable=False,
           server_default="300"),
        _c("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        _c("rollback_strategy", sa.Text(), nullable=True),
        _c("approved", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_recovery_playbooks_trigger",
                 ["workspace_id", "trigger"])])
    op.create_unique_constraint(
        "uq_recovery_playbook_name", "recovery_playbooks",
        ["workspace_id", "name"])

    _t("recovery_attempts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("playbook_id", sa.Integer(), nullable=True),
        _c("trigger", sa.String(64), nullable=False),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("status", sa.String(24), nullable=False, server_default="RUNNING"),
        _c("attempts_so_far", sa.Integer(), nullable=False,
           server_default="1"),
        _c("action_results", sa.Text(), nullable=True),
        _c("escalated_incident_id", sa.Integer(), nullable=True),
        _c("auto_applied", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_recovery_attempts_ws", ["workspace_id", "created_at"]),
                ("ix_recovery_attempts_playbook",
                 ["playbook_id", "created_at"])])
    op.create_unique_constraint(
        "uq_recovery_attempt_idem", "recovery_attempts",
        ["workspace_id", "idempotency_key"])

    # --- self-diagnosis engine ----------------------------------------------
    _t("diagnosis_reports",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("incident_id", sa.Integer(), nullable=True),
        _c("symptom", sa.String(200), nullable=False),
        _c("hypotheses", sa.Text(), nullable=False),
        _c("top_cause", sa.String(200), nullable=True),
        _c("top_confidence", sa.Float(), nullable=True),
        _c("correlated_failures", sa.Integer(), nullable=False,
           server_default="0"),
        _c("report", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_diagnosis_reports_ws", ["workspace_id", "created_at"])])

    # --- self-healing knowledge ---------------------------------------------
    _t("knowledge_recovery_plans",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("issue_kind", sa.String(48), nullable=False),
        _c("target_type", sa.String(32), nullable=False),
        _c("target_id", sa.Integer(), nullable=True),
        _c("plan", sa.Text(), nullable=False),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("status", sa.String(24), nullable=False, server_default="PROPOSED"),
        _c("decision", sa.String(24), nullable=True),
        _c("executed_at", sa.DateTime(timezone=True), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_knowledge_recovery_plans_ws",
                 ["workspace_id", "created_at"])])

    # --- adaptive ingestion / retrieval / RAG --------------------------------
    _t("ingestion_quality_samples",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("document_id", sa.Integer(), nullable=True),
        _c("extraction_quality", sa.Float(), nullable=True),
        _c("ocr_quality", sa.Float(), nullable=True),
        _c("chunk_quality", sa.Float(), nullable=True),
        _c("metadata_completeness", sa.Float(), nullable=True),
        _c("embedding_coverage", sa.Float(), nullable=True),
        _c("processing_latency_ms", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_ingestion_quality_ws", ["workspace_id", "created_at"])])

    _t("ingestion_anomalies",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metric", sa.String(48), nullable=False),
        _c("severity", sa.String(16), nullable=False,
           server_default="MEDIUM"),
        _c("baseline_value", sa.Float(), nullable=True),
        _c("observed_value", sa.Float(), nullable=True),
        _c("drop_percent", sa.Float(), nullable=True),
        _c("recommendation", sa.Text(), nullable=True),
        _c("status", sa.String(16), nullable=False, server_default="OPEN"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_ingestion_anomalies_ws", ["workspace_id", "created_at"])])

    _t("adaptive_candidates",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("domain", sa.String(32), nullable=False),
        _c("change_kind", sa.String(48), nullable=False),
        _c("proposed_value", sa.Text(), nullable=True),
        _c("rationale", sa.Text(), nullable=True),
        _c("evaluation_score", sa.Float(), nullable=True),
        _c("baseline_score", sa.Float(), nullable=True),
        _c("gates", sa.Text(), nullable=True),
        _c("gates_passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("status", sa.String(24), nullable=False,
           server_default="CANDIDATE"),
        _c("promoted", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_adaptive_candidates_ws",
                 ["workspace_id", "domain"])])
    op.create_unique_constraint(
        "uq_adaptive_candidate_idem", "adaptive_candidates",
        ["workspace_id", "idempotency_key"])

    _t("retrieval_drift_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metric", sa.String(48), nullable=False,
           server_default="ndcg_proxy"),
        _c("baseline_value", sa.Float(), nullable=True),
        _c("current_value", sa.Float(), nullable=True),
        _c("drift_percent", sa.Float(), nullable=True),
        _c("degraded", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_retrieval_drift_ws", ["workspace_id", "created_at"])])

    _t("rag_drift_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metric", sa.String(48), nullable=False,
           server_default="groundedness"),
        _c("baseline_value", sa.Float(), nullable=True),
        _c("current_value", sa.Float(), nullable=True),
        _c("drift_percent", sa.Float(), nullable=True),
        _c("degraded", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_rag_drift_ws", ["workspace_id", "created_at"])])

    _t("rag_failure_clusters",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("cause", sa.String(64), nullable=False),
        _c("sample_count", sa.Integer(), nullable=False, server_default="0"),
        _c("representative_question", sa.Text(), nullable=True),
        _c("recommendation", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_rag_failure_clusters_ws",
                 ["workspace_id", "created_at"])])

    # --- model / cost autopilot ---------------------------------------------
    _t("model_performance_samples",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("model", sa.String(120), nullable=False),
        _c("provider", sa.String(64), nullable=False),
        _c("quality_score", sa.Float(), nullable=True),
        _c("latency_ms", sa.Float(), nullable=True),
        _c("cost_usd", sa.Float(), nullable=True),
        _c("availability", sa.Float(), nullable=True),
        _c("tool_reliability", sa.Float(), nullable=True),
        _c("structured_output_reliability", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_model_perf_ws",
                 ["workspace_id", "model", "created_at"])])

    _t("model_drift_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("model", sa.String(120), nullable=True),
        _c("provider", sa.String(64), nullable=True),
        _c("kind", sa.String(32), nullable=False, server_default="MODEL"),
        _c("metric", sa.String(48), nullable=False),
        _c("direction", sa.String(16), nullable=False),
        _c("baseline_value", sa.Float(), nullable=True),
        _c("current_value", sa.Float(), nullable=True),
        _c("drop_percent", sa.Float(), nullable=True),
        _c("recommendation", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_model_drift_ws", ["workspace_id", "created_at"])])

    _t("routing_simulations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("description", sa.String(200), nullable=False),
        _c("workload", sa.Text(), nullable=True),
        _c("current_metrics", sa.Text(), nullable=True),
        _c("simulated_metrics", sa.Text(), nullable=True),
        _c("safety_checks", sa.Text(), nullable=True),
        _c("safe", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("candidate_id", sa.Integer(), nullable=True),
        _c("executed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_routing_sims_ws", ["workspace_id", "created_at"])])

    _t("cost_forecasts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("scope_kind", sa.String(24), nullable=False,
           server_default="WORKSPACE"),
        _c("scope_key", sa.String(120), nullable=True),
        _c("period_days", sa.Integer(), nullable=False, server_default="30"),
        _c("current_spend_usd", sa.Float(), nullable=False,
           server_default="0"),
        _c("forecast_usd", sa.Float(), nullable=False, server_default="0"),
        _c("budget_usd", sa.Float(), nullable=True),
        _c("over_budget", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_forecasts_ws", ["workspace_id", "created_at"])])

    _t("cost_anomalies_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metric", sa.String(48), nullable=False, server_default="spend"),
        _c("severity", sa.String(16), nullable=False,
           server_default="MEDIUM"),
        _c("baseline_value", sa.Float(), nullable=True),
        _c("observed_value", sa.Float(), nullable=True),
        _c("increase_percent", sa.Float(), nullable=True),
        _c("recommendation", sa.Text(), nullable=True),
        _c("status", sa.String(16), nullable=False, server_default="OPEN"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_anomalies_p21_ws", ["workspace_id", "created_at"])])

    _t("cost_guard_decisions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("operation_type", sa.String(64), nullable=False),
        _c("estimated_cost_usd", sa.Float(), nullable=False,
           server_default="0"),
        _c("remaining_budget_usd", sa.Float(), nullable=True),
        _c("policy_level", sa.String(20), nullable=True),
        _c("decision", sa.String(24), nullable=False),
        _c("reason", sa.Text(), nullable=True),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_guard_ws", ["workspace_id", "created_at"])])
    op.create_unique_constraint(
        "uq_cost_guard_idem", "cost_guard_decisions",
        ["workspace_id", "idempotency_key"])

    _t("cost_optimization_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("optimization_kind", sa.String(48), nullable=False),
        _c("policy_level", sa.String(20), nullable=True),
        _c("applied", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("estimated_savings_usd", sa.Float(), nullable=True),
        _c("reason", sa.Text(), nullable=True),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_optim_ws", ["workspace_id", "created_at"])])
    op.create_unique_constraint(
        "uq_cost_optim_idem", "cost_optimization_events",
        ["workspace_id", "idempotency_key"])

    # --- agent autonomy 2.0 --------------------------------------------------
    _t("agent_plan_risks",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("agent_run_id", sa.Integer(), nullable=True),
        _c("plan_summary", sa.String(200), nullable=True),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("risk_factors", sa.Text(), nullable=True),
        _c("simulation", sa.Text(), nullable=True),
        _c("optimizations", sa.Text(), nullable=True),
        _c("applied_optimizations", sa.Text(), nullable=True),
        _c("handed_off", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("decision", sa.String(24), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_agent_plan_risk_ws", ["workspace_id", "created_at"])])

    _t("agent_recovery_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("agent_run_id", sa.Integer(), nullable=True),
        _c("recovery_kind", sa.String(24), nullable=False),
        _c("status", sa.String(24), nullable=False, server_default="RUNNING"),
        _c("detail", sa.Text(), nullable=True),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_agent_recovery_ws", ["workspace_id", "created_at"])])
    op.create_unique_constraint(
        "uq_agent_recovery_idem", "agent_recovery_events",
        ["workspace_id", "idempotency_key"])

    # --- workflow autonomy ---------------------------------------------------
    _t("workflow_risk_assessments",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("workflow_id", sa.Integer(), nullable=True),
        _c("workflow_version_id", sa.Integer(), nullable=True),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("risk_factors", sa.Text(), nullable=True),
        _c("autonomy_decision", sa.String(24), nullable=True),
        _c("requires_approval", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("simulation", sa.Text(), nullable=True),
        _c("optimizations", sa.Text(), nullable=True),
        _c("recurring_failures", sa.Integer(), nullable=False,
           server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_workflow_risk_ws", ["workspace_id", "created_at"])])

    # --- event-driven intelligence ------------------------------------------
    _t("platform_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("event_kind", sa.String(48), nullable=False),
        _c("event_key", sa.String(160), nullable=False),
        _c("payload", sa.Text(), nullable=True),
        _c("deduplicated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("replayed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_platform_events_ws", ["workspace_id", "created_at"]),
                ("ix_platform_events_kind",
                 ["workspace_id", "event_kind"])])
    op.create_unique_constraint(
        "uq_platform_event_key", "platform_events",
        ["workspace_id", "event_key"])

    # --- continuous evaluation ----------------------------------------------
    _t("evaluation_schedules",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("domain", sa.String(32), nullable=False,
           server_default="retrieval"),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("dataset_version", sa.String(32), nullable=False,
           server_default="v1"),
        _c("interval_minutes", sa.Integer(), nullable=False,
           server_default="1440"),
        _c("config", sa.Text(), nullable=True),
        _c("active", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("last_run_at", sa.DateTime(timezone=True), nullable=True),
        _c("last_metrics", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_eval_schedules_ws", ["workspace_id", "created_at"])])

    _t("evaluation_runs_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("schedule_id", sa.Integer(), nullable=True),
        _c("domain", sa.String(32), nullable=False),
        _c("dataset_version", sa.String(32), nullable=False,
           server_default="v1"),
        _c("model", sa.String(120), nullable=True),
        _c("provider", sa.String(64), nullable=True),
        _c("config", sa.Text(), nullable=True),
        _c("metrics", sa.Text(), nullable=True),
        _c("regression_detected", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("gate_passed", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("incident_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_eval_runs_p21_ws", ["workspace_id", "created_at"])])

    # --- AI safety 10.0 / security center 3.0 --------------------------------
    _t("emergency_stops",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("scope", sa.String(24), nullable=False, server_default="ALL"),
        _c("active", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("actor", sa.String(120), nullable=False,
           server_default="operator"),
        _c("reason", sa.Text(), nullable=True),
        _c("lifted_at", sa.DateTime(timezone=True), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_emergency_stops_ws", ["workspace_id", "created_at"])])

    _t("autonomy_abuse_attempts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("abuse_kind", sa.String(48), nullable=False),
        _c("blocked", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("detail", sa.Text(), nullable=True),
        _c("actor", sa.String(120), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_autonomy_abuse_ws", ["workspace_id", "created_at"])])

    _t("tool_safety_violations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("tool_name", sa.String(120), nullable=False),
        _c("violation_kind", sa.String(48), nullable=False),
        _c("blocked", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_tool_safety_ws", ["workspace_id", "created_at"])])

    _t("security_health_scores",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("score", sa.Float(), nullable=False, server_default="100"),
        _c("auth_failures", sa.Integer(), nullable=False,
           server_default="0"),
        _c("authz_failures", sa.Integer(), nullable=False,
           server_default="0"),
        _c("injection_attempts", sa.Integer(), nullable=False,
           server_default="0"),
        _c("ssrf_attempts", sa.Integer(), nullable=False,
           server_default="0"),
        _c("tool_abuse", sa.Integer(), nullable=False, server_default="0"),
        _c("exfiltration_attempts", sa.Integer(), nullable=False,
           server_default="0"),
        _c("suspicious_api", sa.Integer(), nullable=False,
           server_default="0"),
        _c("state", sa.String(16), nullable=False, server_default="HEALTHY"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_security_health_ws", ["workspace_id", "created_at"])])

    _t("security_incidents_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("severity", sa.String(8), nullable=False, server_default="SEV3"),
        _c("category", sa.String(48), nullable=False),
        _c("correlated_count", sa.Integer(), nullable=False,
           server_default="1"),
        _c("summary", sa.String(300), nullable=True),
        _c("recommended_playbook", sa.Text(), nullable=True),
        _c("auto_response", sa.Text(), nullable=True),
        _c("status", sa.String(16), nullable=False, server_default="OPEN"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_sec_incidents_p21_ws", ["workspace_id", "created_at"])])

    # --- data governance 6.0 -------------------------------------------------
    _t("data_classification_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("counts", sa.Text(), nullable=False),
        _c("drifted", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("drift_detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_data_class_ws", ["workspace_id", "created_at"])])

    _t("data_policy_impacts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("affected_models", sa.Text(), nullable=True),
        _c("affected_providers", sa.Text(), nullable=True),
        _c("affected_workflows", sa.Text(), nullable=True),
        _c("affected_connectors", sa.Text(), nullable=True),
        _c("affected_regions", sa.Text(), nullable=True),
        _c("minimization_required", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("residency_violations", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_data_policy_impact_ws",
                 ["workspace_id", "created_at"])])

    # --- multi-region operations ---------------------------------------------
    _t("region_health_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=False),
        _c("region_id", sa.String(64), nullable=False),
        _c("state", sa.String(16), nullable=False, server_default="HEALTHY"),
        _c("workers", sa.Integer(), nullable=False, server_default="0"),
        _c("queue_depth", sa.Integer(), nullable=False, server_default="0"),
        _c("provider_healthy", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("db_healthy", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("failover_ready", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_region_health_p21",
                 ["organization_id", "region_id"])])

    _t("failover_simulations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=False),
        _c("from_region", sa.String(64), nullable=False),
        _c("to_region", sa.String(64), nullable=False),
        _c("residency_ok", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("capacity_ok", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("ready", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("detail", sa.Text(), nullable=True),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("executed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_failover_sims", ["organization_id", "created_at"])])

    _t("residency_guard_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("organization_id", sa.Integer(), nullable=False),
        _c("requested_region", sa.String(64), nullable=False),
        _c("allowed_region", sa.String(64), nullable=True),
        _c("blocked", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_residency_guard", ["organization_id", "created_at"])])

    # --- worker / broker / scheduler / database / cache ----------------------
    _t("worker_health_scores",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("worker_id", sa.String(120), nullable=False),
        _c("region", sa.String(64), nullable=True),
        _c("heartbeat_age_seconds", sa.Integer(), nullable=False,
           server_default="0"),
        _c("throughput", sa.Float(), nullable=False, server_default="0"),
        _c("failure_count", sa.Integer(), nullable=False,
           server_default="0"),
        _c("queue_latency_ms", sa.Float(), nullable=False,
           server_default="0"),
        _c("score", sa.Float(), nullable=False, server_default="100"),
        _c("state", sa.String(16), nullable=False, server_default="HEALTHY"),
        _c("quarantined", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_worker_health_p21",
                 ["workspace_id", "worker_id"])])

    _t("capacity_recommendations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("region", sa.String(64), nullable=True),
        _c("current_workers", sa.Integer(), nullable=False,
           server_default="0"),
        _c("recommended_workers", sa.Integer(), nullable=False,
           server_default="0"),
        _c("fairness_preserved", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("reason", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_capacity_rec_ws", ["workspace_id", "created_at"])])

    _t("broker_health_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("broker", sa.String(24), nullable=False,
           server_default="postgres"),
        _c("depth", sa.Integer(), nullable=False, server_default="0"),
        _c("latency_ms", sa.Float(), nullable=False, server_default="0"),
        _c("visibility_timeouts", sa.Integer(), nullable=False,
           server_default="0"),
        _c("error_count", sa.Integer(), nullable=False, server_default="0"),
        _c("reconnects", sa.Integer(), nullable=False, server_default="0"),
        _c("state", sa.String(16), nullable=False, server_default="HEALTHY"),
        _c("degraded", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_broker_health_ws", ["workspace_id", "created_at"])])

    _t("scheduler_health_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("has_leader", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("leader_age_seconds", sa.Integer(), nullable=False,
           server_default="0"),
        _c("missed_schedules", sa.Integer(), nullable=False,
           server_default="0"),
        _c("recovered_schedules", sa.Integer(), nullable=False,
           server_default="0"),
        _c("dedup_blocked", sa.Integer(), nullable=False,
           server_default="0"),
        _c("state", sa.String(16), nullable=False, server_default="HEALTHY"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_scheduler_health_ws", ["workspace_id", "created_at"])])

    _t("slow_query_records",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("statement", sa.String(300), nullable=False),
        _c("duration_ms", sa.Float(), nullable=False, server_default="0"),
        _c("recommendation", sa.Text(), nullable=True),
        _c("auto_applied", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_slow_queries_ws", ["workspace_id", "created_at"])])

    _t("cache_health_snapshots",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("cache", sa.String(48), nullable=False,
           server_default="embedding_cache"),
        _c("hit_rate", sa.Float(), nullable=False, server_default="0"),
        _c("miss_rate", sa.Float(), nullable=False, server_default="1"),
        _c("stale_entries", sa.Integer(), nullable=False,
           server_default="0"),
        _c("memory_usage_mb", sa.Float(), nullable=False,
           server_default="0"),
        _c("anomaly", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("recommendation", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cache_health_ws", ["workspace_id", "created_at"])])

    # --- graph / memory autonomy ---------------------------------------------
    _t("graph_repair_proposals",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("repair_kind", sa.String(48), nullable=False),
        _c("target_entity_id", sa.Integer(), nullable=True),
        _c("target_relationship_id", sa.Integer(), nullable=True),
        _c("proposal", sa.Text(), nullable=True),
        _c("risk_level", sa.String(16), nullable=False,
           server_default="LOW"),
        _c("impact", sa.Text(), nullable=True),
        _c("status", sa.String(24), nullable=False, server_default="PROPOSED"),
        _c("decision", sa.String(24), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_graph_repairs_ws", ["workspace_id", "created_at"])])

    _t("memory_autonomy_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("memory_id", sa.Integer(), nullable=True),
        _c("event_kind", sa.String(32), nullable=False),
        _c("automatic", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_memory_autonomy_ws", ["workspace_id", "created_at"])])

    # --- personal AI control --------------------------------------------------
    _t("personal_autonomy_settings",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("user_id", sa.Integer(), nullable=False),
        _c("autonomy_level", sa.String(20), nullable=False,
           server_default="RECOMMEND"),
        _c("allow_memory_suppression", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("allow_memory_reset", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("allow_memory_delete", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("activity_feed_enabled", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("explanation_style", sa.String(24), nullable=False,
           server_default="CONCISE"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])
    op.create_unique_constraint(
        "uq_personal_autonomy", "personal_autonomy_settings",
        ["workspace_id", "user_id"])

    _t("ai_activity_items",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("user_id", sa.Integer(), nullable=False),
        _c("item_kind", sa.String(32), nullable=False,
           server_default="SUGGESTION"),
        _c("title", sa.String(200), nullable=False),
        _c("explanation", sa.Text(), nullable=True),
        _c("ref_kind", sa.String(32), nullable=True),
        _c("ref_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_ai_activity_ws_user",
                 ["workspace_id", "user_id", "created_at"])])

    # --- incident management 2.0 ----------------------------------------------
    _t("incidents_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("severity", sa.String(8), nullable=False, server_default="SEV3"),
        _c("title", sa.String(200), nullable=False),
        _c("source", sa.String(32), nullable=False,
           server_default="THRESHOLD"),
        _c("timeline", sa.Text(), nullable=True),
        _c("diagnosis_id", sa.Integer(), nullable=True),
        _c("recovery_attempt_id", sa.Integer(), nullable=True),
        _c("postmortem", sa.Text(), nullable=True),
        _c("postmortem_finalized", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("learnings", sa.Text(), nullable=True),
        _c("status", sa.String(16), nullable=False, server_default="OPEN"),
        _c("resolved_at", sa.DateTime(timezone=True), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_incidents_p21_ws", ["workspace_id", "created_at"])])

    # --- continuous learning ---------------------------------------------------
    _t("learning_dataset_candidates",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("source_kind", sa.String(32), nullable=False,
           server_default="FEEDBACK"),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("dataset_version", sa.String(32), nullable=False,
           server_default="v1"),
        _c("input_text", sa.Text(), nullable=False),
        _c("expected_output", sa.Text(), nullable=True),
        _c("quality", sa.String(16), nullable=False,
           server_default="APPROVED"),
        _c("regression_case_id", sa.Integer(), nullable=True),
        _c("idempotency_key", sa.String(128), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_learning_candidates_ws",
                 ["workspace_id", "created_at"])])
    op.create_unique_constraint(
        "uq_learning_candidate_idem", "learning_dataset_candidates",
        ["workspace_id", "idempotency_key"])

    # --- AI artifact intelligence ----------------------------------------------
    _t("artifact_quality_scores",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("artifact_id", sa.Integer(), nullable=True),
        _c("artifact_kind", sa.String(32), nullable=False,
           server_default="REPORT"),
        _c("version", sa.Integer(), nullable=False, server_default="1"),
        _c("quality_score", sa.Float(), nullable=True),
        _c("groundedness", sa.Float(), nullable=True),
        _c("citation_coverage", sa.Float(), nullable=True),
        _c("completeness", sa.Float(), nullable=True),
        _c("provenance_ok", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("regression", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("comparison", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_artifact_quality_ws", ["workspace_id", "created_at"])])

    # --- API autonomy safety -----------------------------------------------------
    _t("api_abuse_signals_p21",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("abuse_kind", sa.String(48), nullable=False),
        _c("subject", sa.String(160), nullable=True),
        _c("severity", sa.String(16), nullable=False,
           server_default="MEDIUM"),
        _c("rate_limit_recommendation", sa.Text(), nullable=True),
        _c("incident_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_api_abuse_p21_ws", ["workspace_id", "created_at"])])

    # --- cost-aware worker scheduling ---------------------------------------------
    _t("cost_aware_queue_decisions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("job_id", sa.Integer(), nullable=True),
        _c("priority", sa.Integer(), nullable=False, server_default="5"),
        _c("urgency", sa.String(16), nullable=False,
           server_default="NORMAL"),
        _c("estimated_cost_usd", sa.Float(), nullable=False,
           server_default="0"),
        _c("budget_state", sa.String(16), nullable=False,
           server_default="OK"),
        _c("fairness_preserved", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("guard", sa.String(24), nullable=True),
        _c("decision", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_queue_ws", ["workspace_id", "created_at"])])

    # --- autonomous data maintenance -----------------------------------------------
    _t("maintenance_plans",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("plan_kind", sa.String(48), nullable=False),
        _c("targets", sa.Text(), nullable=True),
        _c("destructive", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("dry_run", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("dry_run_result", sa.Text(), nullable=True),
        _c("legal_hold_respected", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("retention_policy_id", sa.Integer(), nullable=True),
        _c("requires_approval", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("status", sa.String(24), nullable=False, server_default="PROPOSED"),
        _c("executed_result", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_maintenance_plans_ws",
                 ["workspace_id", "created_at"])])

    # --- backup / DR ------------------------------------------------------------------
    _t("backup_health_records",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("backup_kind", sa.String(32), nullable=False,
           server_default="SNAPSHOT"),
        _c("status", sa.String(24), nullable=False, server_default="UNKNOWN"),
        _c("last_backup_at", sa.DateTime(timezone=True), nullable=True),
        _c("age_hours", sa.Float(), nullable=True),
        _c("restore_validated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("tenant_isolation_ok", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("rto_target_minutes", sa.Integer(), nullable=True),
        _c("rpo_target_minutes", sa.Integer(), nullable=True),
        _c("rto_observed_minutes", sa.Float(), nullable=True),
        _c("rpo_observed_minutes", sa.Float(), nullable=True),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_backup_health_ws", ["workspace_id", "created_at"])])

    _t("chaos_test_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("scenario", sa.String(48), nullable=False),
        _c("bounded", sa.Boolean(), nullable=False,
           server_default=sa.text("true")),
        _c("simulated", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("passed", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("metrics", sa.Text(), nullable=True),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_chaos_runs_ws", ["workspace_id", "created_at"])])


def downgrade() -> None:
    tables = [
        "chaos_test_runs", "backup_health_records", "maintenance_plans",
        "cost_aware_queue_decisions", "api_abuse_signals_p21",
        "artifact_quality_scores", "learning_dataset_candidates",
        "incidents_p21", "ai_activity_items", "personal_autonomy_settings",
        "memory_autonomy_events", "graph_repair_proposals",
        "cache_health_snapshots", "slow_query_records",
        "scheduler_health_snapshots", "broker_health_snapshots",
        "capacity_recommendations", "worker_health_scores",
        "residency_guard_events", "failover_simulations",
        "region_health_p21", "data_policy_impacts",
        "data_classification_snapshots", "security_incidents_p21",
        "security_health_scores", "tool_safety_violations",
        "autonomy_abuse_attempts", "emergency_stops",
        "evaluation_runs_p21", "evaluation_schedules", "platform_events",
        "workflow_risk_assessments", "agent_recovery_events",
        "agent_plan_risks", "cost_optimization_events",
        "cost_guard_decisions", "cost_anomalies_p21", "cost_forecasts",
        "routing_simulations", "model_drift_events",
        "model_performance_samples", "rag_failure_clusters",
        "rag_drift_snapshots", "retrieval_drift_snapshots",
        "adaptive_candidates", "ingestion_anomalies",
        "ingestion_quality_samples", "knowledge_recovery_plans",
        "diagnosis_reports", "recovery_attempts", "recovery_playbooks",
        "system_health_snapshots", "autonomous_operations",
        "autonomy_transitions", "autonomy_policies",
    ]
    # tables are listed leaf-first (children before parents) so FK
    # constraints are satisfied while dropping
    for name in tables:
        op.drop_table(name)
