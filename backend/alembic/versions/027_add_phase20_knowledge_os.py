"""Phase 20 migration — Self-Improving Enterprise AI Knowledge OS.

Adds the governed self-improvement substrate: improvement proposals with
audited lifecycle transitions, the experiment platform (datasets, immutable
config experiments, runs, comparisons), AI quality scorecards/trends/alerts,
retrieval + RAG failure classification and recommendations, knowledge
health/gaps, document change intelligence, versioned policies + impact,
provider profiles + routing recommendations + anomalies, cost baselines +
token efficiency, agent/workflow/memory/graph intelligence, search quality
events, unified feedback, incidents with timelines, SLO history + error
budgets + reliability scores, prioritized alerts + notification preferences,
versioned reports, retention policies, and API/database health metrics.

Single head; downgrade drops the new tables only (no destructive change to
pre-existing data).
"""

from alembic import op
import sqlalchemy as sa

revision = "027_phase20_knowledge_os"
down_revision = "026_phase19_cloud2"
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
    # --- improvement control plane ------------------------------------------
    _t("improvement_proposals",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("domain", sa.String(40), nullable=False),
        _c("title", sa.String(300), nullable=False),
        _c("problem", sa.Text(), nullable=False),
        _c("evidence", sa.Text(), nullable=True),
        _c("expected_benefit", sa.Text(), nullable=True),
        _c("risk", sa.String(20), nullable=False, server_default="LOW"),
        _c("estimated_cost", sa.Float(), nullable=True),
        _c("proposed_change", sa.Text(), nullable=False),
        _c("evaluation_requirements", sa.Text(), nullable=True),
        _c("author_source", sa.String(100), nullable=True),
        _c("author_user_id", sa.Integer(), nullable=True),
        _c("status", sa.String(24), nullable=False, server_default="PROPOSED"),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_improv_proposals_status", ["status", "domain"]),
                ("ix_improv_proposals_domain", ["domain"])])

    _t("improvement_audits",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("proposal_id", sa.Integer(), nullable=False),
        _c("actor_user_id", sa.Integer(), nullable=True),
        _c("previous_state", sa.String(24), nullable=True),
        _c("new_state", sa.String(24), nullable=False),
        _c("reason", sa.String(1000), nullable=True),
        _c("evidence", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["proposal_id"], ["improvement_proposals.id"],
             {"ondelete": "CASCADE"})],
       indexes=[("ix_improv_audits_proposal", ["proposal_id", "created_at"])])

    _t("improvement_transitions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("from_state", sa.String(24), nullable=False),
        _c("to_state", sa.String(24), nullable=False),
        _c("requires_approval", sa.Boolean(), nullable=False,
           server_default=sa.text("false"))],
       uqs=[("uq_improv_transition", ["from_state", "to_state"])])

    # --- experiment platform -------------------------------------------------
    _t("experiment_datasets",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("name", sa.String(120), nullable=False),
        _c("kind", sa.String(20), nullable=False),
        _c("domain", sa.String(40), nullable=False, server_default="retrieval"),
        _c("items_json", sa.Text(), nullable=False),
        _c("created_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_experiment_dataset_name", ["name"])])

    _t("experiments",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("proposal_id", sa.Integer(), nullable=True),
        _c("name", sa.String(200), nullable=False),
        _c("domain", sa.String(40), nullable=False),
        _c("config_json", sa.Text(), nullable=False),
        _c("config_fingerprint", sa.String(64), nullable=False),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="DRAFT"),
        _c("created_by", sa.Integer(), nullable=True),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["proposal_id"], ["improvement_proposals.id"],
             {"ondelete": "SET NULL"}),
            (["dataset_id"], ["experiment_datasets.id"],
             {"ondelete": "SET NULL"})],
       indexes=[("ix_experiments_status", ["status"]),
                ("ix_experiments_domain", ["domain"])])

    _t("experiment_runs",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("experiment_id", sa.Integer(), nullable=False),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("dataset_name", sa.String(120), nullable=True),
        _c("model", sa.String(120), nullable=True),
        _c("provider", sa.String(120), nullable=True),
        _c("config_json", sa.Text(), nullable=True),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("cost", sa.Float(), nullable=True),
        _c("latency_ms", sa.Float(), nullable=True),
        _c("environment", sa.String(40), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="RUNNING"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["experiment_id"], ["experiments.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_experiment_runs_experiment",
                 ["experiment_id", "created_at"])])

    _t("experiment_comparisons",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("experiment_id", sa.Integer(), nullable=False),
        _c("baseline_run_id", sa.Integer(), nullable=False),
        _c("candidate_run_id", sa.Integer(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("verdict", sa.String(20), nullable=False),
        _c("sample_size", sa.Integer(), nullable=False, server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["experiment_id"], ["experiments.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_experiment_comparisons_exp",
                 ["experiment_id"])])

    # --- AI quality platform 3.0 ---------------------------------------------
    _t("quality_scorecards",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("domain", sa.String(40), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("organization_id", sa.Integer(), nullable=True),
        _c("period", sa.String(10), nullable=False, server_default="daily"),
        _c("scorecard_json", sa.Text(), nullable=False),
        _c("overall_score", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_quality_scorecards_scope", ["domain", "workspace_id"])])

    _t("quality_trends",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("domain", sa.String(40), nullable=False),
        _c("period", sa.String(10), nullable=False),
        _c("window_start", sa.DateTime(timezone=True), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("score", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_quality_trends_domain", ["domain", "period"])])

    _t("quality_alerts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("domain", sa.String(40), nullable=False),
        _c("severity", sa.String(10), nullable=False, server_default="MEDIUM"),
        _c("fingerprint", sa.String(64), nullable=False),
        _c("message", sa.Text(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("resolved", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_quality_alert_fingerprint", ["fingerprint"])],
       indexes=[("ix_quality_alerts_domain", ["domain", "created_at"])])

    # --- retrieval / RAG self-improvement ------------------------------------
    _t("retrieval_failures",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("query_hash", sa.String(64), nullable=False),
        _c("query_preview", sa.String(300), nullable=True),
        _c("failure_class", sa.String(40), nullable=False),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_retrieval_failures_ws", ["workspace_id", "created_at"])])

    _t("retrieval_recommendations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("category", sa.String(40), nullable=False),
        _c("suggestion", sa.Text(), nullable=False),
        _c("rationale", sa.Text(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="PROPOSED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_retrieval_recos_ws", ["workspace_id", "status"])])

    _t("rag_failures",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("execution_id", sa.String(64), nullable=True),
        _c("failure_class", sa.String(40), nullable=False),
        _c("claim", sa.Text(), nullable=True),
        _c("detail", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_rag_failures_ws", ["workspace_id", "created_at"])])

    _t("rag_evaluation_pipelines",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("proposal_id", sa.Integer(), nullable=True),
        _c("dataset_id", sa.Integer(), nullable=True),
        _c("config_json", sa.Text(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=True),
        _c("gate_passed", sa.Boolean(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="RUNNING"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])

    # --- knowledge intelligence ----------------------------------------------
    _t("knowledge_health",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("scope_type", sa.String(20), nullable=False),
        _c("scope_id", sa.Integer(), nullable=False),
        _c("health_json", sa.Text(), nullable=False),
        _c("score", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_knowledge_health_scope", ["scope_type", "scope_id"])])

    _t("knowledge_gap_insights",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("query_hash", sa.String(64), nullable=False),
        _c("query_preview", sa.String(300), nullable=True),
        _c("attempts", sa.Integer(), nullable=False, server_default="0"),
        _c("best_evidence_score", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_kgap_insights_ws", ["workspace_id", "attempts"])])

    _t("doc_change_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("document_id", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("version_from", sa.Integer(), nullable=True),
        _c("version_to", sa.Integer(), nullable=True),
        _c("change_class", sa.String(30), nullable=False),
        _c("impact_json", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_doc_change_events_doc", ["document_id", "created_at"])])

    # --- policy intelligence ---------------------------------------------------
    _t("policy_versions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("scope_type", sa.String(20), nullable=False),
        _c("scope_id", sa.Integer(), nullable=True),
        _c("version", sa.Integer(), nullable=False),
        _c("policy_json", sa.Text(), nullable=False),
        _c("diff_json", sa.Text(), nullable=True),
        _c("actor_user_id", sa.Integer(), nullable=True),
        _c("reason", sa.String(1000), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_policy_version", ["scope_type", "scope_id", "version"])],
       indexes=[("ix_policy_versions_scope", ["scope_type", "scope_id"])])

    _t("policy_impacts",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("policy_version_id", sa.Integer(), nullable=False),
        _c("impact_json", sa.Text(), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_policy_impacts_version", ["policy_version_id"])])

    # --- model / provider optimization -----------------------------------------
    _t("provider_profiles",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("provider", sa.String(120), nullable=False),
        _c("model", sa.String(120), nullable=False),
        _c("period", sa.String(10), nullable=False, server_default="daily"),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_provider_profiles_provider", ["provider", "model"])])

    _t("routing_recommendations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("domain", sa.String(40), nullable=False),
        _c("candidate", sa.Text(), nullable=False),
        _c("rationale", sa.Text(), nullable=True),
        _c("expected_gain", sa.Text(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="PROPOSED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_routing_recos_status", ["status"])])

    _t("provider_anomalies",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("provider", sa.String(120), nullable=False),
        _c("model", sa.String(120), nullable=True),
        _c("anomaly_type", sa.String(30), nullable=False),
        _c("detail", sa.Text(), nullable=True),
        _c("severity", sa.String(10), nullable=False, server_default="MEDIUM"),
        _c("resolved", sa.Boolean(), nullable=False,
           server_default=sa.text("false")),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])

    # --- cost intelligence 5.0 ---------------------------------------------------
    _t("cost_baselines",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("scope_type", sa.String(20), nullable=False),
        _c("scope_id", sa.Integer(), nullable=True),
        _c("dimension", sa.String(60), nullable=True),
        _c("baseline_json", sa.Text(), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_cost_baselines_scope", ["scope_type", "scope_id"])])

    _t("token_efficiency",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("execution_ref", sa.String(64), nullable=True),
        _c("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        _c("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        _c("context_tokens", sa.Integer(), nullable=False, server_default="0"),
        _c("repeated_tokens", sa.Integer(), nullable=False, server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_token_efficiency_ws", ["workspace_id", "created_at"])])

    # --- agent / workflow / memory / graph intelligence ---------------------------
    _t("agent_intelligence",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("execution_id", sa.String(64), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("failure_class", sa.String(30), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_agent_intelligence_ws", ["workspace_id", "created_at"])])

    _t("workflow_intelligence",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("run_id", sa.Integer(), nullable=False),
        _c("node_id", sa.String(120), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_workflow_intel_run", ["run_id", "node_id"])])

    _t("memory_intelligence",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("memory_id", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_memory_intel_memory", ["memory_id"])])

    _t("graph_health",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("metrics_json", sa.Text(), nullable=False),
        _c("score", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_graph_health_ws", ["workspace_id", "created_at"])])

    _t("graph_recommendations",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("kind", sa.String(20), nullable=False),
        _c("candidate", sa.Text(), nullable=False),
        _c("rationale", sa.Text(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="PROPOSED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])

    # --- search intelligence + unified feedback -----------------------------------
    _t("search_quality_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("query_hash", sa.String(64), nullable=False),
        _c("query_preview", sa.String(300), nullable=True),
        _c("event_type", sa.String(30), nullable=False),
        _c("result_count", sa.Integer(), nullable=True),
        _c("latency_ms", sa.Float(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_search_quality_ws", ["workspace_id", "created_at"])])

    _t("feedback_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("workspace_id", sa.Integer(), nullable=False),
        _c("user_id", sa.Integer(), nullable=True),
        _c("source", sa.String(30), nullable=False),
        _c("target_id", sa.String(64), nullable=True),
        _c("rating", sa.Integer(), nullable=True),
        _c("comment", sa.Text(), nullable=True),
        _c("quality_flags", sa.Text(), nullable=True),
        _c("status", sa.String(20), nullable=False, server_default="RECEIVED"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_feedback_events_ws", ["workspace_id", "created_at"])])

    # --- incidents ---------------------------------------------------------------
    _t("incidents",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("title", sa.String(300), nullable=False),
        _c("severity", sa.String(10), nullable=False, server_default="MEDIUM"),
        _c("status", sa.String(20), nullable=False, server_default="OPEN"),
        _c("affected_systems", sa.Text(), nullable=True),
        _c("summary", sa.Text(), nullable=True),
        _c("fingerprint", sa.String(64), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_incidents_status", ["status", "severity"])])

    _t("incident_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("incident_id", sa.Integer(), nullable=False),
        _c("event_type", sa.String(30), nullable=False),
        _c("detail", sa.Text(), nullable=True),
        _c("actor_user_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       fks=[(["incident_id"], ["incidents.id"], {"ondelete": "CASCADE"})],
       indexes=[("ix_incident_events_incident", ["incident_id", "created_at"])])

    # --- SLO 2.0 --------------------------------------------------------------------
    _t("slo_history",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("name", sa.String(120), nullable=False),
        _c("window", sa.String(10), nullable=False, server_default="daily"),
        _c("measured_value", sa.Float(), nullable=True),
        _c("target", sa.Float(), nullable=True),
        _c("met", sa.Boolean(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_slo_history_name", ["name", "created_at"])])

    _t("error_budgets",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("name", sa.String(120), nullable=False),
        _c("period", sa.String(10), nullable=False, server_default="monthly"),
        _c("budget", sa.Float(), nullable=False, server_default="100.0"),
        _c("consumed", sa.Float(), nullable=False, server_default="0.0"),
        _c("status", sa.String(20), nullable=False, server_default="OK"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_error_budget_period", ["name", "period"])])

    _t("reliability_scores",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("service", sa.String(60), nullable=False),
        _c("score", sa.Float(), nullable=False),
        _c("detail_json", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])

    # --- notifications / alerts ------------------------------------------------------
    _t("ops_alert_events",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("category", sa.String(40), nullable=False),
        _c("severity", sa.String(10), nullable=False),
        _c("fingerprint", sa.String(64), nullable=False),
        _c("message", sa.Text(), nullable=False),
        _c("status", sa.String(20), nullable=False, server_default="OPEN"),
        _c("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now()),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_ops_alert_event_fingerprint", ["fingerprint"])],
       indexes=[("ix_ops_alert_events_status", ["status", "severity"])])

    _t("notification_preferences",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("user_id", sa.Integer(), nullable=False),
        _c("workspace_id", sa.Integer(), nullable=True),
        _c("category", sa.String(40), nullable=False),
        _c("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_notification_pref",
             ["user_id", "workspace_id", "category"])])

    # --- reports + retention -----------------------------------------------------------
    _t("report_versions",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("kind", sa.String(30), nullable=False),
        _c("scope_type", sa.String(20), nullable=False, server_default="WORKSPACE"),
        _c("scope_id", sa.Integer(), nullable=True),
        _c("version", sa.Integer(), nullable=False),
        _c("content_json", sa.Text(), nullable=False),
        _c("generated_by", sa.Integer(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_report_version",
             ["kind", "scope_type", "scope_id", "version"])],
       indexes=[("ix_report_versions_kind", ["kind", "scope_type"])])

    _t("improvement_retention",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("category", sa.String(40), nullable=False),
        _c("retention_days", sa.Integer(), nullable=False, server_default="90"),
        _c("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        _c("updated_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       uqs=[("uq_improvement_retention_category", ["category"])])

    # --- API / database health -----------------------------------------------------------
    _t("api_health_metrics",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("endpoint", sa.String(300), nullable=False),
        _c("period", sa.String(10), nullable=False, server_default="hourly"),
        _c("requests", sa.Integer(), nullable=False, server_default="0"),
        _c("errors", sa.Integer(), nullable=False, server_default="0"),
        _c("latency_p50_ms", sa.Float(), nullable=True),
        _c("latency_p95_ms", sa.Float(), nullable=True),
        _c("auth_failures", sa.Integer(), nullable=False, server_default="0"),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())],
       indexes=[("ix_api_health_metric_ep", ["endpoint", "period"])])

    _t("db_health_metrics",
       [_c("id", sa.Integer(), primary_key=True, autoincrement=True),
        _c("metric", sa.String(60), nullable=False),
        _c("value", sa.Float(), nullable=True),
        _c("detail_json", sa.Text(), nullable=True),
        _c("created_at", sa.DateTime(timezone=True), nullable=False,
           server_default=_now())])


def downgrade() -> None:
    tables = [
        "db_health_metrics", "api_health_metrics", "improvement_retention",
        "report_versions", "notification_preferences", "ops_alert_events",
        "reliability_scores", "error_budgets", "slo_history",
        "incident_events", "incidents", "feedback_events",
        "search_quality_events", "graph_recommendations", "graph_health",
        "memory_intelligence", "workflow_intelligence",
        "agent_intelligence", "token_efficiency", "cost_baselines",
        "provider_anomalies", "routing_recommendations", "provider_profiles",
        "policy_impacts", "policy_versions", "doc_change_events",
        "knowledge_gap_insights", "knowledge_health",
        "rag_evaluation_pipelines", "rag_failures",
        "retrieval_recommendations", "retrieval_failures",
        "quality_alerts", "quality_trends", "quality_scorecards",
        "experiment_comparisons", "experiment_runs", "experiments",
        "experiment_datasets", "improvement_transitions",
        "improvement_audits", "improvement_proposals",
    ]
    # tables are listed leaf-first (children before parents) so FK
    # constraints are satisfied while dropping
    for t in tables:
        op.drop_table(t)