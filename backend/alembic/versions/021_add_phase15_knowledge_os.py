"""Add Phase 15 knowledge operating system infrastructure

Revision ID: 021_add_phase15_knowledge_os
Revises: 020_collection_workspace_scope
Create Date: 2026-09-04 09:00:00.000000

Additive only — no existing data is modified or destroyed.
"""

from alembic import op
import sqlalchemy as sa


revision = '021_add_phase15_knowledge_os'
down_revision = '020_collection_workspace_scope'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- AI execution extensions (orchestrator 2.0) ---
    op.add_column('ai_executions', sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='SET NULL'), nullable=True))
    op.add_column('ai_executions', sa.Column('action_id', sa.Integer(), sa.ForeignKey('ai_actions.id', ondelete='SET NULL'), nullable=True))
    op.add_column('ai_executions', sa.Column('execution_type', sa.String(50), nullable=False, server_default='rag'))
    op.add_column('ai_executions', sa.Column('parent_execution_id', sa.String(36), nullable=True))
    op.add_column('ai_executions', sa.Column('priority', sa.String(20), nullable=False, server_default='NORMAL'))
    op.add_column('ai_executions', sa.Column('idempotency_key', sa.String(128), nullable=True))
    op.add_column('ai_executions', sa.Column('request_hash', sa.String(64), nullable=True))
    op.add_column('ai_executions', sa.Column('input_reference', sa.String(255), nullable=True))
    op.add_column('ai_executions', sa.Column('output_reference', sa.String(255), nullable=True))
    op.add_column('ai_executions', sa.Column('trace_id', sa.String(64), nullable=True))
    op.add_column('ai_executions', sa.Column('actual_cost', sa.Float(), server_default='0.0', nullable=False))
    op.add_column('ai_executions', sa.Column('latency_ms', sa.Float(), nullable=True))
    op.add_column('ai_executions', sa.Column('failure_reason', sa.Text(), nullable=True))
    op.add_column('ai_executions', sa.Column('retry_count', sa.Integer(), server_default='0', nullable=False))
    op.create_index('ix_ai_executions_org_id', 'ai_executions', ['organization_id'])
    op.create_index('ix_ai_executions_priority', 'ai_executions', ['priority'])
    op.create_index('ix_ai_executions_parent', 'ai_executions', ['parent_execution_id'])

    # Documents — sensitivity classification
    op.add_column('documents', sa.Column('sensitivity', sa.String(20), nullable=False, server_default='INTERNAL'))

    # Entities 2.0
    op.add_column('entities', sa.Column('confidence', sa.Float(), server_default='1.0', nullable=False))
    op.add_column('entities', sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('entities', sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True))

    # AI actions — dependency graph
    op.add_column('ai_actions', sa.Column('parent_id', sa.Integer(), sa.ForeignKey('ai_actions.id', ondelete='SET NULL'), nullable=True))
    op.add_column('ai_actions', sa.Column('dependency_status', sa.String(20), nullable=False, server_default='NONE'))
    op.add_column('ai_actions', sa.Column('blocked_reason', sa.Text(), nullable=True))

    # --- Execution idempotency + checkpoints ---
    op.create_table(
        'ai_execution_idempotency',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('idempotency_key', sa.String(128), nullable=False),
        sa.Column('request_hash', sa.String(64), nullable=False),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('ai_executions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('result_status', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('workspace_id', 'idempotency_key', name='uq_execution_idempotency_key'),
    )
    op.create_index('ix_execution_idem_expires', 'ai_execution_idempotency', ['expires_at'])

    op.create_table(
        'ai_execution_checkpoints',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('ai_executions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('step_number', sa.Integer(), nullable=False),
        sa.Column('state_json', sa.Text(), nullable=True),
        sa.Column('tool_output_reference', sa.String(255), nullable=True),
        sa.Column('artifact_reference', sa.String(255), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_exec_checkpoints_execution', 'ai_execution_checkpoints', ['execution_id'])

    # --- Knowledge event outbox ---
    op.create_table(
        'knowledge_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('aggregate_type', sa.String(50), nullable=False, server_default='unknown'),
        sa.Column('aggregate_id', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('dedupe_key', sa.String(64), nullable=False, server_default=''),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('workspace_id', 'event_type', 'aggregate_type', 'aggregate_id', 'dedupe_key',
                            name='uq_knowledge_event_dedupe'),
    )
    op.create_index('ix_knowledge_events_status_next_retry', 'knowledge_events', ['status', 'next_retry_at'])
    op.create_index('ix_knowledge_events_workspace', 'knowledge_events', ['workspace_id'])

    # --- Knowledge changes + impact links ---
    op.create_table(
        'knowledge_changes',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('version_from', sa.Integer(), nullable=True),
        sa.Column('version_to', sa.Integer(), nullable=True),
        sa.Column('change_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, server_default='MEDIUM'),
        sa.Column('confidence', sa.String(20), nullable=False, server_default='MEDIUM'),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('old_evidence_json', sa.Text(), nullable=True),
        sa.Column('new_evidence_json', sa.Text(), nullable=True),
        sa.Column('affected_sections_json', sa.Text(), nullable=True),
        sa.Column('affected_entities_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_knowledge_changes_workspace', 'knowledge_changes', ['workspace_id'])
    op.create_index('ix_knowledge_changes_document', 'knowledge_changes', ['document_id'])

    op.create_table(
        'impact_links',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_type', sa.String(30), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('target_type', sa.String(30), nullable=False),
        sa.Column('target_id', sa.Integer(), nullable=False),
        sa.Column('relation', sa.String(20), nullable=False, server_default='EXPLICIT'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_impact_links_source', 'impact_links', ['source_type', 'source_id'])
    op.create_index('ix_impact_links_target', 'impact_links', ['target_type', 'target_id'])

    # --- Policy engine ---
    op.create_table(
        'policy_statements',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('source_version', sa.Integer(), nullable=True),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('requirement_type', sa.String(50), nullable=True),
        sa.Column('applicability', sa.Text(), nullable=True),
        sa.Column('effective_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expiration_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('responsible_party', sa.String(255), nullable=True),
        sa.Column('evidence_reference', sa.String(255), nullable=True),
        sa.Column('source_chunk', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_policy_statements_workspace', 'policy_statements', ['workspace_id'])

    op.create_table(
        'policy_conflicts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('policy_a_id', sa.Integer(), sa.ForeignKey('policy_statements.id', ondelete='CASCADE'), nullable=False),
        sa.Column('policy_b_id', sa.Integer(), sa.ForeignKey('policy_statements.id', ondelete='CASCADE'), nullable=False),
        sa.Column('conflict_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, server_default='MEDIUM'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('conditions_json', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='OPEN'),
        sa.Column('resolution_note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_policy_conflicts_workspace', 'policy_conflicts', ['workspace_id'])

    # --- Temporal knowledge ---
    op.create_table(
        'temporal_facts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('entity_id', sa.Integer(), nullable=True),
        sa.Column('fact_type', sa.String(50), nullable=False),
        sa.Column('fact_value', sa.Text(), nullable=False),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('superseded_by', sa.Integer(), nullable=True),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_temporal_facts_workspace', 'temporal_facts', ['workspace_id'])

    # --- Knowledge snapshots ---
    op.create_table(
        'knowledge_snapshots',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('snapshot_type', sa.String(30), nullable=False, server_default='workspace'),
        sa.Column('data_json', sa.Text(), nullable=False),
        sa.Column('created_by', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_knowledge_snapshots_workspace', 'knowledge_snapshots', ['workspace_id'])

    # --- AI memory ---
    op.create_table(
        'ai_memories',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('memory_type', sa.String(30), nullable=False),
        sa.Column('scope', sa.String(30), nullable=False, server_default='WORKSPACE'),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('confidence', sa.String(20), nullable=False, server_default='MEDIUM'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_ai_memories_workspace', 'ai_memories', ['workspace_id'])
    op.create_index('ix_ai_memories_user', 'ai_memories', ['user_id'])

    # --- Review queue ---
    op.create_table(
        'review_items',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('item_type', sa.String(40), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('priority', sa.String(20), nullable=False, server_default='NORMAL'),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.Column('source_type', sa.String(30), nullable=True),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('assignee_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('due_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('escalated', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('decision_note', sa.Text(), nullable=True),
        sa.Column('decided_by', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_review_items_workspace', 'review_items', ['workspace_id'])
    op.create_index('ix_review_items_status', 'review_items', ['status'])

    # --- Workflow reliability ---
    op.create_table(
        'workflow_node_executions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workflow_execution_id', sa.String(64), nullable=False),
        sa.Column('workflow_id', sa.String(64), nullable=False),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('node_id', sa.String(64), nullable=False),
        sa.Column('attempt', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('input_hash', sa.String(64), nullable=True),
        sa.Column('output_reference', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('workflow_execution_id', 'node_id', 'attempt', name='uq_wf_node_attempt'),
    )
    op.create_index('ix_wf_node_exec_execution', 'workflow_node_executions', ['workflow_execution_id'])

    op.create_table(
        'workflow_compensations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workflow_execution_id', sa.String(64), nullable=False),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('node_id', sa.String(64), nullable=False),
        sa.Column('side_effect_type', sa.String(50), nullable=False),
        sa.Column('reversible', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(20), nullable=False, server_default='RECORDED'),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_wf_comp_execution', 'workflow_compensations', ['workflow_execution_id'])

    # --- Provider health ---
    op.create_table(
        'provider_health',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='UNKNOWN'),
        sa.Column('circuit_state', sa.String(20), nullable=False, server_default='CLOSED'),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_latency_ms', sa.Float(), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('provider', 'model', name='uq_provider_model'),
    )

    # --- AI quality metrics ---
    op.create_table(
        'ai_quality_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('feature', sa.String(50), nullable=True),
        sa.Column('model', sa.String(100), nullable=True),
        sa.Column('provider', sa.String(100), nullable=True),
        sa.Column('value', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_ai_quality_workspace', 'ai_quality_metrics', ['workspace_id'])
    op.create_index('ix_ai_quality_type', 'ai_quality_metrics', ['metric_type'])

    # --- Reconnectable execution streams ---
    op.create_table(
        'ai_stream_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('ai_executions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(30), nullable=False),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('execution_id', 'seq', name='uq_stream_seq'),
    )
    op.create_index('ix_ai_stream_execution', 'ai_stream_events', ['execution_id'])

    # --- Knowledge gaps ---
    op.create_table(
        'knowledge_gaps',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.Integer(), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('gap_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, server_default='MEDIUM'),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('evidence_json', sa.Text(), nullable=True),
        sa.Column('remediation', sa.Text(), nullable=True),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('entity_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='OPEN'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_knowledge_gaps_workspace', 'knowledge_gaps', ['workspace_id'])

    # --- Request deduplication ---
    op.create_table(
        'request_dedup',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', sa.Integer(), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('kind', sa.String(50), nullable=False),
        sa.Column('fingerprint', sa.String(64), nullable=False),
        sa.Column('result_reference', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('workspace_id', 'kind', 'fingerprint', name='uq_request_dedup'),
    )
    op.create_index('ix_request_dedup_expires', 'request_dedup', ['expires_at'])


def downgrade() -> None:
    tables = [
        'request_dedup', 'knowledge_gaps', 'ai_stream_events', 'ai_quality_metrics',
        'provider_health', 'workflow_compensations', 'workflow_node_executions',
        'review_items', 'ai_memories', 'knowledge_snapshots', 'temporal_facts',
        'policy_conflicts', 'policy_statements', 'impact_links', 'knowledge_changes',
        'knowledge_events', 'ai_execution_checkpoints', 'ai_execution_idempotency',
    ]
    for table in tables:
        op.drop_table(table)

    op.drop_index('ix_ai_executions_parent', table_name='ai_executions')
    op.drop_index('ix_ai_executions_priority', table_name='ai_executions')
    op.drop_index('ix_ai_executions_org_id', table_name='ai_executions')
    for column in ('retry_count', 'failure_reason', 'latency_ms', 'actual_cost', 'trace_id',
                   'input_reference', 'output_reference', 'request_hash', 'idempotency_key',
                   'priority', 'parent_execution_id', 'execution_type', 'action_id', 'organization_id'):
        op.drop_column('ai_executions', column)
    op.drop_column('documents', 'sensitivity')
    op.drop_column('entities', 'last_seen_at')
    op.drop_column('entities', 'first_seen_at')
    op.drop_column('entities', 'confidence')
    op.drop_column('ai_actions', 'blocked_reason')
    op.drop_column('ai_actions', 'dependency_status')
    op.drop_column('ai_actions', 'parent_id')