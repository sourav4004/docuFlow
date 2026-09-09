"""Add Phase 14 AI productization tables

Revision ID: 019_add_phase14_ai_product
Revises: 018_add_organization_invitations
Create Date: 2026-09-03 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '019_add_phase14_ai_product'
down_revision = '018_add_organization_invitations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # AI Actions
    op.create_table(
        'ai_actions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('action_type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='SUGGESTED'),
        sa.Column('risk_level', sa.String(20), nullable=False, server_default='LOW'),
        sa.Column('priority', sa.String(20), nullable=False, server_default='NORMAL'),
        sa.Column('source_type', sa.String(50), nullable=False, server_default='manual'),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('execution_id', sa.String(36), nullable=True),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.Column('result_json', sa.Text(), nullable=True),
        sa.Column('estimated_cost', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('approved_by', sa.Integer(), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_actions_workspace_id', 'ai_actions', ['workspace_id'])
    op.create_index('ix_ai_actions_owner_id', 'ai_actions', ['owner_id'])
    op.create_index('ix_ai_actions_status', 'ai_actions', ['status'])
    op.create_index('ix_ai_actions_created_at', 'ai_actions', ['created_at'])
    op.create_index('ix_ai_actions_source', 'ai_actions', ['source_type', 'source_id'])

    # AI Suggestions
    op.create_table(
        'ai_suggestions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('suggestion_type', sa.String(50), nullable=False),
        sa.Column('source_type', sa.String(50), nullable=False, server_default='system'),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('priority', sa.String(20), nullable=False, server_default='NORMAL'),
        sa.Column('status', sa.String(20), nullable=False, server_default='OPEN'),
        sa.Column('evidence_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_suggestions_workspace_id', 'ai_suggestions', ['workspace_id'])
    op.create_index('ix_ai_suggestions_owner_id', 'ai_suggestions', ['owner_id'])
    op.create_index('ix_ai_suggestions_status', 'ai_suggestions', ['status'])
    op.create_index('ix_ai_suggestions_source', 'ai_suggestions', ['source_type', 'source_id'])
    op.create_index('ix_ai_suggestions_created_at', 'ai_suggestions', ['created_at'])

    # Document health
    op.create_table(
        'document_health',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('score', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('factors_json', sa.Text(), nullable=True),
        sa.Column('reasons_json', sa.Text(), nullable=True),
        sa.Column('computed_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_health_workspace_id', 'document_health', ['workspace_id'])
    op.create_index('ix_document_health_document_id', 'document_health', ['document_id'])
    op.create_index('ix_document_health_score', 'document_health', ['score'])
    op.create_unique_constraint('uq_document_health_document', 'document_health', ['document_id'])

    # Deadlines
    op.create_table(
        'deadlines',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source', sa.String(50), nullable=False, server_default='manual'),
        sa.Column('source_reference', sa.Text(), nullable=True),
        sa.Column('confidence', sa.String(20), nullable=False, server_default='UNKNOWN'),
        sa.Column('status', sa.String(20), nullable=False, server_default='UPCOMING'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_deadlines_workspace_id', 'deadlines', ['workspace_id'])
    op.create_index('ix_deadlines_owner_id', 'deadlines', ['owner_id'])
    op.create_index('ix_deadlines_status', 'deadlines', ['status'])
    op.create_index('ix_deadlines_due_date', 'deadlines', ['due_date'])

    # Knowledge insights
    op.create_table(
        'knowledge_insights',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=True),
        sa.Column('insight_type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('importance', sa.String(20), nullable=False, server_default='INFO'),
        sa.Column('evidence_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_insights_workspace_id', 'knowledge_insights', ['workspace_id'])
    op.create_index('ix_insights_type', 'knowledge_insights', ['insight_type'])
    op.create_index('ix_insights_created_at', 'knowledge_insights', ['created_at'])

    # Saved searches
    op.create_table(
        'saved_searches',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('filters_json', sa.Text(), nullable=True),
        sa.Column('is_shared', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_saved_searches_workspace_id', 'saved_searches', ['workspace_id'])
    op.create_index('ix_saved_searches_owner_id', 'saved_searches', ['owner_id'])

    # Search alerts
    op.create_table(
        'search_alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('saved_search_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_match_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['saved_search_id'], ['saved_searches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('saved_search_id', name='uq_alert_per_search')
    )
    op.create_index('ix_search_alerts_workspace_id', 'search_alerts', ['workspace_id'])
    op.create_index('ix_search_alerts_saved_search_id', 'search_alerts', ['saved_search_id'])

    # AI feedback
    op.create_table(
        'ai_feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(36), nullable=True),
        sa.Column('rating', sa.String(20), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_feedback_workspace_id', 'ai_feedback', ['workspace_id'])
    op.create_index('ix_ai_feedback_execution_id', 'ai_feedback', ['execution_id'])
    op.create_index('ix_ai_feedback_user_id', 'ai_feedback', ['user_id'])
    op.create_index('ix_ai_feedback_created_at', 'ai_feedback', ['created_at'])

    # Workflow versions
    op.create_table(
        'workflow_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workflow_id', sa.String(64), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('definition_json', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='DRAFT'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_versions_workflow_id', 'workflow_versions', ['workflow_id'])
    op.create_index('ix_workflow_versions_workspace_id', 'workflow_versions', ['workspace_id'])
    op.create_index('ix_workflow_versions_is_active', 'workflow_versions', ['is_active'])


def downgrade() -> None:
    op.drop_table('workflow_versions')
    op.drop_table('ai_feedback')
    op.drop_table('search_alerts')
    op.drop_table('saved_searches')
    op.drop_table('knowledge_insights')
    op.drop_table('deadlines')
    op.drop_table('document_health')
    op.drop_table('ai_suggestions')
    op.drop_table('ai_actions')