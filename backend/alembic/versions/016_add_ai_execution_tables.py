"""Add AI execution tables

Revision ID: 016_add_ai_execution_tables
Revises: 015_add_phase10_models
Create Date: 2026-09-02 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '016_add_ai_execution_tables'
down_revision = '015_add_phase10_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # AI Executions
    op.create_table(
        'ai_executions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='created'),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('model', sa.String(100), nullable=True),
        sa.Column('provider', sa.String(50), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('output_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('estimated_cost', sa.Float(), nullable=True, server_default='0'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('result_json', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_executions_workspace_id', 'ai_executions', ['workspace_id'])
    op.create_index('ix_ai_executions_user_id', 'ai_executions', ['user_id'])
    op.create_index('ix_ai_executions_status', 'ai_executions', ['status'])
    op.create_index('ix_ai_executions_created_at', 'ai_executions', ['created_at'])

    # AI Execution Steps
    op.create_table(
        'ai_execution_steps',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('execution_id', sa.String(36), nullable=False),
        sa.Column('step_index', sa.Integer(), nullable=False),
        sa.Column('step_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('tool_name', sa.String(100), nullable=True),
        sa.Column('tool_input_json', sa.JSON(), nullable=True),
        sa.Column('tool_output_json', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('output_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['execution_id'], ['ai_executions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_execution_steps_execution_id', 'ai_execution_steps', ['execution_id'])

    # AI Approvals
    op.create_table(
        'ai_approvals',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('execution_id', sa.String(36), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(255), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('risk_level', sa.String(20), nullable=False, server_default='read_only'),
        sa.Column('affected_resources_json', sa.JSON(), nullable=True),
        sa.Column('proposed_parameters_json', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('approved_by', sa.Integer(), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['execution_id'], ['ai_executions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_approvals_execution_id', 'ai_approvals', ['execution_id'])
    op.create_index('ix_ai_approvals_status', 'ai_approvals', ['status'])

    # AI Artifacts
    op.create_table(
        'ai_artifacts',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('artifact_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('content_json', sa.JSON(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('source_document_ids_json', sa.JSON(), nullable=True),
        sa.Column('model_used', sa.String(100), nullable=True),
        sa.Column('provider_used', sa.String(50), nullable=True),
        sa.Column('status', sa.String(20), nullable=True, server_default='draft'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['execution_id'], ['ai_executions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_artifacts_workspace_id', 'ai_artifacts', ['workspace_id'])
    op.create_index('ix_ai_artifacts_execution_id', 'ai_artifacts', ['execution_id'])


def downgrade() -> None:
    op.drop_table('ai_artifacts')
    op.drop_table('ai_approvals')
    op.drop_table('ai_execution_steps')
    op.drop_table('ai_executions')
