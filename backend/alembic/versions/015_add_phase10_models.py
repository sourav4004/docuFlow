"""Add Phase 10 models

Revision ID: 015_add_phase10_models
Revises: 014_add_workspace_to_documents
Create Date: 2026-09-02 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '015_add_phase10_models'
down_revision = '014_add_workspace_to_documents'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Workspace invitations
    op.create_table(
        'workspace_invitations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('inviter_id', sa.Integer(), nullable=False),
        sa.Column('invited_email', sa.String(255), nullable=False),
        sa.Column('role', sa.String(20), nullable=False, server_default='MEMBER'),
        sa.Column('token_hash', sa.String(128), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('accepted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['inviter_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_invitations_workspace_id', 'workspace_invitations', ['workspace_id'])
    op.create_index('ix_invitations_email', 'workspace_invitations', ['invited_email'])
    op.create_index('ix_invitations_status', 'workspace_invitations', ['status'])
    op.create_index('ix_invitations_token_hash', 'workspace_invitations', ['token_hash'], unique=True)

    # Document activities
    op.create_table(
        'document_activities',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_activities_document_id', 'document_activities', ['document_id'])
    op.create_index('ix_document_activities_created_at', 'document_activities', ['created_at'])

    # Notifications
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('notification_type', sa.String(50), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'])
    op.create_index('ix_notifications_read', 'notifications', ['user_id', 'is_read'])
    op.create_index('ix_notifications_created_at', 'notifications', ['created_at'])

    # Document comments
    op.create_table(
        'document_comments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('parent_id', sa.Integer(), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('is_resolved', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_id'], ['document_comments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_comments_document_id', 'document_comments', ['document_id'])
    op.create_index('ix_document_comments_user_id', 'document_comments', ['user_id'])

    # Usage records
    op.create_table(
        'usage_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('usage_type', sa.String(50), nullable=False),
        sa.Column('quantity', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('units', sa.String(20), nullable=False, server_default='count'),
        sa.Column('metadata_json', sa.String(1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_usage_records_workspace_id', 'usage_records', ['workspace_id'])
    op.create_index('ix_usage_records_created_at', 'usage_records', ['created_at'])
    op.create_index('ix_usage_records_type', 'usage_records', ['usage_type'])


def downgrade() -> None:
    op.drop_table('usage_records')
    op.drop_table('document_comments')
    op.drop_table('notifications')
    op.drop_table('document_activities')
    op.drop_table('workspace_invitations')
