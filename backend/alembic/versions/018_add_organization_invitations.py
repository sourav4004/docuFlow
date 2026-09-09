"""Add organization invitations table

Revision ID: 018_add_organization_invitations
Revises: 017_add_phase13_enterprise
Create Date: 2026-09-03 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '018_add_organization_invitations'
down_revision = '017_add_phase13_enterprise'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'organization_invitations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_org_invitations_organization_id', 'organization_invitations', ['organization_id'])
    op.create_index('ix_org_invitations_email', 'organization_invitations', ['invited_email'])
    op.create_index('ix_org_invitations_status', 'organization_invitations', ['status'])
    op.create_index('ix_org_invitations_token_hash', 'organization_invitations', ['token_hash'], unique=True)


def downgrade() -> None:
    op.drop_table('organization_invitations')