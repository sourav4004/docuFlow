"""Add workspace_id to documents table

Revision ID: 014_add_workspace_to_documents
Revises: 013_add_phase9_models
Create Date: 2026-09-02 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '014_add_workspace_to_documents'
down_revision = '013_add_phase9_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('documents', sa.Column('workspace_id', sa.Integer(), nullable=True))
    op.create_index('ix_documents_workspace_id', 'documents', ['workspace_id'])
    op.create_foreign_key(
        'fk_documents_workspace',
        'documents',
        'workspaces',
        ['workspace_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    op.drop_constraint('fk_documents_workspace', 'documents', type_='foreignkey')
    op.drop_index('ix_documents_workspace_id')
    op.drop_column('documents', 'workspace_id')
