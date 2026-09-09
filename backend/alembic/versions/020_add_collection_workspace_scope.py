"""Add workspace scope to collections (Phase 14 enterprise features)

Revision ID: 020_collection_workspace_scope
Revises: 019_add_phase14_ai_product
Create Date: 2026-09-03 14:00:00.000000

Collections remain fully backward compatible: workspace_id is nullable and
legacy collections (NULL) keep their user-scoped behavior.
"""

from alembic import op
import sqlalchemy as sa


revision = '020_collection_workspace_scope'
down_revision = '019_add_phase14_ai_product'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'collections',
        sa.Column(
            'workspace_id',
            sa.Integer(),
            sa.ForeignKey('workspaces.id', ondelete='CASCADE'),
            nullable=True,
        ),
    )
    op.create_index('ix_collections_workspace_id', 'collections', ['workspace_id'])


def downgrade() -> None:
    op.drop_index('ix_collections_workspace_id', table_name='collections')
    op.drop_column('collections', 'workspace_id')
