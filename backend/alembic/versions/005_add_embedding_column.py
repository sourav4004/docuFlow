"""Add embedding column to document_chunks

Revision ID: 005
Revises: 004
Create Date: 2026-08-28 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add embedding column as JSON (stores list of floats).
    # Nullable because embeddings are generated in a separate step
    # after chunk creation.
    op.add_column(
        'document_chunks',
        sa.Column('embedding', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('document_chunks', 'embedding')
