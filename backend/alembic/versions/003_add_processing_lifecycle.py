"""Add processing lifecycle and document_content table

Revision ID: 003
Revises: 002
Create Date: 2026-08-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Migrate existing ACTIVE documents to READY status
    op.execute("UPDATE documents SET status = 'READY' WHERE status = 'ACTIVE'")

    # 2. Create document_content table
    op.create_table(
        'document_content',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('extracted_text', sa.Text(), nullable=False, server_default=''),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('char_count', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=func.now(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id'),
    )
    op.create_index(op.f('ix_document_content_id'), 'document_content', ['id'], unique=False)
    op.create_index(op.f('ix_document_content_document_id'), 'document_content', ['document_id'], unique=False)


def downgrade() -> None:
    # Reverse document_content table
    op.drop_index(op.f('ix_document_content_document_id'), table_name='document_content')
    op.drop_index(op.f('ix_document_content_id'), table_name='document_content')
    op.drop_table('document_content')

    # Restore original status values
    op.execute("UPDATE documents SET status = 'ACTIVE' WHERE status IN ('UPLOADED', 'QUEUED', 'PROCESSING', 'FAILED')")
    op.execute("UPDATE documents SET status = 'READY' WHERE status = 'READY'")
