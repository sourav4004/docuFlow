"""Enable pgvector and convert embedding column to vector type

Revision ID: 006
Revises: 005
Create Date: 2026-08-28 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None

# Dimension must match EMBEDDING_DIMENSION in config (default 384)
VECTOR_DIMENSION = 384


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Drop the old JSON embedding column
    op.drop_column('document_chunks', 'embedding')

    # 3. Add new vector column with correct dimension
    op.execute(
        f"ALTER TABLE document_chunks "
        f"ADD COLUMN embedding vector({VECTOR_DIMENSION})"
    )

    # 4. Add HNSW index for cosine similarity search
    #    HNSW is chosen for good recall/speed tradeoff at small-medium scale.
    #    Lists are not needed (unlike IVFFlat).
    op.execute(
        f"CREATE INDEX ix_document_chunks_embedding "
        f"ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # 1. Drop vector index
    op.drop_index('ix_document_chunks_embedding', table_name='document_chunks')

    # 2. Drop vector column
    op.drop_column('document_chunks', 'embedding')

    # 3. Restore JSON column
    op.add_column(
        'document_chunks',
        sa.Column('embedding', sa.JSON(), nullable=True),
    )

    # Note: we do NOT drop the pgvector extension in downgrade
    # because other tables or extensions might depend on it.
