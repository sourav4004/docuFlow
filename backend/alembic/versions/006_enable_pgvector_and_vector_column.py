"""Enable pgvector and convert embedding column to vector type

Revision ID: 006
Revises: 005
Create Date: 2026-08-28 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None

# Dimension must match EMBEDDING_DIMENSION in config (default 384)
VECTOR_DIMENSION = 384


def _pgvector_available(bind) -> bool:
    """Check if pgvector extension is available."""
    try:
        result = bind.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"))
        return result.fetchone() is not None
    except Exception:
        return False


def upgrade() -> None:
    bind = op.get_bind()
    pgvector_available = _pgvector_available(bind)

    if pgvector_available:
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
        op.execute(
            f"CREATE INDEX ix_document_chunks_embedding "
            f"ON document_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    else:
        # pgvector not available - keep embedding as JSON
        # No GIN index on raw JSON (needs jsonb); just keep JSON column
        print("WARNING: pgvector extension not available. Using JSON embedding column (no vector index).")


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
