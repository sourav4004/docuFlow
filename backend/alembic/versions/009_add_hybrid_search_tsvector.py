"""Add tsvector column, GIN index, and trigger for hybrid search.

Revision ID: 009_add_hybrid_search_tsvector
Revises: c155adc9351b
Create Date: 2026-08-30
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "009_add_hybrid_search_tsvector"
down_revision = "c155adc9351b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add tsvector column for full-text search
    op.add_column(
        "document_chunks",
        sa.Column("search_vector", sa.dialects.postgresql.TSVECTOR()),
    )

    # Create GIN index for fast full-text queries
    op.execute(
        """
        CREATE INDEX ix_document_chunks_search_vector
        ON document_chunks
        USING gin (search_vector)
        """
    )

    # Populate existing rows' search_vector from text column
    op.execute(
        """
        UPDATE document_chunks
        SET search_vector = to_tsvector('english', coalesce(text, ''))
        WHERE search_vector IS NULL
        """
    )

    # Create trigger to auto-update search_vector on INSERT/UPDATE
    op.execute(
        """
        CREATE OR REPLACE FUNCTION document_chunks_search_vector_update()
        RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', coalesce(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )

    op.execute(
        """
        CREATE TRIGGER tsvector_update
        BEFORE INSERT OR UPDATE OF text
        ON document_chunks
        FOR EACH ROW
        EXECUTE FUNCTION document_chunks_search_vector_update()
        """
    )


def downgrade() -> None:
    # Remove trigger and function
    op.execute("DROP TRIGGER IF EXISTS tsvector_update ON document_chunks")
    op.execute(
        "DROP FUNCTION IF EXISTS document_chunks_search_vector_update()"
    )

    # Remove index and column
    op.drop_index(
        "ix_document_chunks_search_vector",
        table_name="document_chunks",
    )
    op.drop_column("document_chunks", "search_vector")
