"""Add collections and collection_documents tables.

Revision ID: 010_add_collections
Revises: 009_add_hybrid_search_tsvector
Create Date: 2026-08-30
"""

from alembic import op
import sqlalchemy as sa


revision = "010_add_collections"
down_revision = "009_add_hybrid_search_tsvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create collections table
    op.create_table(
        "collections",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_collections_user_name", "collections", ["user_id", "name"])

    # Create collection_documents association table
    op.create_table(
        "collection_documents",
        sa.Column("collection_id", sa.Integer(), sa.ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade() -> None:
    op.drop_table("collection_documents")
    op.drop_index("ix_collections_user_name", table_name="collections")
    op.drop_table("collections")
