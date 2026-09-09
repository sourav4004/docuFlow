"""Add processing_jobs table for async document processing.

Revision ID: 011_add_processing_jobs
Revises: 010_add_collections
Create Date: 2026-09-01
"""

from alembic import op
import sqlalchemy as sa


revision = "011_add_processing_jobs"
down_revision = "010_add_collections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create processing_jobs table
    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("job_type", sa.String(50), nullable=False, server_default="document_processing"),
        sa.Column("status", sa.String(50), nullable=False, server_default="QUEUED"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    # Create indexes for common queries
    op.create_index("ix_processing_jobs_status_created", "processing_jobs", ["status", "created_at"])
    op.create_index("ix_processing_jobs_user_status", "processing_jobs", ["user_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_user_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_status_created", table_name="processing_jobs")
    op.drop_table("processing_jobs")
