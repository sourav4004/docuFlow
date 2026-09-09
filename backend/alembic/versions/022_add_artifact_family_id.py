"""add artifact family_id (stable logical artifact identity for versioning)

Revision ID: 022
Revises: 021
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa

revision = "022_add_artifact_family_id"
down_revision = "021_add_phase15_knowledge_os"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("ai_artifacts", sa.Column("family_id", sa.String(length=36), nullable=True))
    op.create_index("ix_ai_artifacts_family_id", "ai_artifacts", ["family_id"])


def downgrade():
    op.drop_index("ix_ai_artifacts_family_id", table_name="ai_artifacts")
    op.drop_column("ai_artifacts", "family_id")
