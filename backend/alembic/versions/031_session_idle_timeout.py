"""Phase 24 migration — session idle-timeout support.

Adds ``user_sessions.last_used_at`` (server-default now(), NOT NULL via
batch-safe backfill) so sessions can be expired after inactivity in addition
to their absolute expiry.

Forward compatible and non-destructive:
- additive column with a server default, so old application code keeps
  inserting sessions without naming it;
- backfill sets existing rows to their ``created_at`` value;
- no data is removed; downgrade drops only the added column.

Revision ID: 031_session_idle_timeout
Revises: 030_phase23_global_cloud
"""

from alembic import op
import sqlalchemy as sa

revision = "031_session_idle_timeout"
down_revision = "030_phase23_global_cloud"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_sessions",
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )
    # Backfill: existing sessions were "last used" at creation time.
    op.execute(
        "UPDATE user_sessions SET last_used_at = created_at WHERE last_used_at IS NULL"
    )
    # Tighten to NOT NULL after the backfill (staged, zero-downtime safe).
    op.alter_column(
        "user_sessions",
        "last_used_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.create_index(
        "ix_user_sessions_last_used_at", "user_sessions", ["last_used_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_user_sessions_last_used_at", table_name="user_sessions")
    op.drop_column("user_sessions", "last_used_at")
