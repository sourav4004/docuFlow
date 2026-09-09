"""Add Phase 13 enterprise SaaS tables

Revision ID: 017_add_phase13_enterprise
Revises: 016_add_ai_execution_tables
Create Date: 2026-09-03 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '017_add_phase13_enterprise'
down_revision = '016_add_ai_execution_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Organizations
    op.create_table(
        'organizations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False),
        sa.Column('description', sa.String(1000), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='ACTIVE'),
        sa.Column('settings_json', sa.String(4000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_organizations_owner_id', 'organizations', ['owner_id'])
    op.create_index('ix_organizations_name', 'organizations', ['name'])
    op.create_index('ix_organizations_slug', 'organizations', ['slug'], unique=True)

    # Organization members
    op.create_table(
        'organization_members',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(20), nullable=False, server_default='MEMBER'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'user_id', name='uq_org_user')
    )
    op.create_index('ix_org_members_user_id', 'organization_members', ['user_id'])

    # Add organization_id to workspaces
    op.add_column('workspaces', sa.Column('organization_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_workspaces_organization_id', 'workspaces', 'organizations', ['organization_id'], ['id'], ondelete='SET NULL')

    # Verified domains
    op.create_table(
        'verified_domains',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('verification_token', sa.String(128), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'domain', name='uq_org_domain')
    )
    op.create_index('ix_verified_domains_domain', 'verified_domains', ['domain'])

    # API keys
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('prefix', sa.String(12), nullable=False),
        sa.Column('key_hash', sa.String(128), nullable=False),
        sa.Column('scopes_json', sa.String(2000), nullable=False, server_default='[]'),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['revoked_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_keys_workspace_id', 'api_keys', ['workspace_id'])
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])
    op.create_index('ix_api_keys_prefix', 'api_keys', ['prefix'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)

    # Webhook endpoints
    op.create_table(
        'webhook_endpoints',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(2000), nullable=False),
        sa.Column('secret_hash', sa.String(128), nullable=False),
        sa.Column('events_json', sa.String(4000), nullable=False, server_default='[]'),
        sa.Column('status', sa.String(20), nullable=False, server_default='ACTIVE'),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_webhook_endpoints_workspace_id', 'webhook_endpoints', ['workspace_id'])
    op.create_index('ix_webhook_endpoints_status', 'webhook_endpoints', ['status'])

    # Webhook events (outbox)
    op.create_table(
        'webhook_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.String(64), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('payload_json', sa.Text(), nullable=False),
        sa.Column('idempotency_key', sa.String(128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id', name='uq_webhook_event_id')
    )
    op.create_index('ix_webhook_events_workspace_id', 'webhook_events', ['workspace_id'])
    op.create_index('ix_webhook_events_event_type', 'webhook_events', ['event_type'])
    op.create_index('ix_webhook_events_created_at', 'webhook_events', ['created_at'])

    # Webhook deliveries
    op.create_table(
        'webhook_deliveries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.Integer(), nullable=False),
        sa.Column('endpoint_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('response_status', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.String(2000), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('last_error', sa.String(1000), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['endpoint_id'], ['webhook_endpoints.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['event_id'], ['webhook_events.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id', 'endpoint_id', name='uq_delivery_event_endpoint')
    )
    op.create_index('ix_webhook_deliveries_event_id', 'webhook_deliveries', ['event_id'])
    op.create_index('ix_webhook_deliveries_endpoint_id', 'webhook_deliveries', ['endpoint_id'])
    op.create_index('ix_webhook_deliveries_status', 'webhook_deliveries', ['status'])
    op.create_index('ix_webhook_deliveries_next_retry', 'webhook_deliveries', ['next_retry_at'])

    # Idempotency keys
    op.create_table(
        'idempotency_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('key', sa.String(255), nullable=False),
        sa.Column('request_hash', sa.String(128), nullable=False),
        sa.Column('response_status', sa.Integer(), nullable=True),
        sa.Column('response_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workspace_id', 'key', name='uq_idempotency_workspace_key')
    )
    op.create_index('ix_idempotency_keys_key', 'idempotency_keys', ['key'])
    op.create_index('ix_idempotency_keys_expires_at', 'idempotency_keys', ['expires_at'])

    # Plans
    op.create_table(
        'plans',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('code', sa.String(50), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('limits_json', sa.String(4000), nullable=False, server_default='{}'),
        sa.Column('features_json', sa.String(4000), nullable=False, server_default='{}'),
        sa.Column('is_default', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_plans_code', 'plans', ['code'], unique=True)

    # Subscriptions
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='ACTIVE'),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['plan_id'], ['plans.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_subscriptions_organization_id', 'subscriptions', ['organization_id'])
    op.create_index('ix_subscriptions_status', 'subscriptions', ['status'])

    # Export jobs
    op.create_table(
        'export_jobs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('export_type', sa.String(50), nullable=False),
        sa.Column('format', sa.String(10), nullable=False, server_default='json'),
        sa.Column('status', sa.String(20), nullable=False, server_default='QUEUED'),
        sa.Column('storage_path', sa.String(1000), nullable=True),
        sa.Column('download_token_hash', sa.String(128), nullable=True),
        sa.Column('download_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_export_jobs_workspace_id', 'export_jobs', ['workspace_id'])
    op.create_index('ix_export_jobs_user_id', 'export_jobs', ['user_id'])
    op.create_index('ix_export_jobs_status', 'export_jobs', ['status'])

    # Security events
    op.create_table(
        'security_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('workspace_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, server_default='INFO'),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('source_ip', sa.String(45), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_security_events_workspace_id', 'security_events', ['workspace_id'])
    op.create_index('ix_security_events_user_id', 'security_events', ['user_id'])
    op.create_index('ix_security_events_event_type', 'security_events', ['event_type'])
    op.create_index('ix_security_events_created_at', 'security_events', ['created_at'])

    # Feature flags
    op.create_table(
        'feature_flags',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('scope_type', sa.String(20), nullable=False, server_default='GLOBAL'),
        sa.Column('scope_id', sa.Integer(), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('rollout_percentage', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_feature_flags_name', 'feature_flags', ['name'])
    op.create_index('ix_feature_flags_scope', 'feature_flags', ['scope_type', 'scope_id'])

    # Integration connections
    op.create_table(
        'integration_connections',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='DISCONNECTED'),
        sa.Column('config_json', sa.Text(), nullable=True),
        sa.Column('credentials_ref', sa.Text(), nullable=True),
        sa.Column('last_connected_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error', sa.String(1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_integration_connections_workspace_id', 'integration_connections', ['workspace_id'])
    op.create_index('ix_integration_connections_provider', 'integration_connections', ['provider'])

    # SSO configurations
    op.create_table(
        'sso_configurations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('provider_type', sa.String(20), nullable=False, server_default='OIDC'),
        sa.Column('issuer', sa.String(500), nullable=False),
        sa.Column('client_id', sa.String(500), nullable=False),
        sa.Column('client_secret_ref', sa.Text(), nullable=True),
        sa.Column('authorization_endpoint', sa.String(1000), nullable=True),
        sa.Column('token_endpoint', sa.String(1000), nullable=True),
        sa.Column('jwks_uri', sa.String(1000), nullable=True),
        sa.Column('redirect_uri', sa.String(1000), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sso_configurations_organization_id', 'sso_configurations', ['organization_id'])

    # SSO states
    op.create_table(
        'sso_states',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('state', sa.String(128), nullable=False),
        sa.Column('nonce', sa.String(128), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('redirect_to', sa.String(1000), nullable=True),
        sa.Column('used', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sso_states_state', 'sso_states', ['state'], unique=True)
    op.create_index('ix_sso_states_expires_at', 'sso_states', ['expires_at'])

    # Retention policies
    op.create_table(
        'retention_policies',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('scope_type', sa.String(20), nullable=False, server_default='WORKSPACE'),
        sa.Column('scope_id', sa.Integer(), nullable=True),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('retention_days', sa.Integer(), nullable=False, server_default='30'),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('last_cleanup_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_retention_policies_scope', 'retention_policies', ['scope_type', 'scope_id'])

    # Usage events
    op.create_table(
        'usage_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('workspace_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('metric', sa.String(50), nullable=False),
        sa.Column('quantity', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('period', sa.String(10), nullable=False),
        sa.Column('event_key', sa.String(128), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_usage_events_workspace_id', 'usage_events', ['workspace_id'])
    op.create_index('ix_usage_events_metric', 'usage_events', ['metric'])
    op.create_index('ix_usage_events_period', 'usage_events', ['period'])
    op.create_index('ix_usage_events_event_key', 'usage_events', ['event_key'])


def downgrade() -> None:
    op.drop_table('usage_events')
    op.drop_table('retention_policies')
    op.drop_table('sso_states')
    op.drop_table('sso_configurations')
    op.drop_table('integration_connections')
    op.drop_table('feature_flags')
    op.drop_table('security_events')
    op.drop_table('export_jobs')
    op.drop_table('subscriptions')
    op.drop_table('plans')
    op.drop_table('idempotency_keys')
    op.drop_table('webhook_deliveries')
    op.drop_table('webhook_events')
    op.drop_table('webhook_endpoints')
    op.drop_table('api_keys')
    op.drop_table('verified_domains')
    op.drop_constraint('fk_workspaces_organization_id', 'workspaces', type_='foreignkey')
    op.drop_column('workspaces', 'organization_id')
    op.drop_table('organization_members')
    op.drop_table('organizations')