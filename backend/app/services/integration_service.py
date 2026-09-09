"""Generic integration provider abstraction and internal event bus.

IntegrationProvider defines a narrow, validated interface for future
third-party connectors (Slack, Google Drive, etc.). Only deterministic
local connectors are implemented; everything else is registered as
COMING SOON and never pretends to work.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.integration import IntegrationConnection
from ..services.audit_service import log_audit_event


@dataclass
class IntegrationDescriptor:
    """Metadata describing an available integration provider."""
    provider: str
    name: str
    status: str  # AVAILABLE, COMING_SOON, CONFIGURATION_REQUIRED
    description: str
    capabilities: list[str] = field(default_factory=list)


class IntegrationProvider(ABC):
    """Abstract base for integration providers."""

    provider_key: str = "generic"

    @abstractmethod
    def test_connection(self, config: dict) -> tuple[bool, Optional[str]]:
        """Test connectivity. Returns (ok, error)."""

    @abstractmethod
    def discover(self, config: dict, limit: int = 100) -> list[dict]:
        """Discover importable resources. Must be bounded and validated."""

    @abstractmethod
    def download(self, config: dict, resource_id: str) -> Optional[bytes]:
        """Download a resource's content. Returns None if not found."""


class LocalFileConnector(IntegrationProvider):
    """Deterministic local connector for tests and development."""

    provider_key = "local_folder"

    def test_connection(self, config: dict) -> tuple[bool, Optional[str]]:
        import os
        path = config.get("path", "")
        if path and os.path.isdir(path):
            return True, None
        return False, "Configured path is not a directory"

    def discover(self, config: dict, limit: int = 100) -> list[dict]:
        import os
        path = config.get("path", "")
        results = []
        if not path or not os.path.isdir(path):
            return results
        for entry in sorted(os.listdir(path))[:limit]:
            full = os.path.join(path, entry)
            if os.path.isfile(full):
                results.append({"id": entry, "name": entry, "size": os.path.getsize(full)})
        return results

    def download(self, config: dict, resource_id: str) -> Optional[bytes]:
        import os
        path = config.get("path", "")
        if not path:
            return None
        # Path traversal protection: resource must be a direct child
        full = os.path.join(path, resource_id)
        if os.path.dirname(full) != os.path.abspath(path):
            return None
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as f:
            return f.read(10 * 1024 * 1024)  # bounded: 10 MB


# Registry of providers — only implemented ones are AVAILABLE
PROVIDER_REGISTRY: dict[str, IntegrationDescriptor] = {
    "local_folder": IntegrationDescriptor(
        provider="local_folder",
        name="Local Folder",
        status="AVAILABLE",
        description="Import documents from a local directory (development/test connector)",
        capabilities=["discover", "download"],
    ),
    "s3": IntegrationDescriptor(
        provider="s3",
        name="S3-compatible Storage",
        status="CONFIGURATION_REQUIRED",
        description="Import from S3-compatible object storage",
        capabilities=["discover", "download"],
    ),
    "slack": IntegrationDescriptor(
        provider="slack",
        name="Slack",
        status="COMING_SOON",
        description="Slack integration",
        capabilities=[],
    ),
    "google_drive": IntegrationDescriptor(
        provider="google_drive",
        name="Google Drive",
        status="COMING_SOON",
        description="Google Drive integration",
        capabilities=[],
    ),
    "dropbox": IntegrationDescriptor(
        provider="dropbox",
        name="Dropbox",
        status="COMING_SOON",
        description="Dropbox integration",
        capabilities=[],
    ),
    "notion": IntegrationDescriptor(
        provider="notion",
        name="Notion",
        status="COMING_SOON",
        description="Notion integration",
        capabilities=[],
    ),
}

_PROVIDER_INSTANCES: dict[str, IntegrationProvider] = {
    "local_folder": LocalFileConnector(),
}


def list_integrations() -> list[dict]:
    """List available integration providers with honest status."""
    return [
        {
            "provider": d.provider,
            "name": d.name,
            "status": d.status,
            "description": d.description,
            "capabilities": d.capabilities,
        }
        for d in PROVIDER_REGISTRY.values()
    ]


def get_provider(provider_key: str) -> Optional[IntegrationProvider]:
    return _PROVIDER_INSTANCES.get(provider_key)


def create_connection(
    db: Session,
    workspace_id: int,
    user_id: int,
    provider: str,
    name: str,
    config: dict,
) -> IntegrationConnection:
    """Create an integration connection (never stores plaintext credentials)."""
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown integration provider: {provider}")
    descriptor = PROVIDER_REGISTRY[provider]
    if descriptor.status == "COMING_SOON":
        raise ValueError(f"Integration '{provider}' is not yet available")

    connection = IntegrationConnection(
        workspace_id=workspace_id,
        user_id=user_id,
        provider=provider,
        name=name,
        config_json=__import__("json").dumps(config, default=str),
        status="DISCONNECTED",
    )
    db.add(connection)
    db.flush()

    provider_impl = get_provider(provider)
    if provider_impl:
        ok, error = provider_impl.test_connection(config)
        connection.status = "CONNECTED" if ok else "ERROR"
        connection.last_error = error
        if ok:
            connection.last_connected_at = datetime.now(timezone.utc)

    log_audit_event(
        db,
        event_type="integration",
        event_action="connect",
        user_id=user_id,
        resource_type="integration_connection",
        resource_id=connection.id,
        details=f"Integration '{provider}' connected",
    )
    return connection


def disconnect_connection(db: Session, connection_id: int, user_id: int) -> None:
    """Disconnect an integration connection."""
    connection = db.query(IntegrationConnection).filter(IntegrationConnection.id == connection_id).first()
    if connection:
        connection.status = "DISCONNECTED"
        connection.last_error = None
        log_audit_event(
            db,
            event_type="integration",
            event_action="disconnect",
            user_id=user_id,
            resource_type="integration_connection",
            resource_id=connection_id,
            details="Integration disconnected",
        )


# ---------------------------------------------------------------------------
# Internal typed event bus
# ---------------------------------------------------------------------------

@dataclass
class DomainEvent:
    """Typed internal domain event consumed by notifications/webhooks/audit."""
    event_type: str
    workspace_id: int
    organization_id: Optional[int] = None
    payload: dict = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Lightweight in-process event bus.

    Events are also persisted to the webhook outbox so delivery survives
    process restarts. Subscribers run after the caller's transaction flushes.
    """

    def __init__(self, db: Session):
        self.db = db
        self._subscribers: list = []

    def subscribe(self, handler) -> None:
        self._subscribers.append(handler)

    def publish(self, event: DomainEvent, idempotency_key: Optional[str] = None) -> None:
        """Persist the event to the webhook outbox and notify subscribers."""
        from ..services.webhook_service import emit_event, enqueue_deliveries
        record = emit_event(
            self.db,
            event.event_type,
            event.workspace_id,
            event.organization_id,
            event.payload,
            idempotency_key=idempotency_key,
        )
        # Create delivery records (actual HTTP delivery is a background concern)
        enqueue_deliveries(self.db, record)
        for handler in self._subscribers:
            handler(event)


def publish_domain_event(
    db: Session,
    event_type: str,
    workspace_id: int,
    organization_id: Optional[int] = None,
    payload: Optional[dict] = None,
    idempotency_key: Optional[str] = None,
) -> None:
    """Convenience function: publish a domain event to the bus + outbox."""
    EventBus(db).publish(
        DomainEvent(
            event_type=event_type,
            workspace_id=workspace_id,
            organization_id=organization_id,
            payload=payload or {},
        ),
        idempotency_key=idempotency_key,
    )