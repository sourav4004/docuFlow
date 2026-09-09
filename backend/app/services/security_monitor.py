"""Rule-based security event detection.

Lightweight, non-invasive rules that produce useful security events for
the security center. No surveillance — only aggregated signals like
repeated failures or unusual bursts.
"""

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.security_event import SecurityEvent


class SecurityMonitor:
    """Detects rule-based security events from observable signals."""

    def __init__(self, db: Session):
        self.db = db

    def _emit(
        self,
        event_type: str,
        severity: str,
        description: str,
        organization_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None,
        source_ip: Optional[str] = None,
    ) -> SecurityEvent:
        import json
        event = SecurityEvent(
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            description=description,
            metadata_json=json.dumps(metadata) if metadata else None,
            source_ip=source_ip,
        )
        self.db.add(event)
        self.db.flush()
        return event

    def _count_recent(self, event_type: str, window_minutes: int, key_field: str, key_value) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        from sqlalchemy import func
        query = self.db.query(func.count(SecurityEvent.id)).filter(
            SecurityEvent.event_type == event_type,
            SecurityEvent.created_at >= cutoff,
        )
        if key_field == "user_id":
            query = query.filter(SecurityEvent.user_id == key_value)
        elif key_field == "source_ip":
            query = query.filter(SecurityEvent.source_ip == key_value)
        return int(query.scalar() or 0)

    def record_failed_login(
        self,
        email: str,
        source_ip: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[SecurityEvent]:
        """Record a failed login; emit a HIGH event after repeated failures."""
        event = self._emit(
            "failed_login",
            "LOW",
            "Failed login attempt",
            user_id=user_id,
            source_ip=source_ip,
            metadata={"email": email},
        )
        # Rule: 5+ failed logins for the same user/email in 15 minutes
        recent = self._count_recent("failed_login", 15, "user_id", user_id) if user_id else 0
        recent_ip = self._count_recent("failed_login", 15, "source_ip", source_ip) if source_ip else 0
        if recent >= 5 or recent_ip >= 10:
            return self._emit(
                "repeated_failed_login",
                "HIGH",
                "Repeated failed login attempts detected",
                user_id=user_id,
                source_ip=source_ip,
            )
        return event

    def record_invalid_api_key(self, source_ip: Optional[str] = None) -> Optional[SecurityEvent]:
        """Record an invalid API key attempt; emit HIGH after repeated failures."""
        event = self._emit(
            "invalid_api_key",
            "LOW",
            "Invalid API key attempt",
            source_ip=source_ip,
        )
        recent = self._count_recent("invalid_api_key", 15, "source_ip", source_ip) if source_ip else 0
        if recent >= 5:
            return self._emit(
                "repeated_invalid_api_key",
                "HIGH",
                "Repeated invalid API key attempts detected",
                source_ip=source_ip,
            )
        return event

    def record_authorization_failure(
        self,
        user_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        source_ip: Optional[str] = None,
    ) -> Optional[SecurityEvent]:
        event = self._emit(
            "authorization_failure",
            "LOW",
            "Authorization failure",
            user_id=user_id,
            workspace_id=workspace_id,
            source_ip=source_ip,
        )
        recent = self._count_recent("authorization_failure", 15, "user_id", user_id) if user_id else 0
        if recent >= 10:
            return self._emit(
                "repeated_authorization_failures",
                "MEDIUM",
                "Repeated authorization failures detected",
                user_id=user_id,
                workspace_id=workspace_id,
                source_ip=source_ip,
            )
        return event

    def record_suspicious_export(self, user_id: int, workspace_id: int) -> SecurityEvent:
        return self._emit(
            "suspicious_export_activity",
            "MEDIUM",
            "Multiple or large export operations recorded",
            workspace_id=workspace_id,
            user_id=user_id,
        )

    def record_webhook_failure(self, endpoint_id: int, workspace_id: int) -> SecurityEvent:
        return self._emit(
            "webhook_delivery_failure",
            "LOW",
            "Webhook delivery failed",
            workspace_id=workspace_id,
            metadata={"endpoint_id": endpoint_id},
        )

    def list_recent(self, limit: int = 100, workspace_id: Optional[int] = None) -> list[SecurityEvent]:
        """List recent security events (workspace-scoped when provided)."""
        query = self.db.query(SecurityEvent).order_by(SecurityEvent.created_at.desc())
        if workspace_id is not None:
            query = query.filter(SecurityEvent.workspace_id == workspace_id)
        return query.limit(limit).all()