"""Audit logging service for security and operational events.

Records important events without exposing sensitive data.
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """Service for recording audit events."""

    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        event_type: str,
        event_action: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[int] = None,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditLog:
        """Record an audit event.
        
        Args:
            event_type: Category of event (auth, document, collection, etc.)
            event_action: Action performed (create, delete, login, etc.)
            user_id: User who performed the action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details (no sensitive data)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            AuditLog: The created audit log entry
        """
        # Sanitize details - remove sensitive data
        sanitized_details = self._sanitize_details(details)

        audit_entry = AuditLog(
            user_id=user_id,
            event_type=event_type,
            event_action=event_action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=sanitized_details,
            ip_address=ip_address,
            user_agent=user_agent[:500] if user_agent else None,
        )
        
        self.db.add(audit_entry)
        self.db.flush()
        
        # Log for observability (without sensitive data)
        logger.info(
            "Audit: %s:%s user=%s resource=%s/%s",
            event_type,
            event_action,
            user_id,
            resource_type,
            resource_id,
        )
        
        return audit_entry

    def _sanitize_details(self, details: Optional[str]) -> Optional[str]:
        """Remove sensitive data from details string."""
        if not details:
            return details
        
        # Remove common sensitive patterns
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "api_key",
            "session",
        ]
        
        sanitized = details
        for pattern in sensitive_patterns:
            # Case-insensitive replacement
            import re
            sanitized = re.sub(
                rf'(?i){pattern}\s*[:=]\s*\S+',
                f'{pattern}: [REDACTED]',
                sanitized,
            )
        
        return sanitized


# Convenience function
def log_audit_event(
    db: Session,
    event_type: str,
    event_action: str,
    user_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AuditLog:
    """Log an audit event."""
    service = AuditService(db)
    return service.log_event(
        event_type=event_type,
        event_action=event_action,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
