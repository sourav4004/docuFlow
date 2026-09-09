"""Webhook management API — endpoints, deliveries, retry."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.webhook import WebhookEndpoint, WebhookDelivery
from ..models.workspace import Workspace
from ..services.webhook_service import (
    WEBHOOK_EVENT_TYPES,
    create_endpoint,
    rotate_endpoint_secret,
    validate_event_types,
    attempt_delivery,
)
from ..services.permission_service import require_permission

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookCreate(BaseModel):
    workspace_id: int
    url: str
    events: list[str] = ["*"]
    description: Optional[str] = None


class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    events: Optional[list[str]] = None
    status: Optional[str] = None
    description: Optional[str] = None


def _endpoint_dict(endpoint: WebhookEndpoint) -> dict:
    return {
        "id": endpoint.id,
        "url": endpoint.url,
        "events": endpoint.events,
        "status": endpoint.status,
        "description": endpoint.description,
        "workspace_id": endpoint.workspace_id,
        "created_at": endpoint.created_at,
    }


@router.get("/events")
def list_event_types(principal: AuthPrincipal = Depends(get_current_principal)):
    """List subscribable webhook event types."""
    return {"events": WEBHOOK_EVENT_TYPES}


@router.post("", status_code=201)
def create_webhook(
    data: WebhookCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Create a webhook endpoint. The signing secret is returned once."""
    workspace = db.query(Workspace).filter(Workspace.id == data.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=data.workspace_id)

    # Basic SSRF guard: reject obviously internal endpoints
    url_lower = data.url.lower()
    blocked_hosts = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1", "169.254.169.254", "metadata")
    if any(h in url_lower for h in blocked_hosts):
        raise HTTPException(status_code=400, detail="Webhook URL must be a public endpoint")

    try:
        events = validate_event_types(data.events)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    endpoint, signing_secret = create_endpoint(
        db,
        data.workspace_id,
        principal.user.id,
        data.url,
        events,
        description=data.description,
    )
    db.commit()
    db.refresh(endpoint)
    result = _endpoint_dict(endpoint)
    result["signing_secret"] = signing_secret  # shown exactly once
    return result


@router.get("")
def list_webhooks(
    workspace_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """List webhook endpoints for a workspace."""
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=workspace_id)
    endpoints = (
        db.query(WebhookEndpoint)
        .filter(WebhookEndpoint.workspace_id == workspace_id)
        .order_by(WebhookEndpoint.created_at.desc())
        .all()
    )
    return {"items": [_endpoint_dict(e) for e in endpoints]}


@router.patch("/{endpoint_id}")
def update_webhook(
    endpoint_id: int,
    data: WebhookUpdate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=endpoint.workspace_id)

    if data.url is not None:
        endpoint.url = data.url
    if data.events is not None:
        try:
            endpoint.events_json = __import__("json").dumps(validate_event_types(data.events))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if data.status is not None:
        if data.status not in ("ACTIVE", "INACTIVE"):
            raise HTTPException(status_code=400, detail="Invalid status")
        endpoint.status = data.status
    if data.description is not None:
        endpoint.description = data.description
    db.commit()
    db.refresh(endpoint)
    return _endpoint_dict(endpoint)


@router.delete("/{endpoint_id}")
def delete_webhook(
    endpoint_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=endpoint.workspace_id)
    db.delete(endpoint)
    db.commit()
    return {"message": "Webhook endpoint deleted"}


@router.post("/{endpoint_id}/rotate-secret")
def rotate_secret(
    endpoint_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Rotate the signing secret. New secret returned once."""
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=endpoint.workspace_id)
    _, new_secret = rotate_endpoint_secret(db, endpoint_id, principal.user.id)
    db.commit()
    return {"message": "Signing secret rotated", "signing_secret": new_secret}


@router.get("/{endpoint_id}/deliveries")
def list_deliveries(
    endpoint_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """List delivery history for an endpoint (no payloads, no secrets)."""
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=endpoint.workspace_id)

    deliveries = (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.endpoint_id == endpoint_id)
        .order_by(WebhookDelivery.created_at.desc())
        .limit(100)
        .all()
    )
    return {
        "items": [
            {
                "id": d.id,
                "event_type": d.event.event_type if d.event else None,
                "status": d.status,
                "attempt_count": d.attempt_count,
                "max_attempts": d.max_attempts,
                "response_status": d.response_status,
                "latency_ms": d.latency_ms,
                "last_error": d.last_error,
                "next_retry_at": d.next_retry_at,
                "created_at": d.created_at,
            }
            for d in deliveries
        ]
    }


@router.post("/{endpoint_id}/deliveries/{delivery_id}/retry")
def retry_delivery(
    endpoint_id: int,
    delivery_id: int,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Manually retry a failed delivery."""
    endpoint = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == endpoint_id).first()
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    require_permission(db, principal.user.id, "webhook:manage", workspace_id=endpoint.workspace_id)

    delivery = (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.id == delivery_id, WebhookDelivery.endpoint_id == endpoint_id)
        .first()
    )
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    if delivery.status == "DELIVERED":
        raise HTTPException(status_code=400, detail="Delivery already succeeded")

    delivery.status = "PENDING"
    delivery.next_retry_at = None
    attempt_delivery(db, delivery)
    db.commit()
    return {"message": "Retry attempted", "status": delivery.status}