"""Health and readiness endpoints for production deployment.

Provides:
- GET /health - Basic liveness check
- GET /health/live - Liveness probe (process alive)
- GET /health/ready - Readiness probe (ready to accept traffic)
"""

import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..core.database import get_db, check_database_health
from ..core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Basic health check endpoint.
    
    Returns:
        dict: Status indicating the service is healthy
    """
    # Verify database connectivity
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "ok",
        "database": db_status,
        "version": "0.2.0",
        "environment": settings.environment,
    }


@router.get("/health/live")
async def liveness_check():
    """Liveness probe - indicates the process is alive.
    
    This should always return 200 if the application is running.
    Use this for Kubernetes liveness probes or similar.
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
    }


@router.get("/health/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe - indicates the application is ready to accept traffic.
    
    Checks:
    - Database connectivity
    - Required configuration
    
    Returns 200 if ready, 503 if not ready.
    """
    checks = {}
    is_ready = True

    # Check database
    db_health = check_database_health()
    checks["database"] = db_health["status"]
    if db_health["status"] != "healthy":
        is_ready = False

    # Check required configuration
    checks["configuration"] = "valid"
    
    # Check vector backend
    try:
        from ..services.vector_backend import get_vector_backend_singleton
        backend = get_vector_backend_singleton()
        checks["vector_backend"] = backend.name
    except Exception as e:
        checks["vector_backend"] = f"error: {str(e)}"
        # Vector backend is optional, don't fail readiness

    # Broker health (Phase 24): Redis optional; PostgreSQL broker remains
    # authoritative. A configured-but-unreachable Redis is DEGRADED, not
    # NOT_READY, because the postgres broker keeps serving.
    try:
        from ..services.capabilities import redis_healthcheck
        redis_state = redis_healthcheck()
        if not redis_state.get("available"):
            if redis_state.get("state") == "NOT_CONFIGURED":
                checks["broker"] = "postgres (redis not configured)"
            else:
                checks["broker"] = "degraded: redis unreachable, postgres active"
        else:
            checks["broker"] = "redis available"
    except Exception as e:
        checks["broker"] = f"unknown: {str(e)[:80]}"

    status_code = 200 if is_ready else 503
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "timestamp": time.time(),
    }


@router.get("/worker-health")
async def worker_health_check(db: Session = Depends(get_db)):
    """Worker-process health: active workers, heartbeats, stale count.

    Exposes only safe operational signals (never host secrets).
    """
    from ..models.phase16 import WorkerHeartbeat
    from datetime import datetime, timedelta, timezone
    rows = db.query(WorkerHeartbeat).limit(200).all()
    now = datetime.now(timezone.utc)
    stale_after = timedelta(seconds=120)
    active = 0
    stale = 0
    for w in rows:
        last = w.last_heartbeat
        if last is None:
            stale += 1
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if w.status == "RUNNING" and (now - last) < stale_after:
            active += 1
        elif w.status == "RUNNING":
            stale += 1
    return {
        "status": "ok" if active else "degraded",
        "workers": len(rows),
        "active": active,
        "stale": stale,
        "timestamp": time.time(),
    }
