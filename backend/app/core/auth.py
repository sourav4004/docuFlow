"""Authentication utilities for session management."""

from datetime import datetime, timedelta
from typing import Optional
import secrets
import json

from fastapi import Request, Response, HTTPException, status, Depends
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from ..models.user import User


# In-memory session store (for simplicity in Phase 1)
# In production, use Redis or a database-backed session store
_sessions: dict[str, dict] = {}


def create_session(user_id: int, response: Response) -> str:
    """
    Create a new session for the user.

    Args:
        user_id: The ID of the authenticated user
        response: FastAPI response to set cookie

    Returns:
        Session ID
    """
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(seconds=settings.session_max_age)

    _sessions[session_id] = {
        "user_id": user_id,
        "expires_at": expires_at,
        "created_at": datetime.utcnow()
    }

    # Set HTTP-only cookie
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_id,
        max_age=settings.session_max_age,
        httponly=True,  # Prevents JavaScript access
        secure=settings.cookie_secure,  # HTTPS only in production
        samesite=settings.cookie_samesite
    )

    return session_id


def get_session(request: Request) -> Optional[dict]:
    """
    Get session data from request cookie.

    Args:
        request: FastAPI request with cookies

    Returns:
        Session data or None if invalid/expired
    """
    session_id = request.cookies.get(settings.session_cookie_name)

    if not session_id:
        return None

    session_data = _sessions.get(session_id)

    if not session_data:
        return None

    # Check expiration
    if datetime.utcnow() > session_data["expires_at"]:
        # Session expired, remove it
        _sessions.pop(session_id, None)
        return None

    return session_data


def invalidate_session(request: Request, response: Response) -> None:
    """
    Invalidate the current session.

    Args:
        request: FastAPI request with cookies
        response: FastAPI response to clear cookie
    """
    session_id = request.cookies.get(settings.session_cookie_name)

    if session_id:
        _sessions.pop(session_id, None)

    # Clear cookie
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.

    Args:
        request: FastAPI request with cookies
        db: Database session

    Returns:
        Current authenticated user

    Raises:
        HTTPException: If not authenticated (401)
    """
    session_data = get_session(request)

    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    user_id = session_data.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional dependency to get current user (doesn't raise exception).

    Args:
        request: FastAPI request
        db: Database session

    Returns:
        Current user or None
    """
    try:
        return get_current_user(request, db)
    except HTTPException:
        return None
