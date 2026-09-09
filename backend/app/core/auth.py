"""Authentication utilities for session management."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import secrets
import logging

from fastapi import Request, Response, HTTPException, status, Depends
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from ..models.user import User
from ..models.session import UserSession

logger = logging.getLogger(__name__)

# Backward-compatible in-memory dict — tests call _sessions.clear() to reset state.
# Actual session storage is in the database (UserSession model).
_sessions: dict = {}


def clear_all_sessions(db=None) -> None:
    """Clear all sessions. Used by tests for isolation.

    If a database session is provided, also clears database-backed sessions.
    """
    _sessions.clear()
    if db is not None:
        try:
            db.query(UserSession).delete()
            db.commit()
        except Exception:
            db.rollback()


def create_session(user_id: int, response: Response, db: Session) -> str:
    """
    Create a new session for the user.

    Args:
        user_id: The ID of the authenticated user
        response: FastAPI response to set cookie
        db: Database session for persistence

    Returns:
        Session ID
    """
    session_id = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=settings.session_max_age)

    user_session = UserSession(
        session_id=session_id,
        user_id=user_id,
        expires_at=expires_at,
    )
    db.add(user_session)
    db.commit()

    # Set HTTP-only cookie
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_id,
        max_age=settings.session_max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
    )

    return session_id


def get_session(request: Request, db: Session) -> Optional[UserSession]:
    """
    Get session from request cookie, querying the database.

    Args:
        request: FastAPI request with cookies
        db: Database session

    Returns:
        UserSession or None if invalid/expired
    """
    session_id = request.cookies.get(settings.session_cookie_name)

    if not session_id:
        return None

    user_session = (
        db.query(UserSession)
        .filter(UserSession.session_id == session_id)
        .first()
    )

    if not user_session:
        return None

    # Check expiration
    now = datetime.now(timezone.utc)
    expires_at = user_session.expires_at
    # Make expires_at timezone-aware if it isn't already
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        # Session expired, remove it
        db.delete(user_session)
        db.commit()
        return None

    # Idle timeout (Phase 24): a session unused for longer than
    # session_idle_timeout is expired even if its absolute expiry has not
    # been reached. Zero/negative disables the idle check (default for
    # compatibility with existing deployments).
    idle_timeout = getattr(settings, "session_idle_timeout", 0)
    if idle_timeout and idle_timeout > 0:
        last_used = user_session.last_used_at or user_session.created_at
        if last_used is not None:
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)
            if (now - last_used).total_seconds() > idle_timeout:
                db.delete(user_session)
                db.commit()
                return None
        # Touch activity. Bounded write: at most once per request that
        # authenticates; keeps idle detection accurate.
        user_session.last_used_at = now
        db.commit()

    return user_session


def invalidate_session(request: Request, response: Response, db: Session) -> None:
    """
    Invalidate the current session.

    Args:
        request: FastAPI request with cookies
        response: FastAPI response to clear cookie
        db: Database session
    """
    session_id = request.cookies.get(settings.session_cookie_name)

    if session_id:
        user_session = (
            db.query(UserSession)
            .filter(UserSession.session_id == session_id)
            .first()
        )
        if user_session:
            db.delete(user_session)
            db.commit()

    # Clear cookie
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
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
    user_session = get_session(request, db)

    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    user = db.query(User).filter(User.id == user_session.user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db),
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


class AuthPrincipal:
    """Authenticated principal — either a session user or an API key caller.

    API-key principals carry the resolved ApiKey so endpoints can enforce
    scope restrictions in addition to workspace isolation.
    """

    def __init__(self, user: User, api_key=None):
        self.user = user
        self.api_key = api_key  # Optional[ApiKey]

    @property
    def is_api_key(self) -> bool:
        return self.api_key is not None

    def has_scope(self, scope: str) -> bool:
        """Check API key scope. Session principals are always allowed."""
        if not self.is_api_key:
            return True
        return self.api_key.has_scope(scope)


def _extract_bearer_token(request: Request) -> Optional[str]:
    """Extract a Bearer token from the Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    return token or None


def get_current_principal(
    request: Request,
    db: Session = Depends(get_db),
) -> AuthPrincipal:
    """Authenticate via API key (Bearer) or session cookie, in that order.

    Raises:
        HTTPException: 401 if neither credential is valid.
    """
    # 1) API key authentication
    bearer = _extract_bearer_token(request)
    if bearer is not None:
        from ..models.api_key import ApiKey
        from ..services.api_key_service import authenticate_api_key, touch_api_key
        api_key = authenticate_api_key(db, bearer)
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key",
            )
        user = db.query(User).filter(User.id == api_key.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key owner not found",
            )
        touch_api_key(db, api_key)
        db.commit()
        return AuthPrincipal(user=user, api_key=api_key)

    # 2) Session authentication (backward compatible)
    user = get_current_user(request, db)
    return AuthPrincipal(user=user, api_key=None)


def get_current_principal_optional(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[AuthPrincipal]:
    """Optional principal — returns None instead of raising 401."""
    try:
        return get_current_principal(request, db)
    except HTTPException:
        return None
