"""Protected endpoints for testing authentication."""

from fastapi import APIRouter, Depends

from ..core.auth import get_current_user
from ..models.user import User

router = APIRouter(tags=["protected"])


@router.get("/protected")
def protected_endpoint(current_user: User = Depends(get_current_user)):
    """
    Protected endpoint for testing authentication.

    Requires valid authentication session.
    """
    return {
        "authenticated": True,
        "user_id": current_user.id,
        "user_email": current_user.email
    }
