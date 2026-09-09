"""Authentication API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import hash_password, verify_password, validate_password, validate_email, normalize_email
from ..core.auth import create_session, invalidate_session, get_current_user
from ..models.user import User
from ..schemas.auth import UserRegister, UserLogin, UserResponse, MessageResponse

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserRegister,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Register a new user.

    - Validates input
    - Normalizes email
    - Checks for duplicate email
    - Hashes password
    - Creates user
    - Establishes authenticated session
    """
    # Validate email
    is_valid_email, email_error = validate_email(user_data.email)
    if not is_valid_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=email_error
        )

    # Validate password
    is_valid_password, password_error = validate_password(user_data.password)
    if not is_valid_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=password_error
        )

    # Normalize email
    normalized_email = normalize_email(user_data.email)

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    password_hash = hash_password(user_data.password)

    # Create user
    new_user = User(
        name=user_data.name,
        email=normalized_email,
        password_hash=password_hash
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create session
    create_session(new_user.id, response, db)

    return new_user


@router.post("/login", response_model=UserResponse)
def login(
    credentials: UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Login with email and password.

    - Validates credentials
    - Verifies password
    - Creates authenticated session
    """
    # Normalize email
    normalized_email = normalize_email(credentials.email)

    # Find user
    user = db.query(User).filter(User.email == normalized_email).first()

    # Verify password (use constant-time comparison to avoid timing attacks)
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Create session
    create_session(user.id, response, db)

    return user


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Logout the current user.

    - Invalidates session
    - Clears authentication cookie
    """
    invalidate_session(request, response, db)

    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.

    Requires authentication.
    """
    return current_user
