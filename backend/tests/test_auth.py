"""Authentication tests."""

from fastapi.testclient import TestClient

from app.main import app
from app.core.database import Base, get_db
from app.models.user import User
from app.models.session import UserSession  # noqa: F401 — ensure table is created
from app.core import auth
from tests.shared_db import engine, TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def setup_function():
    """Clear sessions before each test."""
    auth._sessions.clear()


def test_register_success():
    """Test successful user registration."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "password": "password123"
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "password_hash" not in data
    assert "password" not in data

    # Check cookie is set
    assert "docuflow_session" in response.cookies


def test_register_duplicate_email():
    """Test registration with duplicate email fails."""
    # First registration
    client.post(
        "/auth/register",
        json={
            "name": "First User",
            "email": "duplicate@example.com",
            "password": "password123"
        }
    )

    # Duplicate registration
    response = client.post(
        "/auth/register",
        json={
            "name": "Second User",
            "email": "duplicate@example.com",
            "password": "password456"
        }
    )

    assert response.status_code == 422
    assert "already registered" in response.json()["detail"].lower()


def test_register_invalid_password():
    """Test registration with invalid password fails."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Test User",
            "email": "test2@example.com",
            "password": "short"  # Too short
        }
    )

    # Pydantic returns 422 for validation errors
    assert response.status_code == 422


def test_password_hash_stored():
    """Test that password is hashed, not stored as plaintext."""
    email = "hashtest@example.com"
    password = "mypassword123"

    client.post(
        "/auth/register",
        json={
            "name": "Hash Test",
            "email": email,
            "password": password
        }
    )

    # Check database
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == email).first()

    assert user is not None
    assert user.password_hash != password
    assert user.password_hash.startswith("$argon2")

    db.close()


def test_login_success():
    """Test successful login."""
    email = "login@example.com"
    password = "password123"

    # Register user
    client.post(
        "/auth/register",
        json={
            "name": "Login User",
            "email": email,
            "password": password
        }
    )

    # Login
    response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["email"] == email
    assert "password" not in data
    assert "password_hash" not in data

    # Check cookie is set
    assert "docuflow_session" in response.cookies


def test_login_incorrect_password():
    """Test login with incorrect password fails."""
    email = "wrongpass@example.com"

    # Register user
    client.post(
        "/auth/register",
        json={
            "name": "Wrong Pass User",
            "email": email,
            "password": "correctpassword123"
        }
    )

    # Login with wrong password
    response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": "wrongpassword"
        }
    )

    assert response.status_code == 401


def test_login_nonexistent_user():
    """Test login with non-existent email fails."""
    response = client.post(
        "/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "password123"
        }
    )

    assert response.status_code == 401


def test_get_current_user_authenticated():
    """Test /auth/me with authenticated user."""
    email = "authme@example.com"

    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Auth Me User",
            "email": email,
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Get current user
    response = client.get("/auth/me", cookies=cookies)

    assert response.status_code == 200
    data = response.json()

    assert data["email"] == email
    assert "password_hash" not in data


def test_get_current_user_unauthenticated():
    """Test /auth/me without authentication returns 401."""
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_protected_endpoint_authenticated():
    """Test protected endpoint with authentication."""
    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Protected User",
            "email": "protected@example.com",
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Access protected endpoint
    response = client.get("/protected", cookies=cookies)

    assert response.status_code == 200
    data = response.json()

    assert data["authenticated"] is True
    assert "user_id" in data


def test_protected_endpoint_unauthenticated():
    """Test protected endpoint without authentication returns 401."""
    response = client.get("/protected")

    assert response.status_code == 401


def test_logout():
    """Test logout invalidates session."""
    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Logout User",
            "email": "logout@example.com",
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Verify authenticated
    response = client.get("/auth/me", cookies=cookies)
    assert response.status_code == 200

    # Logout
    logout_response = client.post("/auth/logout", cookies=cookies)
    assert logout_response.status_code == 200

    # Verify session invalidated - use cookies from logout response
    logout_cookies = logout_response.cookies
    response = client.get("/auth/me", cookies=logout_cookies)
    assert response.status_code == 401

    # Verify protected endpoint also returns 401
    response = client.get("/protected", cookies=logout_cookies)
    assert response.status_code == 401


def test_password_not_logged():
    """Test that passwords are not exposed in responses."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Password Test",
            "email": "passtest@example.com",
            "password": "secretpassword123"
        }
    )

    # Check response doesn't contain password
    response_text = response.text.lower()
    assert "secretpassword123" not in response_text
    assert "password_hash" not in response.json()


def test_email_normalization():
    """Test that email is normalized to lowercase."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Email Test",
            "email": "UPPERCASE@EXAMPLE.COM",
            "password": "password123"
        }
    )

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "uppercase@example.com"
    """Test successful user registration."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "password": "password123"
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "password_hash" not in data
    assert "password" not in data

    # Check cookie is set
    assert "docuflow_session" in response.cookies


def test_register_duplicate_email():
    """Test registration with duplicate email fails."""
    # First registration
    client.post(
        "/auth/register",
        json={
            "name": "First User",
            "email": "duplicate@example.com",
            "password": "password123"
        }
    )

    # Duplicate registration
    response = client.post(
        "/auth/register",
        json={
            "name": "Second User",
            "email": "duplicate@example.com",
            "password": "password456"
        }
    )

    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()


def test_register_invalid_password():
    """Test registration with invalid password fails."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Test User",
            "email": "test2@example.com",
            "password": "short"  # Too short
        }
    )

    assert response.status_code == 422


def test_password_hash_stored():
    """Test that password is hashed, not stored as plaintext."""
    email = "hashtest@example.com"
    password = "mypassword123"

    client.post(
        "/auth/register",
        json={
            "name": "Hash Test",
            "email": email,
            "password": password
        }
    )

    # Check database
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == email).first()

    assert user is not None
    assert user.password_hash != password
    assert user.password_hash.startswith("$argon2")

    db.close()


def test_login_success():
    """Test successful login."""
    email = "login@example.com"
    password = "password123"

    # Register user
    client.post(
        "/auth/register",
        json={
            "name": "Login User",
            "email": email,
            "password": password
        }
    )

    # Login
    response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["email"] == email
    assert "password" not in data
    assert "password_hash" not in data

    # Check cookie is set
    assert "docuflow_session" in response.cookies


def test_login_incorrect_password():
    """Test login with incorrect password fails."""
    email = "wrongpass@example.com"

    # Register user
    client.post(
        "/auth/register",
        json={
            "name": "Wrong Pass User",
            "email": email,
            "password": "correctpassword123"
        }
    )

    # Login with wrong password
    response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": "wrongpassword"
        }
    )

    assert response.status_code == 401


def test_login_nonexistent_user():
    """Test login with non-existent email fails."""
    response = client.post(
        "/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "password123"
        }
    )

    assert response.status_code == 401


def test_get_current_user_authenticated():
    """Test /auth/me with authenticated user."""
    email = "authme@example.com"

    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Auth Me User",
            "email": email,
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Get current user
    response = client.get("/auth/me", cookies=cookies)

    assert response.status_code == 200
    data = response.json()

    assert data["email"] == email
    assert "password_hash" not in data


def test_get_current_user_unauthenticated():
    """Test /auth/me without authentication returns 401."""
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_protected_endpoint_authenticated():
    """Test protected endpoint with authentication."""
    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Protected User",
            "email": "protected@example.com",
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Access protected endpoint
    response = client.get("/protected", cookies=cookies)

    assert response.status_code == 200
    data = response.json()

    assert data["authenticated"] is True
    assert "user_id" in data


def test_protected_endpoint_unauthenticated():
    """Test protected endpoint without authentication returns 401."""
    response = client.get("/protected")

    assert response.status_code == 401


def test_logout():
    """Test logout invalidates session."""
    # Register and get session cookie
    register_response = client.post(
        "/auth/register",
        json={
            "name": "Logout User",
            "email": "logout@example.com",
            "password": "password123"
        }
    )

    cookies = register_response.cookies

    # Verify authenticated
    response = client.get("/auth/me", cookies=cookies)
    assert response.status_code == 200

    # Logout
    logout_response = client.post("/auth/logout", cookies=cookies)
    assert logout_response.status_code == 200

    # Verify session invalidated - use cookies from logout response
    logout_cookies = logout_response.cookies
    response = client.get("/auth/me", cookies=logout_cookies)
    assert response.status_code == 401

    # Verify protected endpoint also returns 401
    response = client.get("/protected", cookies=logout_cookies)
    assert response.status_code == 401


def test_password_not_logged():
    """Test that passwords are not exposed in responses."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Password Test",
            "email": "passtest@example.com",
            "password": "secretpassword123"
        }
    )

    # Check response doesn't contain password
    response_text = response.text.lower()
    assert "secretpassword123" not in response_text
    assert "password_hash" not in response.json()


def test_email_normalization():
    """Test that email is normalized to lowercase."""
    response = client.post(
        "/auth/register",
        json={
            "name": "Email Test",
            "email": "UPPERCASE@EXAMPLE.COM",
            "password": "password123"
        }
    )

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "uppercase@example.com"
