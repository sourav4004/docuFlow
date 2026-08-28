from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_exists():
    """Test that the health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_structure():
    """Test that the health endpoint returns expected structure."""
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert data["status"] == "ok"
    assert "database" in data


def test_root_endpoint():
    """Test that the root endpoint works."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data
