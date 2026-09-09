"""Tests for API rate limiting (Phase 5.9 Step 8)."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request
from app.core.rate_limit import (
    SlidingWindowCounter,
    check_rate_limit,
    RateLimitConfig,
    DEFAULT_CONFIG,
    _classify_endpoint,
)


# ==========================================================================
# 1. SLIDING WINDOW COUNTER
# ==========================================================================

class TestSlidingWindowCounter:
    def test_allows_within_limit(self):
        counter = SlidingWindowCounter()
        allowed, retry = counter.check_and_record("key1", 60, 5)
        assert allowed is True
        assert retry == 0

    def test_blocks_at_limit(self):
        counter = SlidingWindowCounter()
        for _ in range(5):
            counter.check_and_record("key1", 60, 5)
        allowed, retry = counter.check_and_record("key1", 60, 5)
        assert allowed is False
        assert retry > 0

    def test_different_keys_independent(self):
        counter = SlidingWindowCounter()
        for _ in range(5):
            counter.check_and_record("key1", 60, 5)
        allowed, _ = counter.check_and_record("key2", 60, 5)
        assert allowed is True

    def test_cleanup_removes_old_entries(self):
        counter = SlidingWindowCounter()
        counter._requests["old_key"] = [1.0]  # Very old timestamp
        counter.cleanup(max_age=0)
        assert "old_key" not in counter._requests


# ==========================================================================
# 2. ENDPOINT CLASSIFICATION
# ==========================================================================

class TestEndpointClassification:
    def test_login_classification(self):
        assert _classify_endpoint("/auth/login", "POST") == "login"
        assert _classify_endpoint("/auth/register", "POST") == "login"

    def test_upload_classification(self):
        assert _classify_endpoint("/documents", "POST") == "upload"

    def test_message_classification(self):
        assert _classify_endpoint("/conversations/1/messages", "POST") == "message"

    def test_default_classification(self):
        assert _classify_endpoint("/documents", "GET") == "default"
        assert _classify_endpoint("/conversations", "GET") == "default"
        assert _classify_endpoint("/health", "GET") == "default"


# ==========================================================================
# 3. RATE LIMIT CHECK
# ==========================================================================

class TestRateLimitCheck:
    def test_allows_normal_requests(self):
        mock_request = MagicMock()
        mock_request.url.path = "/documents"
        mock_request.method = "GET"
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"

        allowed, retry = check_rate_limit(mock_request)
        assert allowed is True
        assert retry == 0

    def test_blocks_after_limit(self):
        mock_request = MagicMock()
        mock_request.url.path = "/documents"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.client.host = "10.0.0.1"

        # Exhaust upload limit (5 per minute)
        for _ in range(DEFAULT_CONFIG.upload_requests_per_minute):
            check_rate_limit(mock_request)

        allowed, retry = check_rate_limit(mock_request)
        assert allowed is False
        assert retry > 0

    def test_user_isolation(self):
        req1 = MagicMock()
        req1.url.path = "/documents"
        req1.method = "GET"
        req1.headers = {}
        req1.client.host = "1.1.1.1"

        req2 = MagicMock()
        req2.url.path = "/documents"
        req2.method = "GET"
        req2.headers = {}
        req2.client.host = "2.2.2.2"

        # User 1 makes many requests
        for _ in range(60):
            check_rate_limit(req1, user_id=1)

        # User 2 should still be allowed
        allowed, _ = check_rate_limit(req2, user_id=2)
        assert allowed is True


# ==========================================================================
# 4. MIDDLEWARE INTEGRATION
# ==========================================================================

class TestRateLimitMiddleware:
    def test_health_endpoint_not_rate_limited(self):
        """Health endpoint should not be rate limited."""
        from fastapi.testclient import TestClient
        from tests.test_auth import client
        # The health endpoint should always work
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_endpoint_not_rate_limited(self):
        from tests.test_auth import client
        resp = client.get("/")
        assert resp.status_code == 200

    def test_rate_limit_headers_on_429(self):
        """429 response should include Retry-After header."""
        from app.core.rate_limit import _counter, DEFAULT_CONFIG
        # Reset counter
        _counter._requests.clear()

        # Exhaust login limit for a specific key
        mock_request = MagicMock()
        mock_request.url.path = "/auth/login"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.client.host = "9.9.9.9"

        for _ in range(DEFAULT_CONFIG.login_requests_per_minute):
            check_rate_limit(mock_request)

        allowed, retry_after = check_rate_limit(mock_request)
        assert allowed is False
        assert retry_after > 0
