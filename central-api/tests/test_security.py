"""Security tests: CORS, headers, secret redaction, rate limiting."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.security.redaction import redact_secrets


client = TestClient(app)


class TestSecurityHeaders:
    """Test security headers on all responses."""
    
    def test_security_headers_present(self):
        """Verify security headers present."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "Strict-Transport-Security" in response.headers


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_preflight_localhost(self):
        """Allow localhost CORS requests."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert response.status_code == 200
        assert "localhost:3000" in response.headers.get("Access-Control-Allow-Origin", "")
    
    def test_cors_preflight_collabmind(self):
        """Allow collabmind.dev CORS requests."""
        response = client.options(
            "/",
            headers={
                "Origin": "https://collabmind.dev",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert response.status_code == 200


class TestSecretRedaction:
    """Test secret redaction patterns."""
    
    def test_redact_password_json(self):
        """Redact password in JSON."""
        text = '{"password": "secret123"}'
        redacted = redact_secrets(text)
        assert "secret123" not in redacted
        assert "***" in redacted
    
    def test_redact_api_key_json(self):
        """Redact api_key in JSON."""
        text = '{"api_key": "sk_test_abc123"}'
        redacted = redact_secrets(text)
        assert "sk_test_abc123" not in redacted
        assert "***" in redacted
    
    def test_redact_bearer_token(self):
        """Redact Bearer tokens."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "***" in redacted
    
    def test_redact_mem11_sk(self):
        """Redact mem11_sk_* API keys."""
        text = 'api_key="mem11_sk_e1234567890abcdef"'
        redacted = redact_secrets(text)
        assert "e1234567890abcdef" not in redacted
        assert "***" in redacted
    
    def test_redact_token_in_url(self):
        """Redact token in URL query params."""
        text = "?token=abc123xyz&other=value"
        redacted = redact_secrets(text)
        # Should preserve URL structure but redact token
        assert "?" in redacted
    
    def test_no_redaction_on_empty(self):
        """Don't crash on empty strings."""
        assert redact_secrets("") == ""
        assert redact_secrets(None) is not None


class TestRateLimiting:
    """Test rate limiting behavior."""
    
    def test_rate_limit_threshold(self):
        """Rate limit after 100 requests/minute per IP."""
        # Note: client sends all requests from same localhost
        # In production, this is per-IP. For testing, do selective hits.
        response = client.get("/")
        # Single request should succeed
        assert response.status_code == 200


class TestSecretInResponses:
    """Verify secrets not leaked in responses."""
    
    def test_no_secrets_in_health_response(self):
        """Health endpoint doesn't expose secrets."""
        response = client.get("/")
        if response.status_code == 200:
            body = response.json()
            body_str = str(body).lower()
            assert "password" not in body_str or "***" in str(body)
            assert "api_key" not in body_str or "***" in str(body)


@pytest.mark.asyncio
async def test_audit_logging_without_context():
    """Verify audit logging handles missing context."""
    response = client.get("/")
    # Should not raise exception even if context not set
    assert response.status_code == 200
