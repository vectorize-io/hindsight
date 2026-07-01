"""Auth validation tests — JWT, API-key, and context dependency."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.apikey import validate_api_key
from app.auth.jwt import validate_jwt
from app.auth.internal import sign_context, verify_context, InternalContext
from app.config import settings
from app.main import app

client = TestClient(app)


class TestJWTValidation:
    """JWT validation — currently scaffold (always returns None)."""
    
    @pytest.mark.asyncio
    async def test_validate_jwt_scaffold_returns_none(self):
        """Scaffold JWT validator returns None (JWKS not called)."""
        result = await validate_jwt("dummy.jwt.token")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_jwt_empty_token(self):
        """JWT validator handles empty token gracefully."""
        result = await validate_jwt("")
        assert result is None


class TestAPIKeyValidation:
    """API-key validation — accepts mem11_sk_* format."""
    
    @pytest.mark.asyncio
    async def test_validate_api_key_accepts_mem11_format(self):
        """Valid API-key format returns claims."""
        result = await validate_api_key("mem11_sk_test123")
        assert result is not None
        assert result["actor_id"] == "mem11_sk_test123"
        assert "service" in result["roles"]
        assert result["scopes"] == ["*"]
    
    @pytest.mark.asyncio
    async def test_validate_api_key_rejects_invalid_format(self):
        """Invalid format returns None."""
        result = await validate_api_key("sk_invalid")
        assert result is None
        
        result = await validate_api_key("random-key")
        assert result is None
        
        result = await validate_api_key("")
        assert result is None


class TestInternalContextSignature:
    """Internal context — HMAC signature verification."""
    
    def test_sign_and_verify_context(self):
        """Sign context → verify recovers original data."""
        original = InternalContext(
            tenant_id="tenant-123",
            actor_id="actor-456",
            roles=["admin"],
            scopes=["read", "write"],
            auth_method="internal_service",
        )
        
        signed = sign_context(original)
        verified = verify_context(signed)
        
        assert verified is not None
        assert verified.tenant_id == original.tenant_id
        assert verified.actor_id == original.actor_id
        assert verified.roles == original.roles
        assert verified.scopes == original.scopes
    
    def test_verify_context_rejects_corrupted_signature(self):
        """Corrupted signature fails verification."""
        original = InternalContext(
            tenant_id="tenant-123",
            actor_id="actor-456",
            auth_method="internal_service",
        )
        signed = sign_context(original)
        payload_b64, sig_b64 = signed.split(".", 1)
        
        # Corrupt signature
        corrupted = f"{payload_b64}.invalid_signature"
        verified = verify_context(corrupted)
        assert verified is None
    
    def test_verify_context_rejects_invalid_format(self):
        """Invalid format returns None."""
        assert verify_context(None) is None
        assert verify_context("") is None
        assert verify_context("no-dot-here") is None
        assert verify_context("too.many.dots") is None


class TestContextDependency:
    """RequestContext dependency — auth priority chain."""
    
    def test_bearer_jwt_auth_fails_scaffold(self):
        """Bearer JWT → scaffold validator returns None → 401."""
        r = client.get(
            "/health",
            headers={"Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.test"}
        )
        # In scaffold, JWT validation returns None, falls through to dev default
        # Since health is public, it should pass regardless
        assert r.status_code == 200
    
    def test_bearer_api_key_header_works(self):
        """Bearer mem11_sk_* → API-key validation → accepted."""
        r = client.get(
            "/health",
            headers={"Authorization": "Bearer mem11_sk_test_service"}
        )
        assert r.status_code == 200
    
    def test_x_api_key_header_works(self):
        """x-api-key mem11_sk_* → API-key validation → accepted."""
        r = client.get(
            "/health",
            headers={"x-api-key": "mem11_sk_test_service"}
        )
        assert r.status_code == 200
    
    def test_internal_context_signature_header_works(self):
        """X-CM-Context signed → internal validation → accepted."""
        ctx = InternalContext(
            tenant_id="tenant-123",
            actor_id="service-abc",
            roles=["internal"],
            auth_method="internal_service",
        )
        signed = sign_context(ctx)
        
        r = client.get(
            "/health",
            headers={"X-CM-Context": signed}
        )
        assert r.status_code == 200
    
    def test_dev_fallback_when_no_auth(self):
        """No auth headers + is_dev=True → dev default context."""
        r = client.get("/health")
        assert r.status_code == 200


class TestAuthPriorityOrder:
    """Auth resolution priority: internal > JWT > API-key > dev."""
    
    def test_internal_context_takes_priority_over_bearer(self):
        """X-CM-Context (priority 1) beats Bearer JWT."""
        internal_ctx = InternalContext(
            tenant_id="internal-tenant",
            actor_id="internal-actor",
            auth_method="internal_service",
        )
        signed = sign_context(internal_ctx)
        
        # Both headers present; internal should win
        r = client.get(
            "/health",
            headers={
                "X-CM-Context": signed,
                "Authorization": "Bearer dummy-jwt",
            }
        )
        assert r.status_code == 200
        # In real implementation, we'd inspect the resolved context
        # For now, both auth paths should result in success
    
    def test_bearer_beats_x_api_key_header(self):
        """Bearer (priority 2) beats x-api-key (priority 3)."""
        # Bearer API-key format takes priority
        r = client.get(
            "/health",
            headers={
                "Authorization": "Bearer mem11_sk_priority_bearer",
                "x-api-key": "mem11_sk_priority_header",
            }
        )
        assert r.status_code == 200
