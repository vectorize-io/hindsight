"""API-key validation for service adapters (mem11_sk_… format)."""

from __future__ import annotations


async def validate_api_key(key: str) -> dict | None:
    """Validate API key and return claims, or None if invalid.
    
    Scaffold: accepts any key matching mem11_sk_* pattern.
    Real implementation: look up in control-plane DB.
    """
    if not key.startswith("mem11_sk_"):
        return None
    
    # Scaffold: return minimal claims for any valid-format key
    # The key value itself becomes the actor_id (no real lookup)
    return {
        "actor_id": key,
        "roles": ["service"],
        "scopes": ["*"],
    }
