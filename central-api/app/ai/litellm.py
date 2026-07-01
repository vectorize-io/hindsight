"""LiteLLM gateway adapter — hardened provider backend.

Proxies /chat/completions, /embeddings, and /models to a LiteLLM proxy instance.
LiteLLM proxy handles routing to OpenAI, Anthropic, Google, xAI, Azure, Vertex,
Cohere, Mistral, etc. — provider-specific keys stay in the LiteLLM proxy.

Health, model inventory, and normalization follow AI provider adapter patterns.
No secrets are returned. Timeout handling and degraded/down states supported.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_CHAT_TIMEOUT = httpx.Timeout(120.0)
_HEALTH_TIMEOUT = httpx.Timeout(5.0)
_MODELS_TIMEOUT = httpx.Timeout(15.0)


def _headers() -> dict[str, str]:
    """Build request headers. Authorization secret never returned to caller."""
    h = {"Content-Type": "application/json"}
    if settings.litellm_api_key:
        h["Authorization"] = f"Bearer {settings.litellm_api_key}"
    return h


async def health() -> dict:
    """Check LiteLLM reachability and return health state.
    
    Returns:
        {"status": "ok"|"degraded"|"down", "code": int, "error": str?}
    """
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            r = await client.get(f"{settings.litellm_url}/health", headers=_headers())
            if r.is_success:
                return {"status": "ok", "code": r.status_code}
            return {"status": "degraded", "code": r.status_code}
    except httpx.TimeoutException:
        logger.warning("litellm health check timeout")
        return {"status": "down", "error": "timeout"}
    except Exception as exc:
        logger.warning("litellm health check failed: %s", exc)
        return {"status": "down", "error": type(exc).__name__}


async def list_models() -> list[dict]:
    """GET /models → LiteLLM, normalized to CollabMind model shape.
    
    Returns list of normalized models. Empty list on error.
    Secrets are never included in normalization.
    """
    try:
        async with httpx.AsyncClient(timeout=_MODELS_TIMEOUT) as client:
            r = await client.get(f"{settings.litellm_url}/models", headers=_headers())
            r.raise_for_status()
            raw = r.json()
    except httpx.TimeoutException:
        logger.warning("litellm list_models timeout")
        return []
    except Exception as exc:
        logger.warning("litellm list_models failed: %s", exc)
        return []
    
    items = raw.get("data", []) if isinstance(raw, dict) else []
    return [_normalize(m) for m in items if isinstance(m, dict)]


def _normalize(raw: dict) -> dict:
    """Normalize LiteLLM model response to CollabMind schema.
    
    Filters out secrets and unknown fields. Returns only safe metadata.
    LiteLLM /models endpoint returns: {"id": "model-name", "object": "model", ...}
    """
    model_id: str = raw.get("id", "unknown")
    
    # Never expose api_key, key, secret, or token fields
    metadata = {}
    for k, v in raw.items():
        if k not in ("id", "object", "created", "owned_by") and not any(
            kw in k.lower() for kw in ("key", "secret", "token", "auth", "credential")
        ):
            metadata[k] = v
    
    return {
        "provider_id": "litellm",
        "model_id": model_id,
        "display_name": model_id.replace("-", " ").replace("_", " ").title(),
        "family": "unknown",
        "capabilities": {
            "chat": True,
            "completion": True,
            "embedding": False,
            "audio": False,
            "tools": True,
            "streaming": True,
            "vision": False,
        },
        "context_window": None,
        "cost": {"input_per_1m": None, "output_per_1m": None, "currency": "USD"},
        "latency_ms": None,
        "health": "unknown",
        "metadata": metadata,
    }


# Chat/embedding methods
async def chat(payload: dict) -> dict:
    """POST /chat/completions → LiteLLM proxy (non-streaming)."""
    # Ensure non-streaming
    chat_payload = {**payload, "stream": False}
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        r = await client.post(
            f"{settings.litellm_url}/chat/completions",
            json=chat_payload,
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json()


async def chat_stream(payload: dict):
    """POST /chat/completions → LiteLLM proxy (streaming).
    
    Yields content chunks as they arrive from LiteLLM.
    
    LiteLLM streaming format (OpenAI-compatible SSE):
    data: {"choices":[{"delta":{"content":"chunk"},"index":0}]}
    data: [DONE]
    """
    # Force streaming
    chat_payload = {**payload, "stream": True}
    
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{settings.litellm_url}/chat/completions",
            json=chat_payload,
            headers=_headers(),
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                # SSE format: "data: {json}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        import json
                        data = json.loads(data_str)
                        
                        # Extract content from delta
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse LiteLLM stream chunk: {e}")
                        continue


async def embed(payload: dict) -> dict:
    """POST /embeddings → LiteLLM proxy."""
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        r = await client.post(
            f"{settings.litellm_url}/embeddings",
            json=payload,
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json()
