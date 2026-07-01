"""LocalAI provider adapter — AI-001.

Governed adapter for LocalAI as the first AI provider backend.
All LLM calls go through central-api governance, never directly from MCP.

Endpoints (OpenAI-compatible subset):
  - GET  /readyz              → health
  - GET  /v1/models            → list models
  - POST /v1/chat/completions  → chat (interface only)
  - POST /v1/embeddings        → embeddings (interface only)
"""

from __future__ import annotations

import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = httpx.Timeout(5.0)
_MODEL_TIMEOUT = httpx.Timeout(30.0)
_CHAT_TIMEOUT = httpx.Timeout(120.0)
_EMBED_TIMEOUT = httpx.Timeout(30.0)

# Regex to guess model family from model id
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gpt", re.IGNORECASE), "gpt"),
    (re.compile(r"claude", re.IGNORECASE), "claude"),
    (re.compile(r"gemini", re.IGNORECASE), "gemini"),
    (re.compile(r"grok", re.IGNORECASE), "grok"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"llama|meta", re.IGNORECASE), "llama"),
    (re.compile(r"mistral|mixtral", re.IGNORECASE), "mistral"),
    (re.compile(r"phi|tinyllama", re.IGNORECASE), "phi"),
    (re.compile(r"deepseek|deep", re.IGNORECASE), "deepseek"),
    (re.compile(r"codestral|code", re.IGNORECASE), "code"),
]


def _headers() -> dict[str, str]:
    """Build request headers; include Bearer token if configured."""
    h = {"Content-Type": "application/json"}
    if settings.localai_api_key:
        h["Authorization"] = f"Bearer {settings.localai_api_key}"
    return h


async def health() -> dict:
    """Check LocalAI reachability via /readyz.

    Returns:
        Dict with status "healthy", "degraded", or "down".
        On success, includes "version" and "details" from LocalAI.
    """
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            r = await client.get(
                f"{settings.localai_base_url}/readyz",
                headers=_headers(),
            )
            if r.is_success:
                info: dict = {"status": "healthy", "ready": True}
                try:
                    body = r.json()
                    if isinstance(body, dict):
                        info["details"] = body
                except Exception:
                    pass
                return info
            else:
                return {
                    "status": "degraded",
                    "ready": False,
                    "code": r.status_code,
                }
    except Exception as exc:
        logger.warning("LocalAI health check failed: %s", exc)
        return {"status": "down", "error": str(exc)}


async def list_models() -> list[dict]:
    """Fetch and normalize all models from LocalAI.

    Returns:
        Normalized model list. Empty list on error (caller handles degradation).
    """
    try:
        async with httpx.AsyncClient(timeout=_MODEL_TIMEOUT) as client:
            r = await client.get(
                f"{settings.localai_base_url}/v1/models",
                headers=_headers(),
            )
            r.raise_for_status()
            body = r.json()

        raw_models = body.get("data", []) if isinstance(body, dict) else body
        if not isinstance(raw_models, list):
            logger.warning("Unexpected LocalAI model response shape: %s", type(raw_models))
            return []

        return [normalize_model(m) for m in raw_models if isinstance(m, dict)]

    except Exception as exc:
        logger.warning("LocalAI model listing failed: %s", exc)
        return []


def normalize_model(raw: dict) -> dict:
    """Convert a raw LocalAI model entry to the CollabMind normalized shape.

    LocalAI /v1/models format (from OpenAI-compatible API):
        {"id": "model-name", "object": "model", "created": ..., "owned_by": "localai"}

    Returns:
        Normalized model dict (see context.md for shape spec).
    """
    model_id = raw.get("id", "unknown")
    display_name = model_id.replace("-", " ").replace("_", " ").title()

    # Guess family from model id
    family: str = "unknown"
    for pattern, fam in _FAMILY_PATTERNS:
        if pattern.search(model_id):
            family = fam
            break

    # Broad capability inference from model id
    is_embed = any(kw in model_id.lower() for kw in ("embed", "text-embedding", "nomic"))
    is_vision = any(kw in model_id.lower() for kw in ("vision", "vision"))
    is_tools = any(kw in model_id.lower() for kw in ("function", "tool", "instruct"))

    return {
        "provider_id": "localai",
        "model_id": model_id,
        "display_name": display_name,
        "family": family,
        "capabilities": {
            "chat": True,
            "completion": True,
            "embedding": is_embed,
            "audio": False,
            "tools": is_tools or model_id.endswith("-instruct"),
            "streaming": True,
            "vision": is_vision,
        },
        "context_window": None,
        "cost": {
            "input_per_1m": None,
            "output_per_1m": None,
            "currency": "USD",
        },
        "latency_ms": None,
        "health": "unknown",
        "metadata": {
            "owned_by": raw.get("owned_by", "localai"),
            "created": raw.get("created"),
        },
    }


async def chat_completion(payload: dict) -> dict:
    """POST /v1/chat/completions → LocalAI (non-streaming).

    OpenAI-compatible endpoint. Returns complete response.
    """
    # Ensure non-streaming
    chat_payload = {**payload, "stream": False}
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        r = await client.post(
            f"{settings.localai_base_url}/v1/chat/completions",
            json=chat_payload,
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json()


async def chat_stream(payload: dict):
    """POST /v1/chat/completions → LocalAI (streaming).
    
    Yields content chunks as they arrive from LocalAI.
    
    LocalAI uses OpenAI-compatible SSE streaming format:
    data: {"choices":[{"delta":{"content":"chunk"},"index":0}]}
    data: [DONE]
    """
    # Force streaming
    chat_payload = {**payload, "stream": True}
    
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{settings.localai_base_url}/v1/chat/completions",
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
                        
                        # Extract content from delta (OpenAI format)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse LocalAI stream chunk: {e}")
                        continue


async def embeddings(payload: dict) -> dict:
    """POST /v1/embeddings → LocalAI."""
    async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
        r = await client.post(
            f"{settings.localai_base_url}/v1/embeddings",
            json=payload,
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json()
