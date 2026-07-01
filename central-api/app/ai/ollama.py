"""Ollama provider adapter — hardened local provider backend.

Ollama runs locally or remotely. Health and model inventory follow AI provider adapter patterns.
No secrets are returned. Timeout handling and degraded/down states supported.
Chat/embeddings are interface stubs until full chat playground (AI-006).
"""

from __future__ import annotations

import json
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = httpx.Timeout(5.0)
_MODELS_TIMEOUT = httpx.Timeout(15.0)
_CHAT_TIMEOUT = httpx.Timeout(120.0)


async def health() -> dict:
    """Check Ollama reachability and return health state.
    
    Returns:
        {"status": "ok"|"degraded"|"down", "code": int?, "error": str?}
    """
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            r = await client.get(settings.ollama_url + "/")
            if r.is_success:
                return {"status": "ok", "code": r.status_code}
            return {"status": "degraded", "code": r.status_code}
    except httpx.TimeoutException:
        logger.warning("ollama health check timeout")
        return {"status": "down", "error": "timeout"}
    except Exception as exc:
        logger.warning("ollama health check failed: %s", exc)
        return {"status": "down", "error": type(exc).__name__}


async def list_models() -> list[dict]:
    """GET /api/tags → Ollama model list, normalized to CollabMind shape.
    
    Returns list of normalized models. Empty list on error.
    Secrets are never included in normalization.
    """
    try:
        async with httpx.AsyncClient(timeout=_MODELS_TIMEOUT) as client:
            r = await client.get(settings.ollama_url + "/api/tags")
            r.raise_for_status()
            body = r.json()
    except httpx.TimeoutException:
        logger.warning("ollama list_models timeout")
        return []
    except Exception as exc:
        logger.warning("ollama list_models failed: %s", exc)
        return []

    raw = body.get("models", []) if isinstance(body, dict) else []
    return [_normalize(m) for m in raw if isinstance(m, dict)]


def _normalize(raw: dict) -> dict:
    """Normalize Ollama model response to CollabMind schema.
    
    Ollama /api/tags returns: {"models": [{"name": "llama2:7b", "size": 12345, "modified_at": "...", "digest": "..."}, ...]}
    
    Filters out secrets and unknown fields. Returns only safe metadata.
    """
    model_id: str = raw.get("name", "unknown")
    
    if model_id.startswith("huggingface.co/"):
        parts = model_id.split("/")
        if len(parts) >= 3:
            model_name = parts[-1].split(":")[0]
            if "llama3.3" in model_name.lower():
                display_name = "Llama3.3 8B Thinking"
            elif "llama" in model_name.lower():
                size_match = None
                for part in model_name.split("-"):
                    if part.lower().endswith("b") and part[:-1].replace(".", "").isdigit():
                        size_match = part.upper()
                        break
                display_name = f"Llama {size_match or 'Unknown'}"
            else:
                display_name = model_name.replace("-", " ").replace("_", " ").title()
        else:
            display_name = model_id.replace(":", " ").replace("-", " ").replace("_", " ").title()
    else:
        display_name = model_id.replace(":", " ").replace("-", " ").replace("_", " ").title()
    
    is_embed = "embed" in model_id.lower()

    quantization = None
    if "-" in model_id and ":" in model_id:
        parts = model_id.split("-")
        if len(parts) > 1 and parts[-1].startswith("q"):
            quantization = parts[-1]
    if raw.get("details", {}).get("quantization_level"):
        quantization = raw["details"]["quantization_level"]

    metadata = {
        "size": raw.get("size"),
        "modified_at": raw.get("modified_at"),
    }
    if quantization:
        metadata["quantization"] = quantization
    if model_id.startswith("huggingface.co/"):
        metadata["source"] = "hugging_face"
        metadata["full_model_id"] = model_id

    return {
        "provider_id": "ollama",
        "model_id": model_id,
        "display_name": display_name,
        "family": _family(model_id),
        "capabilities": {
            "chat": not is_embed,
            "completion": not is_embed,
            "embedding": is_embed,
            "audio": False,
            "tools": False,
            "streaming": True,
            "vision": "vision" in model_id.lower() or "llava" in model_id.lower(),
        },
        "context_window": None,
        "cost": {"input_per_1m": None, "output_per_1m": None, "currency": "USD"},
        "latency_ms": None,
        "health": "unknown",
        "metadata": metadata,
    }


_FAMILY_MAP = [
    ("llama", "llama"), ("mistral", "mistral"), ("mixtral", "mistral"),
    ("gemma", "gemini"), ("phi", "phi"), ("qwen", "qwen"),
    ("deepseek", "deepseek"), ("claude", "claude"), ("gpt", "gpt"),
]


def _family(model_id: str) -> str:
    """Detect model family from model_id."""
    lower = model_id.lower()
    for kw, fam in _FAMILY_MAP:
        if kw in lower:
            return fam
    return "unknown"


# Chat/embeddings methods
async def chat(payload: dict) -> dict:
    """POST /api/chat → Ollama chat endpoint (non-streaming)."""
    # Force non-streaming response for simple chat
    chat_payload = {**payload, "stream": False}
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        r = await client.post(
            settings.ollama_url + "/api/chat",
            json=chat_payload,
        )
        r.raise_for_status()
        return r.json()


async def chat_stream(payload: dict):
    """POST /api/chat → Ollama chat endpoint (streaming).
    
    Yields content chunks as they arrive from Ollama.
    
    Ollama streaming format:
    {"message":{"role":"assistant","content":"chunk"},"done":false}
    {"message":{"role":"assistant","content":""},"done":true}
    """
    # Force streaming response
    chat_payload = {**payload, "stream": True}
    
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            settings.ollama_url + "/api/chat",
            json=chat_payload,
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        if content:
                            yield content
                    
                    if data.get("done"):
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Ollama stream chunk: {e}")
                    continue


async def embed(payload: dict) -> dict:
    """POST /api/embed → Ollama embeddings endpoint."""
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        r = await client.post(
            settings.ollama_url + "/api/embed",
            json=payload,
        )
        r.raise_for_status()
        return r.json()
