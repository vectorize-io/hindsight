"""Chat playground service — AI-006.

Governed chat flow through Central API using router + provider backends.
No memory persistence; playground only.
"""

from __future__ import annotations

import json
import logging
import time
from inspect import isawaitable
from typing import Optional

from app.ai import localai, litellm as litellm_adapter, ollama
from app.ai import providers as registry
from app.ai import router_preview
from app.ai import inventory
from app.ai.schemas import ChatRequest, ChatResponse, RoutePreviewRequest, RouteConstraints
from app.audit.logger import log_event

logger = logging.getLogger(__name__)

_ADAPTERS = {
    "localai": localai,
    "ollama": ollama,
    "litellm": litellm_adapter,
}


async def _preview_route(route_req: RoutePreviewRequest, available_models: list):
    result = router_preview.preview(route_req, available_models)
    if isawaitable(result):
        return await result
    return result


async def handle_chat(req: ChatRequest, tenant_id: str, actor_id: str) -> ChatResponse:
    """Governed chat playground request.
    
    Args:
        req: Chat request with optional provider/model
        tenant_id: Authenticated tenant
        actor_id: Authenticated actor
        
    Returns:
        ChatResponse with response text, provider/model used, and metadata
    """
    # Validate prompt
    if not req.prompt or not req.prompt.strip():
        raise ValueError("prompt is required and cannot be empty")
    
    start_ms = int(time.time() * 1000)
    warnings: list[str] = []
    
    # Provider/model selection
    if req.provider and req.model:
        # Explicit provider/model
        provider_id = req.provider
        model_id = req.model
        decision_id = None
    else:
        # Router-based selection
        route_req = RoutePreviewRequest(
            request_type="chat",
            constraints=RouteConstraints(prefer_local=True),
            record_decision=req.record_decision,
        )
        # Get available models from inventory
        available_models = await inventory.query_models(active_only=True)
        route_result = await _preview_route(route_req, available_models)
        
        if not route_result.selected_provider or not route_result.selected_model:
            return ChatResponse(
                response="No suitable model available",
                provider="none",
                model="none",
                status="error",
                latency_ms=int(time.time() * 1000) - start_ms,
                warnings=["Router: no_selection"],
            )
        
        provider_id = route_result.selected_provider
        model_id = route_result.selected_model
        decision_id = getattr(route_result, "decision_id", None)
    
    # Get provider adapter
    adapter = _ADAPTERS.get(provider_id)
    if not adapter:
        return ChatResponse(
            response="Provider not supported",
            provider=provider_id,
            model=model_id,
            status="error",
            latency_ms=int(time.time() * 1000) - start_ms,
            warnings=[f"Provider {provider_id} not available"],
        )
    
    # Check provider health
    health = await adapter.health()
    if health.get("status") not in ("ok", "unknown"):
        return ChatResponse(
            response="Provider unavailable",
            provider=provider_id,
            model=model_id,
            status="degraded",
            latency_ms=int(time.time() * 1000) - start_ms,
            warnings=[f"Provider health: {health.get('status')}"],
        )
    
    # Build payload for provider
    payload = {
        "model": model_id,
        "messages": [],
    }
    
    if req.system_prompt:
        payload["messages"].append({"role": "system", "content": req.system_prompt})
    
    payload["messages"].append({"role": "user", "content": req.prompt})
    
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    
    # Call provider
    try:
        if provider_id == "localai":
            result = await adapter.chat_completion(payload)
        else:
            result = await adapter.chat(payload)
    except Exception as exc:
        logger.warning("chat call to %s failed: %s", provider_id, exc)
        return ChatResponse(
            response=f"Provider error: {type(exc).__name__}",
            provider=provider_id,
            model=model_id,
            status="error",
            latency_ms=int(time.time() * 1000) - start_ms,
            warnings=[str(exc)],
        )
    
    # Extract response text
    response_text = ""
    if isinstance(result, dict):
        # Try Ollama format first: {"message": {"content": "..."}}
        if "message" in result:
            msg = result.get("message", {})
            response_text = msg.get("content", "")
        # Fall back to OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
        elif "choices" in result:
            choices = result.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                response_text = msg.get("content", "")
    
    latency_ms = int(time.time() * 1000) - start_ms
    
    # Audit log
    log_event(
        tenant_id=tenant_id,
        actor_id=actor_id,
        operation="chat_playground",
        resource_type="chat",
        resource_id=model_id,
        outcome="success" if response_text else "empty_response",
        metadata={
            "provider": provider_id,
            "model": model_id,
            "latency_ms": latency_ms,
            "prompt_length": len(req.prompt),
            "response_length": len(response_text),
            "decision_id": decision_id,
        },
    )
    
    return ChatResponse(
        response=response_text,
        provider=provider_id,
        model=model_id,
        status="ok",
        latency_ms=latency_ms,
        decision_id=decision_id,
        warnings=warnings,
    )


async def handle_chat_stream(req: ChatRequest, tenant_id: str, actor_id: str):
    """Governed streaming chat playground request.
    
    Yields SSE-formatted chunks as they arrive from the provider.
    
    Args:
        req: Chat request with optional provider/model
        tenant_id: Authenticated tenant
        actor_id: Authenticated actor
        
    Yields:
        SSE formatted strings: "data: {json}\n\n"
    """
    # Validate prompt
    if not req.prompt or not req.prompt.strip():
        yield f"data: {json.dumps({'error': 'prompt is required'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    start_ms = int(time.time() * 1000)
    
    # Provider/model selection (same logic as non-streaming)
    if req.provider and req.model:
        provider_id = req.provider
        model_id = req.model
        decision_id = None
    else:
        route_req = RoutePreviewRequest(
            request_type="chat",
            constraints=RouteConstraints(prefer_local=True),
            record_decision=req.record_decision,
        )
        available_models = await inventory.query_models(active_only=True)
        route_result = await _preview_route(route_req, available_models)
        
        if not route_result.selected_provider or not route_result.selected_model:
            yield f"data: {json.dumps({'error': 'No suitable model available'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        provider_id = route_result.selected_provider
        model_id = route_result.selected_model
        decision_id = getattr(route_result, "decision_id", None)
    
    # Get provider adapter
    adapter = _ADAPTERS.get(provider_id)
    if not adapter:
        yield f"data: {json.dumps({'error': f'Provider {provider_id} not supported'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    # Check provider health
    health = await adapter.health()
    if health.get("status") not in ("ok", "unknown"):
        yield f"data: {json.dumps({'error': 'Provider unavailable'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    # Build payload
    payload = {
        "model": model_id,
        "messages": [],
    }
    
    if req.system_prompt:
        payload["messages"].append({"role": "system", "content": req.system_prompt})
    
    payload["messages"].append({"role": "user", "content": req.prompt})
    
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    
    # Stream response
    full_response = ""
    chunk_count = 0
    
    try:
        # Send metadata first
        yield f"data: {json.dumps({'provider': provider_id, 'model': model_id, 'type': 'start'})}\n\n"
        
        # Check if adapter has streaming support
        if not hasattr(adapter, 'chat_stream'):
            # Fallback to non-streaming
            result = await adapter.chat(payload)
            
            # Extract and send response
            response_text = ""
            if isinstance(result, dict):
                if "message" in result:
                    response_text = result.get("message", {}).get("content", "")
                elif "choices" in result:
                    choices = result.get("choices", [])
                    if choices:
                        response_text = choices[0].get("message", {}).get("content", "")
            
            yield f"data: {json.dumps({'content': response_text})}\n\n"
            full_response = response_text
        else:
            # Use streaming
            async for chunk in adapter.chat_stream(payload):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                full_response += chunk
                chunk_count += 1
        
        # Calculate latency
        latency_ms = int(time.time() * 1000) - start_ms
        
        # Send completion metadata
        yield f"data: {json.dumps({'type': 'end', 'latency_ms': latency_ms, 'chunks': chunk_count})}\n\n"
        
        # Audit log
        log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            operation="chat_playground_stream",
            resource_type="chat",
            resource_id=model_id,
            outcome="success",
            metadata={
                "provider": provider_id,
                "model": model_id,
                "latency_ms": latency_ms,
                "prompt_length": len(req.prompt),
                "response_length": len(full_response),
                "chunks": chunk_count,
                "decision_id": decision_id,
            },
        )
        
    except Exception as exc:
        logger.error(f"Streaming chat error: {exc}", exc_info=True)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    
    finally:
        yield "data: [DONE]\n\n"
