"""Hindsight memory webhook handler for Vapi voice AI.

Processes Vapi server events to add persistent memory to voice calls:

- ``assistant-request``: recalled memories are injected into the assistant's
  system prompt via ``assistantOverrides`` returned in the webhook response.
- ``end-of-call-report``: the full call transcript is retained to Hindsight
  asynchronously (fire-and-forget) so it never blocks the webhook response.

Unlike Pipecat (per-turn injection), Vapi memory is injected **once per call**
at call start — there is no per-turn hook in Vapi's architecture.

Basic usage with FastAPI::

    from fastapi import FastAPI, Request
    from hindsight_vapi import HindsightVapiWebhook

    app = FastAPI()
    memory = HindsightVapiWebhook(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )

    @app.post("/webhook")
    async def vapi_webhook(request: Request):
        event = await request.json()
        response = await memory.handle(event)
        return response or {}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hindsight_client import Hindsight

from .config import get_config
from .errors import HindsightVapiError

logger = logging.getLogger(__name__)

_MEMORY_MARKER = "<hindsight_memories>"


def _resolve_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config."""
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else None)
    key = api_key or (config.api_key if config else None)

    if url is None:
        raise HindsightVapiError(
            "No Hindsight API URL configured. "
            "Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


class HindsightVapiWebhook:
    """Webhook handler that adds Hindsight persistent memory to Vapi voice calls.

    Handles two Vapi server events:

    - ``assistant-request``: Recalls relevant memories for the caller and
      returns ``assistantOverrides`` containing a system message with those
      memories. Vapi merges the overrides into the active assistant config.

    - ``end-of-call-report``: Retains the full call transcript to Hindsight
      asynchronously (fire-and-forget). The webhook response is not delayed.

    All other event types are ignored (returns ``None``, Vapi expects HTTP 200).

    Args:
        bank_id: Hindsight memory bank to read from and write to.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key for Hindsight Cloud.
        recall_budget: Recall budget level — ``"low"``, ``"mid"``, ``"high"``.
        recall_max_tokens: Maximum tokens for recall results.
        enable_recall: Inject recalled memories into the assistant system prompt.
        enable_retain: Store call transcripts after each call ends.
        memory_prefix: Text prepended inside the recalled memory block.
    """

    def __init__(
        self,
        bank_id: str,
        *,
        client: Hindsight | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        recall_budget: str = "mid",
        recall_max_tokens: int = 4096,
        enable_recall: bool = True,
        enable_retain: bool = True,
        memory_prefix: str = "Relevant memories from past conversations:\n",
    ) -> None:
        self._bank_id = bank_id
        self._client = _resolve_client(client, hindsight_api_url, api_key)
        config = get_config()
        self._recall_budget = recall_budget or (config.recall_budget if config else "mid")
        self._recall_max_tokens = recall_max_tokens or (
            config.recall_max_tokens if config else 4096
        )
        self._enable_recall = enable_recall
        self._enable_retain = enable_retain
        self._memory_prefix = memory_prefix

    async def handle(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Process a Vapi server event.

        Args:
            event: The parsed JSON body of the Vapi webhook POST request.

        Returns:
            A response dict for Vapi (must be returned as the HTTP response body),
            or ``None`` for events that require no response body (HTTP 200 OK).
        """
        msg = event.get("message", {})
        event_type = msg.get("type")

        if event_type == "assistant-request":
            return await self._handle_assistant_request(msg)
        elif event_type == "end-of-call-report":
            await self._handle_end_of_call(msg)

        return None

    async def build_assistant_overrides(self, query: str) -> dict[str, Any]:
        """Build ``assistantOverrides`` for an outbound call.

        Use this when creating outbound calls via the Vapi API — there is no
        ``assistant-request`` webhook for outbound calls, so memories must be
        injected at call-creation time::

            overrides = await memory.build_assistant_overrides("user preferences")
            vapi.calls.create(
                assistant_id="...",
                assistant_overrides=overrides,
                ...
            )

        Args:
            query: Query string for memory recall (e.g. the caller's name or
                a description of the call topic).

        Returns:
            A dict suitable for passing as ``assistantOverrides``, or ``{}``
            if recall is disabled or returns no results.
        """
        if not self._enable_recall:
            return {}

        memories = await self._recall(query)
        if not memories:
            return {}

        return self._build_overrides(memories)

    async def _handle_assistant_request(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle ``assistant-request``: recall and return assistantOverrides."""
        if not self._enable_recall:
            return {}

        # Use the caller's phone number as the recall query when available.
        caller_number: str | None = msg.get("call", {}).get("customer", {}).get("number")
        query = caller_number or "returning caller"

        memories = await self._recall(query)
        if not memories:
            return {}

        return self._build_overrides(memories)

    async def _handle_end_of_call(self, msg: dict[str, Any]) -> None:
        """Handle ``end-of-call-report``: retain transcript (fire-and-forget)."""
        if not self._enable_retain:
            return

        transcript: str = msg.get("artifact", {}).get("transcript", "")
        if transcript:
            asyncio.create_task(self._retain(transcript))

    def _build_overrides(self, memories: str) -> dict[str, Any]:
        """Build the assistantOverrides dict with a memory system message."""
        memory_block = f"{_MEMORY_MARKER}\n{memories}\n</hindsight_memories>"
        return {
            "assistantOverrides": {
                "model": {
                    "messages": [{"role": "system", "content": memory_block}],
                }
            }
        }

    async def _recall(self, query: str) -> str | None:
        """Call Hindsight recall and return a formatted string, or None."""
        try:
            response = await self._client.arecall(
                bank_id=self._bank_id,
                query=query,
                budget=self._recall_budget,
                max_tokens=self._recall_max_tokens,
            )
            if not response.results:
                return None
            lines = [self._memory_prefix]
            for i, result in enumerate(response.results, 1):
                lines.append(f"{i}. {result.text}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Hindsight recall failed (continuing without memories): {e}")
            return None

    async def _retain(self, content: str) -> None:
        """Call Hindsight retain (fire-and-forget — errors are logged and swallowed)."""
        try:
            await self._client.aretain(
                bank_id=self._bank_id,
                content=content,
            )
        except Exception as e:
            logger.warning(f"Hindsight retain failed: {e}")
