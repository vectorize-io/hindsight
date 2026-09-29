"""Hindsight memory service for Pipecat voice AI pipelines.

Provides a FrameProcessor that slots between the user context aggregator
and the LLM service to add persistent memory via Hindsight.

Placement::

    user_aggregator → HindsightMemoryService → LLM service

On each turn the service:
1. Retains any new complete user+assistant turn pairs (fire-and-forget).
2. Recalls relevant memories for the current user query.
3. Injects recalled memories as a system message into the LLM context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hindsight_client import Hindsight
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .config import get_config
from .errors import HindsightPipecatError

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
        raise HindsightPipecatError(
            "No Hindsight API URL configured. Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


class HindsightMemoryService(FrameProcessor):
    """Pipecat frame processor that adds Hindsight persistent memory.

    Place this between the user context aggregator and the LLM service::

        pipeline = Pipeline([
            transport.input(),
            stt_service,
            user_aggregator,
            HindsightMemoryService(bank_id="user-123", hindsight_api_url="..."),
            llm_service,
            assistant_aggregator,
            tts_service,
            transport.output(),
        ])

    On each ``LLMContextFrame``:

    1. Retains any new complete user+assistant pairs from prior turns
       (non-blocking, fire-and-forget).
    2. Recalls relevant memories using the latest user message as the query.
    3. Injects memories as a ``<hindsight_memories>`` system message into
       the LLM context before forwarding it downstream.

    Args:
        bank_id: Hindsight memory bank to read from and write to.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        recall_budget: Recall budget level — ``"low"``, ``"mid"``, ``"high"``.
        recall_max_tokens: Maximum tokens for recall results.
        enable_recall: Inject recalled memories before each LLM call.
        enable_retain: Store conversation turns after each exchange.
        memory_prefix: Text prepended to the recalled memory block.
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bank_id = bank_id
        self._client = _resolve_client(client, hindsight_api_url, api_key)
        config = get_config()
        self._recall_budget = recall_budget or (config.recall_budget if config else "mid")
        self._recall_max_tokens = recall_max_tokens or (config.recall_max_tokens if config else 4096)
        self._enable_recall = enable_recall
        self._enable_retain = enable_retain
        self._memory_prefix = memory_prefix
        # Track how many messages have already been retained.
        # Messages are stored as a flat list; complete turns are user+assistant pairs.
        self._last_retained_count: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_context_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_context_frame(self, frame: LLMContextFrame) -> None:
        messages: list[dict[str, Any]] = frame.context.messages or []

        # 1. Retain any new complete turns (user+assistant pairs) non-blockingly.
        if self._enable_retain:
            pairs = self._extract_new_turn_pairs(messages)
            for pair_content in pairs:
                asyncio.create_task(self._retain(pair_content))

        # 2. Recall relevant memories for the current user query.
        if self._enable_recall:
            query = self._extract_last_user_message(messages)
            if query:
                memories = await self._recall(query)
                if memories:
                    self._inject_memories(frame.context, memories)

        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _extract_last_user_message(self, messages: list[dict[str, Any]]) -> str | None:
        """Return the content of the last user message, or None."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Multimodal content is a list of parts; extract text parts.
                if isinstance(content, list):
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    return " ".join(texts) if texts else None
        return None

    def _extract_new_turn_pairs(self, messages: list[dict[str, Any]]) -> list[str]:
        """Return formatted strings for any new complete user+assistant turn pairs.

        A complete pair is a user message immediately followed by an assistant
        message. Updates ``_last_retained_count`` to skip already-retained pairs.
        """
        pairs: list[str] = []
        i = self._last_retained_count

        while i < len(messages) - 1:
            msg = messages[i]
            next_msg = messages[i + 1]
            if msg.get("role") == "user" and next_msg.get("role") == "assistant":
                user_text = self._extract_content_text(msg)
                assistant_text = self._extract_content_text(next_msg)
                if user_text and assistant_text:
                    pairs.append(f"User: {user_text}\nAssistant: {assistant_text}")
                self._last_retained_count = i + 2
                i += 2
            else:
                i += 1

        return pairs

    def _extract_content_text(self, msg: dict[str, Any]) -> str:
        """Extract plain text from a message content field."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            return " ".join(texts)
        return ""

    def _inject_memories(self, context: Any, memories: str) -> None:
        """Inject memories as a system message, replacing any prior memory injection."""
        messages: list[dict[str, Any]] = context.messages or []
        memory_block = f"{_MEMORY_MARKER}\n{memories}\n</hindsight_memories>"

        # Replace existing memory system message if present.
        for i, msg in enumerate(messages):
            if msg.get("role") == "system" and _MEMORY_MARKER in msg.get("content", ""):
                messages[i] = {"role": "system", "content": memory_block}
                return

        # Prepend a new system message.
        messages.insert(0, {"role": "system", "content": memory_block})

    async def _recall(self, query: str) -> str | None:
        """Call Hindsight recall and return a formatted string of memories, or None."""
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
