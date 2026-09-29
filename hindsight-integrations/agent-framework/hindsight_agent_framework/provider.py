"""Hindsight memory for Microsoft Agent Framework via a ContextProvider.

A ``HindsightProvider`` plugs into a Microsoft Agent Framework agent through the
context-provider hooks. Before each run it recalls relevant memories from
Hindsight and injects them into the agent's instructions; after each run it
retains the conversation so future runs build on it. No MCP, no tools the model
must remember to call — memory happens automatically each turn.

    from agent_framework.openai import OpenAIChatClient
    from hindsight_agent_framework import HindsightProvider

    agent = OpenAIChatClient().as_agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        context_providers=[HindsightProvider(bank_id="user-123")],
    )

API note: Agent Framework's ``ContextProvider`` API has changed across releases
(``invoking``/``invoked`` → ``before_run``/``after_run``). This targets the
1.x ``before_run``/``after_run`` + ``SessionContext`` contract; the tests
subclass the real base class so any drift fails loudly.
"""

import logging
from typing import Any, Optional

from agent_framework import AgentSession, ContextProvider, Message, SessionContext
from hindsight_client import Hindsight

from ._client import resolve_client
from .config import get_config

logger = logging.getLogger(__name__)

# Heading for the recalled-memory block injected into the agent's instructions.
MEMORY_PROMPT = "## Memories\nConsider the following memories when responding. Ignore any that are not relevant:"


def _role_of(message: Message) -> str:
    """Return a message's role as a plain lowercase string (enum or str)."""
    role = getattr(message, "role", "")
    role = getattr(role, "value", role)  # tolerate an enum or a plain string
    return str(role).lower()


class HindsightProvider(ContextProvider):
    """Automatic long-term memory for Agent Framework agents via Hindsight."""

    def __init__(
        self,
        bank_id: str,
        *,
        client: Optional[Hindsight] = None,
        hindsight_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        budget: Optional[str] = None,
        max_tokens: Optional[int] = None,
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
        recall_tags: Optional[list[str]] = None,
        recall_tags_match: Optional[str] = None,
        mission: Optional[str] = None,
        auto_recall: bool = True,
        auto_retain: bool = True,
        source_id: str = "hindsight",
    ) -> None:
        """Create a Hindsight context provider.

        Args:
            bank_id: Hindsight memory bank for this agent/user/session.
            client: Pre-built Hindsight client (else resolved from url/key/env).
            hindsight_api_url: Hindsight API URL (default: cloud).
            api_key: API key (falls back to the HINDSIGHT_API_KEY env var).
            budget: Recall budget (low/mid/high).
            max_tokens: Max tokens of recalled memories to inject.
            context: Source label for retained memories.
            tags: Tags applied to retained memories.
            recall_tags: Tags to filter recall.
            recall_tags_match: Tag match mode (any/all/any_strict/all_strict).
            mission: Bank mission; creates the bank on first use if set.
            auto_recall: Inject recalled memories before each run.
            auto_retain: Retain the conversation after each run.
            source_id: Provider source id used to tag injected context.
        """
        super().__init__(source_id)
        cfg = get_config()
        self.bank_id = bank_id
        self._client = resolve_client(client, hindsight_api_url, api_key)
        self._budget = budget or (cfg.budget if cfg else "mid")
        self._max_tokens = max_tokens or (cfg.max_tokens if cfg else 4096)
        self._context = context or (cfg.context if cfg else "agent-framework")
        self._tags = tags if tags is not None else (cfg.tags if cfg else None)
        self._recall_tags = recall_tags if recall_tags is not None else (cfg.recall_tags if cfg else None)
        self._recall_tags_match = recall_tags_match or (cfg.recall_tags_match if cfg else "any")
        self._mission = mission or (cfg.mission if cfg else None)
        self._auto_recall = auto_recall
        self._auto_retain = auto_retain
        # Process-level guard (about the bank, not a session) — safe to share.
        self._bank_initialized = False

    # ── Agent Framework ContextProvider hooks ────────────────────────────────

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Recall relevant memories and inject them into the agent's context."""
        if not self._auto_recall:
            return
        query = self._build_query(context)
        if not query:
            return
        try:
            await self._ensure_bank()
            response = await self._client.arecall(
                bank_id=self.bank_id,
                query=query,
                budget=self._budget,
                max_tokens=self._max_tokens,
                tags=self._recall_tags,
                tags_match=self._recall_tags_match,
            )
        except Exception as e:
            # Never block the agent on a recall failure.
            logger.debug("Hindsight recall failed for bank %s: %s", self.bank_id, e)
            return

        memories = [r.text for r in (response.results or []) if getattr(r, "text", None)]
        if not memories:
            return
        block = MEMORY_PROMPT + "\n" + "\n".join(f"- {m}" for m in memories)
        context.extend_instructions(self.source_id, block)

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Retain the run's user input and agent response into Hindsight."""
        if not self._auto_retain:
            return
        # Recalled memories are injected as *instructions*, not messages, so they
        # don't appear here; we also exclude our own source defensively to avoid
        # any retain feedback loop.
        messages = context.get_messages(
            include_input=True,
            include_response=True,
            exclude_sources={self.source_id},
        )
        content = self._format_transcript(messages)
        if not content.strip():
            return
        try:
            await self._ensure_bank()
            await self._client.aretain(
                bank_id=self.bank_id,
                content=content,
                context=self._context,
                metadata={"source": "agent-framework"},
                tags=self._tags,
            )
        except Exception as e:
            logger.debug("Hindsight retain failed for bank %s: %s", self.bank_id, e)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_query(self, context: SessionContext) -> str:
        """Compose a recall query from the run's user input messages."""
        messages = context.get_messages(include_input=True)
        texts = [m.text for m in messages if _role_of(m) == "user" and m.text]
        if not texts:
            texts = [m.text for m in messages if m.text]
        return "\n".join(texts).strip()

    def _format_transcript(self, messages: list[Message]) -> str:
        """Render messages as a ``[role]\\ntext`` transcript for retain."""
        blocks = []
        for m in messages:
            text = (m.text or "").strip()
            if text:
                blocks.append(f"[{_role_of(m)}]\n{text}")
        return "\n\n".join(blocks)

    async def _ensure_bank(self) -> None:
        """Create the bank with its mission on first use (best effort)."""
        if self._bank_initialized or not self._mission:
            return
        try:
            await self._client.acreate_bank(
                bank_id=self.bank_id,
                name=self.bank_id,
                mission=self._mission,
            )
        except Exception as e:
            logger.debug("Bank creation for %s: %s", self.bank_id, e)
        self._bank_initialized = True
