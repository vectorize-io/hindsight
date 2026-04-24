"""Hindsight Agent memory plugin for Hermes.

Lightweight retain-only plugin that reads agent config from
~/.hindsight-agent/config.json and retains conversations to the
correct Hindsight bank. The hindsight-agent CLI handles all
resolution (agent ID → bank, API URL, token).

This is NOT the full Hindsight memory plugin (plugins/memory/hindsight).
It's a thin retain layer designed to work alongside the agent-knowledge
skill, which provides pages, recall, and the self-learning loop.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import httpx

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".hindsight-agent" / "config.json"


def _load_agent_config(agent_id: str) -> dict | None:
    """Load config for a specific agent from ~/.hindsight-agent/config.json."""
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data.get("agents", {}).get(agent_id)
    except Exception:
        return None


def _load_all_agents() -> dict:
    """Load all agents from ~/.hindsight-agent/config.json."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data.get("agents", {})
    except Exception:
        return {}


class HindsightAgentProvider(MemoryProvider):
    """Retain-only memory provider that delegates to hindsight-agent config."""

    def __init__(self) -> None:
        self._agent_id: str | None = None
        self._config: dict | None = None
        self._session_id: str = ""
        self._session_turns: list[dict] = []
        self._sync_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "hindsight_agent"

    def is_available(self) -> bool:
        return CONFIG_PATH.exists()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._session_turns = []

        # Resolve agent ID from Hermes context
        # Try agent_identity first, then fall back to checking all configured agents
        agent_identity = kwargs.get("agent_identity", "")
        logger.info("[hindsight_agent] initialize: session=%s agent_identity=%s kwargs=%s",
                    session_id, agent_identity, list(kwargs.keys()))

        self._config = None
        self._agent_id = None

        # Try exact match on profile name first
        if agent_identity:
            self._config = _load_agent_config(agent_identity)
            if self._config:
                self._agent_id = agent_identity

        # Fallback: if only one hermes agent in config, use it
        # (covers the common single-profile case where agent_identity is "default"
        # but setup used a custom agent ID)
        if not self._config:
            agents = _load_all_agents()
            hermes_agents = {aid: cfg for aid, cfg in agents.items() if cfg.get("harness") == "hermes"}
            if len(hermes_agents) == 1:
                self._agent_id, self._config = next(iter(hermes_agents.items()))
                logger.info("[hindsight_agent] no exact match for '%s', using sole hermes agent '%s'",
                            agent_identity, self._agent_id)
            elif len(hermes_agents) > 1:
                logger.warning(
                    "[hindsight_agent] multiple hermes agents in config (%s) but profile '%s' doesn't match any. "
                    "Run: hindsight-agent setup %s --bank-id <bank> --harness hermes",
                    ", ".join(hermes_agents.keys()), agent_identity, agent_identity,
                )

        if self._config:
            logger.info(
                "[hindsight_agent] initialized: agent=%s bank=%s",
                self._agent_id,
                self._config.get("bank_id"),
            )
        else:
            logger.info(
                "[hindsight_agent] no hermes agent in config, retain disabled",
            )

    def system_prompt_block(self) -> str:
        # No system prompt injection — the skill handles reading pages
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # No prefetch — the skill handles recall via CLI
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Buffer turns for end-of-session retain."""
        if not self._config:
            logger.debug("[hindsight_agent] sync_turn: skipped (no config)")
            return

        self._session_turns.append({"role": "user", "content": user_content})
        self._session_turns.append({"role": "assistant", "content": assistant_content})
        logger.info("[hindsight_agent] sync_turn: buffered %d messages", len(self._session_turns))

    def on_session_end(self, messages: list | None = None, **kwargs: Any) -> None:
        """Retain the full session to Hindsight (async HTTP POST)."""
        logger.info("[hindsight_agent] on_session_end: %d buffered turns, config=%s",
                     len(self._session_turns), bool(self._config))
        if not self._config or not self._session_turns:
            return

        bank_id = self._config["bank_id"]
        api_url = self._config["api_url"].rstrip("/")
        api_token = self._config.get("api_token")

        content = json.dumps(self._session_turns)
        document_id = f"{self._agent_id}:{self._session_id}" if self._session_id else None

        item: dict = {"content": content}
        if document_id:
            item["document_id"] = document_id

        url = f"{api_url}/v1/default/banks/{bank_id}/memories"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        turn_count = len(self._session_turns)

        def _retain() -> None:
            try:
                resp = httpx.post(
                    url,
                    json={"items": [item], "async": True},
                    headers=headers,
                    timeout=30.0,
                )
                if resp.is_success:
                    logger.info(
                        "[hindsight-agent] retained %d messages for %s",
                        turn_count,
                        self._agent_id,
                    )
                else:
                    logger.warning(
                        "[hindsight-agent] retain failed (%d): %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            except Exception as e:
                logger.warning("[hindsight-agent] retain error: %s", e)

        # Run in background thread to not block session teardown
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_retain, daemon=True, name="hindsight-agent-retain")
        self._sync_thread.start()

    def get_tool_schemas(self) -> list[dict]:
        # No tools — the skill provides CLI-based access
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs: Any) -> str:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)


def register(ctx: Any) -> None:
    """Register as a Hermes memory provider plugin."""
    ctx.register_memory_provider(HindsightAgentProvider())
