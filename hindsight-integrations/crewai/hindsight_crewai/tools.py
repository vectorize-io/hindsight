"""CrewAI Tool for Hindsight reflect operations.

Since CrewAI's Storage interface only has save/search/reset,
reflect is exposed as a Tool that agents can call explicitly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr

from ._compat import call_sync
from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


class HindsightReflectTool(BaseTool):
    """CrewAI tool that generates disposition-aware answers from memory.

    Unlike recall (search), reflect synthesizes a coherent, reasoned
    response using the bank's personality/disposition and all relevant
    memories. Use this when agents need a thoughtful, contextual answer
    rather than raw memory facts.

    Args:
        bank_id: The Hindsight memory bank to reflect against.
        hindsight_api_url: Override the configured API URL.
        api_key: Override the configured API key.
        budget: Reflect budget level (low/mid/high).
        reflect_context: Additional context for reflect reasoning.

    Example::

        from hindsight_crewai import HindsightReflectTool
        from crewai import Agent

        reflect_tool = HindsightReflectTool(
            bank_id="my-crew",
            budget="mid",
        )
        agent = Agent(role="Analyst", tools=[reflect_tool], ...)
    """

    name: str = "hindsight_reflect"
    description: str = (
        "Generate a thoughtful, synthesized answer about a topic by reflecting "
        "on all relevant memories. Use this when you need a coherent summary "
        "of what you know, not just raw facts. Input: a question or topic."
    )

    bank_id: str = Field(description="Hindsight memory bank ID")
    hindsight_api_url: str | None = Field(default=None, description="Override API URL")
    api_key: str | None = Field(default=None, description="Override API key")
    budget: str = Field(default="mid", description="Reflect budget (low/mid/high)")
    reflect_context: str | None = Field(
        default=None, description="Additional context for reflect reasoning"
    )

    _local: Any = PrivateAttr(default_factory=threading.local)

    def _get_client(self) -> Any:
        """Get or create a thread-local Hindsight client."""
        client = getattr(self._local, "client", None)
        if client is None:
            from hindsight_client import Hindsight

            config = get_config()
            api_url = self.hindsight_api_url or (
                config.hindsight_api_url if config else "http://localhost:8888"
            )
            api_key = self.api_key or (config.api_key if config else None)

            client = Hindsight(
                base_url=api_url,
                api_key=api_key,
                timeout=30.0,
            )
            self._local.client = client
        return client

    def _run(self, query: str) -> str:
        """Execute the reflect tool.

        Args:
            query: The question or topic to reflect on.

        Returns:
            The synthesized reflect response text.

        Raises:
            HindsightError: If the reflect operation fails.
        """
        reflect_kwargs: dict[str, Any] = {
            "bank_id": self.bank_id,
            "query": query,
            "budget": self.budget,
        }
        if self.reflect_context:
            reflect_kwargs["context"] = self.reflect_context

        def _reflect() -> Any:
            return self._get_client().reflect(**reflect_kwargs)

        try:
            result = call_sync(_reflect)
            text = result.text if hasattr(result, "text") else str(result)

            if not text:
                return "No relevant memories found to reflect on."

            return text

        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            raise HindsightError(f"Reflect failed: {e}") from e
