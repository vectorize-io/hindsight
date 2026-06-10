"""
Nous Portal LLM provider for Hindsight.

Thin wrapper over :class:`OpenAICompatibleLLM`. The Nous Portal speaks the
OpenAI chat-completions wire format, so all request/response handling is
inherited unchanged. The only thing Nous needs on top is a rotating,
inference-scoped JWT (there is no static API key in the Hermes login flow),
which :class:`NousAuthManager` reads from ``~/.hermes/auth.json`` and refreshes
natively — the same pattern as the Codex provider, with no dependency on the
``hermes_cli`` package. See ``nous_auth.py`` for the auth mechanics.

Configure with::

    llm_provider = "nous"
    llm_base_url = "https://inference-api.nousresearch.com/v1"   # or omit
    llm_model    = "deepseek/deepseek-v4-flash"                  # any Nous slug

No API key is set in config; the token comes from the shared Hermes auth store
after a one-time ``hermes portal`` login.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from hindsight_api.engine.providers.nous_auth import (
    NousAuthManager,
    NousNotLoggedInError,
    NousRefreshExpiredError,
)
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

logger = logging.getLogger(__name__)

__all__ = ["NousLLM", "NousAuthManager", "NousNotLoggedInError", "NousRefreshExpiredError"]


class NousLLM(OpenAICompatibleLLM):
    """OpenAI-compatible provider for the Nous Portal with rotating-JWT auth."""

    def __init__(
        self,
        provider: str,
        api_key: str,  # Ignored — the token is read from ~/.hermes/auth.json
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        try:
            self._auth = NousAuthManager.from_file()
        except NousNotLoggedInError as e:
            raise RuntimeError(
                f"Failed to load Nous Portal credentials: {e}\n\n"
                "To set up Nous authentication:\n"
                "1. Install Hermes: https://hermes-agent.nousresearch.com\n"
                "2. Log in to Nous Portal: hermes portal\n"
                "3. Verify: hermes portal status\n\n"
                "Or use a different provider (openai, anthropic, gemini) with an API key."
            ) from e

        # Single-flight async refresh lock — concurrent coroutines racing toward
        # an expired token produce one network refresh.
        self._auth_lock = asyncio.Lock()

        token = self._auth.access_token
        resolved_base = base_url or self._auth.base_url
        # Parent validates provider against a fixed list; present as "openai"
        # (identical wire format) while retaining the true identity for logs.
        super().__init__(
            provider="openai",
            api_key=token,
            base_url=resolved_base,
            model=model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        self._nous_provider_name = provider
        logger.info(
            "Nous LLM initialized: model=%s base_url=%s (rotating inference:invoke JWT)",
            self.model,
            self.base_url,
        )

    # ------------------------------------------------------------------
    # Token lifecycle
    # ------------------------------------------------------------------

    def _rebuild_client(self) -> None:
        """Rebuild the OpenAI SDK client against the current token."""
        self.api_key = self._auth.access_token
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=self.timeout,
        )

    async def _ensure_fresh_token(self) -> None:
        """Proactively refresh if the JWT is near expiry; rebuild on change.

        Cheap when fresh (a JWT exp decode). The blocking refresh (network +
        cross-process file lock) is offloaded to a thread so the event loop is
        never stalled.
        """
        if not self._auth._token_is_stale():
            return
        await self._refresh(reason="proactive (token near expiry)", force=False)

    async def _refresh(self, *, reason: str, force: bool) -> None:
        token_before = self.api_key
        async with self._auth_lock:
            if force:
                if self.api_key != token_before:
                    return  # another coroutine already refreshed
            elif not self._auth._token_is_stale():
                return
            await asyncio.to_thread(lambda: self._auth.refresh_tokens(reason, force=force))
            if self._auth.access_token != self.api_key:
                self._rebuild_client()

    async def _with_auth_retry(self, fn: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        """Run an OpenAI-compatible call, refreshing once on a 401.

        The proactive refresh covers most expiries; a token can still be
        rejected mid-flight if Hermes rotated it out from under us or the exp
        claim was unparseable. One reactive refresh + retry is the safety net.
        """
        await self._ensure_fresh_token()
        try:
            return await fn(*args, **kwargs)
        except APIStatusError as e:
            if getattr(e, "status_code", None) != 401:
                raise
            logger.warning("Nous 401 (%s) — forcing token refresh and retrying once.", label)
            try:
                await self._refresh(reason=f"reactive (HTTP 401 on {label})", force=True)
            except NousRefreshExpiredError as refresh_err:
                raise RuntimeError(
                    "Nous authentication failed and the refresh_token is no longer valid.\n"
                    "Run 'hermes portal' to re-authenticate."
                ) from refresh_err
            return await fn(*args, **kwargs)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    async def verify_connection(self) -> None:
        await self._ensure_fresh_token()
        return await super().verify_connection()

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        return await self._with_auth_retry(super().call, "call", *args, **kwargs)

    async def call_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        return await self._with_auth_retry(super().call_with_tools, "call_with_tools", *args, **kwargs)

    async def cleanup(self) -> None:
        self._auth.close()
        parent_cleanup = getattr(super(), "cleanup", None)
        if parent_cleanup is not None:
            await parent_cleanup()
