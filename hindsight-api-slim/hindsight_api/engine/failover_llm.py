"""Optional failover composite for LLMProvider.

When configured, FailoverLLMProvider wraps a primary and a failover LLMProvider
and re-dispatches calls to the failover when the primary raises after its
internal retries are exhausted. See docs/superpowers/plans/2026-06-19-llm-failover-provider.md
for the design.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .llm_interface import OutputTooLongError

if TYPE_CHECKING:
    from .llm_wrapper import ConfiguredLLMProvider, LLMProvider
    from .response_models import LLMToolCallResult


logger = logging.getLogger(__name__)


def _should_failover(error: Exception) -> bool:
    """True if the failover composite should re-dispatch to the failover provider.

    Re-dispatch on transient/provider errors; propagate deterministic failures
    (output-too-long) unchanged. CancelledError, KeyboardInterrupt, and SystemExit
    are not caught here at all (they inherit from BaseException, not Exception).
    """
    if isinstance(error, OutputTooLongError):
        return False
    return True


class FailoverLLMProvider:
    """Composite wrapping a primary LLMProvider and an optional failover.

    Exposes the same public surface as LLMProvider (call, call_with_tools,
    with_config, verify_connection, cleanup, attribute passthrough). When the
    primary raises a non-deterministic error, the same call is re-dispatched
    to the failover with identical arguments.

    Backwards compatibility note: when failover is None this composite still
    works (no-op fallback path), but MemoryEngine should construct a bare
    LLMProvider in that case to keep the hot path identical.
    """

    def __init__(self, primary: "LLMProvider", failover: "LLMProvider | None") -> None:
        self._primary = primary
        self._failover = failover

    # ── attribute passthrough ──────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal lookup fails — forward to primary for
        # back-compat with code reading .provider, .model, ._client, etc.
        return getattr(self._primary, name)

    # ── core call methods ──────────────────────────────────────────────────────

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        try:
            return await self._primary.call(messages=messages, **kwargs)
        except Exception as primary_error:
            if self._failover is None or not _should_failover(primary_error):
                raise
            logger.warning(
                "Primary LLM %s/%s failed after retries; falling over to %s/%s: %s",
                self._primary.provider,
                self._primary.model,
                self._failover.provider,
                self._failover.model,
                primary_error,
            )
            return await self._failover.call(messages=messages, **kwargs)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMToolCallResult":
        try:
            return await self._primary.call_with_tools(messages=messages, tools=tools, **kwargs)
        except Exception as primary_error:
            if self._failover is None or not _should_failover(primary_error):
                raise
            logger.warning(
                "Primary LLM %s/%s call_with_tools failed; falling over to %s/%s: %s",
                self._primary.provider,
                self._primary.model,
                self._failover.provider,
                self._failover.model,
                primary_error,
            )
            return await self._failover.call_with_tools(messages=messages, tools=tools, **kwargs)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def verify_connection(self) -> None:
        """Verify primary hard, failover soft.

        Primary failure raises (same as today). Failover failure logs a warning
        and continues — a misconfigured failover should not block startup.
        """
        await self._primary.verify_connection()
        if self._failover is None:
            return
        try:
            await self._failover.verify_connection()
        except Exception as e:
            logger.warning(
                "Failover LLM %s/%s verify_connection failed; failover will be unavailable: %s",
                self._failover.provider,
                self._failover.model,
                e,
            )

    async def cleanup(self) -> None:
        await self._primary.cleanup()
        if self._failover is not None:
            await self._failover.cleanup()

    # ── per-bank configuration ─────────────────────────────────────────────────

    def with_config(
        self,
        config: Any,
        *,
        bank_id: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConfiguredFailoverLLMProvider":
        """Mirror LLMProvider.with_config(), returning a configured failover composite."""
        primary_cfg = self._primary.with_config(config, bank_id=bank_id, operation=operation, metadata=metadata)
        failover_cfg = (
            self._failover.with_config(config, bank_id=bank_id, operation=operation, metadata=metadata)
            if self._failover is not None
            else None
        )
        return ConfiguredFailoverLLMProvider(primary_cfg, failover_cfg)


class ConfiguredFailoverLLMProvider:
    """Companion to FailoverLLMProvider for the configured (per-bank) path.

    Holds two ConfiguredLLMProvider instances and applies the same failover
    logic at the call() / call_with_tools() layer.
    """

    def __init__(
        self,
        primary: "ConfiguredLLMProvider",
        failover: "ConfiguredLLMProvider | None",
    ) -> None:
        self._primary = primary
        self._failover = failover

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        try:
            return await self._primary.call(messages=messages, **kwargs)
        except Exception as primary_error:
            if self._failover is None or not _should_failover(primary_error):
                raise
            logger.warning(
                "Primary LLM %s/%s failed (configured path); falling over to %s/%s: %s",
                self._primary.provider,
                self._primary.model,
                self._failover.provider,
                self._failover.model,
                primary_error,
            )
            return await self._failover.call(messages=messages, **kwargs)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMToolCallResult":
        try:
            return await self._primary.call_with_tools(messages=messages, tools=tools, **kwargs)
        except Exception as primary_error:
            if self._failover is None or not _should_failover(primary_error):
                raise
            logger.warning(
                "Primary LLM %s/%s call_with_tools failed (configured path); falling over to %s/%s: %s",
                self._primary.provider,
                self._primary.model,
                self._failover.provider,
                self._failover.model,
                primary_error,
            )
            return await self._failover.call_with_tools(messages=messages, tools=tools, **kwargs)

    def trace_context(self) -> Any | None:
        return self._primary.trace_context()
