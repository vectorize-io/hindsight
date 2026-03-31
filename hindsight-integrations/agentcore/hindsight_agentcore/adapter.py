"""
HindsightRuntimeAdapter — durable memory layer for AgentCore Runtime agents.

Provides:
- before_turn(): recall relevant memories before agent execution
- after_turn(): retain agent output after execution
- run_turn(): high-level wrapper that calls both automatically

Usage:
    from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure

    configure(hindsight_api_url="https://api.hindsight.vectorize.io", api_key="hsk_...")

    adapter = HindsightRuntimeAdapter(agent_name="support-agent")

    # Inside your AgentCore Runtime handler:
    context = TurnContext(
        runtime_session_id=session_id,
        user_id=user_id,
        agent_name="support-agent",
        tenant_id=tenant_id,
    )
    result = await adapter.run_turn(context, payload, agent_callable=run_agent)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .bank import BankResolver, TurnContext, default_bank_resolver
from .config import get_config
from .errors import BankResolutionError

logger = logging.getLogger(__name__)


@dataclass
class RecallPolicy:
    """Controls how memories are retrieved before a turn.

    Attributes:
        mode: 'recall' for deterministic lookup, 'reflect' for LLM synthesis,
              'hybrid' for recall with reflect on retry.
        budget: Hindsight search depth — 'low', 'mid', or 'high'.
        max_tokens: Maximum tokens in the retrieved memory block.
    """

    mode: str = "recall"
    budget: str | None = None
    max_tokens: int | None = None


@dataclass
class RetentionPolicy:
    """Controls how agent output is retained after a turn.

    Attributes:
        context_label: Provenance label stored with each retained memory.
        extra_tags: Additional tags beyond the default TurnContext tags.
        extra_metadata: Additional metadata beyond the default TurnContext metadata.
        include_user_message: Whether to prepend the user message to retained content.
    """

    context_label: str = "agentcore-runtime:conversation_turn"
    extra_tags: list[str] = field(default_factory=list)
    extra_metadata: dict[str, str] = field(default_factory=dict)
    include_user_message: bool = True


class HindsightRuntimeAdapter:
    """Durable memory adapter for Amazon Bedrock AgentCore Runtime agents.

    Bridges AgentCore Runtime invocation identity to Hindsight memory banks.
    Automates pre-turn retrieval and post-turn retention.

    Example — high-level wrapper:
        adapter = HindsightRuntimeAdapter()

        result = await adapter.run_turn(
            context=TurnContext(
                runtime_session_id=session_id,
                user_id=user_id,
                agent_name="support-agent",
                tenant_id=tenant_id,
                request_id=request_id,
            ),
            payload={"prompt": "What happened with my last invoice?"},
            agent_callable=run_my_agent,
        )

    Example — lower-level hooks:
        memory_context = await adapter.before_turn(context, query=payload["prompt"])
        result = await run_my_agent(payload, memory_context=memory_context)
        await adapter.after_turn(context, result=result["output"], query=payload["prompt"])
    """

    def __init__(
        self,
        agent_name: str | None = None,
        *,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        bank_resolver: BankResolver | None = None,
        recall_policy: RecallPolicy | None = None,
        retention_policy: RetentionPolicy | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Create a HindsightRuntimeAdapter.

        Args:
            agent_name: Default agent name used in bank resolution when not
                provided in TurnContext. Optional if TurnContext always has it.
            hindsight_api_url: Hindsight API URL. Falls back to global config,
                then HINDSIGHT_API_URL env var, then Hindsight Cloud.
            api_key: API key for Hindsight Cloud. Falls back to global config,
                then HINDSIGHT_API_KEY / HINDSIGHT_API_TOKEN env vars.
            bank_resolver: Custom bank ID resolver. Defaults to
                default_bank_resolver (tenant:user:agent format).
            recall_policy: Controls how memories are retrieved. Defaults to
                recall mode with 'mid' budget and 1500 max tokens.
            retention_policy: Controls how agent output is retained.
            verbose: Log memory operations. Falls back to global config.
        """
        config = get_config()

        self._agent_name = agent_name
        self._api_url = hindsight_api_url or (
            config.hindsight_api_url if config else "https://api.hindsight.vectorize.io"
        )
        self._api_key = api_key or (config.api_key if config else None)
        self._bank_resolver = bank_resolver or default_bank_resolver
        self._recall_policy = recall_policy or RecallPolicy(
            budget=config.recall_budget if config else "mid",
            max_tokens=config.recall_max_tokens if config else 1500,
        )
        self._retention_policy = retention_policy or RetentionPolicy()
        self._retain_async = config.retain_async if config else True
        self._timeout = config.timeout if config else 15.0
        self._default_tags = config.tags if config else []
        self._verbose = verbose if verbose is not None else (config.verbose if config else False)

        # Thread-local storage for the Hindsight client
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def before_turn(
        self,
        context: TurnContext,
        *,
        query: str,
    ) -> str:
        """Retrieve relevant memories before agent execution.

        Args:
            context: AgentCore Runtime invocation identity.
            query: The user's message or task description — used as the
                recall query. Typically the prompt text.

        Returns:
            Formatted memory string to inject into the agent prompt,
            or empty string if no relevant memories exist.
        """
        if not query.strip():
            return ""

        try:
            bank_id = self._resolve_bank(context)
        except BankResolutionError:
            logger.exception("Bank resolution failed — skipping recall")
            return ""

        if self._verbose:
            logger.info("[Hindsight] Recalling from bank '%s'", bank_id)

        try:
            client = self._get_client()
            policy = self._recall_policy
            budget = policy.budget or "mid"
            max_tokens = policy.max_tokens or 1500

            if policy.mode == "reflect":
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.reflect(
                        bank_id=bank_id,
                        query=query,
                        budget=budget,
                        max_tokens=max_tokens,
                    ),
                )
                return resp.answer if resp and resp.answer else ""

            # Default: recall
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.recall(
                    bank_id=bank_id,
                    query=query,
                    budget=budget,
                    max_tokens=max_tokens,
                ),
            )
            if not resp or not resp.results:
                return ""
            return _format_memories(resp.results)

        except Exception:
            # Graceful degradation — memory is enhancement, not infrastructure
            logger.warning("[Hindsight] Recall failed — continuing without memory", exc_info=True)
            return ""

    async def after_turn(
        self,
        context: TurnContext,
        *,
        result: str,
        query: str | None = None,
    ) -> None:
        """Retain agent output after execution.

        Fails silently — retention failure never surfaces to the user turn.

        Args:
            context: AgentCore Runtime invocation identity.
            result: The agent's output to store.
            query: Optional user message to prepend to the retained content.
        """
        if not result.strip():
            return

        if self._retain_async:
            asyncio.ensure_future(self._retain(context, result=result, query=query))
        else:
            await self._retain(context, result=result, query=query)

    async def run_turn(
        self,
        context: TurnContext,
        payload: dict[str, Any],
        *,
        agent_callable: Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]],
        query_key: str = "prompt",
        result_key: str = "output",
    ) -> dict[str, Any]:
        """High-level wrapper: recall → execute → retain.

        Args:
            context: AgentCore Runtime invocation identity.
            payload: The request payload passed to the agent.
            agent_callable: Async callable ``(payload, memory_context) -> result_dict``.
                Receives the payload plus recalled memories as a string.
            query_key: Key in payload containing the user's prompt/query.
            result_key: Key in the result dict containing the agent's output.

        Returns:
            The result dict returned by agent_callable.

        Example:
            async def my_agent(payload, memory_context: str) -> dict:
                prompt = payload["prompt"]
                if memory_context:
                    prompt = f"Past context:\\n{memory_context}\\n\\n{prompt}"
                output = await llm.invoke(prompt)
                return {"output": output}

            result = await adapter.run_turn(context, payload, agent_callable=my_agent)
        """
        # Resolve agent_name from context or adapter default
        if not context.agent_name and self._agent_name:
            context = TurnContext(
                runtime_session_id=context.runtime_session_id,
                user_id=context.user_id,
                agent_name=self._agent_name,
                tenant_id=context.tenant_id,
                request_id=context.request_id,
            )

        query = payload.get(query_key, "")

        # Pre-turn: recall
        memory_context = await self.before_turn(context, query=str(query))

        # Execute agent
        result = await agent_callable(payload, memory_context)

        # Post-turn: retain
        output = result.get(result_key, "")
        if output:
            await self.after_turn(context, result=str(output), query=str(query))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_bank(self, context: TurnContext) -> str:
        """Resolve bank ID, filling in default agent_name if needed."""
        if not context.agent_name and self._agent_name:
            context = TurnContext(
                runtime_session_id=context.runtime_session_id,
                user_id=context.user_id,
                agent_name=self._agent_name,
                tenant_id=context.tenant_id,
                request_id=context.request_id,
            )
        return self._bank_resolver(context)

    def _get_client(self) -> Any:
        """Return a thread-local Hindsight client."""
        client = getattr(self._local, "client", None)
        if client is None:
            from hindsight_client import Hindsight  # type: ignore[import]

            client = Hindsight(
                base_url=self._api_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
            self._local.client = client
        return client

    async def _retain(
        self,
        context: TurnContext,
        *,
        result: str,
        query: str | None,
    ) -> None:
        """Internal: call Hindsight retain with full metadata."""
        try:
            bank_id = self._resolve_bank(context)
        except BankResolutionError:
            logger.exception("Bank resolution failed — skipping retention")
            return

        policy = self._retention_policy
        content = result
        if policy.include_user_message and query and query.strip():
            content = f"User: {query}\nAssistant: {result}"

        tags = context.as_tags() + self._default_tags + policy.extra_tags
        metadata = {**context.as_metadata(), **policy.extra_metadata}

        if self._verbose:
            logger.info("[Hindsight] Retaining to bank '%s'", bank_id)

        try:
            client = self._get_client()
            document_id = context.request_id or f"{context.runtime_session_id}:{context.user_id}"

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.retain(
                    bank_id=bank_id,
                    content=content,
                    context=policy.context_label,
                    document_id=document_id,
                    tags=tags,
                    metadata=metadata,
                ),
            )
        except Exception:
            # Never surface retention failures to the user turn
            logger.warning("[Hindsight] Retain failed — memory not stored", exc_info=True)


def _format_memories(results: list[Any]) -> str:
    """Format a list of Hindsight memory results as a bullet list."""
    lines: list[str] = []
    for r in results:
        text = getattr(r, "text", str(r))
        type_ = getattr(r, "type", None)
        date = getattr(r, "mentioned_at", None) or getattr(r, "mentionedAt", None)
        type_str = f" [{type_}]" if type_ else ""
        date_str = f" ({date})" if date else ""
        lines.append(f"- {text}{type_str}{date_str}")
    return "\n\n".join(lines)
