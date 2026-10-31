"""
Claude Code LLM provider using Claude Agent SDK.

This provider enables using Claude Pro/Max subscriptions for API calls
via the Claude CLI authentication. It uses the Claude Agent SDK, which
authenticates with a server-held `claude setup-token` token when one is
configured as the API key, and otherwise with the host's `claude auth login`.
"""

import asyncio
import json
import logging
import re
import tempfile
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ValidationError

from hindsight_api.engine.llm_interface import (
    LLMInterface,
    ProviderContentPolicyError,
    ProviderRateLimitResetError,
)
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.llm_wrapper import parse_llm_json
from hindsight_api.engine.providers.openai_compatible_llm import _strip_code_fences
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.structured_output import provider_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

from ..response_models import LLMCallResult

logger = logging.getLogger(__name__)


class _SdkUsage(BaseModel):
    """The Anthropic-shaped usage block the CLI reports on its messages."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


def _sdk_token_usage(usage: dict[str, Any] | None, messages: list[dict[str, Any]], text: str) -> TokenUsage:
    """Token counts from the usage block the CLI reports.

    Anthropic counts cache reads and writes apart from ``input_tokens``; the
    ``TokenUsage`` contract counts them in it, with reads also in ``cached_tokens``.
    Falls back to a char/4 estimate when the run reported no usage.
    """
    if not usage:
        estimated_input = sum(len(str(m.get("content", ""))) for m in messages) // 4
        estimated_output = len(text) // 4
        return TokenUsage(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            total_tokens=estimated_input + estimated_output,
        )
    parsed = _SdkUsage.model_validate(usage)
    cache_read = parsed.cache_read_input_tokens or 0
    input_tokens = (parsed.input_tokens or 0) + cache_read + (parsed.cache_creation_input_tokens or 0)
    output_tokens = parsed.output_tokens or 0
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cache_read,
    )


# Isolation env passed to the spawned `claude` CLI. CLAUDE_CONFIG_DIR
# redirects the subprocess away from the host's ~/.claude/, so any
# operator-installed plugins (e.g. hindsight-memory) and their Stop hooks do
# not fire inside our LLM-call subprocesses. Without this, retain/reflect/
# consolidation LLM calls would trigger a Stop-hook retain of the subprocess
# transcript back into the same bank — a recursive feedback loop (issue #1751).
# This only covers *user* settings: the call sites also pin cwd to this dir and
# pass setting_sources=[], or project settings from the daemon's cwd reopen the
# same loop (issue #5542).
# CLAUDE_SECURESTORAGE_CONFIG_DIR="" forces the CLI's keychain service name
# back to the canonical un-suffixed entry that `claude auth login` wrote;
# otherwise it would be namespaced by sha256(CLAUDE_CONFIG_DIR) and OAuth
# lookup would fail. Requires bundled CLI >= 2.1.150 (claude-agent-sdk 0.2.82).
# When a setup-token is configured, _claude_env() below drops that override on
# purpose so the host login is NOT found and the token is used instead.
_isolated_claude_env: dict[str, str] | None = None


def _get_isolated_claude_env() -> dict[str, str]:
    """Return a process-lifetime env dict that isolates the spawned CLI from user plugins."""
    global _isolated_claude_env
    if _isolated_claude_env is None:
        path = tempfile.mkdtemp(prefix="hindsight-claude-code-")
        _isolated_claude_env = {
            "CLAUDE_CONFIG_DIR": path,
            "CLAUDE_SECURESTORAGE_CONFIG_DIR": "",
        }
        logger.debug(f"Claude Code: isolated CLAUDE_CONFIG_DIR={path}")
    return _isolated_claude_env


def _claude_env(oauth_token: str | None) -> dict[str, str]:
    """Isolation env, plus the server-held ``claude setup-token`` token when one is configured.

    With a token in ``HINDSIGHT_API_LLM_API_KEY`` (or a per-operation / per-bank
    key) the server owns the auth session: the CLI authenticates from
    ``CLAUDE_CODE_OAUTH_TOKEN`` instead of the host's ``claude auth login``.
    The CLI prefers a stored login over that variable, so the token path drops
    CLAUDE_SECURESTORAGE_CONFIG_DIR="": the storage lookup stays namespaced to the
    empty isolated dir, finds no login, and falls through to the token.

    Built by copying the isolation env and removing that one key, rather than
    re-listing the keys to keep: a third isolation variable added above would
    otherwise be silently missing from the token path only.
    """
    if not oauth_token:
        return _get_isolated_claude_env()
    env = dict(_get_isolated_claude_env())
    del env["CLAUDE_SECURESTORAGE_CONFIG_DIR"]
    env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


def _result_error_detail(message: Any) -> str:
    """Build an actionable error string from an ``is_error`` ResultMessage.

    The CLI can report a failure with ``is_error=True`` while ``subtype``
    still reads ``"success"``, putting the real detail in ``result`` (e.g.
    quota exhaustion: ``You've hit your weekly limit · resets ...`` with
    ``api_error_status: 429``). The SDK's own fallback exception surfaces
    only the subtype, producing the misleading "Claude Code returned an
    error result: success" (issue #2702) — so prefer ``result``.
    """
    detail = (message.result or "").strip() or message.subtype or "unknown error"
    return f"Claude Code reported an error: {detail}"


#: The link a content-policy (AUP) refusal always carries, e.g. "API Error:
#: Sonnet 4.5 can't help with this. Start a new session to continue.\n\nLearn
#: more: https://www.anthropic.com/legal/aup". Matching the link rather than the
#: apologetic prose keeps the test narrow: a false positive would turn an
#: ordinary transient error into a permanent one (issue #3690).
_POLICY_REFUSAL_MARKER = "anthropic.com/legal/aup"


#: The reset clause of a subscription limit, e.g. "You've hit your session limit ·
#: resets 12:20am (UTC)" or "... · resets Oct 9, 5pm (Europe/Rome)".
_LIMIT_RESET_RE = re.compile(
    r"\blimit\b.*?\bresets\s+(?:(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2}),?\s+(?:at\s+)?)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)


def _limit_reset_at(text: str, now: datetime) -> datetime | None:
    """When a subscription limit named in ``text`` reopens, or None if it names none.

    The CLI prints a wall-clock time in a named zone, with a date only when it is not
    within the next day, so a bare time means its next occurrence after ``now``.
    """
    match = _LIMIT_RESET_RE.search(text)
    if match is None:
        return None
    try:
        tz = ZoneInfo(match["tz"].strip())
    except (ZoneInfoNotFoundError, ValueError):
        return None
    hour = int(match["hour"]) % 12 + (12 if match["ampm"].lower() == "pm" else 0)
    minute = int(match["minute"] or 0)
    local_now = now.astimezone(tz)
    try:
        if match["month"]:
            month = datetime.strptime(match["month"].title(), "%b").month
            reset = local_now.replace(
                month=month, day=int(match["day"]), hour=hour, minute=minute, second=0, microsecond=0
            )
            if reset <= local_now:
                reset = reset.replace(year=reset.year + 1)
        else:
            reset = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if reset <= local_now:
                reset += timedelta(days=1)
    except ValueError:
        return None
    return reset.astimezone(UTC)


def _result_error(message: Any) -> Exception:
    """Build the exception for an ``is_error`` ResultMessage.

    A content-policy refusal is permanent, so it gets its own type: the retry
    loops below re-raise it untouched instead of replaying the same prompt, and
    the worker fails the task rather than rescheduling it.

    A subscription limit with a reset time becomes the engine's defer signal, so
    the worker parks the task until the limit lifts instead of failing it — a
    failed knowledge-page refresh used to pause that page for good (#5394).
    """
    text = _result_error_detail(message)
    if _POLICY_REFUSAL_MARKER in text.lower():
        return ProviderContentPolicyError(text)
    retry_at = _limit_reset_at(text, datetime.now(UTC))
    if retry_at is not None:
        return ProviderRateLimitResetError(retry_at=retry_at, message=text)
    return RuntimeError(text)


class ClaudeCodeLLM(LLMInterface):
    """
    LLM provider using Claude Code authentication.

    Authenticates using Claude Pro/Max credentials via `claude auth login`
    and makes API calls through the Claude Agent SDK.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,  # Optional `claude setup-token` token; empty = CLI login
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Claude Code LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)
        self._warn_reasoning_effort_unsupported()
        self._env = _claude_env(api_key.strip() if api_key else None)
        # The key can also come from a per-operation or per-bank LLM config, so the
        # hint names the config rather than only the env var an operator may not have set.
        self._auth_hint = (
            "The configured Claude Code API key was rejected. Generate a new token with "
            "'claude setup-token' and update HINDSIGHT_API_LLM_API_KEY (or the bank/operation LLM key)."
            if "CLAUDE_CODE_OAUTH_TOKEN" in self._env
            else "Run 'claude auth login', or set HINDSIGHT_API_LLM_API_KEY to a token from 'claude setup-token'."
        )

        # Verify Claude Agent SDK is available
        try:
            self._verify_claude_code_available()
            auth = (
                "CLAUDE_CODE_OAUTH_TOKEN from config" if "CLAUDE_CODE_OAUTH_TOKEN" in self._env else "claude auth login"
            )
            logger.info(f"Claude Code: Using Claude Agent SDK (authentication via {auth})")
            # Automated calls on a Claude subscription may conflict with Anthropic's
            # terms; point operators at the docs notice rather than starting silently (#5444).
            logger.warning(
                "Claude Code provider runs automated model calls on your Claude subscription, which may "
                "conflict with Anthropic's terms. Read the notice before using it beyond personal local "
                "development, and prefer the 'anthropic' provider with an API key for servers: "
                "https://hindsight.vectorize.io/developer/models#claude-code-setup-claude-promax"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Claude Code provider: {e}\n\n"
                "To set up Claude Code authentication:\n"
                "1. Install Claude Code CLI: npm install -g @anthropics/claude-code\n"
                "2. Login with your Pro/Max plan: claude auth login\n"
                "   (or run 'claude setup-token' and set HINDSIGHT_API_LLM_API_KEY to the token)\n"
                "3. Verify authentication: claude --version\n\n"
                "Or use a different provider (anthropic, openai, gemini) with API keys."
            ) from e

        # Metrics collector is imported at module level

    def _verify_claude_code_available(self) -> None:
        """
        Verify that Claude Agent SDK can be imported and is properly configured.

        Raises:
            ImportError: If Claude Agent SDK is not installed.
            RuntimeError: If Claude Code is not authenticated.
        """
        try:
            # Import Claude Agent SDK
            # Reduce Claude Agent SDK logging verbosity
            import logging as sdk_logging

            from claude_agent_sdk import query  # noqa: F401  # type: ignore[unresolved-import]

            sdk_logging.getLogger("claude_agent_sdk").setLevel(sdk_logging.WARNING)
            sdk_logging.getLogger("claude_agent_sdk._internal").setLevel(sdk_logging.WARNING)

            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Run: uv add claude-agent-sdk or pip install claude-agent-sdk"
            ) from e

        # SDK will automatically check for authentication when first used
        # No need to verify here - let it fail gracefully on first call with helpful error

    async def verify_connection(self) -> None:
        """
        Verify that the Claude Code provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        from ...config import get_config

        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=10,
                temperature=get_config().llm_temperature_verification,
                scope="verification",
                max_retries=0,
            )
            logger.info("Claude Code connection verified successfully")
        except Exception as e:
            logger.error(f"Claude Code connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify Claude Code connection: {e}") from e

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMCallResult:
        """
        Make an LLM API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response (ignored by Claude Agent SDK).
            temperature: Sampling temperature (ignored by Claude Agent SDK).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Use strict JSON schema enforcement (not supported).

        Returns:

        Raises:
            OutputTooLongError: If output exceeds token limits (not supported by Claude Agent SDK).
            Exception: Re-raises API errors after retries exhausted.
        """
        from claude_agent_sdk import (  # type: ignore[unresolved-import]
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        start_time = time.perf_counter()

        # Build system prompt
        system_prompt = ""
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt += ("\n\n" + content) if system_prompt else content
            elif role == "user":
                user_content += ("\n\n" + content) if user_content else content
            elif role == "assistant":
                # Claude Agent SDK doesn't support multi-turn easily in query()
                # For now, prepend assistant messages to user content
                user_content += f"\n\n[Previous assistant response: {content}]"

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = provider_json_schema(response_format)
            schema_instruction = (
                f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
                "Respond with ONLY the JSON, no markdown formatting."
            )
            user_content += schema_instruction

        # Configure SDK options
        #
        # tools=[] is required: with `tools` left at its default (None -> full
        # "claude_code" built-in preset), allowed_tools=[] alone does not stop
        # the CLI from loading the full built-in toolset and deferring into
        # ToolSearch before answering, which burns the single max_turns=1
        # budget on a tool-deferral step instead of a text response. Without
        # this, single-turn calls intermittently fail with "Reached maximum
        # number of turns (1)" even though the prompt itself needs no tools.
        options = ClaudeAgentOptions(
            system_prompt=system_prompt if system_prompt else None,
            max_turns=1,  # Single-turn for API-style interactions
            tools=[],  # Disable built-in tools so nothing forces a ToolSearch deferral
            allowed_tools=[],  # Disable tools for standard LLM calls
            # Pin the configured model (issue #2881). Without this the spawned CLI
            # runs its own default model — an Opus-class model on Pro/Max OAuth —
            # regardless of HINDSIGHT_API_*_LLM_MODEL, while metrics/logs still print
            # self.model, so the mismatch is invisible. The CLI loads no settings
            # files (setting_sources=[] below), so a host settings.json can't set the
            # model either: passing it through here is the only channel.
            model=self.model or None,
            env=self._env,
            # CLAUDE_CONFIG_DIR only moves *user* settings. The CLI still loads
            # ./.claude/settings.json from its cwd as project settings, and the
            # subprocess inherits the daemon's cwd: started from $HOME, that file
            # IS ~/.claude/settings.json with the host's hooks, and the #1751
            # loop comes back (issue #5542). Run in the empty isolated dir and
            # load no settings files at all.
            cwd=self._env["CLAUDE_CONFIG_DIR"],
            setting_sources=[],
        )

        # Call Claude Agent SDK
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Collect streaming response
                full_text = ""
                sdk_usage: dict[str, Any] | None = None

                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.claude_code.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                    async for message in query(prompt=user_content, options=options):
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    full_text += block.text
                        elif isinstance(message, ResultMessage):
                            if message.is_error:
                                # Surface the CLI's actual error text (e.g. quota
                                # exhaustion) instead of the SDK's subtype-based
                                # fallback exception (issue #2702).
                                raise _result_error(message)
                            # The run's total usage, as the CLI billed it.
                            sdk_usage = message.usage

                # Stashed before parsing so a later parse/validate failure traces
                # the same counts the success path does (#2387).
                token_usage = _sdk_token_usage(sdk_usage, messages, full_text)
                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=token_usage.input_tokens,
                        output_tokens=token_usage.output_tokens,
                        cached_tokens=token_usage.cached_tokens,
                    )
                )

                # Handle structured output
                if response_format is not None:
                    # Models may wrap JSON in markdown; the shared helper strips
                    # fences line-based, so a JSON value that itself contains the
                    # text "```json" is no longer truncated (#4819).
                    try:
                        json_data = json.loads(_strip_code_fences(full_text))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Claude Code JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        if attempt < max_retries:
                            backoff = min(initial_backoff * (2**attempt), max_backoff)
                            await asyncio.sleep(backoff)
                            last_exception = e
                            continue
                        # Retry budget spent: structural repair as a last resort, as
                        # openai_compatible_llm.py does since #3935. A quotation copied
                        # verbatim leaves an unescaped inner '"' that every retry
                        # reproduces (#5472). We only get here after the CLI's
                        # ResultMessage arrived without an error, so the body is complete.
                        # Raises JSONDecodeError again if unrecoverable.
                        json_data = parse_llm_json(full_text)

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = full_text

                # Record metrics
                duration = time.perf_counter() - start_time
                metrics = get_metrics_collector()

                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=token_usage.input_tokens,
                    output_tokens=token_usage.output_tokens,
                    success=True,
                    cached_input_tokens=token_usage.cached_tokens,
                )

                # Record trace span
                try:
                    from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                    span_recorder = get_span_recorder()
                    span_recorder.record_llm_call(
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                        messages=messages,
                        response_content=_serialize_for_span(result),
                        input_tokens=token_usage.input_tokens,
                        output_tokens=token_usage.output_tokens,
                        duration=duration,
                        finish_reason=None,
                        error=None,
                        cached_tokens=token_usage.cached_tokens,
                    )
                except Exception as span_error:
                    # Tracing must remain best-effort, but expose instrumentation
                    # bugs that would otherwise silently erase spans (#3025).
                    logger.debug("Claude Code span recording failed: %s", span_error, exc_info=True)

                # Log slow calls
                if duration > 10.0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, time={duration:.3f}s"
                    )

                return LLMCallResult(content=result, usage=token_usage)

            except ValidationError:
                # Pydantic schema validation failure — retrying with the same
                # input won't produce a different schema.  Raise immediately
                # instead of burning quota on identical calls (#1412).
                raise

            except (ProviderContentPolicyError, ProviderRateLimitResetError):
                # Content-policy refusal: the model declined this exact content,
                # so every replay earns the same refusal. Raise immediately
                # instead of spending the full retry budget on it (#3690). A
                # subscription limit will not lift within the backoff either (#5394).
                raise

            except Exception as e:
                last_exception = e

                # Check for authentication errors
                error_str = str(e).lower()
                if "auth" in error_str or "login" in error_str or "credential" in error_str:
                    logger.error(f"Claude Code authentication error: {e}")
                    raise RuntimeError(f"Claude Code authentication failed: {e}\n\n{self._auth_hint}") from e

                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(f"Claude Code error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Claude Code error after {max_retries + 1} attempts: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Claude Code call failed after all retries")

    async def cleanup(self) -> None:
        """Clean up resources (no HTTP client to close for Claude Agent SDK)."""
        pass

    def supports_attempt_scoped_concurrency(self) -> bool:
        return True
