"""
OpenAI Codex LLM provider using ChatGPT Plus/Pro OAuth authentication.

This provider enables using ChatGPT Plus/Pro subscriptions for API calls
without separate OpenAI Platform API credits. It uses OAuth tokens from the
Codex ``auth.json`` (``$CODEX_HOME/auth.json``, or ``~/.codex/auth.json`` when
``CODEX_HOME`` is unset) and communicates with the ChatGPT backend API.

Tokens are refreshed automatically: the provider decodes the access_token
JWT's ``exp`` claim and proactively refreshes via
``POST https://auth.openai.com/oauth/token`` ~60s before expiry. It also
reactively refreshes once on a 401/403 from the Codex backend before giving
up. The refresh request shape mirrors the canonical ``@openai/codex`` CLI
implementation (codex-rs/login/src/auth/manager.rs on github.com/openai/codex)
so that future server-side changes affect both clients identically.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.metrics import get_metrics_collector

from .codex_auth import (
    _CODEX_CLIENT_ID,
    _CODEX_REFRESH_TOKEN_URL,
    _CODEX_TERMINAL_REFRESH_ERROR_CODES,
    _CODEX_TOKEN_REFRESH_SKEW_SECONDS,
    CodexAuthManager,
    CodexRefreshExpiredError,
    default_codex_auth_file,
)

# Re-export for backward compatibility (tests import from this module).
__all__ = [
    "CodexLLM",
    "CodexRefreshExpiredError",
    "CodexAuthManager",
    "_CODEX_REFRESH_TOKEN_URL",
    "_CODEX_CLIENT_ID",
    "_CODEX_TOKEN_REFRESH_SKEW_SECONDS",
    "_CODEX_TERMINAL_REFRESH_ERROR_CODES",
]

logger = logging.getLogger(__name__)

# Newer Codex models are gated on the first-party client identity; the previous
# browser-shaped User-Agent returned "Model not found" for Luna (#2643).
# Use a neutral version because Hindsight must not claim a specific Codex release.
_CODEX_ORIGINATOR = "codex_cli_rs"
_CODEX_USER_AGENT = "codex_cli_rs/0.0.0 (Hindsight)"

# Name of the single forced function tool used to carry structured output when
# strict_schema is on. The Codex backend speaks the OpenAI Responses API, so a
# forced function call gives us constrained decoding straight into the response
# schema — no prompt-injected schema, no raw json.loads on free-form model text,
# no invalid-\escape retry storm (issue #2504, same class as #1002 / #2339).
_STRUCTURED_TOOL_NAME = "structured_response"

# Valid JSON string escape characters (the char that may follow a backslash).
_VALID_JSON_ESCAPE_CHARS = set('"\\/bfnrtu')


def _repair_invalid_json_escapes(text: str) -> str:
    """Best-effort repair of invalid ``\\escape`` sequences in a JSON string.

    Escape-heavy content (code, serial/CLI commands, Windows paths, regexes)
    makes weaker models emit backslashes that aren't valid JSON escapes (e.g.
    ``\\d``, ``\\s``, ``C:\\Users``), so ``json.loads`` fails deterministically
    and every retry re-fails the same way (issue #2504). This doubles any
    backslash that isn't part of a valid escape so the payload parses. It is a
    lenient fallback only — the strict_schema forced-tool path is the real fix.
    """
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in _VALID_JSON_ESCAPE_CHARS:
                # Preserve the valid escape (both chars) verbatim.
                result.append(ch)
                result.append(nxt)
                i += 2
                continue
            # Invalid escape: escape the lone backslash so JSON parses.
            result.append("\\\\")
            i += 1
            continue
        if ch == "\\" and i + 1 == n:
            # Trailing lone backslash — escape it.
            result.append("\\\\")
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


class CodexLLM(LLMInterface):
    """
    LLM provider using OpenAI Codex OAuth authentication.

    Authenticates using ChatGPT Plus/Pro credentials stored in the Codex
    ``auth.json`` (honoring ``CODEX_HOME``, default ``~/.codex``) and makes API
    calls to chatgpt.com/backend-api/codex/responses.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,  # Will be ignored, reads from the Codex auth.json (CODEX_HOME or ~/.codex)
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """Initialize Codex LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Single-flight async refresh lock. Multiple concurrent coroutines
        # racing toward an expired token should produce one network refresh.
        self._auth_lock = asyncio.Lock()

        # Load Codex OAuth credentials (keep these methods for test patching).
        try:
            access_token, account_id = self._load_codex_auth()
            refresh_token = self._load_codex_refresh_token()
            logger.info(f"Loaded Codex OAuth credentials for account: {account_id}")
        except Exception as e:
            auth_file = default_codex_auth_file()
            raise RuntimeError(
                f"Failed to load Codex OAuth credentials from {auth_file}: {e}\n\n"
                "To set up Codex authentication:\n"
                "1. Install Codex CLI: npm install -g @openai/codex\n"
                "2. Login: codex auth login\n"
                f"3. Verify: ls {auth_file}\n\n"
                "(Set CODEX_HOME to use a credentials directory other than ~/.codex.)\n\n"
                "Or use a different provider (openai, anthropic, gemini) with API keys."
            ) from e

        self._auth_manager = CodexAuthManager(
            access_token=access_token,
            account_id=account_id,
            refresh_token=refresh_token,
            auth_file=default_codex_auth_file(),
        )

        # Use ChatGPT backend API endpoint. Codex auth is tied to
        # chatgpt.com/backend-api, not the OpenAI-compatible base URL used by
        # other providers. Deployments often set a global LLM_BASE_URL for an
        # OpenAI-compatible proxy; ignore that inherited value unless the user
        # explicitly provides a Codex backend URL.
        if not self.base_url or self.base_url.rstrip("/").endswith("/v1"):
            self.base_url = "https://chatgpt.com/backend-api"
        else:
            self.base_url = self.base_url.rstrip("/")

        # Normalize model name (strip openai/ prefix if present)
        if self.model.startswith("openai/"):
            self.model = self.model[len("openai/") :]

        # Map reasoning effort to Codex reasoning summary format
        # Codex supports: "auto", "concise", "detailed"
        self.reasoning_summary = self._map_reasoning_effort(reasoning_effort)

        # HTTP client for SSE streaming
        self._client = httpx.AsyncClient(timeout=120.0)

    # ------------------------------------------------------------------
    # Properties — delegate to _auth_manager (preserves test-visible API)
    # ------------------------------------------------------------------

    @property
    def access_token(self) -> str:
        return self._auth_manager.access_token

    @access_token.setter
    def access_token(self, v: str) -> None:
        self._auth_manager.access_token = v

    @property
    def account_id(self) -> str:
        return self._auth_manager.account_id

    def _build_request_headers(self) -> httpx.Headers:
        return httpx.Headers(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "OpenAI-Account-ID": self.account_id,
                "User-Agent": _CODEX_USER_AGENT,
                "Origin": "https://chatgpt.com",
                "originator": _CODEX_ORIGINATOR,
            }
        )

    @property
    def refresh_token(self) -> str | None:
        return self._auth_manager.refresh_token

    @refresh_token.setter
    def refresh_token(self, v: str | None) -> None:
        self._auth_manager.refresh_token = v

    @property
    def _auth_file(self) -> Path:
        return self._auth_manager._auth_file

    @_auth_file.setter
    def _auth_file(self, v: Path) -> None:
        self._auth_manager._auth_file = v

    # ------------------------------------------------------------------
    # Forwarding methods (keep surface area for tests / subclasses)
    # ------------------------------------------------------------------

    def _load_codex_auth(self) -> tuple[str, str]:
        """
        Load OAuth credentials from the Codex ``auth.json`` (CODEX_HOME or ~/.codex).

        Returns:
            Tuple of (access_token, account_id).

        Raises:
            FileNotFoundError: If auth file doesn't exist.
            ValueError: If auth file is invalid.
        """
        auth_file = default_codex_auth_file()

        if not auth_file.exists():
            raise FileNotFoundError(
                f"Codex auth file not found: {auth_file}\nRun 'codex auth login' to authenticate with ChatGPT Plus/Pro."
            )

        with open(auth_file) as f:
            data = json.load(f)

        # Validate auth structure
        auth_mode = data.get("auth_mode")
        if auth_mode != "chatgpt":
            raise ValueError(f"Expected auth_mode='chatgpt', got: {auth_mode}")

        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id")

        if not access_token:
            raise ValueError("No access_token found in Codex auth file. Run 'codex auth login' again.")

        return access_token, account_id

    def _load_codex_refresh_token(self) -> str | None:
        """Read ``tokens.refresh_token`` from the configured auth file.

        Kept as an instance method so existing tests that patch
        ``CodexLLM._load_codex_refresh_token`` continue to work. Works both
        pre- and post-``__init__`` because it does not depend on
        ``_auth_manager`` being constructed yet.
        """
        auth_file = self._auth_manager._auth_file if hasattr(self, "_auth_manager") else default_codex_auth_file()
        return CodexAuthManager.load_refresh_token_from_file(auth_file)

    @staticmethod
    def _decode_jwt_exp_unixtime(token: str) -> int | None:
        """Delegate to ``CodexAuthManager._decode_jwt_exp_unixtime``."""
        return CodexAuthManager._decode_jwt_exp_unixtime(token)

    def _token_is_stale(self, skew_seconds: int = _CODEX_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        """Delegate to ``_auth_manager._token_is_stale``."""
        return self._auth_manager._token_is_stale(skew_seconds)

    def _persist_auth_atomic(self, updated_tokens: dict[str, Any]) -> None:
        """Delegate to ``_auth_manager._persist_auth_atomic``."""
        return self._auth_manager._persist_auth_atomic(updated_tokens)

    async def _refresh_oauth_tokens(self, reason: str = "", *, force: bool = False) -> None:
        """Async single-flight OAuth token refresh.

        Outer asyncio.Lock preserves single-flight semantics for concurrent
        coroutines; the actual network call is offloaded to a thread via
        ``asyncio.to_thread`` so the event loop stays unblocked.

        Args:
            reason: Free-form string included in log lines for diagnostics.
            force: When True, refresh even if the JWT exp claim looks fresh.
                Used by the reactive 401 path.

        Raises:
            CodexRefreshExpiredError: when the server returns a terminal
                error code or any 401 on the refresh endpoint.
            RuntimeError: for other refresh failures (network, 5xx, etc.).
        """
        token_before_lock = self.access_token
        async with self._auth_lock:
            if force:
                if self.access_token != token_before_lock:
                    return
            else:
                if not self._auth_manager._token_is_stale():
                    return
            await asyncio.to_thread(lambda: self._auth_manager.refresh_tokens(reason, force=force))

    async def _ensure_fresh_token(self) -> None:
        """Refresh the access_token proactively if it is near or past expiry.

        Called at the top of every API-bound method. Cheap when the token is
        fresh (just decodes the JWT exp claim and returns).
        """
        if self._auth_manager._token_is_stale():
            try:
                await self._refresh_oauth_tokens(reason="proactive (token near expiry)")
            except CodexRefreshExpiredError:
                raise

    def _map_reasoning_effort(self, effort: str) -> str:
        """
        Map standard reasoning effort to Codex reasoning summary format.

        Args:
            effort: Standard effort level ("low", "medium", "high", "xhigh").

        Returns:
            Codex reasoning summary: "concise", "detailed", or "auto".
        """
        mapping = {
            "low": "concise",
            "medium": "auto",
            "high": "detailed",
            "xhigh": "detailed",
        }
        return mapping.get(effort.lower(), "auto")

    def _normalize_tool_choice(self, tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
        """Normalize forced function tool choice for the Codex Responses API.

        Older agent paths may still pass OpenAI chat-completions style named
        tool choice payloads such as:

            {"type": "function", "function": {"name": "recall"}}

        Codex Responses expects the named function at the top level instead:

            {"type": "function", "name": "recall"}
        """
        if not isinstance(tool_choice, dict):
            return tool_choice
        if str(tool_choice.get("type") or "").strip() != "function":
            return tool_choice
        function_payload = tool_choice.get("function")
        if isinstance(function_payload, dict):
            function_name = str(function_payload.get("name") or "").strip()
            if function_name:
                return {"type": "function", "name": function_name}
        function_name = str(tool_choice.get("name") or "").strip()
        if function_name:
            return {"type": "function", "name": function_name}
        return tool_choice

    async def verify_connection(self) -> None:
        """Verify Codex connection by making a simple test call."""
        try:
            logger.info(f"Verifying Codex LLM: model={self.model}, account={self.account_id}...")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=10,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info(f"Codex LLM verified: {self.model}")
        except Exception as e:
            # 429 means quota exhausted, not a configuration error — warn but allow startup
            if "429" in str(e) or "usage_limit_reached" in str(e):
                logger.warning(f"Codex LLM quota exhausted for {self.model}, continuing startup: {e}")
                return
            raise RuntimeError(f"Codex LLM connection verification failed for {self.model}: {e}") from e

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
        return_usage: bool = False,
    ) -> Any:
        """Make API call to Codex backend with SSE streaming.

        Args:
            strict_schema: Route structured output through a single forced
                function tool (constrained decoding) instead of prompt-injecting
                the schema and parsing free-form text. The Codex backend speaks
                the OpenAI Responses API, so the forced function call emits the
                response schema directly as tool arguments — eliminating the
                invalid-``\\escape`` retry storm (issue #2504). When False, falls
                back to schema-in-prompt + JSON parse, now hardened with a lenient
                invalid-escape repair before giving up.
        """
        start_time = time.time()

        # Proactively refresh the OAuth access_token if it's near expiry.
        # Cheap when fresh: a JWT exp decode + comparison.
        await self._ensure_fresh_token()

        # Tracks whether we've already attempted a reactive refresh in
        # response to a 401 from the backend. Set once on the first auth
        # failure so we retry exactly once after refresh, not in a loop.
        attempted_refresh_after_auth_error = False

        # Prepare system instructions
        system_instruction = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction += ("\n\n" + content) if system_instruction else content
            else:
                user_messages.append(msg)

        # Structured output: prefer a single forced function tool (constrained
        # decoding) over text-injecting the schema and parsing the reply. The
        # forced tool guarantees schema-shaped JSON in the tool arguments,
        # eliminating the invalid-\escape retry storm (issue #2504). When
        # strict_schema is off we keep the schema-in-prompt + json.loads
        # fallback (now hardened with a lenient escape repair) for callers that
        # can't force tools.
        schema = None
        use_forced_tool = False
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = strict_json_schema(response_format) if strict_schema else response_format.model_json_schema()
            if strict_schema:
                use_forced_tool = True
            else:
                schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"
                system_instruction += schema_msg

        # gpt-5.2-codex only supports "detailed" reasoning summary
        reasoning_summary = "detailed" if "5.2" in self.model else self.reasoning_summary

        # Build Codex request payload
        payload = {
            "model": self.model,
            "instructions": system_instruction,
            "input": [
                {
                    "type": "message",
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
                for msg in user_messages
            ],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"summary": reasoning_summary},
            "store": False,  # Codex uses stateless mode
            "stream": True,  # SSE streaming
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        if use_forced_tool and schema is not None:
            # Single function tool whose parameters ARE the response schema;
            # force it via tool_choice so the backend does constrained decoding.
            payload["tools"] = [
                {
                    "type": "function",
                    "name": _STRUCTURED_TOOL_NAME,
                    "description": "Return the structured response.",
                    "parameters": schema,
                }
            ]
            payload["tool_choice"] = {"type": "function", "name": _STRUCTURED_TOOL_NAME}
            payload["parallel_tool_calls"] = False

        headers = self._build_request_headers()

        url = f"{self.base_url}/codex/responses"

        # Manual attempt tracking instead of ``for attempt in range(...)`` so
        # that the reactive-refresh path can retry once without consuming a
        # normal-retry budget slot. The refresh-retry is conceptually a
        # separate auth-recovery attempt that shouldn't compete with backoff.
        attempt = 0
        while True:
            try:
                response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)
                response.raise_for_status()

                # Forced-tool path: read structured output from the function-call
                # arguments (already a JSON string in a dedicated channel) rather
                # than from free-form assistant text.
                if use_forced_tool:
                    text_content, tool_calls = await self._parse_sse_tool_stream(response)
                    content = text_content or ""
                else:
                    tool_calls = []
                    content = await self._parse_sse_stream(response)

                # Codex SSE carries no usage block; stash the same char/4 estimate
                # the success path traces so a later parse/validate failure records
                # consistent (estimated) token counts rather than zero (#2387).
                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
                        output_tokens=len(content) // 4,
                    )
                )

                # Handle structured output
                if use_forced_tool:
                    tool_input = None
                    for tc in tool_calls:
                        if tc.name == _STRUCTURED_TOOL_NAME:
                            tool_input = tc.arguments if isinstance(tc.arguments, dict) else None
                            break
                    if tool_input is None:
                        # Model ignored the forced tool (rare — e.g. a gateway that
                        # drops tool_choice). Retry so we don't hard-fail.
                        logger.warning(
                            f"Codex forced structured tool missing from response "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        if attempt < max_retries:
                            backoff = min(initial_backoff * (2**attempt), max_backoff)
                            await asyncio.sleep(backoff)
                            attempt += 1
                            continue
                        raise RuntimeError("Codex did not return the forced structured_response tool call")
                    content = json.dumps(tool_input)
                    result = tool_input if skip_validation else response_format.model_validate(tool_input)
                elif response_format is not None:
                    # Models may wrap JSON in markdown
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        # Escape-heavy content deterministically re-fails every
                        # retry (issue #2504). Try a lenient invalid-escape repair
                        # before burning a retry / re-raising.
                        try:
                            json_data = json.loads(_repair_invalid_json_escapes(clean_content))
                            logger.info("Codex JSON parsed after repairing invalid escape sequences")
                        except json.JSONDecodeError:
                            logger.warning(f"Codex JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                            if attempt < max_retries:
                                backoff = min(initial_backoff * (2**attempt), max_backoff)
                                await asyncio.sleep(backoff)
                                attempt += 1
                                continue
                            raise

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=0,  # Codex doesn't report token counts in SSE
                    output_tokens=0,
                    success=True,
                )

                # Record trace span
                try:
                    from hindsight_api.tracing import get_span_recorder

                    # Estimate tokens for tracing
                    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
                    estimated_output = len(content) // 4
                    span_recorder = get_span_recorder()
                    span_recorder.record_llm_call(
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                        messages=messages,
                        response_content=result if isinstance(result, str) else result.model_dump_json(),
                        input_tokens=estimated_input,
                        output_tokens=estimated_output,
                        duration=duration,
                        finish_reason=None,
                        error=None,
                    )
                except Exception:
                    pass  # logging failure must never affect the operation

                if return_usage:
                    # Codex doesn't provide token counts, estimate based on content
                    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
                    estimated_output = len(content) // 4
                    token_usage = TokenUsage(
                        input_tokens=estimated_input,
                        output_tokens=estimated_output,
                        total_tokens=estimated_input + estimated_output,
                    )
                    return result, token_usage

                return result

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Auth error: try one OAuth refresh + retry before giving up.
                # The proactive refresh at the top of this method catches most
                # expiries, but a token can also become invalid mid-request if
                # another process rotates auth.json out from under us, or if
                # the JWT exp claim is unparseable and we never knew it was
                # stale. Reactive refresh is the safety net.
                if status_code in (401, 403):
                    if not attempted_refresh_after_auth_error:
                        attempted_refresh_after_auth_error = True
                        try:
                            await self._refresh_oauth_tokens(
                                reason=f"reactive (HTTP {status_code} from codex backend)",
                                force=True,
                            )
                            # Rebuild the Authorization header with the new
                            # token and retry without consuming a normal-retry
                            # budget slot — this is a dedicated auth-recovery
                            # attempt that shouldn't compete with backoff.
                            headers["Authorization"] = f"Bearer {self.access_token}"
                            logger.info("Codex auth refreshed after auth error; retrying request once")
                            continue
                        except CodexRefreshExpiredError as refresh_err:
                            logger.error("Codex refresh_token is permanently invalid; cannot recover from auth error")
                            raise RuntimeError(
                                "Codex authentication failed and the refresh_token is no longer valid.\n"
                                "Run 'codex auth login' to re-authenticate."
                            ) from refresh_err
                        except Exception as refresh_err:
                            logger.error(
                                f"Codex token refresh attempt failed: {type(refresh_err).__name__}: {refresh_err}"
                            )
                            # Fall through to the original raise below.
                    logger.error(f"Codex auth error (HTTP {status_code}): {e.response.text[:200]}")
                    raise RuntimeError(
                        "Codex authentication failed. Your OAuth token may have expired.\n"
                        "Run 'codex auth login' to re-authenticate."
                    ) from e

                # Log the actual error message from the API
                error_detail = e.response.text[:500] if hasattr(e.response, "text") else str(e)

                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(
                        f"Codex HTTP error {status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_detail}"
                    )
                    await asyncio.sleep(backoff)
                    attempt += 1
                    continue
                else:
                    logger.error(
                        f"Codex HTTP error after {max_retries + 1} attempts: Status {status_code}, Detail: {error_detail}"
                    )
                    raise

            except httpx.RequestError as e:
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(f"Codex connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(backoff)
                    attempt += 1
                    continue
                else:
                    logger.error(f"Codex connection error after {max_retries + 1} attempts: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected Codex error: {type(e).__name__}: {e}")
                raise

    async def _parse_sse_stream(self, response: httpx.Response) -> str:
        """
        Parse Server-Sent Events (SSE) stream from Codex API.

        Args:
            response: HTTP response with SSE stream.

        Returns:
            Extracted text content from stream.
        """
        full_text = ""
        event_type = None

        async for line in response.aiter_lines():
            if not line:
                continue

            # Track event type
            if line.startswith("event: "):
                event_type = line[7:]

            # Parse data
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract content based on event type
                    if event_type == "response.text.delta" and "delta" in data:
                        full_text += data["delta"]
                    elif event_type == "response.content_part.delta" and "delta" in data:
                        full_text += data["delta"]
                    # Check for item content
                    elif "item" in data:
                        item = data["item"]
                        if "content" in item:
                            content = item["content"]
                            if isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and "text" in part:
                                        full_text += part["text"]
                            elif isinstance(content, str):
                                full_text += content

                except json.JSONDecodeError:
                    # Skip malformed JSON events
                    pass

        return full_text

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """
        Make API call with tool calling support.

        Parses Codex SSE stream to extract tool calls from response.output_item.done events.
        Tools are converted from OpenAI format to Codex format (flat structure at top level).

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools - "auto", "none", "required", or a specific function.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        start_time = time.time()

        # Proactively refresh the OAuth access_token if it's near expiry.
        # Same rationale as in ``call()`` — keeps the request from leaving
        # the client carrying a token that's already past ``exp``.
        await self._ensure_fresh_token()

        # Prepare system instructions
        system_instruction = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction += ("\n\n" + content) if system_instruction else content
            elif role == "tool":
                # Handle tool results
                user_messages.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": f"Tool result: {content}",
                    }
                )
            else:
                user_messages.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content,
                    }
                )

        # Convert tools to Codex format
        # Codex expects tools with type and name/description/parameters at top level
        codex_tools = []
        for tool in tools:
            func = tool.get("function", {})
            codex_tools.append(
                {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )

        # gpt-5.2-codex only supports "detailed" reasoning summary
        reasoning_summary = "detailed" if "5.2" in self.model else self.reasoning_summary

        payload = {
            "model": self.model,
            "instructions": system_instruction,
            "input": user_messages,
            "tools": codex_tools,
            "tool_choice": self._normalize_tool_choice(tool_choice),
            "parallel_tool_calls": True,
            "reasoning": {"summary": reasoning_summary},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        headers = self._build_request_headers()

        url = f"{self.base_url}/codex/responses"

        # Debug logging for troubleshooting
        logger.debug(f"Codex tool call request: url={url}, model={payload['model']}, tools={len(codex_tools)}")

        # One reactive refresh attempt on auth failure, mirroring call().
        # ``call_with_tools`` doesn't have a retry loop, so we hand-roll a
        # single retry after refreshing the token. Any non-auth error still
        # surfaces immediately to keep behavior identical for callers.
        attempted_refresh_after_auth_error = False

        try:
            response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)

            if response.status_code in (401, 403) and not attempted_refresh_after_auth_error:
                attempted_refresh_after_auth_error = True
                try:
                    await self._refresh_oauth_tokens(
                        reason=f"reactive (HTTP {response.status_code} from codex backend in call_with_tools)",
                        force=True,
                    )
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    logger.info("Codex auth refreshed after auth error; retrying tool-call request once")
                    response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)
                except CodexRefreshExpiredError as refresh_err:
                    logger.error(
                        "Codex refresh_token is permanently invalid; cannot recover from auth error in tool-call path"
                    )
                    raise RuntimeError(
                        "Codex authentication failed and the refresh_token is no longer valid.\n"
                        "Run 'codex auth login' to re-authenticate."
                    ) from refresh_err
                except Exception as refresh_err:
                    logger.error(
                        f"Codex token refresh attempt failed in tool-call path: {type(refresh_err).__name__}: {refresh_err}"
                    )
                    # Fall through to the normal error path below.

            # Log response details on error
            if response.status_code != 200:
                logger.error(f"Codex API error {response.status_code}: {response.text[:500]}")

            response.raise_for_status()

            # Parse SSE for tool calls and content
            content, tool_calls = await self._parse_sse_tool_stream(response)

            duration = time.time() - start_time
            metrics = get_metrics_collector()
            metrics.record_llm_call(
                provider=self.provider,
                model=self.model,
                scope=scope,
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                success=True,
            )

            # Record OpenTelemetry span
            try:
                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                # Convert LLMToolCall objects to dicts for span recording
                tool_calls_dict = (
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                )
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    messages=messages,
                    response_content=content,
                    input_tokens=0,  # Codex doesn't provide token counts
                    output_tokens=0,
                    duration=duration,
                    finish_reason="tool_calls" if tool_calls else "stop",
                    error=None,
                    tool_calls=tool_calls_dict,
                )
            except Exception:
                pass  # logging failure must never affect the operation

            return LLMToolCallResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                input_tokens=0,
                output_tokens=0,
            )

        except Exception as e:
            logger.error(f"Codex tool call error: {e}")
            raise

    async def _parse_sse_tool_stream(self, response: httpx.Response) -> tuple[str | None, list[LLMToolCall]]:
        """
        Parse SSE stream for tool calls and content.

        Returns:
            Tuple of (content, tool_calls).
        """
        content = ""
        tool_calls: list[LLMToolCall] = []
        event_type = None

        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith("event: "):
                event_type = line[7:]

            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract text content
                    if event_type == "response.text.delta" and "delta" in data:
                        content += data["delta"]

                    # Extract completed tool calls from response.output_item.done
                    elif event_type == "response.output_item.done":
                        item = data.get("item", {})
                        if item.get("type") == "function_call" and item.get("status") == "completed":
                            tool_name = item.get("name", "")
                            arguments_str = item.get("arguments", "{}")
                            call_id = item.get("call_id", "")

                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                # Escape-heavy content can emit invalid \escape
                                # sequences (issue #2504); repair before giving up.
                                try:
                                    arguments = json.loads(_repair_invalid_json_escapes(arguments_str))
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                                    arguments = {}

                            tool_calls.append(
                                LLMToolCall(
                                    id=call_id,
                                    name=tool_name,
                                    arguments=arguments,
                                )
                            )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse SSE data: {e}, data_str: {data_str[:200]}")

        return content if content else None, tool_calls

    async def cleanup(self) -> None:
        """Clean up HTTP clients."""
        await self._client.aclose()
        self._auth_manager.close()
