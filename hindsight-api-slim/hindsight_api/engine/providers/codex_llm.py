"""
OpenAI Codex LLM provider using ChatGPT Plus/Pro OAuth authentication.

This provider enables using ChatGPT Plus/Pro subscriptions for API calls
without separate OpenAI Platform API credits. It uses OAuth tokens from
~/.codex/auth.json and communicates with the ChatGPT backend API.

Tokens are refreshed automatically: the provider decodes the access_token
JWT's ``exp`` claim and proactively refreshes via
``POST https://auth.openai.com/oauth/token`` ~60s before expiry. It also
reactively refreshes once on a 401/403 from the Codex backend before giving
up. The refresh request shape mirrors the canonical ``@openai/codex`` CLI
implementation (codex-rs/login/src/auth/manager.rs on github.com/openai/codex)
so that future server-side changes affect both clients identically.
"""

import asyncio
import base64
import binascii
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


# OAuth refresh endpoint and client id, mirrored from the canonical
# ``@openai/codex`` CLI (codex-rs/login/src/auth/manager.rs on
# github.com/openai/codex). The endpoint is overridable via env var so that
# future Codex changes or staging environments can be pointed at without a
# code change — same env var name the upstream CLI uses.
_CODEX_REFRESH_TOKEN_URL = os.environ.get("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "https://auth.openai.com/oauth/token")
_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Proactively refresh this many seconds before the JWT ``exp`` claim. The
# upstream Codex CLI uses no skew (it refreshes at ``exp <= now``); the
# extra window reduces races where a request leaves the client with a token
# that the server has already declared expired by the time it arrives.
_CODEX_TOKEN_REFRESH_SKEW_SECONDS = 60

# OAuth error codes that the refresh endpoint returns when the refresh_token
# itself is no longer usable. These are terminal — retrying refresh will not
# succeed; the user must re-run ``codex auth login``.
_CODEX_TERMINAL_REFRESH_ERROR_CODES = frozenset(
    {"refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated"}
)


class CodexRefreshExpiredError(RuntimeError):
    """Raised when the Codex refresh_token itself is no longer valid.

    The user must re-run ``codex auth login`` to obtain new credentials.
    Callers should surface a clear remediation message and stop retrying.
    """


class CodexLLM(LLMInterface):
    """
    LLM provider using OpenAI Codex OAuth authentication.

    Authenticates using ChatGPT Plus/Pro credentials stored in ~/.codex/auth.json
    and makes API calls to chatgpt.com/backend-api/codex/responses.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,  # Will be ignored, reads from ~/.codex/auth.json
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """Initialize Codex LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Path is fixed at ~/.codex/auth.json — matches the upstream CLI.
        # Storing it on self lets the refresh path re-read after another
        # process (e.g. a sidecar) rotates the file out from under us.
        self._auth_file = Path.home() / ".codex" / "auth.json"

        # Single-flight refresh lock. Multiple concurrent requests racing
        # toward an expired token should produce one network refresh, not N.
        self._auth_lock = asyncio.Lock()

        # Load Codex OAuth credentials
        try:
            self.access_token, self.account_id = self._load_codex_auth()
            self.refresh_token = self._load_codex_refresh_token()
            logger.info(f"Loaded Codex OAuth credentials for account: {self.account_id}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Codex OAuth credentials from ~/.codex/auth.json: {e}\n\n"
                "To set up Codex authentication:\n"
                "1. Install Codex CLI: npm install -g @openai/codex\n"
                "2. Login: codex auth login\n"
                "3. Verify: ls ~/.codex/auth.json\n\n"
                "Or use a different provider (openai, anthropic, gemini) with API keys."
            ) from e

        # Use ChatGPT backend API endpoint
        if not self.base_url:
            self.base_url = "https://chatgpt.com/backend-api"

        # Normalize model name (strip openai/ prefix if present)
        if self.model.startswith("openai/"):
            self.model = self.model[len("openai/") :]

        # Map reasoning effort to Codex reasoning summary format
        # Codex supports: "auto", "concise", "detailed"
        self.reasoning_summary = self._map_reasoning_effort(reasoning_effort)

        # HTTP client for SSE streaming
        self._client = httpx.AsyncClient(timeout=120.0)

    def _load_codex_auth(self) -> tuple[str, str]:
        """
        Load OAuth credentials from ~/.codex/auth.json.

        Returns:
            Tuple of (access_token, account_id).

        Raises:
            FileNotFoundError: If auth file doesn't exist.
            ValueError: If auth file is invalid.
        """
        auth_file = Path.home() / ".codex" / "auth.json"

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
        """Load ``tokens.refresh_token`` from ``~/.codex/auth.json``.

        Returns None when the auth file is unreadable or omits the field —
        the provider still functions as a one-shot loader in that case, it
        just can't refresh when the access_token expires. This deliberately
        does not raise so that ``__init__`` keeps the existing failure mode
        of raising only on missing ``access_token``.
        """
        try:
            with open(self._auth_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Codex auth file unreadable when loading refresh_token: {type(e).__name__}. "
                "Token refresh will not be available; the access_token in memory will be used until it expires."
            )
            return None
        return data.get("tokens", {}).get("refresh_token")

    @staticmethod
    def _decode_jwt_exp_unixtime(token: str) -> int | None:
        """Return the JWT ``exp`` claim as a unix timestamp, or None on parse failure.

        ChatGPT/Codex access_tokens are JWTs whose payload includes ``exp``
        (RFC 7519). We need the expiry to schedule proactive refresh — the
        ``auth.json`` file does not persist a separate ``expires_at`` field
        in the upstream CLI's shape, so decoding the JWT itself is the
        canonical way to know when the token is stale.

        We do not verify the signature — the server is the source of truth
        on whether the token is actually accepted, and the only thing this
        method affects is the *timing* of refresh, not whether to trust the
        token contents.
        """
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload_b64 = parts[1]
            # JWT uses base64url without padding. Re-pad before decoding.
            padding = "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
            payload = json.loads(payload_bytes.decode("utf-8"))
            exp = payload.get("exp")
            return int(exp) if exp is not None else None
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return None

    def _token_is_stale(self, skew_seconds: int = _CODEX_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        """True when the cached access_token is past expiry (with skew).

        Returns False when expiry cannot be determined — we'd rather use a
        possibly-expired token and recover via the reactive 401 path than
        refresh aggressively on every request when ``exp`` parsing fails.
        """
        exp = self._decode_jwt_exp_unixtime(self.access_token)
        if exp is None:
            return False
        return exp <= int(time.time()) + skew_seconds

    def _persist_auth_atomic(self, updated_tokens: dict[str, Any]) -> None:
        """Write the rotated tokens back to ``~/.codex/auth.json`` atomically.

        Strategy: re-read the on-disk auth.json (so we don't clobber fields
        another process may have added), patch ``tokens.*`` and
        ``last_refresh``, write to a tempfile in the same directory with
        mode 0600, then ``os.replace`` onto the target. ``os.replace`` is
        atomic within the same filesystem on POSIX and Windows, so a
        concurrent reader will see either the old file or the fully-written
        new file — never a partial truncate, which is the upstream CLI's
        worst-case race.

        On non-Unix platforms the chmod is a best-effort no-op; the parent
        directory permissions still bound access.
        """
        current: dict[str, Any]
        try:
            with open(self._auth_file) as f:
                loaded = json.load(f)
            # auth.json should always be a JSON object at the top level; if
            # someone has hand-edited it into a non-object shape, fall back
            # to the minimal default rather than crashing the refresh path.
            current = loaded if isinstance(loaded, dict) else {"auth_mode": "chatgpt", "tokens": {}}
        except (OSError, json.JSONDecodeError):
            # If the file became unreadable between our last read and now,
            # construct a minimal shape rather than refusing to persist.
            current = {"auth_mode": "chatgpt", "tokens": {}}

        existing_tokens = current.get("tokens")
        tokens: dict[str, Any] = existing_tokens if isinstance(existing_tokens, dict) else {}
        for key in ("access_token", "refresh_token", "id_token", "account_id"):
            if key in updated_tokens and updated_tokens[key] is not None:
                tokens[key] = updated_tokens[key]
        current["tokens"] = tokens
        current["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Write to a sibling tempfile so the rename is same-filesystem.
        parent = self._auth_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".auth.", suffix=".json.tmp", dir=str(parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(current, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass  # best-effort on platforms that don't support chmod
            os.replace(tmp_path, self._auth_file)
        except Exception:
            # Clean up the orphaned tempfile if rename fails.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    async def _refresh_oauth_tokens(self, reason: str = "", *, force: bool = False) -> None:
        """Refresh the OAuth access_token using the stored refresh_token.

        Single-flight: serialized through ``self._auth_lock`` so concurrent
        callers produce one network request. The first caller refreshes; the
        rest wake up and observe that either (a) the in-memory token is no
        longer stale (proactive case) or (b) the in-memory token has changed
        since they entered (reactive case), and return without re-refreshing.

        Args:
            reason: Free-form string included in log lines for diagnostics.
            force: When True, refresh even if the JWT exp claim looks fresh.
                Used by the reactive 401 path — the server rejected the
                token, so we cannot trust the JWT's self-reported expiry.

        Raises:
            CodexRefreshExpiredError: when the server returns a terminal
                error code (refresh_token_expired/reused/invalidated) or any
                401 on the refresh endpoint itself.
            RuntimeError: for other refresh failures (network, 5xx, etc.).
        """
        # Capture the token we'd be refreshing BEFORE acquiring the lock so
        # that we can detect mid-wait rotation by another coroutine.
        token_before_lock = self.access_token
        async with self._auth_lock:
            if force:
                # Reactive: skip only if another coroutine already rotated
                # the token while we were waiting on the lock.
                if self.access_token != token_before_lock:
                    return
            else:
                # Proactive: skip if the token is no longer stale (the
                # canonical "another coroutine refreshed first" check).
                if not self._token_is_stale():
                    return

            if not self.refresh_token:
                raise RuntimeError(
                    "Codex access_token is expired but no refresh_token is available. "
                    "Run 'codex auth login' to re-authenticate."
                )

            log_reason = f" ({reason})" if reason else ""
            logger.info(f"Refreshing Codex OAuth access_token{log_reason}")

            request_body = {
                "client_id": _CODEX_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }
            try:
                response = await self._client.post(
                    _CODEX_REFRESH_TOKEN_URL,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
            except httpx.RequestError as e:
                raise RuntimeError(f"Codex OAuth refresh network error: {type(e).__name__}") from e

            if response.status_code == 401:
                # Classify by ``error.code`` (or top-level ``error`` string) — same
                # mapping as the upstream Rust CLI's request_chatgpt_token_refresh.
                error_code = self._extract_oauth_error_code(response)
                if error_code in _CODEX_TERMINAL_REFRESH_ERROR_CODES:
                    raise CodexRefreshExpiredError(
                        f"Codex refresh_token is permanently invalid (error.code={error_code}). "
                        "Run 'codex auth login' to re-authenticate."
                    )
                # Unknown 401 — treat as terminal too, matching the upstream classification.
                raise CodexRefreshExpiredError(
                    f"Codex OAuth refresh returned 401 with unrecognized error code "
                    f"({error_code or 'none'}). Run 'codex auth login' to re-authenticate."
                )

            if response.status_code >= 400:
                # 5xx and other 4xx are transient/retryable from the caller's
                # perspective; surface as RuntimeError without leaking the
                # request body in logs.
                raise RuntimeError(f"Codex OAuth refresh failed with HTTP {response.status_code}")

            try:
                body = response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Codex OAuth refresh returned non-JSON body: {e}") from e

            new_access = body.get("access_token")
            if not new_access:
                raise RuntimeError("Codex OAuth refresh returned no access_token")

            # The refresh_token may rotate on each refresh — adopt the new
            # one if the server sent it, otherwise keep the existing.
            new_refresh = body.get("refresh_token") or self.refresh_token
            new_id_token = body.get("id_token")

            # Update in-memory state first so callers waiting on the lock
            # see fresh credentials immediately, even if disk write fails.
            self.access_token = new_access
            self.refresh_token = new_refresh

            persisted = {
                "access_token": new_access,
                "refresh_token": new_refresh,
            }
            if new_id_token:
                persisted["id_token"] = new_id_token

            try:
                self._persist_auth_atomic(persisted)
            except OSError as e:
                # In-memory creds are valid; warn but don't fail the request
                # path. Future process starts will fall back to the stale
                # on-disk auth.json and immediately refresh.
                logger.warning(
                    f"Codex OAuth refresh succeeded but persisting auth.json failed: {type(e).__name__}. "
                    "In-memory credentials are up to date; on-disk file is stale."
                )

            logger.info("Codex OAuth access_token refreshed successfully")

    @staticmethod
    def _extract_oauth_error_code(response: "httpx.Response") -> str | None:
        """Pull the OAuth error code out of a 4xx response body, if present.

        The refresh endpoint returns shapes like
        ``{"error": "...", "error_code": "..."}`` or
        ``{"error": {"code": "..."}}``. We don't fail the call if the body
        is unparseable — the caller falls back to a generic "unknown" error.
        """
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        # Shape 1: error is a nested object with "code"
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                return code
        # Shape 2: top-level error_code string
        code = body.get("error_code")
        if isinstance(code, str):
            return code
        # Shape 3: error is itself a string code
        if isinstance(err, str):
            return err
        return None

    async def _ensure_fresh_token(self) -> None:
        """Refresh the access_token proactively if it is near or past expiry.

        Called at the top of every API-bound method. Cheap when the token is
        fresh (just decodes the JWT exp claim and returns).
        """
        if self._token_is_stale():
            try:
                await self._refresh_oauth_tokens(reason="proactive (token near expiry)")
            except CodexRefreshExpiredError:
                # Surface to the caller as the same RuntimeError shape the
                # request loop has historically raised, so existing error
                # handling paths keep working.
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
        """Make API call to Codex backend with SSE streaming."""
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

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
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

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Account-ID": self.account_id,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Origin": "https://chatgpt.com",
        }

        url = f"{self.base_url}/codex/responses"
        last_exception = None

        # Manual attempt tracking instead of ``for attempt in range(...)`` so
        # that the reactive-refresh path can retry once without consuming a
        # normal-retry budget slot. The refresh-retry is conceptually a
        # separate auth-recovery attempt that shouldn't compete with backoff.
        attempt = 0
        while True:
            try:
                response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)
                response.raise_for_status()

                # Parse SSE stream
                content = await self._parse_sse_stream(response)

                # Handle structured output
                if response_format is not None:
                    # Models may wrap JSON in markdown
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Codex JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        if attempt < max_retries:
                            backoff = min(initial_backoff * (2**attempt), max_backoff)
                            await asyncio.sleep(backoff)
                            last_exception = e
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
                last_exception = e
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
                last_exception = e
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

        if last_exception:
            raise last_exception
        raise RuntimeError("Codex call failed after all retries")

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

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Account-ID": self.account_id,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Origin": "https://chatgpt.com",
        }

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
        """Clean up HTTP client."""
        await self._client.aclose()
