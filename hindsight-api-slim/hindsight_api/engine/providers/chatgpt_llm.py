"""``chatgpt`` provider: the public Responses API billed to the user's ChatGPT plan.

Uses the OAuth access token from OpenAI's "Sign in with ChatGPT" flow for
open-source apps (see ``chatgpt_auth.py``) as the bearer credential for
``POST https://api.openai.com/v1/responses``. Request translation, structured
output, tool calls and usage accounting are inherited from
``OpenAIResponsesLLM``; this class only changes the transport:

- every request is sent with ``store: false`` and ``stream: true`` and is
  successful only on ``response.completed``,
- fields the preview route rejects are stripped and ``system`` messages are sent
  as ``developer`` messages,
- the token is refreshed before each call, and once more on a 401,
- ChatGPT plan error codes map onto Hindsight's handling: a usage-limit hit defers
  the operation (``ProviderRateLimitResetError``), temporary unavailability
  retries with backoff, and unsupported-capability / permission errors fail
  without retrying.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError

from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.providers.chatgpt_auth import (
    RESOURCE,
    ChatGPTAuth,
    ChatGPTAuthError,
    ChatGPTLoginRequiredError,
    default_home,
)
from hindsight_api.engine.providers.openai_compatible_llm import _rate_limit_retry_at
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM, RetryableResponsesError
from hindsight_api.engine.remote_retry import is_transient_remote_error

logger = logging.getLogger(__name__)

# The only endpoint this flow supports. Never inherited from HINDSIGHT_API_LLM_BASE_URL,
# which would hand the ChatGPT bearer token to whatever proxy that names.
_API_BASE_URL = RESOURCE

# Request fields the ChatGPT plan preview route rejects. ``service_tier`` is not on
# the documented list but is named as an unsupported override in the error docs.
_UNSUPPORTED_FIELDS = frozenset(
    {
        "background",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "multi_agent",
        "prompt",
        "prompt_cache_retention",
        "safety_identifier",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "previous_response_id",
        "service_tier",
    }
)

USAGE_LIMIT_CODE = "subscription_sharing_usage_limit_exceeded"
UNSUPPORTED_CAPABILITY_CODE = "subscription_sharing_unsupported_capability"
# Permission/policy failures: retrying the same request cannot succeed.
_NON_RETRYABLE_CODES = frozenset(
    {
        "subscription_sharing_user_not_eligible",
        "subscription_sharing_route_not_supported",
        "subscription_sharing_invalid_user",
        "chatpass_v2_scope_not_authorized",
        "chatpass_v2_invalid_authorization_context",
    }
)

# Failure codes inside a stream that a later attempt can clear. Any other code (an
# invalid or too-long prompt, a policy refusal) fails the same way every time, and
# retrying it would only spend the user's shared plan quota.
_TRANSIENT_STREAM_CODES = frozenset(
    {
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
        "server_error",
        "rate_limit_exceeded",
    }
)

# A usage-limit error carries no reset time and OpenAI says not to infer one, so
# deferred work comes back after this pause (or a later Retry-After, if sent).
USAGE_LIMIT_DEFER_SECONDS = 30 * 60
USAGE_URL = "https://chatgpt.com/settings/usage"


class ChatGPTRequestError(RuntimeError):
    """The ChatGPT plan route refused the request in a way retrying will not fix."""


def _plan_error(
    code: str | None, message: str, param: str | None, *, label: str, retry_at: datetime | None = None
) -> Exception | None:
    """Map a ChatGPT plan error code to the exception to raise, or None when it is not plan-specific."""
    if code == USAGE_LIMIT_CODE:
        pause_until = datetime.now(UTC) + timedelta(seconds=USAGE_LIMIT_DEFER_SECONDS)
        resume_at = max(retry_at, pause_until) if retry_at else pause_until
        return ProviderRateLimitResetError(
            retry_at=resume_at,
            message=(
                f"ChatGPT plan usage limit reached ({label}); pausing until {resume_at.isoformat()}. "
                f"Check {USAGE_URL}. {message}"
            ),
        )
    if code == UNSUPPORTED_CAPABILITY_CODE:
        return ChatGPTRequestError(
            f"ChatGPT plan route does not support part of this request ({label}, param={param!r}): {message}"
        )
    if code in _NON_RETRYABLE_CODES:
        return ChatGPTRequestError(f"ChatGPT plan usage refused ({label}, code={code}): {message}")
    return None


def _stream_failure(code: str | None, message: str, param: str | None, *, label: str) -> Exception:
    """Exception for a failure reported inside the stream (``response.failed`` / ``error``)."""
    mapped = _plan_error(code, message, param, label=label)
    if mapped is not None:
        return mapped
    if code is None or code in _TRANSIENT_STREAM_CODES:
        return RetryableResponsesError(f"ChatGPT plan response failed ({label}, code={code}): {message}")
    return ChatGPTRequestError(f"ChatGPT plan response failed ({label}, code={code}): {message}")


def _plan_request_body(params: dict[str, Any]) -> dict[str, Any]:
    """Adapt a built Responses request to the ChatGPT plan route's preview limits."""
    body = {k: v for k, v in params.items() if k not in _UNSUPPORTED_FIELDS}
    if isinstance(body.get("extra_body"), dict):
        # ``store``/``stream`` too: the route requires the values forced below.
        body["extra_body"] = {
            k: v for k, v in body["extra_body"].items() if k not in _UNSUPPORTED_FIELDS and k not in ("store", "stream")
        }
    body["input"] = [
        {**item, "role": "developer"} if isinstance(item, dict) and item.get("role") == "system" else item
        for item in body.get("input", [])
    ]
    body["store"] = False
    body["stream"] = True
    return body


class ChatGPTLLM(OpenAIResponsesLLM):
    """OpenAI Responses API on a ChatGPT plan via Sign in with ChatGPT credentials."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # Loaded first: the parent requires an api_key, and a missing login should
        # fail at startup with the command that fixes it.
        try:
            self._auth = ChatGPTAuth.shared(default_home())
        except ChatGPTAuthError as e:
            raise RuntimeError(f"chatgpt provider unavailable: {e}") from e
        if model.startswith("openai/"):
            model = model[len("openai/") :]
        super().__init__(
            provider=provider,
            # Placeholder; each request carries the current OAuth access token instead.
            api_key="chatgpt-oauth",
            base_url=_API_BASE_URL,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            extra_body=extra_body,
        )
        # Drop the OpenAI-Organization / OpenAI-Project headers the SDK would add from
        # OPENAI_ORG_ID / OPENAI_PROJECT_ID: those name an API-key org, not this grant.
        self._client.organization = None
        self._client.project = None

    def _supports_reasoning_model(self) -> bool:
        # Every model a ChatGPT plan serves here is a reasoning model.
        return True

    async def _create_response(self, params: dict[str, Any]) -> Any:
        body = _plan_request_body(params)
        try:
            token = await self._auth.get_access_token()
            try:
                return await self._stream(body, token)
            except AuthenticationError as e:
                # A ChatGPT plan code means the token was accepted but plan usage was
                # refused; rotating the refresh token would not change that.
                if str(getattr(e, "code", None) or "").startswith(("subscription_sharing_", "chatpass_")):
                    raise
                logger.info("ChatGPT plan request returned 401; refreshing the access token once")
                await self._auth.refresh(rejected_token=token)
                return await self._stream(body, await self._auth.get_access_token())
        except ChatGPTLoginRequiredError:
            raise
        except ChatGPTAuthError as e:
            raise RetryableResponsesError(str(e)) from e
        except APIStatusError as e:
            mapped = _plan_error(
                getattr(e, "code", None),
                e.message,
                getattr(e, "param", None),
                label=f"{self.provider}/{self.model}, HTTP {e.status_code}",
                retry_at=_rate_limit_retry_at(e) if e.status_code == 429 else None,
            )
            if mapped is not None:
                raise mapped from e
            raise
        except APIConnectionError:
            raise
        except APIError as e:
            # The SDK raises a bare APIError for a stream event whose payload is
            # ``{"error": {...}}`` rather than a typed ``error`` event.
            label = f"{self.provider}/{self.model}"
            raise _stream_failure(getattr(e, "code", None), e.message, getattr(e, "param", None), label=label) from e

    async def _stream(self, body: dict[str, Any], token: str) -> Any:
        """Stream one request to its terminal event and return the final ``Response``."""
        label = f"{self.provider}/{self.model}"
        try:
            return await self._consume(body, token, label)
        except (APIError, RetryableResponsesError, ChatGPTRequestError, ProviderRateLimitResetError):
            raise
        except Exception as e:
            # A connection dropped mid-stream surfaces as a raw transport error: the SDK
            # only wraps failures that happen while sending the request.
            if is_transient_remote_error(e):
                raise RetryableResponsesError(f"ChatGPT plan stream interrupted ({label}): {type(e).__name__}") from e
            raise

    async def _consume(self, body: dict[str, Any], token: str, label: str) -> Any:
        done_items: list[Any] = []
        # The bearer goes per request on the one shared client: with_options() would
        # build a client copy each call and re-read OPENAI_ORG_ID / OPENAI_PROJECT_ID.
        headers = {**(body.get("extra_headers") or {}), "Authorization": f"Bearer {token}"}
        async with await self._client.responses.create(**{**body, "extra_headers": headers}) as stream:
            async for event in stream:
                kind = getattr(event, "type", None)
                if kind == "response.output_item.done":
                    done_items.append(event.item)
                elif kind == "response.completed":
                    response = event.response
                    if not response.output and done_items:
                        response.output = done_items
                    return response
                elif kind == "response.incomplete":
                    response = event.response
                    reason = getattr(response.incomplete_details, "reason", None)
                    if reason == "max_output_tokens":
                        return response  # the parent maps this to OutputTooLongError
                    raise ChatGPTRequestError(f"ChatGPT plan response incomplete ({label}): {reason}")
                elif kind in ("response.failed", "error"):
                    error = event.response.error if kind == "response.failed" else event
                    raise _stream_failure(
                        getattr(error, "code", None),
                        getattr(error, "message", "") or "",
                        getattr(error, "param", None),
                        label=label,
                    )
        raise RetryableResponsesError(f"ChatGPT plan stream ended without response.completed ({label})")
