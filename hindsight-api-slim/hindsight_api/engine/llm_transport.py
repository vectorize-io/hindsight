"""Transport-level concerns shared by the SDK-backed LLM providers.

The OpenAI and Anthropic SDKs both sit on ``httpx``, and both hide the same two
things from an operator staring at a stalled call:

* **Which phase stalled.** A bare float timeout means "all four httpx phases", and
  ``APITimeoutError`` stringifies to ``"Request timed out."`` whether the request
  died waiting for the TCP handshake, waiting for a pool slot, mid-write, or
  waiting for the first response byte. Those are four different faults with four
  different owners. :func:`describe_transport_error` recovers the distinction from
  ``__cause__``.
* **How long the connect phase may take.** Passing ``llm_timeout`` as a float also
  raises the connect timeout to the full request budget -- above the OpenAI SDK's
  own 5 s default -- so an endpoint that never completes its handshake burns the
  entire budget instead of failing fast. :func:`build_sdk_timeout` caps it.

Both were diagnosed from issue #3881, where ~50% of reflect calls stalled for
exactly ``llm_timeout`` and the logs could not say which phase was stuck.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import aiohttp

# Only to configure the third-party SDKs built on httpx (openai, anthropic); our own
# HTTP calls go through aiohttp.
import httpx  # noqa: TID251

from ..config import (
    DEFAULT_LLM_HTTP_LOG_LEVEL,
    ENV_LLM_HTTP_LOG_LEVEL,
    get_config,
)
from .aiohttp_session import per_phase_timeout

logger = logging.getLogger(__name__)

# Cap on the deepest transport message echoed into a log line. Transport errors are
# short by nature; the cap only exists so a pathological one can't flood the log.
_MESSAGE_CAP = 200

# Returned when an error wraps no transport-level cause.
_NO_CAUSE = "<no cause>"

# Floor on the per-request deadline of the LLM call in flight, set by a caller that
# knows the call is legitimately long. Reflect's final synthesis is the case that
# motivated it (issue #4568): its prompt is 30-90k tokens of tool results and takes
# 5-48 s on a healthy provider, while the tool-calling turns before it answer in
# 1-5 s. One fixed per-call deadline cannot fit both -- the 30 s reflect default killed
# healthy synthesis calls, and a value that fits synthesis lets one stalled tool turn
# outlive the caller -- so the deadline follows the prompt instead.
#
# It is a *floor*, not a replacement: :func:`effective_request_timeout` can only
# lengthen the configured deadline. An operator's explicit ``HINDSIGHT_API_*_LLM_TIMEOUT``
# is therefore never cut below what they set, and a background refresh running the same
# agent under a 2700 s deadline keeps it. A ContextVar rather than a new ``call()``
# argument: the caller holds an ``LLMProvider`` -- an ``LLMConfig``, a multi-LLM chain,
# or a bare provider class -- and every one of those signatures would have to grow the
# argument; the request-context / usage / queue-wait plumbing in ``llm_trace.py``
# already crosses the same boundary this way.
_request_timeout_floor_ctx: ContextVar[float | None] = ContextVar("hindsight_llm_request_timeout_floor", default=None)


@contextmanager
def request_timeout_floor(seconds: float | None) -> Iterator[None]:
    """Arm a deadline floor for the LLM calls made inside the block; ``None`` arms nothing.

    Scoped to the block so a floor computed for one large prompt cannot leak into the
    next, smaller call on the same task.
    """
    token = _request_timeout_floor_ctx.set(seconds)
    try:
        yield
    finally:
        _request_timeout_floor_ctx.reset(token)


def effective_request_timeout(configured: float) -> float:
    """The deadline a provider arms for this call: ``configured``, lengthened to the floor if one is set."""
    floor = _request_timeout_floor_ctx.get()
    if floor is None:
        return configured
    return max(configured, floor)


class RequestDeadlineExceeded:
    """Marker base for provider errors that mean "the per-request deadline expired".

    Some providers cannot raise a ``TimeoutError`` for that: Codex's runaway-stream
    error must stay an ``aiohttp.ClientPayloadError`` so the transport retry ladder
    classifies it as transient. Mixing this marker in lets
    :func:`is_request_deadline_error` recognise it without importing the provider.
    It carries no state, so it cannot conflict with the exception base's layout.
    """


def is_request_deadline_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its ``__cause__`` chain) is a per-request deadline expiry.

    Covers the three shapes the providers produce: a bare ``TimeoutError`` (asyncio's
    ``wait_for``/``timeout``, Gemini, LiteLLM, Cursor), an SDK error raised ``from`` an
    ``httpx.TimeoutException`` (OpenAI-compatible, Anthropic), and a provider error
    carrying the :class:`RequestDeadlineExceeded` marker (Codex). Only ``__cause__`` is
    followed: ``__context__`` would also pick up an unrelated timeout that happened to
    be in flight when a different error was raised.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (TimeoutError, httpx.TimeoutException, RequestDeadlineExceeded)):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def build_sdk_timeout(total: float) -> httpx.Timeout:
    """Per-phase httpx timeout for an SDK client, with the connect phase capped.

    ``total`` is the resolved per-request LLM timeout and stays in force for the
    read, write and pool phases. Connect is capped at ``HINDSIGHT_API_LLM_CONNECT_TIMEOUT``
    (10 s by default) so an unreachable or wedged endpoint surfaces in seconds
    rather than consuming the whole request budget. Setting that variable to 0
    restores the old behaviour of one value across all four phases.
    """
    connect_cap = get_config().llm_connect_timeout
    if connect_cap <= 0:
        return httpx.Timeout(total)
    return httpx.Timeout(total, connect=min(connect_cap, total))


def build_aiohttp_timeout(total: float) -> aiohttp.ClientTimeout:
    """:func:`build_sdk_timeout` for the providers that talk HTTP through aiohttp directly."""
    connect_cap = get_config().llm_connect_timeout
    if connect_cap <= 0:
        return per_phase_timeout(total)
    return per_phase_timeout(total, connect=min(connect_cap, total))


def _qualified(exc: BaseException) -> str:
    """``module.ClassName`` for an exception, trimmed to the top-level module."""
    module = type(exc).__module__.split(".")[0]
    name = type(exc).__qualname__
    return f"{module}.{name}" if module and module != "builtins" else name


def describe_transport_error(err: BaseException, *, max_depth: int = 5) -> str:
    """Name the transport exceptions an SDK connection error wraps.

    Returns a chain like ``httpx.ReadTimeout <- httpcore.ReadTimeout`` with the
    deepest non-empty message appended, or ``"<no cause>"`` when the SDK error
    wraps nothing. Never raises -- a diagnostic must not break the request path.
    """
    chain: list[str] = []
    message = ""
    seen: set[int] = {id(err)}
    current = err.__cause__ or err.__context__
    while current is not None and len(chain) < max_depth and id(current) not in seen:
        seen.add(id(current))
        chain.append(_qualified(current))
        try:
            text = str(current).strip()
        except Exception:  # a __str__ that raises must not break error logging
            text = ""
        if text:
            message = text
        current = current.__cause__ or current.__context__
    if not chain:
        return _NO_CAUSE
    described = " <- ".join(chain)
    if message:
        described = f"{described}: {message[:_MESSAGE_CAP]}"
    return described


def describe_llm_error(err: BaseException) -> str:
    """Render an LLM-call failure as ``Class: message [cause chain]``.

    Provider-agnostic, and the reason it exists: several of the exceptions that end a
    stalled call stringify to the *empty string*. A bare ``asyncio.TimeoutError`` is the
    common one -- ``[REFLECT ...] LLM error on iteration 2:  (120002ms)`` in the wild --
    and it names neither the failure nor the provider. Logging the class as well means
    every provider gets the phase named, not just the ones on an httpx client we build.
    """
    text = ""
    try:
        text = str(err).strip()
    except Exception:  # a __str__ that raises must not break error logging
        pass
    described = f"{_qualified(err)}: {text[:_MESSAGE_CAP]}" if text else _qualified(err)
    cause = describe_transport_error(err)
    return described if cause == _NO_CAUSE else f"{described} [{cause}]"


def configure_http_logging() -> None:
    """Apply the configured level to the ``httpx`` and ``httpcore`` loggers.

    Default WARNING keeps a per-request line out of normal operation. DEBUG turns
    ``httpcore`` into the instrument that names the phase a stalled request is stuck
    in (``connect_tcp``, ``send_request_headers``, ``receive_response_headers``),
    which is what tells a hung LLM call apart from a slow one.
    """
    raw = get_config().llm_http_log_level.strip().upper()
    level = logging.getLevelName(raw)
    if not isinstance(level, int):
        logger.warning(f"{ENV_LLM_HTTP_LOG_LEVEL}={raw!r} is not a log level; using {DEFAULT_LLM_HTTP_LOG_LEVEL}")
        level = logging.getLevelName(DEFAULT_LLM_HTTP_LOG_LEVEL)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(level)
