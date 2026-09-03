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
import os

import httpx

from ..config import (
    DEFAULT_LLM_CONNECT_TIMEOUT,
    DEFAULT_LLM_HTTP_LOG_LEVEL,
    ENV_LLM_CONNECT_TIMEOUT,
    ENV_LLM_HTTP_LOG_LEVEL,
)

logger = logging.getLogger(__name__)

# Cap on the deepest transport message echoed into a log line. Transport errors are
# short by nature; the cap only exists so a pathological one can't flood the log.
_MESSAGE_CAP = 200

# Returned when an error wraps no transport-level cause.
_NO_CAUSE = "<no cause>"


def build_sdk_timeout(total: float) -> httpx.Timeout:
    """Per-phase httpx timeout for an SDK client, with the connect phase capped.

    ``total`` is the resolved per-request LLM timeout and stays in force for the
    read, write and pool phases. Connect is capped at ``HINDSIGHT_API_LLM_CONNECT_TIMEOUT``
    (10 s by default) so an unreachable or wedged endpoint surfaces in seconds
    rather than consuming the whole request budget. Setting that variable to 0
    restores the old behaviour of one value across all four phases.
    """
    connect_cap = float(os.getenv(ENV_LLM_CONNECT_TIMEOUT, str(DEFAULT_LLM_CONNECT_TIMEOUT)))
    if connect_cap <= 0:
        return httpx.Timeout(total)
    return httpx.Timeout(total, connect=min(connect_cap, total))


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
    raw = os.getenv(ENV_LLM_HTTP_LOG_LEVEL, DEFAULT_LLM_HTTP_LOG_LEVEL).strip().upper()
    level = logging.getLevelName(raw)
    if not isinstance(level, int):
        logger.warning(f"{ENV_LLM_HTTP_LOG_LEVEL}={raw!r} is not a log level; using {DEFAULT_LLM_HTTP_LOG_LEVEL}")
        level = logging.getLevelName(DEFAULT_LLM_HTTP_LOG_LEVEL)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(level)
