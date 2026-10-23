"""LLM Request extension: choose the LLM, and the credentials, of each LLM call.

The server's LLM settings (``HINDSIGHT_API_LLM_*``) are fixed at startup, so every
call goes to one model on one account. This extension decides per call instead:

- **Credentials only** — the same LLM, a different key and/or headers. For
  deployments that front their spend with a gateway (LiteLLM proxy, Bifrost,
  OpenRouter, Portkey, ...) and give each customer a virtual key there.
- **A whole LLM** — a different provider, model, endpoint and key. For per-bank or
  per-operation model choice (a cheap model for one bank's retain, a strong one for
  another's reflect), or per-customer cloud accounts.

It is asked once per LLM call, with the tenant schema, bank and operation the call
runs for — retain, reflect, consolidation, mental-model refresh, in the API process
and in the worker alike — and answers with :class:`LLMRequestOverrides`, or with
``None`` to use the server's own LLM. Raising refuses the call.

Nothing reaches this hook from the API caller: the choice is made server-side, by
operator code, from server-side state. An extension that derives its answer from
something a caller can write (bank config, metadata) must validate it — that is
where an endpoint or a key would otherwise come from user input.

Load it with::

    HINDSIGHT_API_LLM_REQUEST_EXTENSION=mypackage.keys:MyKeys
    HINDSIGHT_API_LLM_REQUEST_SOME_OPTION=value   # -> config["some_option"]
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from hindsight_api.config import LLMMemberConfig
from hindsight_api.extensions.base import Extension
from hindsight_api.schema_context import get_current_bank_id, get_current_schema

if TYPE_CHECKING:
    from hindsight_api.engine.llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRequestTarget:
    """What an LLM call is for — everything the extension may decide on.

    ``bank_id`` is ``None`` for a call made outside any bank (rare: e.g. a bank-less
    maintenance step). An extension that bills per bank should refuse such a call
    rather than guess.
    """

    schema: str
    """The tenant schema the call runs in. Bank ids are unique per schema only, so a
    multi-tenant deployment must match on this too."""
    bank_id: str | None
    operation: str | None
    """``retain``, ``reflect``, ``consolidation``, ``mental_model_delta_ops``, ...;
    ``None`` when the call was not attributed to an operation."""
    provider: str
    """The server's LLM for this call — what an answer would replace."""
    model: str


@dataclass(frozen=True)
class LLMRequestOverrides:
    """How one LLM call should be sent.

    ``llm`` sends the call to a different LLM altogether — provider, model, endpoint
    and key, in the same shape as a multi-LLM chain member (``LLMMemberConfig``).
    Fields it leaves unset inherit the server's global LLM defaults, exactly as a
    chain member's do; timeouts and retries inherit the operation's.

    ``api_key`` replaces the key of the LLM the call goes to (the server's, or
    ``llm``), in whatever form the provider expects (``Authorization: Bearer`` for
    OpenAI-compatible endpoints, ``x-api-key`` for Anthropic). ``headers`` are added
    to the request, for gateways that take a virtual key or a customer id in a
    header of their own. Both work without ``llm``, and then cost no new client.

    All three are excluded from ``repr`` so a log line or traceback that prints this
    object does not print a credential.
    """

    llm: LLMMemberConfig | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.llm is not None and not isinstance(self.llm, LLMMemberConfig):
            raise ValueError("LLMRequestOverrides.llm must be an LLMMemberConfig or None")
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ValueError("LLMRequestOverrides.api_key must be a non-empty string or None")
        for name, value in self.headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError("LLMRequestOverrides.headers must map strings to strings")
            # A CR/LF would let a value smuggle an extra header (or a body) into the request.
            if any(c in name or c in value for c in "\r\n\0"):
                raise ValueError(f"LLMRequestOverrides header {name!r} contains a control character")
        # Frozen copy: the extension may keep and mutate the dict it handed over.
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


class LLMRequestExtension(Extension, ABC):
    """Decide the LLM and credentials of every LLM call the server makes.

    Called on every call (retries of a call reuse the answer), so cache anything
    slow. Must be safe to call concurrently.
    """

    @abstractmethod
    async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides | None:
        """Return how to send this call, or ``None`` for the server's own LLM.

        Raise to refuse the call; the error fails the operation that made it.
        """
        ...


class LLMRequestRefused(RuntimeError):
    """An LLM call the installed extension could not, or would not, send."""


# ── runtime plumbing (used by the engine, not by extensions) ────────────────────

_installed: LLMRequestExtension | None = None

_overrides_ctx: ContextVar[LLMRequestOverrides | None] = ContextVar("hindsight_llm_request_overrides", default=None)


def install_llm_request_extension(extension: LLMRequestExtension | None) -> None:
    """Make ``extension`` decide every LLM call in this process (``None`` removes it).

    Process-wide on purpose: a per-provider handle would have to be threaded to every
    LLM the engine builds (per operation, per multi-LLM member, the vision model, ...),
    and one that was missed would silently send calls on the server's own LLM.
    """
    global _installed
    _installed = extension


def installed_llm_request_extension() -> LLMRequestExtension | None:
    return _installed


def current_llm_request_overrides() -> LLMRequestOverrides | None:
    """The overrides bound for the LLM call in flight, read by the providers."""
    return _overrides_ctx.get()


def request_client(client: Any) -> Any:
    """``client`` (an OpenAI or Anthropic SDK client) carrying the credentials bound for this call.

    ``with_options`` returns a copy that shares the client's connection pool, so a
    per-request key costs an object, not a connection. Applying the key here — not
    as ``extra_headers`` in the call params — keeps it out of everything that logs
    or dumps those params.
    """
    overrides = _overrides_ctx.get()
    if overrides is None:
        return client
    options: dict[str, Any] = {}
    if overrides.api_key is not None:
        options["api_key"] = overrides.api_key
    if overrides.headers:
        options["default_headers"] = dict(overrides.headers)
    return client.with_options(**options) if options else client


# Header names that carry a credential. The SDKs apply custom headers after their own
# auth header, so one of these among a provider's default headers would override a
# per-call key and send the call on the server's credentials.
_CREDENTIAL_HEADER_HINTS = ("auth", "key", "token", "secret", "cookie")


def _credential_headers(headers: Mapping[str, str] | None) -> list[str]:
    return [name for name in (headers or {}) if any(hint in name.lower() for hint in _CREDENTIAL_HEADER_HINTS)]


@dataclass
class BoundLLMRequest:
    """What :func:`bind_llm_request` decided for one call. Pass to :meth:`reset` when done."""

    llm: LLMProvider | None
    """The LLM to send this call to instead of the caller's, or ``None`` to send it as usual."""
    _overrides_token: Token | None = None

    def reset(self) -> None:
        if self._overrides_token is not None:
            _overrides_ctx.reset(self._overrides_token)


_NOTHING_BOUND = BoundLLMRequest(llm=None)


async def bind_llm_request(provider: LLMProvider) -> BoundLLMRequest:
    """Ask the installed extension how to send the call about to go to ``provider``.

    Binds the answer's credentials for the providers to read, and returns the LLM to
    run the call on when the answer chose one — the caller runs it there directly,
    without asking again. Raises :class:`LLMRequestRefused`
    when the answer carries credentials the receiving provider cannot apply: sending
    the call on the server's key instead would bill the wrong account, which is the
    one thing this extension exists to prevent.
    """
    extension = _installed
    if extension is None:
        return _NOTHING_BOUND

    # Lazy: memory_engine imports the provider modules that import this one.
    from hindsight_api.engine.llm_trace import current_trace_context

    trace = current_trace_context()
    target = LLMRequestTarget(
        schema=get_current_schema(),
        # The trace binding is the explicit one (set by with_config for this operation);
        # the engine-wide bank binding covers calls made outside with_config.
        bank_id=(trace.bank_id if trace is not None and trace.bank_id else None) or get_current_bank_id(),
        operation=trace.operation if trace is not None else None,
        provider=provider.provider,
        model=provider.model,
    )
    overrides = await extension.resolve(target)
    if overrides is not None and not isinstance(overrides, LLMRequestOverrides):
        raise LLMRequestRefused(
            f"{type(extension).__name__}.resolve() returned {type(overrides).__name__}, expected LLMRequestOverrides or None"
        )

    delegate = _llm_for(overrides, provider) if overrides is not None and overrides.llm is not None else None
    if overrides is not None and overrides.llm is not None and overrides.llm.api_key is None:
        # _llm_for built the chosen LLM with the answer's key, so it is no longer a
        # per-call override: providers that cannot take one need not refuse it.
        overrides = dataclasses.replace(overrides, api_key=None)
    receiver = delegate or provider
    if overrides is not None and (overrides.api_key is not None or overrides.headers):
        if not receiver._provider_impl.supports_request_overrides():
            raise LLMRequestRefused(
                f"The LLM request extension chose credentials for bank {target.bank_id!r}, but provider "
                f"'{receiver.provider}' cannot send a call with per-request credentials. Refusing rather than "
                f"sending it on its configured credentials."
            )
        # Names only in the message: the values are the credentials.
        clashing = [h for h in _credential_headers(receiver.default_headers) if h not in overrides.headers]
        if overrides.api_key is not None and clashing:
            raise LLMRequestRefused(
                f"The LLM request extension chose a key for bank {target.bank_id!r}, but the LLM's default "
                f"headers {clashing} carry a credential that would override it. Refusing rather than sending "
                f"the call on those; override the header in the answer's headers, or drop it from the defaults."
            )
    return BoundLLMRequest(
        llm=delegate,
        _overrides_token=_overrides_ctx.set(overrides),
    )


# ── LLMs chosen by an answer, built once and reused ─────────────────────────────

# ponytail: fixed LRU bound. Each entry holds SDK clients with their own connection
# pool, so an extension answering with an open-ended set of configs (one per user)
# must stay bounded; an evicted LLM is dropped, not closed, because a call may still
# be using it. Make the size configurable if a deployment needs more live configs.
_MAX_CHOSEN_LLMS = 64
_chosen_llms: OrderedDict[str, LLMProvider] = OrderedDict()
# threading, not asyncio: the cache is process-wide, and every critical section is await-free.
_chosen_llms_lock = threading.Lock()


def _llm_for(overrides: LLMRequestOverrides, caller: LLMProvider) -> LLMProvider:
    """The LLM for ``overrides.llm``, built like a multi-LLM chain member of ``caller``'s operation.

    Unlike a chain member, it inherits none of the server's credentials: the server's
    default headers, Vertex service account, Codex home and LiteLLM router config would
    otherwise bill the server's account, or send its gateway key to the chosen endpoint.
    A member without a key takes the answer's ``api_key``.
    """
    from hindsight_api.config import StaticConfigProxy, _get_raw_config
    from hindsight_api.engine.memory_engine import _LLMCallDefaults, _member_to_llm

    assert overrides.llm is not None
    member = overrides.llm
    if member.api_key is None and overrides.api_key is not None:
        member = dataclasses.replace(member, api_key=overrides.api_key)
    defaults = _LLMCallDefaults(
        timeout=caller.timeout,
        max_retries=caller.max_retries,
        initial_backoff=caller.initial_backoff,
        max_backoff=caller.max_backoff,
    )
    # Hashed so the cache's keys are not a second copy of every credential in memory.
    identity = json.dumps(
        {"member": dataclasses.asdict(member), "defaults": dataclasses.asdict(defaults)}, sort_keys=True, default=str
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    with _chosen_llms_lock:
        llm = _chosen_llms.get(key)
        if llm is not None:
            _chosen_llms.move_to_end(key)
            return llm

    # Built outside the lock: constructing a provider can be slow (SDK import, service
    # account parsing). Two calls racing on a new config may both build; the first
    # one stored wins and the other is dropped unused.
    no_server_credentials = dataclasses.replace(
        _get_raw_config(),
        llm_default_headers=None,
        llm_vertexai_service_account_key=None,
        llm_codex_home=None,
        llm_litellmrouter_config=None,
    )
    built = _member_to_llm(member, StaticConfigProxy(no_server_credentials), defaults)
    with _chosen_llms_lock:
        llm = _chosen_llms.setdefault(key, built)
        _chosen_llms.move_to_end(key)
        while len(_chosen_llms) > _MAX_CHOSEN_LLMS:
            _chosen_llms.popitem(last=False)
    return llm
