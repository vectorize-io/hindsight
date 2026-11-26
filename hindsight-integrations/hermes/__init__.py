"""Hindsight memory plugin — MemoryProvider with knowledge graph, entity resolution
and multi-strategy retrieval; cloud (API key), local_external, or local_embedded.

Config: $HERMES_HOME/hindsight/config.json (profile-scoped), else ~/.hindsight/
config.json (legacy, shared), else env: HINDSIGHT_API_KEY / BANK_ID / BUDGET /
API_URL / MODE / TIMEOUT / IDLE_TIMEOUT / RETAIN_TAGS / RETAIN_OBSERVATION_SCOPES /
RETAIN_SOURCE / RETAIN_STRATEGY / RETAIN_USER_PREFIX / RETAIN_ASSISTANT_PREFIX, and
HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT (config.json port_health_grace_timeout).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import agent.secret_scope as _secret_scope
from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread
from agent.secret_scope import UnscopedSecretError, get_secret
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from pydantic import BaseModel, ValidationError, field_validator
from tools.registry import tool_error
from utils import read_json_or_empty

from .embedded import (
    _CLIENT_MODULES,
    _RETRIABLE_CONNECTION_MARKERS,
    _build_embedded_profile_env,
    _check_local_runtime,
    _daemon_is_running,
    _embedded_llm_api_key,
    _embedded_profile_env_path,
    _export_port_health_grace_timeout,
    _installed_api_binary_exists,
    _load_simple_env,
    _local_runtime_hint,
    _materialize_embedded_profile_env,
    _may_rewrite_profile_env,
    _profile_env_drifted,
    _start_daemon,
    _stop_daemon,
)
from .settings import (
    _DEFAULT_API_URL,
    _DEFAULT_IDLE_TIMEOUT,
    _DEFAULT_LOCAL_URL,
    _DEFAULT_OBSERVATION_SCOPES,
    _DEFAULT_RETAIN_SOURCE,
    _DEFAULT_RETAIN_STRATEGY,
    _DEFAULT_TIMEOUT,
    _HINDSIGHT_GLYPH,
    _MIN_CLIENT_VERSION,
    _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
    _PROVIDER_DEFAULT_MODELS,
    _VALID_BUDGETS,
    ChatScope,
    ResolvedConfig,
    _clears_min_scores,
    _compile_line_patterns,
    _daemon_llm_provider,
    _discover_cwd_bank_id,
    _fill_tag_templates,
    _format_memory,
    _normalize_min_scores,
    _normalize_observation_scopes,
    _normalize_string_list,
    _parse_float_setting,
    _parse_int_setting,
    _parse_session_context,
    _project_name,
    _repository_root,
    _resolve_bank_id_template,
    _strip_query_lines,
    _template_fields,
    resolve_chat_config,
)

logger = logging.getLogger(__name__)

_LOCAL_MODES = {"local", "local_embedded"}
# Share of the operation timeout an extra recall bank may use, so a slow extra bank
# always gives up before the operation that also carries the primary's result does.
_EXTRA_BANK_TIMEOUT_SHARE = 0.8
# How long an extra bank is skipped after it fails, for recall and writes alike, so an
# unreachable bank costs one timeout per cooldown instead of one on every turn.
_EXTRA_BANK_COOLDOWN = 300.0
_RETAIN_CONTEXT_DEFAULT = "conversation between Hermes Agent and the User"
# Added after every retain context, a custom one included. Without it, a reply that repeats a
# fetched article or video in its author's first person ("I tested...") is stored as the
# agent's or the User's own experience (#6818). The tool message only names what was fetched:
# the model needs this sentence to read the reply as relayed.
_RELAYED_CONTENT_NOTE = (
    "Tool messages show what the assistant fetched (pages, videos, files), written by other people; "
    "when the Assistant repeats or summarises them, their claims and first person belong to that source, "
    "not to the User or the Assistant."
)
# A turn's tool calls are sent as one line each, never their output (too costly to retain).
_MAX_TOOL_CALLS_PER_TURN = 20
_MAX_TOOL_ARG_CHARS = 120


def _with_relayed_note(context: str | None) -> str:
    context = (context or "").strip().rstrip(".")
    return f"{context}. {_RELAYED_CONTENT_NOTE}" if context else _RELAYED_CONTENT_NOTE


class _ToolFunction(BaseModel):
    name: str = "unknown"
    arguments: Any = None  # a JSON string in OpenAI's shape; some providers send the object


class _ToolCall(BaseModel):
    function: _ToolFunction = _ToolFunction()


class _TranscriptMessage(BaseModel):
    """One row of the OpenAI-style transcript Hermes passes to ``sync_turn``; other keys ignored."""

    role: str = ""
    tool_calls: List[_ToolCall] = []

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _none_is_empty(cls, v: Any) -> Any:
        return v or []


def _parse_transcript(messages: List[Any]) -> List[_TranscriptMessage]:
    """A row that is not a message (or a tool call of an unknown shape) is skipped, not fatal:
    a malformed transcript must never cost the turn its retain."""
    parsed = []
    for message in messages:
        try:
            parsed.append(_TranscriptMessage.model_validate(message))
        except ValidationError:
            continue
    return parsed


def _format_arguments(arguments: Any) -> str:
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        args = arguments
    if isinstance(args, dict):
        return ", ".join(f"{k}={str(v)[:_MAX_TOOL_ARG_CHARS]}" for k, v in args.items())
    return str(args or "")[:_MAX_TOOL_ARG_CHARS]


def _tool_call_lines(messages: List[Any] | None) -> List[str]:
    """One "Tool call: name(arg=value, ...)" line per tool call the agent made this turn, i.e.
    after the last user message of Hermes' transcript. Arguments are cut short; outputs dropped."""
    transcript = _parse_transcript(messages or [])
    last_user = max((i for i, m in enumerate(transcript) if m.role == "user"), default=-1)
    calls = [call.function for m in transcript[last_user + 1 :] if m.role == "assistant" for call in m.tool_calls]
    lines = [f"Tool call: {fn.name}({_format_arguments(fn.arguments)})" for fn in calls[:_MAX_TOOL_CALLS_PER_TURN]]
    if len(calls) > _MAX_TOOL_CALLS_PER_TURN:
        lines.append(f"... and {len(calls) - _MAX_TOOL_CALLS_PER_TURN} more tool calls")
    return lines


def _bot_label(turn_author: Dict[str, Any] | None) -> str | None:
    """The label for a turn another agent wrote, from Hermes' ``turn_author`` ({"id", "name",
    "is_bot"}); None when a human wrote it or Hermes sent nothing. "Bot" when it has no name."""
    if not (isinstance(turn_author, dict) and turn_author.get("is_bot")):
        return None
    return str(turn_author.get("name") or "").strip() or "Bot"


# Injected instead of nothing when recall raises, so the agent can tell "memory is down" from
# "nothing found" (#6482). Fixed text, never the exception: errors can carry hosts and URLs. It is
# worded so a customer-facing agent need not announce an outage, only not read the gap as fact.
_RECALL_FAILED_NOTE = (
    "# Hindsight Memory\n"
    "Memory lookup failed this turn, so context from earlier sessions is missing. Don't conclude "
    "the user never told you something just because it is not here."
)


def _plugin_version() -> str:
    """Read the version from plugin.yaml — Hermes loads this directory, not a pip package."""
    try:
        # Explicit utf-8: the locale default (cp936, cp932...) can't decode it on Windows (#5704).
        text = (Path(__file__).resolve().parent / "plugin.yaml").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return "0.0.0"
    for line in text.splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip().strip("'\"") or "0.0.0"
    return "0.0.0"


_USER_AGENT = f"hindsight-hermes/{_plugin_version()}"


def _scoped_setting(name: str, default: str = "") -> str:
    """Profile-scoped read of a retain SHAPING value, with the provider's own default on a miss.

    Under ``gateway.multiplex_profiles`` ``os.environ`` holds the DEFAULT profile's ``.env``, so a miss
    is a miss — never ``os.environ`` (same rule as the daemon's key and base URL in ``embedded.py``).
    Single-profile deployments are unchanged: with no scope installed ``get_secret`` still reads the
    process env, where the value IS this profile's own.

    Deliberately narrower than a bare ``get_secret``: this helper is only for presentation shaping
    (retain source label, speaker prefixes, tags). The isolation-critical values — ``mode``,
    ``apiKey`` and the ``bankId`` data partition — read through bare ``get_secret`` above and so
    still fail loud on a scopeless multiplexed read, matching the other scoped credential readers.
    In ``_load_config`` that read happens FIRST, so a missing scope raises on ``HINDSIGHT_MODE``
    before this helper is ever reached; swallowing here therefore cannot mask an isolation failure.
    What it does avoid is losing the whole memory provider (``initialize`` failing, and the manager
    logging + dropping it) because a speaker prefix could not be resolved.
    """
    try:
        value = get_secret(name, default)
    except UnscopedSecretError:
        return default
    return default if value is None else value


def _scoped_flag(name: str, default: bool) -> bool:
    """A boolean ``_scoped_setting``: ``true``/``1``/``yes``/``on`` (any case) are on; anything else
    set is off; unset keeps *default*."""
    value = str(_scoped_setting(name, "") or "").strip().lower()
    return default if not value else value in {"true", "1", "yes", "on"}


def _cloud_api_key(config: dict) -> str:
    return config.get("apiKey") or config.get("api_key") or get_secret("HINDSIGHT_API_KEY", "")


def _warn_if_client_outdated() -> None:
    """Warn when the installed ``hindsight-client`` is below what this plugin needs.

    This used to try to fix it by calling ``tools.lazy_deps.install_specs``, which on
    package-manager Hermes hands the process to the updater instead of installing (the loop in
    NousResearch/hermes-agent#126494; upstream af26acab73 now raises ImportError). There is
    nothing to repair at runtime anyway: the floor is declared in this plugin's
    ``pyproject.toml``, so a version below it means the environment was built or stripped without
    it, which a plugin update fixes and a lazy install would only paper over.
    """
    try:
        from importlib.metadata import version as pkg_version

        from packaging.version import Version

        installed = pkg_version("hindsight-client")
        if Version(installed) < Version(_MIN_CLIENT_VERSION):
            logger.warning(
                "hindsight-client %s is older than this plugin needs (>=%s). "
                "Run 'hermes plugins update hindsight' to prepare its declared dependencies.",
                installed,
                _MIN_CLIENT_VERSION,
            )
    except Exception:
        pass  # packaging or metadata unavailable — proceed anyway


# update_mode='append' capability (Hindsight >= 0.5.0), cached per (API URL, key fingerprint)
# per process so every provider on the same API+key shares one /version round trip. Only an answer
# the server actually gave is cached; an unreachable server is asked again (the coding-agents client
# does the same by probing per write). Caching that failure too disabled append
# for the whole process after one network blip at startup (#5221). The key includes the credential
# so one profile's 401 cannot downgrade a sibling profile that shares the URL with a valid key.
_append_capability_cache: Dict[tuple[str, str | None], bool] = {}
_append_capability_lock = threading.Lock()
# When the probe last found the server unreachable. The probe runs on the agent's turn, so a hung
# server is asked again at most once per window rather than stalling every turn on its timeout.
_append_probe_unreachable_at: Dict[tuple[str, str | None], float] = {}
_APPEND_PROBE_RETRY_S = 60.0


class _VersionProbeUnreachable(Exception):
    """No version path answered: the server is down or slow, so its version is still unknown."""


def _fetch_hindsight_api_version(api_url: str, api_key: str | None = None, timeout: float = 5.0) -> str | None:
    """The server's version, or None when it answered without one (an API older than /version).

    ``/v1/version`` first: an ingress that routes only ``/v1`` to the API hands a bare ``/version``
    to whatever else it fronts, whose 404 read as "no version" (same order as coding-agents, #5286).
    ``/api/version`` last: a proxy that fronts the control plane at the root answers there with the
    API's own version, which it relays (#6137).
    Raises ``_VersionProbeUnreachable`` when no version came back and either path failed to answer
    (connection error, timeout, 5xx), so the caller can tell "unknown" from "old": only a 4xx or a
    versionless body from every path means the server is old.
    """
    import urllib.error
    import urllib.request

    if not api_url:
        return None
    headers = {"User-Agent": _USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    unanswered = False
    for path in ("/v1/version", "/version", "/api/version"):
        url = api_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:  # noqa: S310
                data = json.loads(resp.read(65536).decode("utf-8", errors="replace"))  # capped (#5598)
        except urllib.error.HTTPError as exc:
            unanswered = unanswered or exc.code >= 500  # a 4xx is an answer: this path is not served
            logger.debug("Hindsight version probe %s answered %s", url, exc.code)
            continue
        except ValueError as exc:  # not our JSON (an HTML page behind a proxy): an answer, not an outage
            logger.debug("Hindsight version probe %s returned no JSON: %s", url, exc)
            continue
        except Exception as exc:
            unanswered = True
            logger.debug("Hindsight version probe failed for %s: %s", url, exc)
            continue
        version = (data.get("version") or data.get("api_version")) if isinstance(data, dict) else None
        if version:
            return str(version)
    if unanswered:
        raise _VersionProbeUnreachable(api_url)
    return None


def _check_api_supports_update_mode_append(api_url: str, api_key: str | None = None) -> bool:
    """Cached ``update_mode='append'`` check for *api_url*. False when the server is too old or
    unreachable (safe default: per-process document_id, no update_mode = resume-overwrite fix
    intact); only the server's own answer is cached. See #6654, #5221.
    """
    if not api_url:
        return False
    from agent.credential_persistence import fingerprint_secret_value

    cache_key = (api_url, fingerprint_secret_value(api_key))
    with _append_capability_lock:
        if cache_key in _append_capability_cache:
            return _append_capability_cache[cache_key]
        failed_at = _append_probe_unreachable_at.get(cache_key)
        if failed_at is not None and time.monotonic() - failed_at < _APPEND_PROBE_RETRY_S:
            return False
    try:
        version = _fetch_hindsight_api_version(api_url, api_key)
    except _VersionProbeUnreachable:
        with _append_capability_lock:
            _append_probe_unreachable_at[cache_key] = time.monotonic()
        logger.debug(
            "Hindsight API %s unreachable for the version probe; retrying in %ss", api_url, _APPEND_PROBE_RETRY_S
        )
        return False
    try:  # missing/invalid version -> unsupported
        from packaging.version import Version

        supported = bool(version) and Version(version) >= Version(_MIN_VERSION_FOR_UPDATE_MODE_APPEND)
    except Exception:
        supported = False
    with _append_capability_lock:
        # A concurrent probe may have filled the cache meanwhile; its answer wins.
        supported = _append_capability_cache.setdefault(cache_key, supported)
    if supported:
        logger.debug("Hindsight API %s version %s supports update_mode='append'", api_url, version)
    else:
        logger.warning(
            "Hindsight API at %s reports version %r, older than %s. "
            "Falling back to per-process document_id — retains across "
            "processes/sessions create separate documents instead of "
            "appending to a session-scoped one. Upgrade Hindsight to "
            "%s+ to enable update_mode='append' deduplication.",
            api_url,
            version,
            _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
            _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
        )
    return supported


# One long-lived event loop per process for Hindsight async calls; ephemeral
# loops would leak aiohttp sessions.
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# Pushed to the per-provider retain queue to wake the writer for a clean exit.
_WRITER_SENTINEL = object()
# A queued retain that fails on a transient error is kept and sent again ahead of the next one
# (#5218); after this many tries, or on an error that can never succeed, it is dropped loudly.
_RETAIN_MAX_ATTEMPTS = 5
# Failed retains kept for resending while the server is down; the oldest go first past this.
_RETAIN_BACKLOG_LIMIT = 100
# An async retain the server accepted and then FAILED is re-queued server-side up to this many
# times (#5591). The server still holds its payload, so nothing is resent from here.
_SERVER_RETRY_LIMIT = 3
# Async retain operations still watched for a server-side failure; the oldest go first past this.
_UNSETTLED_OPS_LIMIT = 200


class _PendingRetain(BaseModel):
    """One automatic retain as plain data, so what is still unsent when Hermes exits can be
    written to disk and sent by the next session (#6915); it was a closure before, which could
    not outlive the process. Every input is fixed when it is queued: the writer runs after later
    turns move the session state on. ``item`` is the aretain_batch item, open-ended by design."""

    item: Dict[str, Any]
    bank_ids: List[str]
    retain_async: bool
    # Fixed per bank, so a resend the server already got is a replay, not a second copy (#5218).
    operation_ids: Dict[str, str]
    label: str = "retain"
    document_id: str | None = None
    # Where turns go when the bank refuses append (#5599); fixed so a resend reuses it.
    own_document_id: str | None = None
    track_ops: bool = True


@dataclass
class _QueuedRetain:
    """A queued retain and how many times it has failed."""

    retain: _PendingRetain
    failures: int = 0


# Retains still unsent when a provider shut down, one JSON file per shutdown, sent by the next initialize().
_UNSENT_RETAINS_DIR = ("hindsight", "unsent-retains")


def _retain_error_is_permanent(exc: BaseException) -> bool:
    """A 4xx other than timeout/rate limit: the same request fails the same way every time."""
    status = getattr(exc, "status", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


def _is_append_without_document_text(exc: BaseException) -> bool:
    """The server's 400 for update_mode='append' on a bank with store_document_text off
    (InvalidRetainRequestError in memory_engine.py; matched on its message, #5599)."""
    return getattr(exc, "status", None) == 400 and "document text storage" in str(getattr(exc, "body", "") or exc)


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
            daemon=True,
            name="hindsight-loop",
        )
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = _DEFAULT_TIMEOUT):
    """Schedule *coro* on the shared loop and block until done."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(coro, _get_loop())
    if future is None:
        raise RuntimeError("Hindsight loop unavailable")
    return future.result(timeout=timeout)


RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional per-call tags to merge with configured default retain tags.",
            },
            "occurred_at": {
                "type": "string",
                "description": (
                    "When the remembered event actually happened, as an ISO-8601 date "
                    "or datetime (e.g. '2026-08-20' or '2026-08-20T14:30:00+02:00'). "
                    "Pass this whenever the memory references a specific event time "
                    "('yesterday', 'last Tuesday', 'on March 3rd') so Hindsight can "
                    "anchor it on the timeline. Omit for timeless facts/preferences."
                ),
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "description": "What to search for."}},
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "description": "The question to reflect on."}},
    },
}


SEARCH_PAGES_SCHEMA = {
    "name": "hindsight_search_pages",
    "description": (
        "Search the user's pages: documents Hindsight writes from memory and keeps up to date, "
        "each answering one question (a project, a person, a long-running topic). Use it when "
        "the user asks where something stands or what was decided. Returns one line per page: "
        "name, id and its opening text. Read a page in full with hindsight_read_page."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer", "description": "How many pages to return (1-20, default 5)."},
        },
    },
}

READ_PAGE_SCHEMA = {
    "name": "hindsight_read_page",
    "description": "Read one page in full, as markdown. Pass an id returned by hindsight_search_pages or hindsight_list_pages.",
    "parameters": {
        "type": "object",
        "required": ["page_id"],
        "properties": {"page_id": {"type": "string", "description": "The page id."}},
    },
}

LIST_PAGES_SCHEMA = {
    "name": "hindsight_list_pages",
    "description": (
        "List every page and folder by name and id, without their text. Prefer "
        "hindsight_search_pages to find pages about a topic; use this when the user asks what pages exist."
    ),
    "parameters": {"type": "object", "properties": {}},
}

CREATE_PAGE_SCHEMA = {
    "name": "hindsight_create_page",
    "description": (
        "Create a page that Hindsight writes from memory and keeps up to date on its own. Every "
        "page costs model calls each time it refreshes, so create one ONLY when the user explicitly "
        'asks for a page ("keep a page on X"), never on your own initiative. The page is built in '
        "the background, so it is empty for a short while."
    ),
    "parameters": {
        "type": "object",
        "required": ["name", "question"],
        "properties": {
            "name": {"type": "string", "description": "Short page title, e.g. 'House renovation'."},
            "question": {
                "type": "string",
                "description": (
                    "The question the page answers, re-asked on every refresh, e.g. "
                    "'What is the current state of the house renovation, and what is still open?'"
                ),
            },
        },
    },
}

_PAGE_GUIDANCE = (
    " Use hindsight_search_pages for where something stands or what was decided: pages are "
    "organized, kept-current summaries."
)
_PAGE_SNIPPET_CHARS = 200
_PAGE_LIST_MAX_LINES = 100


def _pages_method(client: Any, name: str, legacy: str) -> Callable[..., Any]:
    """The client's page method, or its pre-1.2.1 "knowledge" name (the client floor is 0.10.1)."""
    return getattr(client, name, None) or getattr(client, legacy)


def _api_error_detail(exc: Exception) -> str:
    """The server's ``detail`` message for a failed API call; the client exception's own text
    carries the HTTP response headers, which is noise in a tool result."""
    with contextlib.suppress(Exception):
        detail = json.loads(getattr(exc, "body", None) or "").get("detail")
        if isinstance(detail, list):  # a 422 lists one entry per invalid field
            detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
        if isinstance(detail, str) and detail:
            return detail
    status = getattr(exc, "status", None)
    return f"HTTP {status}" if status else (str(exc) or type(exc).__name__)


def _bank_not_written_yet(exc: Exception) -> bool:
    """Hindsight answers 404 for a bank nothing has been retained to yet. Read off the status rather
    than importing the client's exception class here: that import, run from the prefetch thread while
    the writer thread first imports the client, deadlocks on the package's import lock."""
    return getattr(exc, "status", None) == 404


def _load_config() -> dict:
    """$HERMES_HOME/hindsight/config.json (profile-scoped), else ~/.hindsight/config.json
    (legacy, shared; standalone profiles only), else environment variables."""
    paths = [get_hermes_home() / "hindsight" / "config.json"]
    # A routed profile never falls back to the process user's shared file (#6754). Hermes
    # releases before v2026.9.21 can't tell, so they keep the legacy lookup.
    if not getattr(_secret_scope, "serves_routed_profile", lambda: False)():
        paths.append(Path.home() / ".hindsight" / "config.json")
    for path in paths:
        # A corrupt (or empty) file falls through to the next source, as before the dedup.
        if path.exists() and (data := read_json_or_empty(path)):
            return data
    # Mode, bank (the data partition), endpoint and retain shaping are per-profile .env values like
    # the key beside them: read through the secret scope so a multiplexed secondary never inherits
    # the default profile's bank/mode. Tuning knobs (timeouts, budget) stay process-global.
    return {
        "mode": get_secret("HINDSIGHT_MODE", "") or "cloud",
        "apiKey": get_secret("HINDSIGHT_API_KEY", ""),
        "timeout": _parse_int_setting(os.environ.get("HINDSIGHT_TIMEOUT"), _DEFAULT_TIMEOUT),
        "idle_timeout": _parse_int_setting(os.environ.get("HINDSIGHT_IDLE_TIMEOUT"), _DEFAULT_IDLE_TIMEOUT),
        "retain_tags": get_secret("HINDSIGHT_RETAIN_TAGS", "") or "",
        "observation_scopes": get_secret("HINDSIGHT_RETAIN_OBSERVATION_SCOPES", "") or "",
        "retain_source": _scoped_setting("HINDSIGHT_RETAIN_SOURCE", _DEFAULT_RETAIN_SOURCE),
        "retain_strategy": _scoped_setting("HINDSIGHT_RETAIN_STRATEGY", _DEFAULT_RETAIN_STRATEGY),
        "retain_user_prefix": _scoped_setting("HINDSIGHT_RETAIN_USER_PREFIX", "User"),
        "retain_assistant_prefix": _scoped_setting("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant"),
        "retain_context": _scoped_setting("HINDSIGHT_RETAIN_CONTEXT", _RETAIN_CONTEXT_DEFAULT),
        "retain_indicator": _scoped_flag("HINDSIGHT_RETAIN_INDICATOR", True),
        "recall_indicator": _scoped_flag("HINDSIGHT_RECALL_INDICATOR", True),
        "recall_sync": _scoped_flag("HINDSIGHT_RECALL_SYNC", False),
        "banks": {
            "hermes": {
                "bankId": get_secret("HINDSIGHT_BANK_ID", "") or "hermes",
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
        "reflect_budget": os.environ.get("HINDSIGHT_REFLECT_BUDGET", ""),
    }


def _event_timestamp() -> str:
    """Configured Hermes event time with an explicit UTC offset."""
    event_time = _hermes_now()
    # hermes_time.now() is aware; guard a replacement clock emitting offset-less dates.
    if event_time.tzinfo is None or event_time.utcoffset() is None:
        event_time = event_time.astimezone()
    return event_time.isoformat(timespec="seconds")


def _mint_document_id(session_id: str) -> str:
    """Per-process document id: reusing session_id alone overwrote the document on
    /resume (the reloaded session's first retain replaced the stored content)."""
    return f"{session_id}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


# Settings the local_embedded daemon's profile env is built from (embedded._build_embedded_profile_env).
_DAEMON_ENV_SETTINGS = ("llm_provider", "llm_api_key", "llmApiKey", "llm_model", "llm_base_url", "idle_timeout")

# initialize() kwargs copied verbatim (str, stripped) onto ``self._<name>``.
_SESSION_KWARGS = (
    "platform",
    "user_id",
    "user_name",
    "chat_id",
    "chat_name",
    "chat_type",
    "thread_id",
    "agent_identity",
    "agent_workspace",
    "cwd",
)
# aiohttp's message for a pooled keep-alive connection the server closed while it sat idle.
_STALE_KEEPALIVE_MARKER = "server disconnected"
# Retain metadata keys, each stamped from the attribute of the same name when set.
_METADATA_ATTRS = (
    "session_id",
    "platform",
    "user_id",
    "user_name",
    "chat_id",
    "chat_name",
    "chat_type",
    "thread_id",
    "agent_identity",
)
# Retain metadata that differs on every call; left off appends to a session document (#6241).
_PER_CALL_METADATA = frozenset({"retained_at", "message_count", "turn_index"})
# Naming Hermes' built-in session_search tells the model when to pick each tool. With only
# the tool names listed, models defaulted to session_search (raw transcripts) for every
# history question and skipped Hindsight's deduplicated memories.
_TOOL_GUIDANCE = (
    "For cross-session facts, user preferences, and past decisions, "
    "prefer hindsight_recall over session_search — it returns "
    "deduplicated, high-density observations across sessions. "
    "Use hindsight_reflect for cross-session pattern synthesis. "
    "Use session_search only when you need verbatim transcripts or "
    "exact wording from a specific conversation."
)
# A wrong memory left alone is served again next session; a stated correction outranks it.
_RETAIN_GUIDANCE = (
    " Use hindsight_retain to store facts."
    " When the user says a memory is wrong or out of date, or you find it is, store the correction"
    " with hindsight_retain: what memory said, what is true now, and why (e.g. 'User moved from"
    " Porto to Lisbon; Porto is no longer their city')."
)
_CONTEXT_NOTE = "Relevant memories are automatically injected into context."
_SYSTEM_PROMPT_TAILS = {
    "context": _CONTEXT_NOTE,
    "tools": _TOOL_GUIDANCE + _RETAIN_GUIDANCE,
    "hybrid": f"{_CONTEXT_NOTE} {_TOOL_GUIDANCE}{_RETAIN_GUIDANCE}",
}


# A session_context read runs before the first prompt is built, so it gets its own short bound.
_SESSION_CONTEXT_TIMEOUT = 5.0


class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    # Each server-side op status poll is a round trip — coarser than the 0.05s queue poll.
    _RETAIN_OP_POLL_INTERVAL_S = 0.5

    def backup_paths(self) -> List[str]:
        """Legacy shared config + embedded-mode profile env files live under ~/.hindsight."""
        with contextlib.suppress(Exception):
            return [str(Path.home() / ".hindsight")]
        return []

    def __init__(self):
        self._config = self._api_key = self._client = None
        self._embedded_url = None
        self._client_lock = threading.Lock()
        self._api_url, self._mode = _DEFAULT_API_URL, "cloud"
        self._timeout, self._idle_timeout = _DEFAULT_TIMEOUT, _DEFAULT_IDLE_TIMEOUT
        self._bank_id, self._budget, self._bank_id_template = "hermes", "mid", ""
        self._reflect_budget = "mid"
        self._mirror_to_own_bank = False
        self._static_bank_id = "hermes"
        self._project, self._bank_source = "", "bank_id"
        self._chat_route = ResolvedConfig({}, "")
        self._extra_bank_down_until: dict[tuple[str, str], float] = {}
        self._additional_bank_ids: list[str] = []
        self._recall_additional_bank_ids: list[str] = []
        self._write_bank_ids: list[str] = ["hermes"]
        self._memory_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._prefetch_method = "recall"  # "recall" or "reflect"
        for name in _SESSION_KWARGS:
            setattr(self, f"_{name}", "")
        self._session_id = self._parent_session_id = self._document_id = ""
        self._status_callback: Optional[Callable[[str], None]] = None
        # Set from initialize() kwargs; defaults keep _start_embedded_daemon safe when a
        # host constructs the provider without calling initialize() (availability probes do).
        self._warning_callback: Optional[Callable[[str], None]] = None
        self._platform: str = "cli"

        # Retain: single-writer model — sync_turn() enqueues, one writer thread
        # drains sequentially (ad-hoc threads raced interpreter shutdown:
        # "cannot schedule new futures" / "Unclosed client session").
        self._retain_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._sync_thread = None  # legacy alias external callers may join; points at the writer
        self._shutting_down = threading.Event()
        self._atexit_registered = False
        self._retain_tags: List[str] = []
        self._tags: list[str] | None = None
        self._observation_scopes: Any = None
        self._retain_source = _DEFAULT_RETAIN_SOURCE
        self._retain_strategy = _DEFAULT_RETAIN_STRATEGY
        self._retain_user_prefix, self._retain_assistant_prefix = "User", "Assistant"
        self._turn_counter = self._turn_index = 0
        self._session_turns: list[str] = []  # ALL turns for the session
        self._last_retained_turn_count = 0  # append-mode delta watermark
        # Server-side async retain ops still in flight: aretain_batch returns on
        # *acceptance*, not durability, so the prefetch gates on these via
        # get_operation_status (a drained local queue is not a read-after-write signal).
        self._pending_retain_ops: set[tuple[str, str]] = set()
        self._pending_retain_ops_lock = threading.Lock()
        # Every async op of an automatic retain until the server settles it, with how many
        # times we asked the server to retry it. Unlike _pending_retain_ops (a visibility
        # wait, dropped at its deadline), this is about durability: a failed op is re-queued
        # (#5591). Guarded by _pending_retain_ops_lock; insertion order = age.
        self._unsettled_retain_ops: dict[tuple[str, str], int] = {}
        # Retains the writer is working through: failed ones (#5218) then the one in flight, oldest
        # first. Guarded by _retain_backlog_lock: shutdown() takes what is left to disk (#6915).
        self._retain_backlog: list[_QueuedRetain] = []
        self._retain_backlog_lock = threading.Lock()
        self._unsent_spooled = threading.Event()  # set by shutdown(): the writer sends nothing more
        # Retains given up on (writer thread increments), and how many the status line has
        # already reported (turn path). Lets the indicator tell the truth instead of "saving…" (#6819).
        self._retains_dropped = 0
        self._retains_dropped_shown = 0
        # Set once the primary bank rejects append because it keeps no document text (#5599).
        self._append_unsupported = False
        self._apply_retain_policy({})

        # Recall: pending prefetch block + count, and the indicator state (recall_status()).
        self._prefetch_result, self._prefetch_count = "", 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        # Bumped by every new prefetch, session switch and shutdown: a worker writes its result
        # only if no newer one started since, so a slow recall can't overwrite a fresher one or
        # land in the next session.
        self._prefetch_generation = 0
        self._last_recall_returned, self._last_recall_count = False, 0
        self._apply_recall_settings({})

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in _LOCAL_MODES:
                # The availability gate is the only place worth self-healing from: agent_init drops
                # the provider outright when this returns False, and every other runtime probe below
                # runs after it has already passed.
                return _check_local_runtime().available
            # Hermes keeps a provider whose client cannot import and lets every recall and retain
            # fail; a gateway started before the install runs without it (#6063).
            if not _check_local_runtime(_CLIENT_MODULES).available:
                return False
            # Cloud needs a key: a URL alone used to pass, and every request then went out
            # unauthenticated (#6438). A keyless self-hosted server is what local_external is for.
            return mode == "local_external" or bool(_cloud_api_key(cfg))
        except Exception:
            logger.debug("Hindsight availability check failed", exc_info=True)
            return False

    def unavailable_reason(self) -> str:
        """Install hint when this process can't import what the mode needs (is_available() gates
        initialize() out, so the hint it would log never fires; agent_init shows this).

        ``is_available()`` returns False for local modes when the embedded runtime can't be imported, so
        ``initialize()`` — and the hint it would log — is never reached (#7718). Surface the install
        guidance here, where agent_init warns about an unavailable provider.
        """
        try:
            cfg = _load_config()
        except Exception:
            return ""
        mode = cfg.get("mode", "cloud")
        status = _check_local_runtime() if mode in _LOCAL_MODES else _check_local_runtime(_CLIENT_MODULES)
        if not status.available:
            return _local_runtime_hint(status.reason).strip()
        if mode == "cloud" and not _cloud_api_key(cfg):
            return (
                "Hindsight cloud mode needs an API key: set HINDSIGHT_API_KEY (or api_key in "
                "hindsight/config.json), or run `hermes memory setup`. For a self-hosted server "
                "without a key, use mode local_external."
            )
        return ""

    def save_config(self, values, hermes_home):
        """Merge *values* into $HERMES_HOME/hindsight/config.json."""
        from utils import atomic_json_write

        config_path = Path(hermes_home) / "hindsight" / "config.json"
        atomic_json_write(config_path, {**read_json_or_empty(config_path), **values}, mode=0o600)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard — installs only the deps needed for the selected mode."""
        from .setup import run_setup

        run_setup(self, hermes_home, config)

    def get_config_schema(self):
        return [
            {
                "key": "mode",
                "description": "Connection mode",
                "default": "cloud",
                "choices": ["cloud", "local_embedded", "local_external"],
            },
            # Cloud mode
            {
                "key": "api_url",
                "description": "Hindsight Cloud API URL",
                "default": _DEFAULT_API_URL,
                "when": {"mode": "cloud"},
            },
            {
                "key": "api_key",
                "description": "Hindsight Cloud API key",
                "secret": True,
                "env_var": "HINDSIGHT_API_KEY",
                "url": "https://ui.hindsight.vectorize.io",
                "when": {"mode": "cloud"},
            },
            # Local external mode
            {
                "key": "api_url",
                "description": "Hindsight API URL",
                "default": _DEFAULT_LOCAL_URL,
                "when": {"mode": "local_external"},
            },
            {
                "key": "api_key",
                "description": "API key (optional)",
                "secret": True,
                "env_var": "HINDSIGHT_API_KEY",
                "when": {"mode": "local_external"},
            },
            # Local embedded mode
            {
                "key": "llm_provider",
                "description": "LLM provider",
                "default": "openai",
                "choices": [
                    "openai",
                    "anthropic",
                    "gemini",
                    "groq",
                    "openrouter",
                    "minimax",
                    "ollama",
                    "lmstudio",
                    "openai_compatible",
                ],
                "when": {"mode": "local_embedded"},
            },
            {
                "key": "llm_base_url",
                "description": "Endpoint URL (e.g. http://192.168.1.10:8080/v1)",
                "default": "",
                "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"},
            },
            {
                "key": "llm_api_key",
                "description": "LLM API key (optional for openai_compatible)",
                "secret": True,
                "env_var": "HINDSIGHT_LLM_API_KEY",
                "when": {"mode": "local_embedded"},
            },
            {
                "key": "llm_model",
                "description": "LLM model",
                "default": "gpt-5-nano",
                "default_from": {"field": "llm_provider", "map": _PROVIDER_DEFAULT_MODELS},
                "when": {"mode": "local_embedded"},
            },
            {
                "key": "bank_id",
                "description": "Memory bank name (static fallback when bank_id_template is unset)",
                "default": "hermes",
            },
            {
                "key": "bank_id_template",
                "description": "Optional template to derive bank_id dynamically. Placeholders: {profile}, {workspace}, {project}, {platform}, {user}, {session}. {project} is the git repository name (shared across worktrees, empty outside a repository). Example: hermes-{project}",
                "default": "",
            },
            {
                "key": "recall_budget",
                "description": "Recall thoroughness",
                "default": "mid",
                "choices": ["low", "mid", "high"],
            },
            {
                "key": "reflect_budget",
                "description": "Reflect thoroughness (unset = same as recall_budget)",
                "choices": ["low", "mid", "high"],
            },
            {
                "key": "memory_mode",
                "description": "Memory integration mode",
                "default": "hybrid",
                "choices": ["hybrid", "context", "tools"],
            },
            {
                "key": "recall_prefetch_method",
                "description": "Auto-recall method",
                "default": "recall",
                "choices": ["recall", "reflect"],
            },
            {
                "key": "retain_tags",
                "description": "Default tags applied to retained memories (comma-separated)",
                "default": "",
            },
            {
                "key": "observation_scopes",
                "description": "How observations are scoped during consolidation: 'shared' (default — one scope for the bank, so observations merge across sessions; the session:/parent: tags stay on the facts), 'combined' (one scope per exact tag set, i.e. per session), 'per_tag' (one isolated observation per tag), 'all_combinations' (every tag subset — expensive), or a JSON list of tag-lists for explicit custom scopes.",
                "default": _DEFAULT_OBSERVATION_SCOPES,
            },
            {
                "key": "retain_source",
                "description": "Metadata source value attached to retained memories (identifies the client that stored them)",
                "default": _DEFAULT_RETAIN_SOURCE,
            },
            {
                "key": "retain_strategy",
                "description": (
                    "Named retain strategy applied to every stored item, steering extraction for "
                    "this content type. The bank must define the name under retain_strategies; an "
                    "unknown name is ignored by the server and the item falls back to unmissioned "
                    "extraction. Empty means let the bank decide."
                ),
                "default": _DEFAULT_RETAIN_STRATEGY,
            },
            {
                "key": "retain_user_prefix",
                "description": "Label used before user turns in retained transcripts",
                "default": "User",
            },
            {
                "key": "retain_assistant_prefix",
                "description": "Label used before assistant turns in retained transcripts",
                "default": "Assistant",
            },
            {
                "key": "recall_tags",
                "description": "Tags to filter when searching memories (comma-separated)",
                "default": "",
            },
            {
                "key": "recall_tags_match",
                "description": "Tag matching mode for recall",
                "default": "any",
                "choices": ["any", "all", "any_strict", "all_strict"],
            },
            {
                "key": "recall_exclude_tags",
                "description": "Tags to keep out of recall (comma-separated or a list): a memory carrying any of them is skipped by auto-recall and the hindsight_recall tool but stays in the bank; untagged memories are unaffected",
                "default": "",
            },
            {
                "key": "reflect_tags",
                "description": "Tags to filter what reflect reasons over (comma-separated or a list); unset searches the whole bank",
                "default": "",
            },
            {
                "key": "reflect_tags_match",
                "description": "Tag matching mode for reflect",
                "default": "any",
                "choices": ["any", "all", "any_strict", "all_strict"],
            },
            {
                "key": "recall_types",
                "description": "Fact types to surface on recall — applies to both auto-recall and the hindsight_recall tool (comma-separated or list). Defaults to observation-only — observations are Hindsight's consolidated, deduplicated, evidence-grounded knowledge layer; raw world/experience facts are the supporting evidence observations already summarize. Set to e.g. 'observation,world,experience' to also include raw facts.",
                "default": "observation",
            },
            {
                "key": "recall_min_scores",
                "description": 'Minimum relevance per score field, as a JSON object (e.g. {"reranker": 0.25}). Recall always returns up to recall_max_tokens of the best-ranked memories even when none is relevant; memories under a floor are dropped from auto-recall and the hindsight_recall tool, so an off-topic turn injects nothing. Default: no floor',
                "default": "",
            },
            {"key": "auto_recall", "description": "Automatically recall memories before each turn", "default": True},
            {
                "key": "recall_sync",
                "description": "Recall synchronously against the current message before each turn (higher relevance, adds recall latency to the turn). Default off: recall runs in the background and is injected on the next turn.",
                "default": False,
            },
            {
                "key": "recall_indicator",
                "description": "Show a '💭 Hindsight — recalled N memories' status line when auto-recall injects memory (turn off for customer-facing agents)",
                "default": True,
            },
            {
                "key": "retain_indicator",
                "description": "Show a '💭 Hindsight — saving to memory…' status line when a turn is saved to memory (turn off for customer-facing agents)",
                "default": True,
            },
            {"key": "auto_retain", "description": "Automatically retain conversation turns", "default": True},
            {
                "key": "retain_cron",
                "description": "Also auto-retain scheduled (cron) runs; off by default so job output stays out of the user's memory",
                "default": False,
            },
            {
                "key": "retain_kanban",
                "description": "Let kanban workers write to memory (auto-retain, the retain tool, built-in memory mirror); off by default so workers only read the shared bank",
                "default": False,
            },
            {
                "key": "mirror_builtin_memory",
                "description": "Copy built-in memory writes (MEMORY.md, USER.md adds and replaces) into the bank; turn off to keep those notes out of a curated bank",
                "default": True,
            },
            {
                "key": "recall_cron",
                "description": "Also auto-recall in scheduled (cron) runs; off by default because the query would be Hermes' fixed cron notice, not the job",
                "default": False,
            },
            {"key": "retain_every_n_turns", "description": "Retain every N turns (1 = every turn)", "default": 1},
            {
                "key": "retain_async",
                "description": "Process retain asynchronously on the Hindsight server",
                "default": True,
            },
            {
                "key": "prefetch_waits_for_retain",
                "description": "Have the background next-turn prefetch wait for the just-completed retain to become recall-visible on the server (local queue drain + async operation completion) before recalling, so recall includes the just-completed turn (runs off the reply path, adds no response latency)",
                "default": True,
            },
            {
                "key": "prefetch_retain_drain_timeout",
                "description": "Max seconds the background prefetch waits for the retain to become recall-visible (queue drain + server-side completion) before recalling anyway",
                "default": 10.0,
            },
            {
                "key": "retain_context",
                "description": "Context label for retained memories",
                "default": "conversation between Hermes Agent and the User",
            },
            {"key": "recall_max_tokens", "description": "Maximum tokens for recall results", "default": 4096},
            {
                "key": "recall_max_tokens_mode",
                "description": "How recall_max_tokens applies when recall searches several banks: per_bank gives each bank the full budget; split divides it evenly across the banks searched, so the total stays under recall_max_tokens",
                "default": "per_bank",
                "choices": ["per_bank", "split"],
            },
            {
                "key": "recall_max_input_chars",
                "description": "Maximum input query length for auto-recall",
                "default": 800,
            },
            {
                "key": "recall_query_strip_patterns",
                "description": "Regexes, one per line: matching lines are removed from the auto-recall query before recall_max_input_chars is applied (the message the model sees is unchanged)",
                "default": "",
            },
            {"key": "recall_prompt_preamble", "description": "Custom preamble for recalled memories in context"},
            {
                "key": "session_context",
                "description": "Mental models and pages to put in the system prompt at session start, in order (comma-separated mental_model:<id> or page:<id>; a bare id is a mental model)",
                "default": "",
            },
            {
                "key": "page_tools",
                "description": "Give the agent tools to search, read and list pages (pages Hindsight writes from memory)",
                "default": True,
            },
            {
                "key": "page_create_tool",
                "description": "Also let the agent create pages when the user asks; off by default because every page costs model calls on each refresh",
                "default": False,
            },
            {"key": "timeout", "description": "API request timeout in seconds", "default": _DEFAULT_TIMEOUT},
            {
                "key": "idle_timeout",
                "description": "Embedded daemon idle timeout in seconds (0 disables auto-shutdown)",
                "default": _DEFAULT_IDLE_TIMEOUT,
                "when": {"mode": "local_embedded"},
            },
            {
                "key": "port_health_grace_timeout",
                "description": "Seconds to wait for a slow daemon /health before treating it as stale (raise on busy/low-resource hosts; blank uses the 30s default)",
                "default": "",
                "when": {"mode": "local_embedded"},
            },
        ]

    # -- client -------------------------------------------------------------

    def _int_setting(self, key: str, env_var: str, default: int, env_default=None) -> int:
        """Config value if set (explicit 0 preserved), else env var, else default."""
        value = self._config.get(key)
        return _parse_int_setting(os.environ.get(env_var, env_default) if value is None else value, default)

    def _new_embedded_client(self):
        """Client for local_embedded: start the daemon out-of-process, then talk HTTP to it.

        This used to be ``from hindsight import HindsightEmbedded``, which required the
        ``hindsight-all`` package — the entire hindsight-api-slim server tree — inside Hermes'
        venv, where it cannot resolve against Hermes' pinned extras. ``HindsightEmbedded`` is
        only ``hindsight_client.Hindsight`` + ``hindsight_embed.get_embed_manager()``, so we
        compose the same two packages ourselves. Same daemon, same profile, same profile ``.env``,
        same pg0 database: nothing of an existing user's data or config moves.
        """
        status = _check_local_runtime()
        if not status.available:
            raise RuntimeError(
                "Hindsight local runtime is unavailable" + (f": {status.reason}" if status.reason else "")
            )
        from hindsight_client import Hindsight

        cfg = self._config
        profile = str(cfg.get("profile", "hermes") or "hermes")
        self._idle_timeout = self._int_setting(
            "idle_timeout",
            "HINDSIGHT_IDLE_TIMEOUT",
            _DEFAULT_IDLE_TIMEOUT,
            env_default=self._idle_timeout,
        )
        # Only explicitly-configured keys: an omitted one is resolved by the daemon from the
        # profile .env, then the parent environment (#3253). This mirrors HindsightEmbedded's
        # own config assembly so the daemon sees exactly what it saw before.
        daemon_config = _build_embedded_profile_env(cfg)
        daemon_config["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(self._idle_timeout)
        logger.debug(
            "Starting embedded Hindsight daemon (profile=%s, provider=%s)",
            profile,
            daemon_config.get("HINDSIGHT_API_LLM_PROVIDER", ""),
        )
        self._announce_slow_first_start(profile)
        self._embedded_url = _start_daemon(daemon_config, profile)
        logger.info("Connected to embedded Hindsight daemon at %s", self._embedded_url)
        # A daemon running ApiKeyTenantExtension rejects every keyless call with 401 (#5023). Its key
        # lives in the profile env the daemon boots from; a configured apiKey is the fallback.
        api_key = _load_simple_env(_embedded_profile_env_path(cfg)).get("HINDSIGHT_API_TENANT_API_KEY") or self._api_key
        return Hindsight(base_url=self._embedded_url, api_key=api_key or None)

    def _new_cloud_client(self):
        from hindsight_client import Hindsight

        kwargs = {
            "base_url": self._api_url,
            "timeout": float(self._timeout or _DEFAULT_TIMEOUT),
            "user_agent": _USER_AGENT,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        logger.debug(
            "Creating Hindsight cloud client (url=%s, has_key=%s, timeout=%s)",
            self._api_url,
            bool(self._api_key),
            kwargs["timeout"],
        )
        return Hindsight(**kwargs)

    def _announce_slow_first_start(self, profile: str) -> None:
        """Tell the user once when the embedded daemon is about to be fetched, not just started.

        With no ``hindsight-api`` binary in the environment the daemon runs through
        ``uvx hindsight-api@<version>``, which downloads the server (and its ML stack) into uv's
        cache before it can boot. Until now nothing reached the user during that: the manager's
        Rich panel is written to ~/.hermes/logs/hindsight-embed.log and only after the attempt
        ends, so a first run looked like Hermes hanging, and #4936 reported a reply that took
        6m23s, retained nothing and printed no error. Same sink as the root-refusal notice.
        """
        if _daemon_is_running(profile) or _installed_api_binary_exists():
            return
        msg = (
            "Hindsight is downloading its local memory server (0.5-3 GB on first use, depending "
            "on platform). "
            "Memory will start working once it finishes; progress is in "
            f"{get_hermes_home() / 'logs' / 'hindsight-embed.log'}."
        )
        logger.info(msg)
        with contextlib.suppress(Exception):
            if self._warning_callback is not None:
                self._warning_callback(msg)
            else:
                from gateway.warning_notifications import render_notification

                render_notification(
                    lambda: print(f"  ⏳ {msg}", file=sys.stderr, flush=True),
                    platform=self._platform,
                )

    def _get_client(self):
        """Return the cached Hindsight client, creating it at most once.

        Locked because in local_embedded the creation *starts the daemon*: the background start
        worker and the first memory operation both land here, and an unguarded check-then-set let
        them each start one and build one client, of which the loser was dropped without being
        closed (an unclosed aiohttp session). threading.Lock, not asyncio: these are threads, and
        the guarded section never awaits.
        """
        with self._client_lock:
            if self._client is None:
                self._client = (
                    self._new_embedded_client() if self._mode == "local_embedded" else self._new_cloud_client()
                )
            return self._client

    def _run_sync(self, coro):
        """Schedule *coro* on the shared loop using the configured timeout."""
        return _run_sync(coro, timeout=self._timeout)

    def _run_hindsight_operation(self, operation, *, repeatable: bool = True):
        """Run an async client operation, recreating the client and retrying once when its
        connection went stale: for local_embedded, a daemon that is no longer there; in any mode,
        a pooled connection the server already closed (#5734). The server may have read a request
        it dropped mid-way, so that retry is only for a *repeatable* operation."""
        try:
            return self._run_sync(operation(self._get_client()))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}".lower()
            stale_daemon = self._mode == "local_embedded" and any(m in text for m in _RETRIABLE_CONNECTION_MARKERS)
            if not stale_daemon and not (repeatable and _STALE_KEEPALIVE_MARKER in text):
                raise
            logger.info("Hindsight connection went stale; recreating client and retrying once: %s", exc)
            with self._client_lock:
                stale, self._client = self._client, None
            if stale is not None:
                # Close it, or its aiohttp session leaks ("Unclosed client session").
                with contextlib.suppress(Exception):
                    self._run_sync(stale.aclose())
            client = self._get_client()
            return self._run_sync(operation(client))

    # -- retain writer thread + server-side visibility -------------------------

    def _ensure_writer(self) -> None:
        """Lazy-start the single retain-writer thread (tools-only providers never pay for it)."""
        if (thread := self._writer_thread) is not None and thread.is_alive():
            return
        # A previous writer may have exited after shutdown(); allow the fresh one to drain.
        self._shutting_down.clear()
        self._unsent_spooled.clear()
        thread = spawn_context_thread(self._writer_loop, name="hindsight-writer")
        self._writer_thread = self._sync_thread = thread
        thread.start()

    def _register_atexit(self) -> None:
        """Idempotent atexit drain: a CLI exit that skips MemoryManager.shutdown_all()
        must not race interpreter teardown."""
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self._atexit_shutdown)

    def _writer_loop(self) -> None:
        """Drain the retain queue serially until the sentinel. A failing job can't
        kill the writer; task_done() always fires so queue.join() works."""
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    self._run_retains([])  # a last try for retains that failed earlier
                    return
                self._run_retains([job])
            except Exception as exc:
                logger.warning("Hindsight retain failed: %s", exc, exc_info=True)
            finally:
                self._retain_queue.task_done()

    def _run_retains(self, jobs: list[_PendingRetain]) -> None:
        """Run the backlog of failed retains, then *jobs*, in order (writer thread).

        A transient failure (unreachable server, timeout, 5xx) keeps that retain and every
        one after it for the next call, so turns are not lost when a write fails after the
        turn buffer moved on (#5218), and a down server costs one timeout per turn rather
        than one per queued retain. Resending is safe for async retains: each carries its
        own operation_id, which the server answers as a replay rather than new work. A sync
        retain whose reply was lost may be stored twice; that beats losing it."""
        self._settle_retain_ops()
        with self._retain_backlog_lock:
            self._retain_backlog.extend(_QueuedRetain(job) for job in jobs)
            if (excess := len(self._retain_backlog) - _RETAIN_BACKLOG_LIMIT) > 0:
                logger.warning("Hindsight retain backlog full; dropping the %d oldest unsent retain(s)", excess)
                del self._retain_backlog[:excess]
                self._retains_dropped += excess
        while True:
            with self._retain_backlog_lock:
                # Once shutdown() took the backlog to disk, the next session sends it instead.
                if not self._retain_backlog or self._unsent_spooled.is_set():
                    return
                entry = self._retain_backlog[0]
            try:
                self._send_retain(entry.retain)
            except Exception as exc:
                entry.failures += 1
                if not (_retain_error_is_permanent(exc) or entry.failures >= _RETAIN_MAX_ATTEMPTS):
                    logger.warning(
                        "Hindsight retain failed (attempt %d), will resend it with the next retain: %s",
                        entry.failures,
                        exc,
                    )
                    return
                logger.warning(
                    "Hindsight retain failed after %d attempt(s), dropping it: %s", entry.failures, exc, exc_info=True
                )
                self._retains_dropped += 1
            with self._retain_backlog_lock:
                if self._retain_backlog and self._retain_backlog[0] is entry:
                    self._retain_backlog.pop(0)

    def _atexit_shutdown(self) -> None:
        try:
            if not self._shutting_down.is_set():
                self.shutdown()
        except Exception as exc:
            logger.debug("Hindsight atexit shutdown failed: %s", exc)

    def _track_retain_ops(self, retain_response, bank_id: str, *, wait: bool = True) -> None:
        """Record the async ``operation_id``/``operation_ids`` of an aretain_batch reply:
        watched until the server settles it (a failed op is re-queued), and with *wait*
        also pending until recall-visible (the next prefetch waits on it). No id
        (older API / sync completion) leaves only the local queue drain as a signal. Keyed
        per ``(bank_id, op_id)`` so multi-bank writes poll the status endpoint of the bank
        each op targets."""
        raw_ids = [
            getattr(retain_response, "operation_id", None),
            *(getattr(retain_response, "operation_ids", None) or []),
        ]
        if ids := [str(op) for op in raw_ids if op]:
            with self._pending_retain_ops_lock:
                if wait:
                    self._pending_retain_ops.update((bank_id, op_id) for op_id in ids)
                for op_id in ids:
                    self._unsettled_retain_ops.setdefault((bank_id, op_id), 0)
                for key in list(self._unsettled_retain_ops)[:-_UNSETTLED_OPS_LIMIT]:
                    del self._unsettled_retain_ops[key]

    def _settle_retain_ops(self) -> None:
        """One status check per watched async retain op (writer thread, before each write).

        A server-side ``failed`` used to count as done, and with the turn buffer already
        cleared that turn was lost (#5591). The server keeps a failed op's payload, so it is
        re-queued there with POST /operations/{id}/retry, up to _SERVER_RETRY_LIMIT times.
        Ops that end after this process exits are not retried. The first status check that
        errors ends the pass: the server is likely down, and every further check would cost
        a full timeout."""
        from hindsight_client_api.exceptions import NotFoundException

        with self._pending_retain_ops_lock:
            watched = list(self._unsettled_retain_ops.items())
        settled: list[tuple[str, str]] = []
        retried: list[tuple[str, str]] = []
        for (bank_id, op_id), retries in watched:
            try:
                resp = self._run_hindsight_operation(
                    lambda client: client.operations.get_operation_status(bank_id=bank_id, operation_id=op_id)
                )
            except NotFoundException:
                settled.append((bank_id, op_id))
                continue
            except Exception as exc:
                logger.debug("Hindsight retain op status check failed for %s: %s", op_id, exc)
                break
            status = str(getattr(resp, "status", "") or "").lower()
            if status in {"completed", "cancelled"}:
                settled.append((bank_id, op_id))
            elif status == "failed":
                settled.append((bank_id, op_id))
                if retries >= _SERVER_RETRY_LIMIT:
                    logger.warning(
                        "Hindsight retain operation %s failed on the server %d times; giving up on it "
                        "(it can still be retried from the control plane)",
                        op_id,
                        retries + 1,
                    )
                    self._retains_dropped += 1
                    continue
                try:
                    self._run_hindsight_operation(
                        lambda client: client.operations.retry_operation(bank_id=bank_id, operation_id=op_id)
                    )
                except Exception as exc:
                    logger.warning("Hindsight retain operation %s failed and could not be retried: %s", op_id, exc)
                    continue
                logger.warning("Hindsight retain operation %s failed on the server; asked it to retry", op_id)
                retried.append((bank_id, op_id))
        with self._pending_retain_ops_lock:
            for key in settled:
                retries = self._unsettled_retain_ops.pop(key, 0)
                if key in retried:
                    self._unsettled_retain_ops[key] = retries + 1

    def _is_retain_op_complete(self, bank_id: str, op_id: str) -> bool:
        """True when a server-side retain op is done or gone (completed ops are evicted,
        so 404 = no longer pending). Transient errors -> False, caller keeps waiting."""
        from hindsight_client_api.exceptions import NotFoundException

        try:
            resp = self._run_hindsight_operation(
                lambda client: client.operations.get_operation_status(bank_id=bank_id, operation_id=op_id)
            )
        except NotFoundException:
            return True
        except Exception as exc:
            logger.debug("Prefetch: operation status check failed for %s: %s", op_id, exc)
            return False
        return str(getattr(resp, "status", "") or "").lower() in {"completed", "failed"}

    def _wait_for_retains_drained(self, timeout: float) -> bool:
        """Block up to *timeout* s for the last retain to become recall-visible
        (prefetch thread only, never the reply path). Two barriers on one budget:
        (1) the writer queue drains — polls ``unfinished_tasks`` rather than
        ``queue.join()`` so a wedged write can't hang the prefetch; (2) the
        server-side async ops complete (async retain returns on acceptance, not
        durability). False on timeout/shutdown."""
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        expired = lambda: deadline is not None and time.monotonic() >= deadline  # noqa: E731
        while self._retain_queue.unfinished_tasks > 0:
            if self._shutting_down.is_set():
                return False
            if expired():
                logger.debug(
                    "Prefetch: retain drain timed out after %.1fs (%d pending)",
                    timeout,
                    self._retain_queue.unfinished_tasks,
                )
                return False
            time.sleep(0.05)
        return self._wait_for_server_retain_ops(expired, timeout)

    def _wait_for_server_retain_ops(self, _expired: Callable[[], bool], timeout: float) -> bool:
        """Poll tracked async retain ops until complete or *_expired()* (deadline
        predicate). Ops still pending at the deadline are DROPPED: keeping them
        would let a permanently failing status endpoint burn the full timeout on
        EVERY later prefetch (a per-turn latency penalty via prefetch()'s bounded
        join). Trades a possibly-stale recall for liveness; WARNING once per prefetch."""
        while True:
            with self._pending_retain_ops_lock:
                pending = list(self._pending_retain_ops)
            if not pending:
                return True
            if self._shutting_down.is_set():
                return False
            done: set[tuple[str, str]] = set()
            for bank_id, op_id in pending:
                if self._shutting_down.is_set():
                    return False
                if _expired():
                    break
                if self._is_retain_op_complete(bank_id, op_id):
                    done.add((bank_id, op_id))
            with self._pending_retain_ops_lock:
                self._pending_retain_ops.difference_update(done)
                if not self._pending_retain_ops:
                    return True
                dropped = len(self._pending_retain_ops) if _expired() else 0
                if dropped:
                    self._pending_retain_ops.clear()
            if dropped:
                logger.warning(
                    "Prefetch: server retain visibility timed out after %.1fs; "
                    "dropping %d unresolved op(s) so later prefetches stay "
                    "bounded (recall may miss the just-completed turn)",
                    timeout,
                    dropped,
                )
                return False
            time.sleep(self._RETAIN_OP_POLL_INTERVAL_S)

    # -- retain target -----------------------------------------------------------

    def _resolve_retain_target(self, fallback_document_id: str) -> tuple[str, str | None]:
        """(document_id, update_mode) from live API capability: >= 0.5.0 reuses the
        stable session-scoped id with ``update_mode='append'``; older APIs get
        *fallback_document_id* (per-process unique) and no update_mode — the only
        way the resume-overwrite fix works there. The /version probe targets the
        embedded client's dynamic per-profile port when running, else api_url.

        On Hindsight ≥ 0.5.0 the API supports ``update_mode='append'``, which lets us reuse a stable
        session-scoped ``document_id`` across process lifecycles without overwriting prior turns. On older
        APIs we fall back to *fallback_document_id* (the per-process unique ``f"{session_id}-{start_ts}"``
        minted at initialize / switch time) and don't pass ``update_mode`` at all — that's the only way the
        resume-overwrite fix (#6654) keeps working on legacy servers.
        """
        url = self._embedded_url if self._mode == "local_embedded" else None
        probe_url = str(url) if url else (self._api_url or "")
        if self._append_unsupported:
            return fallback_document_id, None
        if self._session_id and _check_api_supports_update_mode_append(probe_url, self._api_key):
            return self._session_id, "append"
        return fallback_document_id, None

    # -- lifecycle ---------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "").strip()
        self._parent_session_id = str(kwargs.get("parent_session_id", "") or "").strip()
        # "primary" | "subagent" | "cron" | "flush" from Hermes, plus our own "kanban" below;
        # read by _apply_retain_policy (#4945).
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        # Kanban workers start as plain "primary" CLI sessions; Hermes marks them only in the env (#5555).
        if self._agent_context == "primary" and os.environ.get("HERMES_KANBAN_TASK"):
            self._agent_context = "kanban"
        # Status channel for the retain indicator (recall reports via recall_status()).
        if callable(kwargs.get("status_callback")):
            self._status_callback = kwargs["status_callback"]
        # Gated presentation for automatic startup warnings (agent._emit_warning on CLI).
        self._warning_callback = kwargs.get("warning_callback") if callable(kwargs.get("warning_callback")) else None
        # session_id stays in tags so processes for one session remain filterable together.
        self._document_id = _mint_document_id(self._session_id)
        _warn_if_client_outdated()

        for name in _SESSION_KWARGS:
            setattr(self, f"_{name}", str(kwargs.get(name) or "").strip())
        self._platform = self._platform or "cli"  # after the loop, which would blank it again
        # The one config every setting below reads: the file with this chat's ``chats`` entry over it.
        self._chat_route = resolve_chat_config(
            _load_config(), ChatScope(self._platform, self._chat_id, self._thread_id)
        )
        self._config = cfg = self._chat_route.values
        self._turn_index = self._last_retained_turn_count = 0
        self._session_turns = []
        self._mode = cfg.get("mode", "cloud")
        self._timeout = self._int_setting("timeout", "HINDSIGHT_TIMEOUT", _DEFAULT_TIMEOUT)
        self._idle_timeout = self._int_setting("idle_timeout", "HINDSIGHT_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT)
        if self._mode == "local":  # legacy alias
            self._mode = "local_embedded"
        if self._mode == "local_embedded":
            # Must precede the daemon_embed_manager import, which reads it at import time.
            _export_port_health_grace_timeout(cfg)
            status = _check_local_runtime()
            if not status.available:
                logger.warning(
                    "Hindsight local mode disabled because its runtime could not be imported: %s.%s",
                    status.reason,
                    _local_runtime_hint(status.reason),
                )
                self._mode = "disabled"
                return
        self._apply_connection_settings(cfg)
        self._apply_retain_settings(cfg)
        self._apply_recall_settings(cfg)

        client_version = "unknown"
        with contextlib.suppress(Exception):
            from importlib.metadata import version as pkg_version

            client_version = pkg_version("hindsight-client")
        logger.info(
            "Hindsight initialized: chat=%s (%s), mode=%s, api_url=%s, bank=%s (from %s), budget=%s, memory_mode=%s, "
            "prefetch_method=%s, client=%s",
            self._chat_route.chat_key,
            f"chats[{self._chat_route.matched!r}]" if self._chat_route.matched else "no chats entry",
            self._mode,
            self._api_url,
            self._bank_id,
            self._bank_source,
            self._budget,
            self._memory_mode,
            self._prefetch_method,
            client_version,
        )
        if self._bank_id_template or self._bank_source == "repository config":
            logger.debug(
                "Hindsight bank %s from %s (template %r: profile=%s workspace=%s project=%s platform=%s user=%s, cwd=%s)",
                self._bank_id,
                self._bank_source,
                self._bank_id_template,
                self._agent_identity,
                self._agent_workspace,
                self._project,
                self._platform,
                self._user_id,
                self._cwd,
            )
        logger.debug(
            "Hindsight config: auto_retain=%s, auto_recall=%s, retain_every_n=%d, "
            "retain_async=%s, retain_context=%s, recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, recall_tags=%s",
            self._auto_retain,
            self._auto_recall,
            self._retain_every_n_turns,
            self._retain_async,
            self._retain_context,
            self._recall_max_tokens,
            self._recall_max_input_chars,
            self._tags,
            self._recall_tags,
        )

        if self._mode == "local_embedded":
            overrides = self._chat_route.overrides or {}
            if "profile" not in overrides and any(key in overrides for key in _DAEMON_ENV_SETTINGS):
                # Each chat would rewrite the shared daemon's env back and forth, restarting it every time.
                logger.warning(
                    "Hindsight chats[%r] changes daemon settings without its own profile: the %r daemon "
                    "restarts whenever sessions switch between chats. Give the entry a profile.",
                    self._chat_route.matched,
                    cfg.get("profile", "hermes"),
                )
            self._start_embedded_daemon()
        if not self._writes_disabled:
            self._resend_unsent_retains()

    def _build_write_bank_ids(self) -> list[str]:
        """The deduped, ordered set of banks this provider writes to (and
        recalls from, primary-first).

        Order: primary ``_bank_id`` → own static bank when ``mirror_to_own_bank``
        and different → each ``additional_banks`` entry. Deduped throughout.
        With no multi-bank config this is exactly ``[_bank_id]``.
        """
        ordered: list[str] = [self._bank_id]
        if self._mirror_to_own_bank and self._static_bank_id not in ordered:
            ordered.append(self._static_bank_id)
        for bank in self._additional_bank_ids:
            if bank not in ordered:
                ordered.append(bank)
        return ordered

    def _write_to_banks(self, bank_ids: list[str], write: Callable[[str], Any]) -> None:
        """Write to the primary bank, then to each extra bank with failures isolated.

        A failing primary raises straight away, before any extra bank is touched, so a
        failed retain is reported exactly as on the single-bank path and a retry cannot
        duplicate the item in the extra banks. A failing extra bank is logged and skipped,
        then left out for the cooldown; it misses the writes made during that time.
        """
        write(bank_ids[0])
        for bank_id in bank_ids[1:]:
            if not self._extra_bank_available("retain", bank_id):
                continue
            try:
                write(bank_id)
            except Exception as exc:
                self._log_bank_failure("retain", bank_id, exc)
            else:
                self._mark_bank_ok("retain", bank_id)

    def _extra_bank_available(self, operation: str, bank_id: str) -> bool:
        """False while an extra bank is cooling down after a failed *operation* on it."""
        return time.monotonic() >= self._extra_bank_down_until.get((operation, bank_id), 0.0)

    def _log_bank_failure(self, operation: str, bank_id: str, exc: BaseException) -> None:
        """Start a cooldown for *operation* on a failing extra bank, warning once per outage.
        Recall and writes cool down separately: a slow search must not stop writes."""
        key = (operation, bank_id)
        if key not in self._extra_bank_down_until:
            logger.warning(
                "Hindsight %s: skipping bank %s for %.0fs: %r", operation, bank_id, _EXTRA_BANK_COOLDOWN, exc
            )
        else:
            logger.debug("Hindsight %s: bank %s still failing: %r", operation, bank_id, exc)
        self._extra_bank_down_until[key] = time.monotonic() + _EXTRA_BANK_COOLDOWN

    def _mark_bank_ok(self, operation: str, bank_id: str) -> None:
        """End an extra bank's outage for *operation* on its first success."""
        if self._extra_bank_down_until.pop((operation, bank_id), None) is not None:
            logger.info("Hindsight %s: bank %s is answering again", operation, bank_id)

    def _build_recall_bank_ids(self) -> list[str]:
        """The banks recall searches: the write set (primary first), then each
        ``recall_additional_banks`` entry not already in it.

        Recall-only banks are read and never written, so a shared bank can be
        consulted without this provider's conversations landing in it. A bank
        listed in both ``additional_banks`` and ``recall_additional_banks`` stays
        writable. With no recall-only banks this is exactly the write set.
        """
        ordered = list(self._write_bank_ids)
        for bank in self._recall_additional_bank_ids:
            if bank not in ordered:
                ordered.append(bank)
        return ordered

    def _resolve_template_bank(self) -> str:
        return _resolve_bank_id_template(
            self._bank_id_template,
            fallback=self._static_bank_id,
            profile=self._agent_identity,
            workspace=self._agent_workspace,
            project=self._project,
            platform=self._platform,
            user=self._user_id,
            session=self._session_id,
            chat=self._chat_id,
            thread=self._thread_id,
        )

    def _tag_setting(self, value: Any) -> List[str]:
        """A tag list setting (retain, recall, recall-exclude and reflect tags) with its placeholders filled in."""
        return _fill_tag_templates(
            _normalize_string_list(value),
            profile=self._agent_identity,
            workspace=self._agent_workspace,
            project=self._project,
            platform=self._platform,
            user=self._user_id,
            chat=self._chat_id,
            thread=self._thread_id,
        )

    def _apply_connection_settings(self, cfg: dict) -> None:
        """Endpoint, bank and mode selectors from *cfg* (env fallbacks where documented)."""
        self._api_key = _cloud_api_key(cfg)
        default_url = _DEFAULT_LOCAL_URL if self._mode in {"local_embedded", "local_external"} else _DEFAULT_API_URL
        self._api_url = cfg.get("api_url") or get_secret("HINDSIGHT_API_URL", "") or default_url

        banks = cfg_get(cfg, "banks", "hermes", default={})
        self._bank_id_template = cfg.get("bank_id_template", "") or ""
        # The static bank: the template's fallback and the mirror target, from one place.
        # A non-empty HINDSIGHT_BANK_ID wins over config.json's bank, as documented (#5171); without
        # config.json, _load_config already builds the bank from it.
        # A chats entry's bank_id is this chat's bank: it beats HINDSIGHT_BANK_ID and repository config.
        chat_bank = self._chat_route.chat_bank_id
        self._static_bank_id = (
            chat_bank or get_secret("HINDSIGHT_BANK_ID", "") or cfg.get("bank_id") or banks.get("bankId") or "hermes"
        )
        # Precedence: chats entry bank_id → closest .hindsight/config.toml in a trusted repository →
        # template → static bank_id.
        # The repository is only looked up when something needs it, and only once.
        trusted_dirs = _normalize_string_list(cfg.get("trusted_project_dirs"))
        # {project} may also appear in a tag setting, which is filled from self._project too.
        tag_settings = [cfg.get(key) for key in ("recall_tags", "recall_exclude_tags", "reflect_tags", "retain_tags")]
        tag_settings.append(_scoped_setting("HINDSIGHT_RETAIN_TAGS", ""))
        uses_project = "project" in _template_fields(self._bank_id_template) or any(
            "project" in _template_fields(tag) for value in tag_settings for tag in _normalize_string_list(value)
        )
        root, cwd_bank, self._project = None, None, ""
        if self._cwd and (trusted_dirs or uses_project):
            try:
                root = _repository_root(self._cwd)
            except Exception as exc:
                logger.warning("hindsight: repository lookup for %s failed: %r", self._cwd, exc)
        if root is not None and trusted_dirs:
            cwd_bank = _discover_cwd_bank_id(self._cwd, root, trusted_dirs)
        if root is not None and uses_project:
            try:
                self._project = _project_name(root)
            except Exception as exc:
                logger.warning("hindsight: cannot name the project at %s: %r — {project} is empty", root, exc)
        if chat_bank and not self._bank_id_template:
            self._bank_id, self._bank_source = chat_bank, f"chats[{self._chat_route.matched!r}]"
        elif cwd_bank:
            self._bank_id, self._bank_source = cwd_bank, "repository config"
        else:
            self._bank_id = self._resolve_template_bank()
            self._bank_source = "template" if self._bank_id_template else "bank_id"
        # Multi-bank write/recall (optional, default off): the deduped ordered
        # set used for write fan-out and prioritized recall merge.
        self._mirror_to_own_bank = bool(cfg.get("mirror_to_own_bank", False))
        self._additional_bank_ids = _normalize_string_list(cfg.get("additional_banks"))
        self._write_bank_ids = self._build_write_bank_ids()
        # Recall-only banks (optional, default off): searched after the write
        # set, never written. ``recallAdditionalBanks`` is the name the
        # claude-code and omo integrations use for the same setting.
        self._recall_additional_bank_ids = _normalize_string_list(
            cfg.get("recall_additional_banks") or cfg.get("recallAdditionalBanks")
        )
        budget = cfg.get("recall_budget") or cfg.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"
        # Reflect runs an agentic loop per budget step, so the recall budget that suits search can
        # push reflect past the server's timeout on a big bank (#6246). Unset follows recall_budget,
        # which is what reflect used before this setting existed.
        reflect_budget = cfg.get("reflect_budget")
        self._reflect_budget = reflect_budget if reflect_budget in _VALID_BUDGETS else self._budget
        memory_mode = cfg.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in _SYSTEM_PROMPT_TAILS else "hybrid"
        prefetch_method = cfg.get("recall_prefetch_method") or cfg.get("prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in {"recall", "reflect"} else "recall"

    def _apply_retain_settings(self, cfg: dict) -> None:
        def _cfg_or_env(key: str, env_var: str, default: str = "") -> Any:
            # The env half is the same per-profile value ``_load_config`` resolves through the scope;
            # a raw read here handed a multiplexed secondary the DEFAULT profile's retain shaping back.
            return cfg.get(key) or _scoped_setting(env_var, default)

        self._retain_tags = self._tag_setting(_cfg_or_env("retain_tags", "HINDSIGHT_RETAIN_TAGS"))
        self._tags = self._retain_tags or None
        self._observation_scopes = _normalize_observation_scopes(
            _cfg_or_env("observation_scopes", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES", _DEFAULT_OBSERVATION_SCOPES)
        )
        self._retain_source = str(
            _cfg_or_env("retain_source", "HINDSIGHT_RETAIN_SOURCE", _DEFAULT_RETAIN_SOURCE)
        ).strip()
        self._retain_strategy = str(
            _cfg_or_env("retain_strategy", "HINDSIGHT_RETAIN_STRATEGY", _DEFAULT_RETAIN_STRATEGY)
        ).strip()
        self._retain_user_prefix = (
            str(_cfg_or_env("retain_user_prefix", "HINDSIGHT_RETAIN_USER_PREFIX", "User")).strip() or "User"
        )
        self._retain_assistant_prefix = (
            str(_cfg_or_env("retain_assistant_prefix", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant")).strip()
            or "Assistant"
        )
        self._apply_retain_policy(cfg)

    def _apply_retain_policy(self, cfg: dict) -> None:
        """Pure-config retain knobs (no env/secret reads; ``{}`` yields the defaults)."""
        # Hermes' provider contract: skip automatic writes for non-primary runs (#5555). retain_cron and
        # retain_kanban opt those runs back in; sub-agents report to a parent that retains its own turns.
        context = getattr(self, "_agent_context", "primary")
        retain_kanban = bool(cfg.get("retain_kanban", False))
        skip = (
            context == "subagent"
            or (context == "kanban" and not retain_kanban)
            or (context == "cron" and not cfg.get("retain_cron", False))
        )
        self._auto_retain = bool(cfg.get("auto_retain", True)) and not skip
        # A kanban worker shares its profile's bank to READ it. Skipping auto-retain alone still let the
        # retain tool and the built-in memory mirror write task chatter into it, mostly duplicate reports
        # the parent already retains (#6124). Unless retain_kanban opts it in (#6419), it writes nothing.
        self._writes_disabled = context == "kanban" and not retain_kanban
        # A curated bank may not want MEMORY.md/USER.md notes copied in, in any context (#6725).
        self._mirror_builtin_memory = bool(cfg.get("mirror_builtin_memory", True))
        self._retain_every_n_turns = max(1, _parse_int_setting(cfg.get("retain_every_n_turns"), 1))
        self._retain_context = cfg.get("retain_context", _RETAIN_CONTEXT_DEFAULT)
        self._retain_async = cfg.get("retain_async", True)
        # On by default so the user SEES memory working whether or not the model
        # mentions it; off switch for customer-facing agents (recall_indicator too).
        self._retain_indicator = bool(cfg.get("retain_indicator", True))
        # The next turn's warm prefetch could read BEFORE an async retain is
        # recall-visible; when True it first waits (bounded, off the reply path)
        # for the queue to drain AND the server-side op(s) to complete.
        self._prefetch_waits_for_retain = cfg.get("prefetch_waits_for_retain", True)
        self._prefetch_retain_drain_timeout = _parse_float_setting(cfg.get("prefetch_retain_drain_timeout"), 10.0)

    def _apply_recall_settings(self, cfg: dict) -> None:
        """Recall knobs are pure config too (``{}`` yields the defaults)."""
        # Documented as comma-separated; the client wants a list (a raw string is a 422).
        self._recall_tags = self._tag_setting(cfg.get("recall_tags")) or None
        self._recall_tags_match = cfg.get("recall_tags_match", "any")
        self._recall_exclude_tags = self._tag_setting(cfg.get("recall_exclude_tags")) or None
        # Reflect's own scope, opt-in: reflect searched the whole bank before recall_tags existed,
        # so inheriting them would quietly narrow it for every install that sets them (#5170).
        self._reflect_tags = self._tag_setting(cfg.get("reflect_tags")) or None
        self._reflect_tags_match = cfg.get("reflect_tags_match", "any")
        # A cron prompt opens with Hermes' fixed ~1.1k-char cron notice, so the first
        # recall_max_input_chars of every cron query were that notice, not the job: each fire injected
        # ~4k tokens of unrelated memories (#6145). recall_cron opts back in; the tools still work.
        cron_skip = getattr(self, "_agent_context", "primary") == "cron" and not cfg.get("recall_cron", False)
        self._auto_recall = cfg.get("auto_recall", True) and not cron_skip
        self._recall_sync = bool(cfg.get("recall_sync", False))
        self._recall_max_tokens = _parse_int_setting(cfg.get("recall_max_tokens"), 4096)
        self._recall_split_tokens = cfg.get("recall_max_tokens_mode") == "split"
        self._recall_max_input_chars = _parse_int_setting(cfg.get("recall_max_input_chars"), 800)
        self._recall_query_strip_patterns = _compile_line_patterns(cfg.get("recall_query_strip_patterns"))
        self._session_context_refs = _parse_session_context(cfg.get("session_context"))
        self._page_tools = bool(cfg.get("page_tools", True))
        # Off by default: each page costs model calls on every refresh, so an agent that creates pages
        # on its own initiative runs up the user's bill.
        self._page_create_tool = bool(cfg.get("page_create_tool", False))
        self._session_context_block: str | None = None  # read once, on the first system prompt
        # None -> observation-only (Hindsight's consolidated, deduplicated layer; raw
        # world/experience facts re-ship the evidence they summarize and burn the
        # recall_max_tokens budget); a comma-separated string is accepted for parity
        # with recall_tags; an explicit list broadens or disables the filter.
        configured_types = cfg.get("recall_types")
        if isinstance(configured_types, str):
            self._recall_types = [t.strip() for t in configured_types.split(",") if t.strip()]
        else:
            self._recall_types = list([] if configured_types is None else configured_types) or ["observation"]
        self._recall_min_scores = _normalize_min_scores(cfg.get("recall_min_scores"))
        self._recall_prompt_preamble = cfg.get("recall_prompt_preamble", "")
        self._recall_indicator = bool(cfg.get("recall_indicator", True))

    def _start_embedded_daemon(self) -> None:
        """Start the embedded daemon on a background thread (Rich output -> log file)."""
        # PostgreSQL's initdb refuses root; without this guard the start thread
        # retries forever, reloading embedding models (~958MB RAM, ~33% CPU)
        # with no user-visible error.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            msg = (
                "Hindsight local_embedded mode cannot run as root "
                "(PostgreSQL initdb refuses root). Skipping the embedded "
                "memory daemon. Run Hermes as a non-root user, or switch "
                "to cloud / local_external mode via 'hermes memory setup'."
            )
            logger.warning(msg)
            # Surface to the terminal too — a daemon that never starts would otherwise fail silently and
            # the user would only see Hermes get sluggish (issue #13125). This is an automatic
            # startup diagnostic: it goes through the agent's gated warning sink when wired,
            # otherwise through the shared render boundary; the log line above never does.
            with contextlib.suppress(Exception):
                if self._warning_callback is not None:
                    self._warning_callback(msg)
                else:
                    from gateway.warning_notifications import render_notification

                    render_notification(
                        lambda: print(f"  ⚠ {msg}", file=sys.stderr, flush=True),
                        platform=self._platform,
                    )
            self._mode = "disabled"
            return
        spawn_context_thread(self._daemon_start_worker, name="hindsight-daemon-start").start()

    def _daemon_start_worker(self) -> None:
        import traceback

        log_path = get_hermes_home() / "logs" / "hindsight-embed.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _log(text: str) -> None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text)

        try:
            # Rich console -> our log file (redirecting global fds would capture other threads).
            import hindsight_embed.daemon_embed_manager as dem
            from rich.console import Console

            dem.console = Console(file=open(log_path, "a", encoding="utf-8"), force_terminal=False)

            profile = self._config.get("profile", "hermes")
            # Profile .env out of sync with config -> rewrite and restart a running daemon.
            # Fail-closed on key material: when this process holds no key (no secret
            # scope on this thread) but the file does, a rewrite would destroy the
            # only key copy the daemon subprocess can read. Skip the write AND the
            # stop: restarting the daemon now would boot it keyless, which is the
            # exact outage this guards against. _get_client() below sends the daemon
            # whatever key WAS available (config, secret scope, or the file itself).
            if _profile_env_drifted(self._config):
                if _may_rewrite_profile_env(self._config):
                    _materialize_embedded_profile_env(self._config)
                    if _daemon_is_running(profile):
                        _log("\n=== Config changed, restarting daemon ===\n")
                        _stop_daemon(profile)
                else:
                    logger.warning(
                        "Hindsight profile env for %r holds an LLM API key this process cannot see "
                        "(no secret scope); leaving the file untouched so the daemon keeps its key.",
                        profile,
                    )
                    _log("\n=== Profile env has a key this process cannot see; left untouched ===\n")
            # Builds the client, which starts (or reuses) the daemon. This runs AFTER the
            # profile-env reconciliation above on purpose: the daemon reads that file at boot,
            # so starting first would pin the stale values for the life of the process.
            self._get_client()
            _log("\n=== Daemon started successfully ===\n")
        except Exception as e:
            _log(f"\n=== Daemon startup failed: {e} ===\n" + traceback.format_exc())

    def system_prompt_block(self) -> str:
        mode = self._memory_mode if self._memory_mode in _SYSTEM_PROMPT_TAILS else "hybrid"
        label = "" if mode == "hybrid" else f" ({mode} mode)"
        # With auto_recall off nothing is injected, so hybrid must not promise injected context.
        tail = (
            _TOOL_GUIDANCE + _RETAIN_GUIDANCE
            if mode == "hybrid" and not self._auto_recall
            else _SYSTEM_PROMPT_TAILS[mode]
        )
        if mode != "context" and self._page_tools:
            tail += _PAGE_GUIDANCE
        if self._writes_disabled:  # the retain tool is not offered to a read-only kanban worker
            tail = tail.replace(_RETAIN_GUIDANCE, "")
        block = f"# Hindsight Memory\nActive{label}. Bank: {self._bank_id}, budget: {self._budget}.\n{tail}"
        if self._session_context_block is None:
            self._session_context_block = self._load_session_context()
        return f"{block}\n\n{self._session_context_block}" if self._session_context_block else block

    def _load_session_context(self) -> str:
        """The configured mental models and pages, in order, read once per session (#5285).

        Read once and kept, so the system prompt stays byte-identical for the whole session and
        prompt caching keeps working; a refreshed model shows up in the next session. Each read is
        a stored-document read (no LLM call), bounded to a few seconds; one that fails is skipped
        with a warning. In local_embedded mode a daemon that is not up yet is not started for this:
        the session goes without and the next one picks it up."""
        if not self._session_context_refs:
            return ""
        if self._mode == "local_embedded" and not _daemon_is_running(self._config.get("profile", "hermes")):
            logger.info("Hindsight session_context skipped: the local daemon is not running yet")
            return ""
        sections = []
        for entry in self._session_context_refs:
            try:
                client = self._get_client()
                if entry.kind == "page":
                    get_page = _pages_method(client, "aget_page", "aget_knowledge_page")
                    doc = _run_sync(get_page(bank_id=self._bank_id, page_id=entry.ref), _SESSION_CONTEXT_TIMEOUT)
                    content = getattr(doc, "body", None)
                else:
                    doc = _run_sync(
                        client.aget_mental_model(bank_id=self._bank_id, mental_model_id=entry.ref, detail="content"),
                        _SESSION_CONTEXT_TIMEOUT,
                    )
                    content = getattr(doc, "content", None)
            except Exception as exc:
                logger.warning("Hindsight session_context: could not read %s %s: %s", entry.kind, entry.ref, exc)
                continue
            if not (content or "").strip():
                logger.warning("Hindsight session_context: %s %s has no content yet", entry.kind, entry.ref)
                continue
            name = getattr(doc, "name", None) or entry.ref
            sections.append(f"## {name}\n{content.strip()}")
        if not sections:
            return ""
        return "# Hindsight Session Context\nBackground from memory, read at session start.\n\n" + "\n\n".join(sections)

    # -- recall ------------------------------------------------------------------

    def _recall_disabled(self) -> bool:
        """Guards shared by the async and synchronous recall paths."""
        why = (
            "tools-only mode"
            if self._memory_mode == "tools"
            else "auto_recall disabled"
            if not self._auto_recall
            else "shutting down"
            if self._shutting_down.is_set()
            else None
        )
        if why:
            logger.debug("Prefetch: skipped (%s)", why)
        return why is not None

    def _recall(self, query: str) -> list:
        """Semantic recall across the write set (primary bank first, then
        mirrors/additional banks), then any recall-only banks. A result whose text an
        earlier bank already returned is dropped.

        All banks are queried concurrently, each with the configured budget and
        ``recall_max_tokens`` (an even share of it when ``recall_max_tokens_mode``
        is ``split``). A primary-bank failure raises, exactly as the single-bank
        recall does; a failing extra bank is skipped with a warning.
        Single-bank configs query exactly ``_bank_id``.
        """
        bank_ids = [
            bank_id
            for i, bank_id in enumerate(self._build_recall_bank_ids())
            if i == 0 or self._extra_bank_available("recall", bank_id)
        ]
        max_tokens = self._recall_max_tokens
        # Split over the banks actually queried: an extra bank paused after failures takes no
        # share. Each bank keeps at least one token so a tiny budget still asks every bank (#5426).
        if self._recall_split_tokens:
            max_tokens = max(1, max_tokens // len(bank_ids))
        kwargs: dict = {"query": query, "budget": self._budget, "max_tokens": max_tokens}
        if self._recall_exclude_tags:
            # The server takes tags or tag_groups, not both, and ANDs top-level groups: the include
            # filter moves into the groups. any_strict in the NOT skips only memories that carry an
            # excluded tag, so untagged history stays recallable (#5375).
            groups: list = [{"not": {"tags": self._recall_exclude_tags, "match": "any_strict"}}]
            if self._recall_tags:
                groups.insert(0, {"tags": self._recall_tags, "match": self._recall_tags_match})
            kwargs["tag_groups"] = groups
        elif self._recall_tags:
            kwargs.update(tags=self._recall_tags, tags_match=self._recall_tags_match)
        if self._recall_types:
            kwargs["types"] = self._recall_types
        if self._recall_min_scores:
            kwargs["min_scores"] = self._recall_min_scores

        async def _recall_all(client):
            extra_timeout = float(self._timeout or _DEFAULT_TIMEOUT) * _EXTRA_BANK_TIMEOUT_SHARE
            primary = asyncio.ensure_future(client.arecall(bank_id=bank_ids[0], **kwargs))
            extras = [
                asyncio.ensure_future(asyncio.wait_for(client.arecall(bank_id=bank_id, **kwargs), extra_timeout))
                for bank_id in bank_ids[1:]
            ]
            try:
                # Raising keeps the primary's single-bank contract and lets
                # _run_hindsight_operation retry a stale embedded daemon.
                first = await primary
            except BaseException as exc:
                if not _bank_not_written_yet(exc):
                    for task in extras:
                        task.cancel()
                    raise
                first = exc  # nothing retained to the primary yet; the extra banks still answer
            return [first, *await asyncio.gather(*extras, return_exceptions=True)]

        results, seen = [], set()
        for bank_id, resp in zip(bank_ids, self._run_hindsight_operation(_recall_all)):
            is_extra = bank_id != bank_ids[0]
            if isinstance(resp, BaseException):
                # A bank nothing was retained to answers 404: it has nothing to find, it is not down.
                if not _bank_not_written_yet(resp):
                    self._log_bank_failure("recall", bank_id, resp)
                elif is_extra:
                    self._mark_bank_ok("recall", bank_id)
                continue
            if is_extra:
                self._mark_bank_ok("recall", bank_id)
            bank_results = resp.results or []
            if self._recall_min_scores:
                bank_results = [r for r in bank_results if _clears_min_scores(r, self._recall_min_scores)]
            # Drop only what an earlier bank already returned; one bank's own answer passes
            # through untouched, so a single-bank recall is exactly the server's response.
            kept = [r for r in bank_results if getattr(r, "text", None) not in seen]
            results.extend(kept)
            seen.update(text for r in kept if (text := getattr(r, "text", None)))
        return results

    def _reflect(self, query: str) -> str | None:
        kwargs: Dict[str, Any] = {"bank_id": self._bank_id, "query": query, "budget": self._reflect_budget}
        if self._reflect_tags:
            kwargs.update(tags=self._reflect_tags, tags_match=self._reflect_tags_match)
        try:
            resp = self._run_hindsight_operation(lambda client: client.areflect(**kwargs))
        except Exception as exc:
            if _bank_not_written_yet(exc):
                return None
            raise
        return resp.text

    def _do_recall(self, query: str) -> tuple[str, int]:
        """One recall/reflect for *query* (background prefetch and ``recall_sync`` paths)
        -> (text, memory count); the count is 0 for reflect (synthesis) and on error, where the text
        is ``_RECALL_FAILED_NOTE``."""
        # Boilerplate lines (a provenance header on every message) go first, or they fill the
        # recall_max_input_chars window and recall finds nothing (#5639).
        query = _strip_query_lines(query, self._recall_query_strip_patterns)
        if self._recall_max_input_chars:
            query = query[: self._recall_max_input_chars]
        try:
            if self._prefetch_method == "reflect":
                logger.debug("Recall: calling reflect (bank=%s, query_len=%d)", self._bank_id, len(query))
                return self._reflect(query) or "", 0
            logger.debug(
                "Recall: calling recall (bank=%s, query_len=%d, budget=%s)", self._bank_id, len(query), self._budget
            )
            results = self._recall(query)
            logger.debug("Recall: returned %d results", len(results))
            tz = _hermes_now().tzinfo
            return "\n".join(f"- {_format_memory(r, tz)}" for r in results if r.text), len(results)
        except Exception as e:
            logger.warning("Hindsight recall failed: %s", e, exc_info=True)
            return _RECALL_FAILED_NOTE, 0

    def _finish_prefetch(self, result: str, count: int) -> str:
        """Record indicator state (cleared on empty turns, never a stale count); format the block."""
        if result == _RECALL_FAILED_NOTE:
            self._last_recall_returned, self._last_recall_count = False, 0
            return result
        self._last_recall_returned, self._last_recall_count = bool(result), count if result else 0
        if not result:
            logger.debug("Prefetch: no results available")
            return ""
        logger.debug("Prefetch: returning %d chars of context", len(result))
        header = self._recall_prompt_preamble or (
            "# Hindsight Memory (persistent cross-session context)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def _join_prefetch(self, timeout: float, *, log: bool = False) -> None:
        if not (self._prefetch_thread and self._prefetch_thread.is_alive()):
            return
        if log:
            logger.debug("Prefetch: waiting for background thread to complete")
        self._prefetch_thread.join(timeout=timeout)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Opt-in: recall synchronously against the *current* message so the
        # injected memories match this turn's query, not the previous turn's.
        # See NousResearch/hermes-agent#5820.
        if self._recall_sync:
            return self._finish_prefetch(*(("", 0) if self._recall_disabled() else self._do_recall(query)))
        # Default: the background worker's result for the previous turn (capped join).
        self._join_prefetch(3.0, log=True)
        with self._prefetch_lock:
            result, count = self._prefetch_result, self._prefetch_count
            self._prefetch_result, self._prefetch_count = "", 0
        return self._finish_prefetch(result, count)

    def recall_status(self) -> Optional[RecallStatus]:
        """Count injected by the last prefetch; None if nothing injected or ``recall_indicator=false``."""
        if not self._recall_indicator or not self._last_recall_returned:
            return None
        return RecallStatus(provider_label="Hindsight", count=self._last_recall_count, glyph=_HINDSIGHT_GLYPH)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Sync mode recalls live each turn — nothing to prime in the background.
        if self._recall_sync or self._recall_disabled():
            return
        # A late call for a session we already left must not prime (or bump the generation of) this one (#5628).
        if session_id and session_id != self._session_id:
            return

        with self._prefetch_lock:
            self._prefetch_generation += 1
            generation = self._prefetch_generation

        def _run():
            # Wait (bounded, off the reply path) for the just-completed turn's
            # retain to be recall-visible so the warmed context includes it.
            if self._prefetch_waits_for_retain:
                self._wait_for_retains_drained(self._prefetch_retain_drain_timeout)
            text, count = self._do_recall(query)
            if text:
                with self._prefetch_lock:
                    if generation == self._prefetch_generation:
                        self._prefetch_result, self._prefetch_count = text, count

        self._prefetch_thread = spawn_context_thread(_run, name="hindsight-prefetch")
        self._prefetch_thread.start()

    # -- retain ------------------------------------------------------------------

    def _build_turn_messages(
        self,
        user_content: str,
        assistant_content: str,
        bot_label: str | None = None,
        tool_calls: List[str] | None = None,
    ) -> List[Dict[str, str]]:
        """*bot_label* names the agent that wrote the user side (None: a human did). It becomes
        that message's role and label, so extraction doesn't read a bot's words as the User's
        (#5374). The server reads role as plain text, so any name is accepted. *tool_calls*
        (see ``_tool_call_lines``) go between the two as one ``tool`` message (#6818)."""
        now = _event_timestamp()  # one turn -> every message shares the event timestamp
        user_role, user_label = ("user", self._retain_user_prefix) if bot_label is None else (bot_label, bot_label)
        messages = [{"role": user_role, "content": f"{user_label}: {user_content}", "timestamp": now}]
        if tool_calls:
            messages.append({"role": "tool", "content": "\n".join(tool_calls), "timestamp": now})
        messages.append(
            {"role": "assistant", "content": f"{self._retain_assistant_prefix}: {assistant_content}", "timestamp": now}
        )
        return messages

    def _build_metadata(self, *, message_count: int, turn_index: int) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            # UTC write/audit time (event time lives on the item timestamp).
            "retained_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "message_count": str(message_count),
            "turn_index": str(turn_index),
        }
        if self._retain_source:
            metadata["source"] = self._retain_source
        metadata.update({name: value for name in _METADATA_ATTRS if (value := getattr(self, f"_{name}"))})
        return metadata

    def _build_retain_kwargs(
        self,
        content: str,
        *,
        context: str | None = None,
        metadata: Dict[str, str] | None = None,
        tags: List[str] | None = None,
        occurred_at: str | None = None,
        update_mode: str | None = None,
    ) -> Dict[str, Any]:
        """Build one aretain_batch item. The server resolves occurred_start/end (incl.
        relative phrases in content) from the item timestamp: explicit occurred_at
        wins, else the configured event clock."""
        item: Dict[str, Any] = {
            # See #93568.
            "content": content,
            "metadata": metadata or self._build_metadata(message_count=1, turn_index=self._turn_index),
            "timestamp": (occurred_at or "").strip() or _event_timestamp(),
        }
        merged_tags = _normalize_string_list(list(self._retain_tags) + _normalize_string_list(tags))
        item.update({k: v for k, v in (("context", context), ("update_mode", update_mode)) if v is not None})
        item.update({k: v for k, v in (("tags", merged_tags), ("observation_scopes", self._observation_scopes)) if v})
        # Per-item strategy: overrides the bank default for this item only. Omitted when
        # unset so the bank keeps deciding.
        if self._retain_strategy:
            item["strategy"] = self._retain_strategy
        return item

    def _retain_batch(
        self,
        item: dict,
        *,
        bank_id: str,
        document_id: str | None = None,
        retain_async: bool | None = None,
        operation_id: str | None = None,
    ):
        """Dispatch one item via aretain_batch (bank_id/document_id/retain_async/operation_id
        are call-level args, never item keys). *operation_id* only applies to async retains."""
        kwargs: Dict[str, Any] = {
            "bank_id": bank_id,
            "items": [item],
            "document_id": document_id,
            "retain_async": retain_async,
            "operation_id": operation_id if retain_async else None,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # A retain is only safe to resend under its operation id, which the server replays.
        return self._run_hindsight_operation(
            lambda client: client.aretain_batch(**kwargs), repeatable="operation_id" in kwargs
        )

    def _make_turn_retain_job(
        self, turns: list[str], *, document_id: str, update_mode: str | None, label: str, track_ops: bool = True
    ) -> _PendingRetain:
        """The retain shipping *turns* as one document. Inputs are snapshotted NOW: the
        writer runs after later sync_turn() calls mutate _session_turns/_turn_index/_session_id."""
        # One flat message array: the server only chunks by turn (and merges appends)
        # when the document is a list of messages, not a list of per-turn lists.
        messages = [message for turn in turns for message in json.loads(turn)]
        content = json.dumps(messages, ensure_ascii=False)
        metadata = self._build_metadata(message_count=len(messages), turn_index=self._turn_index)
        if update_mode == "append":
            # An append sets the document's metadata on every memory the session already has, so a
            # value that changes per call would rewrite all of them on each turn (#6241).
            metadata = {k: v for k, v in metadata.items() if k not in _PER_CALL_METADATA}
        # A session named as its own parent (in-place compaction) is no lineage; tagging it
        # would flip the document's tag set and re-consolidate it from scratch.
        parent = self._parent_session_id if self._parent_session_id != self._session_id else ""
        lineage = (("session", self._session_id), ("parent", parent))
        bank_ids = list(self._write_bank_ids)
        return _PendingRetain(
            item=self._build_retain_kwargs(
                content,
                context=_with_relayed_note(self._retain_context),
                metadata=metadata,
                tags=[f"{kind}:{sid}" for kind, sid in lineage if sid] or None,
                occurred_at=_event_timestamp(),
                update_mode=update_mode,
            ),
            bank_ids=bank_ids,
            retain_async=self._retain_async,
            # One id can't serve two banks: the server answers 409.
            operation_ids={bank_id: str(uuid.uuid4()) for bank_id in bank_ids},
            label=label,
            document_id=document_id,
            own_document_id=f"{document_id}-{uuid.uuid4().hex[:12]}",
            track_ops=track_ops,
        )

    def _send_retain(self, retain: _PendingRetain) -> None:
        """Send one queued retain to each of its banks (writer thread)."""
        item, update_mode = retain.item, retain.item.get("update_mode")

        def _write(bank_id: str) -> None:
            logger.debug(
                "Hindsight %s: bank=%s, doc=%s, mode=%s, async=%s, content_len=%d",
                retain.label,
                bank_id,
                retain.document_id,
                update_mode,
                retain.retain_async,
                len(item["content"]),
            )

            def _send(payload: dict, target_document_id: str | None):
                return self._retain_batch(
                    payload,
                    bank_id=bank_id,
                    document_id=target_document_id,
                    retain_async=retain.retain_async,
                    operation_id=retain.operation_ids[bank_id],
                )

            try:
                resp = _send(item, retain.document_id)
            except Exception as exc:
                # A bank with store_document_text off accepts a session's first append (it
                # creates the document) and refuses every later one. Stop appending, and
                # send these turns as their own document so they are not lost.
                if update_mode != "append" or not _is_append_without_document_text(exc):
                    raise
                if not self._append_unsupported:
                    logger.warning(
                        "Hindsight bank %s keeps no document text, so it cannot append to a "
                        "session document; retaining each batch of turns as its own document",
                        bank_id,
                    )
                    self._append_unsupported = True
                # Same operation_id: the refused append was never queued, so the id is unused.
                resp = _send({k: v for k, v in item.items() if k != "update_mode"}, retain.own_document_id)
            # Async retains are only *accepted* here; track the op id(s) so the
            # next-turn prefetch can wait for true server-side completion.
            if retain.retain_async:
                self._track_retain_ops(resp, bank_id, wait=retain.track_ops and self._prefetch_waits_for_retain)

        self._write_to_banks(retain.bank_ids, _write)
        logger.debug("Hindsight %s succeeded", retain.label)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        turn_author: Dict[str, Any] | None = None,
        messages: List[Any] | None = None,
    ) -> None:
        """Enqueue a retain for the current turn (non-blocking; writer thread). Dropped
        once shutdown() fired so post-exit retains never reach aiohttp during teardown.
        *turn_author* (``{"id", "name", "is_bot"}``) is who wrote the user side; *messages* is
        Hermes' transcript so far, read only for this turn's tool calls. Hermes sends each only
        to a signature that names it."""
        why = (
            "auto_retain disabled"
            if not self._auto_retain
            else "shutting down"
            if self._shutting_down.is_set()
            else None
        )
        if why:
            logger.debug("sync_turn: skipped (%s)", why)
            return
        if session_id:
            self._session_id = str(session_id).strip()

        self._session_turns.append(
            json.dumps(
                self._build_turn_messages(
                    user_content, assistant_content, _bot_label(turn_author), _tool_call_lines(messages)
                ),
                ensure_ascii=False,
            )
        )
        self._turn_counter = self._turn_index = self._turn_counter + 1
        if remainder := self._turn_counter % self._retain_every_n_turns:
            logger.debug(
                "sync_turn: buffered turn %d (will retain at turn %d)",
                self._turn_counter,
                self._turn_counter + (self._retain_every_n_turns - remainder),
            )
            return

        document_id, update_mode = self._resolve_retain_target(self._document_id)
        # Append-capable APIs get only the delta since the last retain; legacy /
        # overwrite APIs need the whole session because each retain replaces the document.
        start = self._last_retained_turn_count if update_mode == "append" else 0
        turns_to_retain = self._session_turns[start:]
        if not turns_to_retain:
            logger.debug("sync_turn: skipped append retain; no new turns since last retain")
            return
        logger.debug(
            "sync_turn: retaining %d/%d turns, payload %d chars",
            len(turns_to_retain),
            len(self._session_turns),
            sum(len(t) for t in turns_to_retain),
        )

        job = self._make_turn_retain_job(
            turns_to_retain, document_id=document_id, update_mode=update_mode, label="retain"
        )
        # Indicator fires only past every skip/buffer gate: solely on turns that persist.
        # Model-independent status line; no-op without retain_indicator/status channel.
        if self._retain_indicator and self._status_callback is not None:
            try:
                self._status_callback(f"{_HINDSIGHT_GLYPH} Hindsight — {self._retain_status()}")
            except Exception:
                logger.debug("Retain indicator emit failed (non-fatal)", exc_info=True)
        self._enqueue_retain(job)
        # Advance the watermark only after the delta is queued so a later retain
        # doesn't re-ship turns already handed to the writer.
        if update_mode == "append":
            # Every buffered turn has now been shipped (the retain content was
            # snapshotted into the closure above). Append retains only ever read
            # the un-retained tail — sync_turn slices from the watermark and
            # flush-on-switch flushes what's left — so drop the retained turns
            # instead of letting the buffer grow for the whole session. Overwrite
            # mode is deliberately untouched: it resends the full session each
            # retain and must keep every turn.
            self._session_turns.clear()
            self._last_retained_turn_count = 0

    def on_memory_write(self, action: str, target: str, content: str, metadata: Dict[str, Any] | None = None) -> None:
        """Retain built-in memory adds and replaces (``target``: memory | user) on the writer
        thread, so an entry the agent later prunes from the size-capped file stays in the bank.
        Removals are not retained."""
        if (
            action not in ("add", "replace")
            or not content
            or self._shutting_down.is_set()
            or self._writes_disabled
            or not self._mirror_builtin_memory
        ):
            return
        item = self._build_retain_kwargs(
            content,
            context=f"Hermes built-in {target} memory entry",
            tags=["builtin-memory", f"builtin-target:{target}", f"builtin-action:{action}"],
        )
        bank_ids = list(self._write_bank_ids)
        self._enqueue_retain(
            _PendingRetain(
                item=item,
                bank_ids=bank_ids,
                retain_async=self._retain_async,
                operation_ids={bank_id: str(uuid.uuid4()) for bank_id in bank_ids},  # replay-safe resend (#5218)
                label="builtin-memory",
                track_ops=False,
            )
        )

    def _retain_status(self) -> str:
        """The retain indicator's text: says so when earlier saves failed, not just "saving…".

        Reports each drop once: calling it marks the drops counted so far as shown. Called on the
        turn path only; the writer thread only ever increments ``_retains_dropped``."""
        if (dropped := self._retains_dropped - self._retains_dropped_shown) > 0:
            self._retains_dropped_shown += dropped
            return f"{dropped} earlier save(s) to memory failed and were dropped (see the Hermes log)"
        with self._retain_backlog_lock:  # the writer pops from it
            waiting = sum(1 for entry in self._retain_backlog if entry.failures)
        if waiting:
            return f"couldn't save to memory yet, will retry ({waiting} waiting)"
        return "saving to memory…"

    def _enqueue_retain(self, job: _PendingRetain) -> None:
        """Hand *job* to the (lazily started) writer and arm the atexit drain."""
        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(job)

    # -- tools -------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        schemas = (
            [RECALL_SCHEMA, REFLECT_SCHEMA] if self._writes_disabled else [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]
        )
        if self._page_tools:
            schemas += [SEARCH_PAGES_SCHEMA, READ_PAGE_SCHEMA, LIST_PAGES_SCHEMA]
        if self._page_create_tool and not self._writes_disabled:
            schemas.append(CREATE_PAGE_SCHEMA)
        return schemas

    def read_only_tool_names(self) -> List[str]:
        """Tools a read-only delegated child may use (Hermes ``delegation.memory_read``, #6601).

        Recall and the page reads: retain and page creation write to the bank, and reflect may
        consolidate server-side. Hermes intersects this with :meth:`get_tool_schemas`, so ``context``
        mode exposes none."""
        return ["hindsight_recall", "hindsight_search_pages", "hindsight_read_page", "hindsight_list_pages"]

    def _tool_retain(self, args: dict) -> str:
        if self._writes_disabled:
            return "hindsight_retain is disabled for kanban workers; set retain_kanban to allow it."
        content, context = args["content"], args.get("context")
        item = self._build_retain_kwargs(
            content, context=context, tags=args.get("tags"), occurred_at=args.get("occurred_at")
        )

        def _write(bank_id: str) -> None:
            logger.debug(
                "Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                bank_id,
                len(content),
                context,
            )
            resp = self._retain_batch(item, bank_id=bank_id, retain_async=self._retain_async)
            if self._retain_async:
                self._track_retain_ops(resp, bank_id)

        self._write_to_banks(list(self._write_bank_ids), _write)
        logger.debug("Tool hindsight_retain: success")
        # An async retain is only accepted here; the memory is searchable once the server processes it.
        return "Memory queued for storage." if self._retain_async else "Memory stored successfully."

    def _tool_recall(self, args: dict) -> str:
        query = args["query"]
        logger.debug("Tool hindsight_recall: bank=%s, query_len=%d, budget=%s", self._bank_id, len(query), self._budget)
        results = self._recall(query)
        logger.debug("Tool hindsight_recall: %d results", len(results))
        tz = _hermes_now().tzinfo
        return (
            "\n".join(f"{i}. {_format_memory(r, tz)}" for i, r in enumerate(results, 1))
            or "No relevant memories found."
        )

    def _tool_reflect(self, args: dict) -> str:
        query = args["query"]
        logger.debug(
            "Tool hindsight_reflect: bank=%s, query_len=%d, budget=%s", self._bank_id, len(query), self._reflect_budget
        )
        text = self._reflect(query) or ""
        logger.debug("Tool hindsight_reflect: response_len=%d", len(text))
        return text or "No relevant memories found."

    def _tool_search_pages(self, args: dict) -> str:
        try:
            limit = min(max(int(args.get("limit") or 5), 1), 20)
        except (TypeError, ValueError):
            limit = 5
        resp = self._run_hindsight_operation(
            lambda c: _pages_method(c, "asearch_pages", "asearch_knowledge_base")(
                bank_id=self._bank_id, q=args["query"], limit=limit
            )
        )
        lines = []
        for hit in resp.results or []:
            # The snippet is the page's opening markdown; heading marks are noise on one line.
            snippet = " ".join(re.sub(r"(?m)^\s*#+\s*", "", hit.snippet or "").split())
            if len(snippet) > _PAGE_SNIPPET_CHARS:
                snippet = snippet[:_PAGE_SNIPPET_CHARS].rstrip() + "…"
            lines.append(f"- {hit.name} ({hit.id}){f': {snippet}' if snippet else ''}")
        return "\n".join(lines) or "No pages match."

    def _tool_read_page(self, args: dict) -> str:
        page = self._run_hindsight_operation(
            lambda c: _pages_method(c, "aget_page", "aget_knowledge_page")(
                bank_id=self._bank_id, page_id=args["page_id"]
            )
        )
        # The body opens with its own title, so it goes out as-is (a "# name" line doubled it).
        return (page.body or "").strip() or f"Page {page.name} is not built yet."

    def _tool_list_pages(self, args: dict) -> str:
        # The pre-1.2.1 tree method takes no tag filters, so pass the bank only.
        tree = self._run_hindsight_operation(
            lambda c: _pages_method(c, "aget_page_tree", "aget_knowledge_base_tree")(bank_id=self._bank_id)
        )
        lines: list[str] = []

        def _walk(nodes: list, depth: int) -> None:
            for node in nodes or []:
                label = f"{node.name}/" if node.kind == "folder" else f"{node.name} ({node.id})"
                lines.append(f"{'  ' * depth}- {label}")
                _walk(getattr(node, "children", None), depth + 1)

        _walk(tree.roots, 0)
        if not lines:
            return "No pages yet."
        if len(lines) > _PAGE_LIST_MAX_LINES:
            more = len(lines) - _PAGE_LIST_MAX_LINES
            lines = [*lines[:_PAGE_LIST_MAX_LINES], f"… {more} more; use hindsight_search_pages to find one."]
        return "\n".join(lines)

    def _tool_create_page(self, args: dict) -> str:
        if not self._page_create_tool or self._writes_disabled:
            return "hindsight_create_page is disabled; set page_create_tool to allow it."
        name = args["name"]
        resp = self._run_hindsight_operation(
            lambda c: _pages_method(c, "acreate_page", "acreate_knowledge_page")(
                bank_id=self._bank_id, name=name, source_query=args["question"]
            ),
            repeatable=False,  # a retried create would make a second page
        )
        return f"Created page {name} ({resp.page_id}). Hindsight builds it in the background and keeps it up to date."

    # tool name -> (required args, handler, user-facing failure prefix)
    _TOOL_HANDLERS = {
        "hindsight_retain": (("content",), _tool_retain, "Failed to store memory"),
        "hindsight_recall": (("query",), _tool_recall, "Failed to search memory"),
        "hindsight_reflect": (("query",), _tool_reflect, "Failed to reflect"),
        "hindsight_search_pages": (("query",), _tool_search_pages, "Failed to search pages"),
        "hindsight_read_page": (("page_id",), _tool_read_page, "Failed to read page"),
        "hindsight_list_pages": ((), _tool_list_pages, "Failed to list pages"),
        "hindsight_create_page": (("name", "question"), _tool_create_page, "Failed to create page"),
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        required, handler, failure = self._TOOL_HANDLERS[tool_name]
        for name in required:
            if not args.get(name, ""):
                return tool_error(f"Missing required parameter: {name}")
        try:
            # ensure_ascii=False: escaped CJK costs ~3x the tokens in the tool result (#6561).
            return json.dumps({"result": handler(self, args)}, ensure_ascii=False)
        except Exception as e:
            logger.warning("%s failed: %s", tool_name, e, exc_info=True)
            return tool_error(f"{failure}: {_api_error_detail(e)}")

    # -- session lifecycle -------------------------------------------------------

    def _flush_buffered_turns(self, label: str) -> None:
        """Queue the buffered turns under the CURRENT session/document ids."""
        if not self._session_turns or self._shutting_down.is_set():
            return
        document_id, update_mode = self._resolve_retain_target(self._document_id)
        job = self._make_turn_retain_job(
            list(self._session_turns), document_id=document_id, update_mode=update_mode, label=label, track_ops=False
        )
        # Same writer queue as sync_turn: FIFO behind queued retains, no two threads
        # racing aretain_batch on one document, shutdown drain intact, and a failed flush
        # resent like any other retain.
        self._enqueue_retain(job)

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs
    ) -> None:
        """Rotate per-session state (/resume, /branch, /reset, /new, compression) so
        writes don't land in the previous session's document. Always: flush buffered
        turns under the OLD ids first (``retain_every_n_turns > 1`` would silently
        lose them), join the in-flight prefetch and drop its result (no stale recall
        for the new session), then set ``_session_id``, mint a fresh ``_document_id``
        and clear the batch buffers. ``reset`` is accepted but unneeded: buffer
        clearing is correct for every switch.

        Without this hook, initialize()-cached state (``_session_id``, ``_document_id``, ``_session_turns``,
        ``_turn_counter``) would keep pointing at the previous session and writes would land in the wrong
        document. See hermes-agent#6672.
        Always update ``_session_id`` so metadata and tags on subsequent retains reflect the active session.
        Always clear the accumulated batch buffers (``_session_turns``, ``_turn_counter``, ``_turn_index``)
        — even for /resume and /branch, the new session's batching must start from zero so an in-flight
        retain doesn't flush under the wrong ``_document_id``. See #1303.
        """
        new_id = str(new_session_id or "").strip()
        if not new_id:
            return

        # 1. Flush buffered turns under the OLD identifiers, resolved BEFORE the
        # rotation (legacy: per-process unique; >=0.5.0: session-scoped + append).
        self._flush_buffered_turns("flush-on-switch")

        # 2. Drain the old session's in-flight prefetch and drop its result.
        self._join_prefetch(3.0)
        with self._prefetch_lock:
            self._prefetch_generation += 1  # a worker still running past the join must not land here
            self._prefetch_result = ""

        # 3. Rotate to the new session.
        if parent_session_id:
            self._parent_session_id = str(parent_session_id).strip()
        self._session_id, self._document_id = new_id, _mint_document_id(new_id)
        # A {session} bank template names a bank per session: follow the switch.
        if self._bank_source == "template" and "session" in _template_fields(self._bank_id_template):
            self._bank_id = self._resolve_template_bank()
            self._write_bank_ids = self._build_write_bank_ids()
        self._session_turns = []
        self._turn_counter = self._turn_index = self._last_retained_turn_count = 0
        # The new session reads session_context afresh: the bank may have changed above, and a
        # mental model refreshed since the last session is picked up here, as the docs promise.
        self._session_context_block = None
        logger.debug(
            "Hindsight on_session_switch: new_session=%s parent=%s reset=%s doc=%s",
            self._session_id,
            self._parent_session_id,
            reset,
            self._document_id,
        )

    # Hermes pre-compress checkpoint API v2: with compression.checkpoint_required on, Hermes
    # compacts only after on_pre_compress returns, and keeps the full transcript if it raises.
    pre_compress_checkpoint_api_version = 2

    def on_pre_compress(
        self, messages: list[dict[str, Any]], *, require_checkpoint: bool = False, **kwargs: Any
    ) -> str:
        """Best effort: nothing to do, sync_turn already retained these turns. Required: store the
        messages about to be compacted, synchronously, before Hermes drops them; raise on failure."""
        if not require_checkpoint:
            return ""
        if self._writes_disabled:
            raise RuntimeError("Hindsight cannot write from this run, so it cannot checkpoint")
        # default=str: an odd value in a message (a multimodal part object) must not raise here, or
        # Hermes would refuse every compression of this conversation for a reason no retry can fix.
        content = json.dumps(messages, ensure_ascii=False, default=str)
        # Same messages -> same document: a retried compaction overwrites, it doesn't duplicate.
        document_id = f"{self._session_id}-checkpoint-{uuid.uuid5(uuid.NAMESPACE_OID, content).hex[:16]}"
        item = self._build_retain_kwargs(
            content,
            context=_with_relayed_note(self._retain_context),
            metadata=self._build_metadata(message_count=len(messages), turn_index=self._turn_index),
            tags=[f"session:{self._session_id}", "checkpoint:pre-compress"],
        )
        # A primary-bank failure raises (Hermes then refuses to compact); extra banks are best effort.
        self._write_to_banks(
            list(self._write_bank_ids),
            lambda bank_id: self._retain_batch(item, bank_id=bank_id, document_id=document_id, retain_async=False),
        )
        return ""

    def _close_client(self) -> None:
        """Both modes now hold a plain ``hindsight_client.Hindsight``, so one aclose on the
        shared loop is the whole story. The old local_embedded branch existed because
        ``HindsightEmbedded.close()`` closed its inner sync client from the calling thread
        ("attached to a different loop" before aiohttp released the session); there is no
        wrapper to unwind any more. The daemon deliberately outlives us — it is shared with
        the hindsight-embed CLI and other profiles' clients.
        """
        self._run_sync(self._client.aclose())

    def _spool_unsent_retains(self) -> None:
        """Write the retains not sent yet (failed, in flight, or still queued) to disk for the
        next session, and say so once (#6915). The one in flight may still land; resending it
        is a replay for an async retain, and a possible second copy beats a lost turn."""
        with self._retain_backlog_lock:
            self._unsent_spooled.set()
            unsent = [entry.retain for entry in self._retain_backlog]
            self._retain_backlog = []
        while True:
            try:
                job = self._retain_queue.get_nowait()
            except queue.Empty:
                break
            if job is not _WRITER_SENTINEL:
                unsent.append(job)
        if not unsent:
            return
        folder = get_hermes_home().joinpath(*_UNSENT_RETAINS_DIR)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps([r.model_dump() for r in unsent], ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)  # a half-written file is never picked up
        except OSError as exc:
            logger.warning("Hindsight could not save %d unsent retain(s), they are lost: %s", len(unsent), exc)
            return
        msg = f"Hindsight: {len(unsent)} memory save(s) not sent yet, will send them next time."
        logger.warning("%s (%s)", msg, path)
        if self._warning_callback is not None:
            with contextlib.suppress(Exception):
                self._warning_callback(msg)

    def _resend_unsent_retains(self) -> None:
        """Queue the retains an earlier session could not send before it exited (#6915). Each
        file is claimed by renaming it first, so two sessions starting together never both send it."""
        folder = get_hermes_home().joinpath(*_UNSENT_RETAINS_DIR)
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            claimed = path.with_suffix(f".claimed-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            try:
                path.rename(claimed)
            except OSError:
                continue  # another session took it
            try:
                retains = [
                    _PendingRetain.model_validate(raw) for raw in json.loads(claimed.read_text(encoding="utf-8"))
                ]
            except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
                logger.warning("Hindsight could not read unsent retains in %s, skipping them: %s", claimed, exc)
                continue
            logger.info("Hindsight: sending %d memory save(s) left over from an earlier session", len(retains))
            for retain in retains:
                self._enqueue_retain(retain)
            claimed.unlink(missing_ok=True)  # queued now; a new shutdown spools them again if needed

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: stopping writer + waiting for background threads")
        # Ship turns still waiting for the retain_every_n_turns cadence. Skipped on a
        # cadence boundary: overwrite mode keeps every turn buffered, all already sent.
        if self._turn_counter % self._retain_every_n_turns:
            self._flush_buffered_turns("flush-on-shutdown")
        # Stop accepting retain jobs first so late sync_turn() calls are dropped.
        self._shutting_down.set()
        # The writer finishes in-flight work then exits on the sentinel; the
        # bounded join keeps shutdown predictable even if the daemon is wedged.
        if (writer := self._writer_thread) is not None and writer.is_alive():
            self._retain_queue.put(_WRITER_SENTINEL)
            writer.join(timeout=10.0)
            if writer.is_alive():
                logger.warning("Hindsight writer did not stop within 10s")
        self._spool_unsent_retains()
        self._join_prefetch(5.0)
        with self._prefetch_lock:
            self._prefetch_generation += 1
        if self._atexit_registered:
            # The hook holds this provider alive until exit; the gateway makes one per chat session.
            atexit.unregister(self._atexit_shutdown)
            self._atexit_registered = False
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._close_client()
            self._client = None
        # The module-global loop is intentionally NOT stopped: it's shared by every
        # provider in the process (one per gateway chat session); stopping it would
        # strand siblings' aiohttp sessions ("Unclosed client session"). Daemon
        # thread, reclaimed at process exit.


# The module-global background event loop (_loop / _loop_thread) is intentionally NOT stopped here. It is
# shared across every HindsightMemoryProvider instance in the process — the plugin loader creates a new
# provider per AIAgent, and the gateway creates one AIAgent per concurrent chat session. Stopping the loop
# from one provider's shutdown() strands the aiohttp ClientSession + TCPConnector owned by every sibling
# provider on a dead loop, which surfaces as the "Unclosed client session" / "Unclosed connector" warnings
# reported in #11923. The loop runs on a daemon thread and is reclaimed on process exit; per-session cleanup
# happens via self._client.aclose() above.
def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())
