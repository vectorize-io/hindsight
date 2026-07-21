"""Opt-in diagnostic: dump the exact request behind an LLM 4xx rejection.

Some ``400 INVALID_ARGUMENT`` / ``400 Bad Request`` rejections of structured-output
calls are not reproducible by reconstructing the request after the fact — the failing
factor lives in the request as it was actually assembled at runtime. Reconstructed
replays of the same inputs return ``200``, so the only reliable way to see what the
model rejected is to capture the real request at the moment it fails.

This helper is provider-agnostic. Every provider's error handler calls
``dump_request_on_4xx`` with whatever it assembled — a Pydantic config
(google-genai ``GenerateContentConfig``), a kwargs dict (OpenAI / Anthropic /
LiteLLM ``**call_params``), etc. — plus the raised error. The helper self-gates:
it is a no-op unless the ``llm_debug_dump_4xx`` config flag
(``HINDSIGHT_API_LLM_DEBUG_DUMP_4XX``) is enabled AND the error carries a 4xx
status, so callers can drop one unconditional call into each ``except`` block.

Safety / scope:
- Off by default — the config flag is unset in normal operation.
- The serialized config omits message bodies (the ``messages``/``contents``/``input``
  keys are stripped); message previews are length-capped, so an enabled dump can't
  flood logs or spill large bodies.
- Never raises — diagnostics must not break the request path (falls back to ``repr``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Top-level request keys whose values are message bodies. Stripped from the config
# view so the dump never spills large user content — previews are logged separately.
_CONTENT_KEYS = ("messages", "contents", "input")

_PREVIEW_CHARS = 1500
_CONFIG_REPR_CAP = 8000
_ERR_CAP = 200


def _enabled() -> bool:
    from hindsight_api.config import get_config

    return bool(get_config().llm_debug_dump_4xx)


def status_code_of(err: Any) -> int | None:
    """Best-effort HTTP status of a provider error, across SDK error shapes.

    OpenAI/Anthropic expose ``status_code``; google-genai uses ``code``; some wrap the
    status on a ``response``. Returns None when no integer status is discoverable.
    """
    for attr in ("status_code", "code", "http_status"):
        value = getattr(err, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(err, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _serialize_config(request: Any) -> str:
    """Render the request config to a string without message bodies, never raising."""
    try:
        if request is None:
            return "null"
        # Pydantic models (google-genai GenerateContentConfig, SDK params objects).
        dump = getattr(request, "model_dump_json", None)
        if callable(dump):
            return dump(exclude_none=True)
        if isinstance(request, dict):
            view = {k: v for k, v in request.items() if k not in _CONTENT_KEYS}
            return json.dumps(view, ensure_ascii=False, default=str)
        return repr(request)[:_CONFIG_REPR_CAP]
    except Exception:
        return repr(request)[:_CONFIG_REPR_CAP]


@dataclass
class _MessagePreview:
    """A message rendered for the dump: role + extracted text (not yet length-capped)."""

    role: str
    text: str


def _message_preview(msg: Any) -> _MessagePreview:
    """Extract role + text from a message across dict and provider-object shapes."""
    # OpenAI / Anthropic dict: {"role": ..., "content": str | list[block]}
    if isinstance(msg, dict):
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        if isinstance(content, str):
            return _MessagePreview(role, content)
        if isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict):
                    text += block.get("text") or ""
                else:
                    text += getattr(block, "text", "") or ""
            return _MessagePreview(role, text)
        return _MessagePreview(role, "" if content is None else str(content))
    # google-genai Content: role + parts[].text
    role = str(getattr(msg, "role", "?"))
    text = ""
    for part in getattr(msg, "parts", None) or []:
        text += getattr(part, "text", None) or ""
    if not text:
        text = getattr(msg, "content", "") or ""
    return _MessagePreview(role, text)


def _resolve_messages(request: Any, messages: Any) -> Any:
    """Where per-message previews come from: explicit ``messages``, else inside ``request``."""
    if messages is not None:
        return messages
    if isinstance(request, dict):
        for key in _CONTENT_KEYS:
            if key in request:
                return request[key]
    return []


def dump_request_on_4xx(
    *,
    scope: str,
    provider: str,
    model: str,
    err: Any,
    request: Any = None,
    messages: Any = None,
) -> None:
    """Log the exact request behind an LLM 4xx when the diagnostic is enabled.

    No-op unless ``HINDSIGHT_API_LLM_DEBUG_DUMP_4XX`` is truthy and ``err`` carries a
    4xx status. ``request`` is whatever the provider assembled (a Pydantic config, a
    kwargs dict, ...); ``messages`` overrides where the per-message previews come from
    (defaults to the message list found inside ``request``).
    """
    if not _enabled():
        return
    code = status_code_of(err)
    if code is None or not (400 <= code < 500):
        return
    try:
        cfg_repr = _serialize_config(request)
        summary = []
        for msg in _resolve_messages(request, messages) or []:
            m = _message_preview(msg)
            summary.append({"role": m.role, "chars": len(m.text), "preview": m.text[:_PREVIEW_CHARS]})
        logger.error(
            "[LLM_4XX_DUMP] provider=%s model=%s scope=%s code=%s err=%s config=%s contents=%s",
            provider,
            model,
            scope,
            code,
            str(err)[:_ERR_CAP],
            cfg_repr,
            json.dumps(summary, ensure_ascii=False),
        )
    except Exception as dump_exc:  # never let diagnostics break the request path
        logger.warning("[LLM_4XX_DUMP] failed to serialize rejected request: %s", dump_exc)
