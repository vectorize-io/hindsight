"""Per-bank provider cost attribution.

Two transport mechanisms, both opt-in:

1. OpenAI ``user`` field (``HINDSIGHT_API_LLM_SEND_BANK_AS_USER``) — shared by the
   OpenAI-compatible LLM path and the OpenAI embeddings path. Downstream cost
   gateways (OpenRouter usage accounting, LiteLLM, Helicone) key spend on the
   OpenAI ``user`` field.
2. ``X-Hindsight-Bank-Id`` header (``HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER``)
   — shared by the Cohere-compatible remote rerank path (Cohere base_url,
   OpenRouter, ZeroEntropy, SiliconFlow, Alibaba) and the LiteLLM proxy rerank
   path. The Cohere rerank wire format has no OpenAI ``user`` field, so rerank
   attribution rides a header instead. Does not apply to local, TEI, native
   Cohere SDK, LiteLLM SDK, or Google rerankers.

Note: when enabled, the bank id is transmitted to the upstream provider. Banks
that are themselves end-user identifiers are therefore forwarded to the
provider — operators should opt in with that in mind, and only against trusted
gateways/proxies for the header path.
"""

from typing import Any

RERANK_BANK_HEADER = "X-Hindsight-Bank-Id"


def apply_bank_attribution(request: dict[str, Any]) -> None:
    """Tag ``request`` with ``user=<bank_id>`` for per-bank cost attribution.

    Mutates ``request`` in place. No-op when the flag is off, no bank is in context,
    or the caller already set ``user`` — we never override an explicit value.
    """
    if "user" in request:
        return
    # Lazy imports: memory_engine imports the embeddings/provider modules that call
    # this, so a top-level import of memory_engine here would be circular.
    from ..config import get_config
    from .memory_engine import get_current_bank_id

    if not get_config().llm_send_bank_as_user:
        return
    bank_id = get_current_bank_id()
    if bank_id:
        request["user"] = bank_id


def bank_attribution_headers() -> dict[str, str]:
    """Per-bank attribution headers for outbound remote rerank requests.

    Empty dict when the flag is off or no bank is bound — callers splat it
    unconditionally. The Cohere rerank wire format has no OpenAI ``user`` field,
    so rerank attribution rides a header instead of the body.
    """
    from ..config import get_config  # lazy: same circular-import reason as above
    from .memory_engine import get_current_bank_id

    if not get_config().reranker_send_bank_as_header:
        return {}
    bank_id = get_current_bank_id()
    # Header values must be ASCII per httpx; bank ids are free-form path params, so
    # non-ASCII ids are skipped rather than percent-encoded — attribution is
    # best-effort, and an encoded id wouldn't match billing dashboards anyway.
    if not bank_id or not bank_id.isascii():
        return {}
    return {RERANK_BANK_HEADER: bank_id}
