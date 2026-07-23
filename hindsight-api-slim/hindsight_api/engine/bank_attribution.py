"""Per-bank provider cost attribution via the OpenAI ``user`` field.

Shared by the OpenAI-compatible LLM path and the OpenAI embeddings path so both
tag outbound requests identically. Opt-in via ``HINDSIGHT_API_LLM_SEND_BANK_AS_USER``;
downstream cost gateways (OpenRouter usage accounting, LiteLLM, Helicone) key spend
on the OpenAI ``user`` field.

Note: when enabled, the bank id is transmitted to the upstream provider as the
end-user identifier. Banks that are themselves end-user identifiers are therefore
forwarded to the provider — which is exactly what the OpenAI ``user`` field is for,
but operators should opt in with that in mind.
"""

from typing import Any

RERANKER_BANK_ID_HEADER = "X-Hindsight-Bank-Id"


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


def reranker_bank_attribution_headers() -> dict[str, str]:
    """Return the fixed per-bank header for trusted remote reranker endpoints."""
    from ..config import get_config
    from .memory_engine import get_current_bank_id

    if not get_config().reranker_send_bank_as_header:
        return {}
    bank_id = get_current_bank_id()
    return {RERANKER_BANK_ID_HEADER: bank_id} if bank_id else {}
