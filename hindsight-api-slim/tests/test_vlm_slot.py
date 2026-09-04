"""A vision model for the chunks that need one, and the retain LLM for the rest.

Inline attachments made a vision-capable retain LLM mandatory for the whole
bank: one screenshot in one document and every text-only chunk — the
overwhelming majority — was billed against a vision model too. The vision slot
(`HINDSIGHT_API_VLM_*`) exists so that cost follows the pictures.

Everything here is deterministic: which config a chunk is sent to is a routing
decision, not a model judgement, so it is asserted directly rather than judged.
"""

import base64
import inspect
from datetime import datetime

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.response_models import LLMCallResult
from hindsight_api.engine.retain.attachment_content import (
    LoadedAttachment,
    attachment_placeholder,
    compute_attachment_hash,
    short_attachment_id,
)
from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
ATTACHMENT_ID = short_attachment_id(compute_attachment_hash(PNG))


class _RecordingConfig:
    """Stands in for an LLMConfig, remembering that it was the one called."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.provider = name
        self.model = name
        self._calls = calls

    async def call(self, **kwargs):
        self._calls.append(self.provider)
        from hindsight_api.engine.response_models import TokenUsage

        return LLMCallResult(content={"facts": []}, usage=TokenUsage())

    async def get_or_create_cached_prefix(self, *a, **k):
        return None

    def supports_prompt_caching(self) -> bool:
        return False


class _Loader:
    def __init__(self, attachments):
        self._attachments = attachments

    async def load(self, ids):
        return {i: self._attachments[i] for i in ids if i in self._attachments}


def test_the_vision_slot_falls_back_to_the_retain_llm_when_unset():
    """Leaving HINDSIGHT_API_VLM_* unset must change nothing at all."""
    config = HindsightConfig.from_env()

    assert config.vlm_provider is None
    assert config.vlm_model is None
    assert config.vlm_api_key is None
    assert config.vlm_base_url is None


def test_the_key_and_base_url_are_credential_protected():
    """They carry a secret and an internal host, like every other LLM slot's."""
    assert "vlm_api_key" in HindsightConfig._CREDENTIAL_FIELDS
    assert "vlm_base_url" in HindsightConfig._CREDENTIAL_FIELDS


def test_the_slot_is_server_level_like_the_other_per_operation_llms():
    """Matches retain_llm_provider/model, which are deliberately not per-bank."""
    assert "vlm_provider" not in HindsightConfig._CONFIGURABLE_FIELDS
    assert "vlm_model" not in HindsightConfig._CONFIGURABLE_FIELDS


def test_a_base_url_follows_its_own_provider_not_the_retain_one():
    """Naming a vlm_provider without a URL must not inherit retain's host.

    Inheriting would send the request to the wrong endpoint carrying the wrong
    key — a failure that looks like an auth error from a provider the operator
    never configured.
    """
    from hindsight_api.engine.memory_engine import _provider_default_base_url

    assert _provider_default_base_url("groq") == "https://api.groq.com/openai/v1"
    assert _provider_default_base_url("ollama") == "http://localhost:11434/v1"
    assert _provider_default_base_url("openai") == ""
    assert _provider_default_base_url(None) == ""


@pytest.mark.asyncio
async def test_only_a_chunk_carrying_an_attachment_reaches_the_vision_model():
    """The routing decision, asserted on both sides of the branch."""
    calls: list[str] = []
    retain = _RecordingConfig("retain-llm", calls)
    vision = _RecordingConfig("vision-llm", calls)
    loader = _Loader({ATTACHMENT_ID: LoadedAttachment(media_type="image/png", data=PNG)})
    config = HindsightConfig.from_env()

    await _extract_facts_from_chunk(
        chunk="Plain prose with nothing attached to it.",
        chunk_index=0,
        total_chunks=1,
        event_date=datetime(2026, 1, 1),
        context="",
        llm_config=retain,
        config=config,
        attachment_loader=loader,
        vlm_config=vision,
    )
    assert calls == ["retain-llm"], "a text-only chunk must not pay for the vision model"

    await _extract_facts_from_chunk(
        chunk=f"Here is the diagram: {attachment_placeholder(ATTACHMENT_ID)}",
        chunk_index=0,
        total_chunks=1,
        event_date=datetime(2026, 1, 1),
        context="",
        llm_config=retain,
        config=config,
        attachment_loader=loader,
        vlm_config=vision,
    )
    assert calls == ["retain-llm", "vision-llm"], "an attachment-bearing chunk must use the vision model"


def test_a_configured_vision_slot_does_not_inherit_the_retain_fallback_chain():
    """A vision call must not fail over to a text model.

    `_build_llm` would make the vision model member 0 of the retain chain and
    append the retain fallbacks behind it. Those fallbacks are text models —
    that is *why* a separate vision slot was configured — so a failed vision
    call would quietly fail over to one, extract from the prose and drop the
    picture. That is the same silent omission the 422 gate refuses up front, and
    it must not return as a fallback.

    Asserted against the source because the hazard is a single call that is easy
    to reintroduce while copying the retain slot's construction, and it has no
    observable signature until a vision provider is actually down.
    """
    from hindsight_api.engine.memory_engine import MemoryEngine

    assignments = [
        line for line in inspect.getsource(MemoryEngine.__init__).splitlines() if "self._vlm_config =" in line
    ]
    assert assignments, "the vision slot is no longer assigned in __init__"
    assert not any("_build_llm" in line for line in assignments), (
        "the vision slot is being wrapped in a fallback chain — a failed vision call "
        "would fail over to a text model and silently drop the attachment"
    )
