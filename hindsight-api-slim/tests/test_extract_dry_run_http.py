"""HTTP + engine tests for dry-run fact extraction.

POST /memories/dry-run-extract runs extraction ONLY (no resolution/links/embeddings/persistence) and
returns candidate facts (a subset of the memory-unit shape) plus LLM token usage. Uses the
deterministic mock-LLM `memory` fixture, so extraction yields canned facts without a real provider.
"""

import base64
import os
import uuid
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.config import clear_config_cache
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.extensions import (
    OperationValidatorExtension,
    PrecheckContext,
    ValidationResult,
)

# Dry-run facts are a subset of the memory-unit shape — only fields a fresh extraction produces
# (no storage/consolidation/curation fields, since nothing is persisted).
FACT_KEYS = {
    "text",
    "fact_type",
    "occurred_start",
    "occurred_end",
    "entities",
    # Not a storage field: which of the returned `chunks` this fact came from, so a
    # caller can group facts under the chunk that produced them.
    "chunk_index",
    "attachments",
}


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_dry_run_extracts_without_persisting(api_client, memory):
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    before = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={
            "content": "Alice moved to Berlin in 2021 and works as a nurse.",
            "retain_mission": "Capture where people live and their jobs.",
            "retain_chunk_size": 4000,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["facts"], list) and body["facts"], "expected candidate facts"

    for fact in body["facts"]:
        # A subset of the memory-unit shape — no persistence/curation fields leak in. Null fields
        # are omitted from responses API-wide (#2204), so the optional date fields may be absent;
        # assert no UNEXPECTED key appears and the always-present ones are there.
        assert set(fact) <= FACT_KEYS, f"unexpected keys: {set(fact) - FACT_KEYS}"
        assert {"text", "fact_type", "entities"} <= set(fact)
        assert fact["fact_type"] in ("world", "experience")
        assert isinstance(fact["entities"], list)  # raw extraction → array, not a joined string

    # Token usage is reported alongside the facts.
    assert set(body["usage"]) >= {"input_tokens", "output_tokens", "total_tokens"}

    # No persistence: the bank's stored memory count is unchanged.
    after = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    assert after["total"] == before["total"]


@pytest.mark.asyncio
async def test_dry_run_honors_free_form_entities_override(api_client, memory):
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={
            "content": "Alice moved to Berlin in 2021 and works as a nurse.",
            "entities_allow_free_form": True,
        },
    )

    assert resp.status_code == 200, resp.text
    entities = [entity for fact in resp.json()["facts"] for entity in fact["entities"]]
    assert entities
    assert all(isinstance(entity, str) for entity in entities)


@pytest.mark.asyncio
async def test_dry_run_does_not_create_missing_bank(api_client, memory):
    bank_id = f"dryrun-missing-{uuid.uuid4().hex[:8]}"
    request_context = RequestContext()

    assert (
        await memory.get_bank_profile(
            bank_id=bank_id,
            request_context=request_context,
        )
        is None
    )

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": "Alice moved to Berlin in 2021."},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["facts"]

    assert (
        await memory.get_bank_profile(
            bank_id=bank_id,
            request_context=request_context,
        )
        is None
    )


@pytest.mark.asyncio
async def test_dry_run_rejects_empty_content(api_client, memory):
    """Empty/whitespace-only content is rejected by request validation (422) before the
    billable LLM extraction call runs — matching retain (RetainItem.content) and recall
    (RecallRequest.query), which already reject empty input."""
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    before = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    for content in ("", "   ", "\n\t "):
        resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
            json={"content": content},
        )
        assert resp.status_code == 422, resp.text

    # Rejected before extraction: nothing was persisted.
    after = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    assert after["total"] == before["total"]


@pytest.mark.asyncio
async def test_dry_run_disabled_returns_404(api_client, memory):
    """With HINDSIGHT_API_ENABLE_DRY_RUN_EXTRACT=false the endpoint is removed (returns 404)."""
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    try:
        with patch.dict(os.environ, {"HINDSIGHT_API_ENABLE_DRY_RUN_EXTRACT": "false"}):
            clear_config_cache()  # force get_config() to re-read the patched env
            resp = await api_client.post(
                f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
                json={"content": "Alice moved to Berlin in 2021."},
            )
        assert resp.status_code == 404, resp.text
        assert "disabled" in resp.json()["detail"].lower()
    finally:
        clear_config_cache()  # env restored on with-exit; reset so later tests see the default


@pytest.mark.asyncio
async def test_dry_run_rejects_unknown_override(memory):
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    with pytest.raises(ValueError, match="Unsupported extraction override"):
        await memory.extract_dry_run(
            bank_id,
            "some content",
            overrides={"embeddings_provider": "evil"},
            request_context=RequestContext(),
        )


class _DryRunRejectingValidator(OperationValidatorExtension):
    """Operation validator that rejects the dry-run-extract precheck.

    Models an extension that gates LLM-billable routes (revoked key / exhausted
    balance / rate-limited tenant). It rejects only the ``dry_run_extract``
    operation so the test asserts the dry-run route is actually wired to the
    precheck, not that the validator rejects everything.
    """

    async def precheck(self, ctx: PrecheckContext) -> ValidationResult:
        if ctx.operation == "dry_run_extract":
            return ValidationResult.reject("dry-run extraction not allowed", status_code=402)
        return ValidationResult.accept()

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()


@pytest.mark.asyncio
async def test_dry_run_honors_operation_precheck(api_client, memory):
    """dry-run-extract makes a real LLM call, so it must run the same billing/quota/rate-limit
    precheck the other LLM-billable POST routes (retain/recall/reflect/mental_model_*/files_retain)
    already wire. A validator that rejects the operation must short-circuit the request before any
    extraction runs — without the precheck dependency the route would proceed to a 200."""
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    previous = getattr(memory, "_operation_validator", None)
    memory._operation_validator = _DryRunRejectingValidator({})
    try:
        resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
            json={"content": "Alice moved to Berlin in 2021."},
        )
        assert resp.status_code == 402, resp.text
        assert "not allowed" in resp.json()["detail"].lower()
    finally:
        memory._operation_validator = previous


@pytest.mark.asyncio
async def test_dry_run_disabled_returns_404_even_with_validator(api_client, memory):
    """A disabled dry-run route must 404 before the billing/quota precheck runs, even with a
    configured validator — the feature-flag gate is declared as a dependency before the precheck,
    so it preserves the original "disabled → 404" contract instead of leaking a 402/401/429."""
    bank_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())

    previous = getattr(memory, "_operation_validator", None)
    memory._operation_validator = _DryRunRejectingValidator({})
    try:
        with patch.dict(os.environ, {"HINDSIGHT_API_ENABLE_DRY_RUN_EXTRACT": "false"}):
            clear_config_cache()  # force get_config() to re-read the patched env
            resp = await api_client.post(
                f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
                json={"content": "Alice moved to Berlin in 2021."},
            )
        assert resp.status_code == 404, resp.text
        assert "disabled" in resp.json()["detail"].lower()
    finally:
        clear_config_cache()  # env restored on with-exit; reset so later tests see the default
        memory._operation_validator = previous


async def _config_used_by_dry_run(api_client, memory, bank_id, body):
    """The config the dry run actually hands the extractor.

    Asserted directly rather than through the returned facts: what the strategy
    changes is the settings extraction runs under, and the MockLLM's echo of the
    prompt is a slow, indirect proxy for that.
    """
    captured = {}
    real = fact_extraction.extract_facts_from_text

    async def spy(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return await real(*args, **kwargs)

    with patch.object(fact_extraction, "extract_facts_from_text", spy):
        resp = await api_client.post(f"/v1/default/banks/{bank_id}/memories/dry-run-extract", json=body)
    assert resp.status_code == 200, resp.text
    return captured["config"]


@pytest.mark.asyncio
async def test_dry_run_applies_the_banks_default_strategy(api_client, memory):
    """Retain resolves through `_resolve_retain_config`, which applies the bank's
    `retain_default_strategy` when a caller names none. The dry run resolved the
    config directly and skipped strategies entirely, so it extracted under settings a
    real retain would not have used."""
    bank_id = f"dryrun-strategy-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await memory.update_bank_config(
        bank_id=bank_id,
        updates={
            "retain_strategies": {"wordy": {"retain_extraction_mode": "verbose"}},
            "retain_default_strategy": "wordy",
        },
        request_context=RequestContext(),
    )

    config = await _config_used_by_dry_run(api_client, memory, bank_id, {"content": "Alice moved to Berlin."})

    assert config.retain_extraction_mode == "verbose"


@pytest.mark.asyncio
async def test_dry_run_honors_a_named_strategy(api_client, memory):
    bank_id = f"dryrun-named-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await memory.update_bank_config(
        bank_id=bank_id,
        updates={
            "retain_strategies": {
                "wordy": {"retain_extraction_mode": "verbose"},
                "terse": {"retain_extraction_mode": "verbatim"},
            },
            "retain_default_strategy": "wordy",
        },
        request_context=RequestContext(),
    )

    config = await _config_used_by_dry_run(
        api_client, memory, bank_id, {"content": "Alice moved to Berlin.", "strategy": "terse"}
    )

    assert config.retain_extraction_mode == "verbatim"


@pytest.mark.asyncio
async def test_dry_run_in_chunks_mode_returns_the_chunks_not_llm_facts(api_client, memory):
    """chunks mode never reaches an LLM in a real retain — each chunk is stored
    verbatim — but that branch lives in `extract_facts_from_contents`, which the dry
    run does not go through. It called the model and showed extracted facts for a
    configuration that produces none."""
    bank_id = f"dryrun-chunks-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await memory.update_bank_config(
        bank_id=bank_id,
        updates={"retain_extraction_mode": "chunks"},
        request_context=RequestContext(),
    )
    text = "Alice moved to Berlin in 2021 and works as a nurse."

    resp = await api_client.post(f"/v1/default/banks/{bank_id}/memories/dry-run-extract", json={"content": text})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [f["text"] for f in body["facts"]] == [text], "the chunk is the memory, verbatim"
    # No model call, so nothing to bill. Null fields are omitted API-wide (#2204).
    assert (body.get("usage") or {}).get("total_tokens", 0) == 0


@pytest.mark.asyncio
async def test_dry_run_returns_the_chunks_it_extracted_from(api_client, memory):
    """Already computed on every path and previously discarded. Without them
    `retain_chunk_size` is a number with no visible effect — you can change it and see
    nothing, which is the opposite of what a tester is for."""
    bank_id = f"dryrun-chunks-md-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await memory.update_bank_config(
        bank_id=bank_id, updates={"retain_chunk_size": 60}, request_context=RequestContext()
    )
    text = "Alice moved to Berlin in 2021. " * 6

    resp = await api_client.post(f"/v1/default/banks/{bank_id}/memories/dry-run-extract", json={"content": text})

    assert resp.status_code == 200, resp.text
    chunks = resp.json()["chunks"]
    assert len(chunks) > 1, "a 60-char chunk size must cut this text up"
    assert all(c["text"] for c in chunks)
    assert all("fact_count" in c for c in chunks)


@pytest.mark.asyncio
async def test_each_fact_names_the_chunk_it_came_from(api_client, memory):
    """Derived from the per-chunk counts both extraction paths already return, so the
    real retain pipeline carries nothing extra for a preview's benefit."""
    bank_id = f"dryrun-attr-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await memory.update_bank_config(
        bank_id=bank_id, updates={"retain_chunk_size": 60}, request_context=RequestContext()
    )

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": "Alice moved to Berlin in 2021. " * 6},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    chunks, facts = body["chunks"], body["facts"]
    assert len(chunks) > 1

    # Every fact points at a real chunk, and the per-chunk totals add up: a fact
    # attributed to the wrong chunk would still pass a "not null" check.
    for fact in facts:
        assert 0 <= fact["chunk_index"] < len(chunks)
    for index, chunk in enumerate(chunks):
        attributed = [f for f in facts if f["chunk_index"] == index]
        assert len(attributed) == chunk["fact_count"], f"chunk {index} count disagrees"


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4 mock pdf content"


def _image_block(data: bytes = PNG_BYTES) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(data).decode()},
    }


def _file_block(data: bytes = PDF_BYTES, filename: str = "receipt.pdf") -> dict:
    return {
        "type": "file",
        "filename": filename,
        "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(data).decode()},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


@pytest.mark.parametrize(
    "supports_vision_val,expected_detail",
    [
        (False, "cannot read images"),
        (None, "HINDSIGHT_API_LLM_VISION=true"),
    ],
)
@pytest.mark.asyncio
async def test_dry_run_multimodal_vision_check_raises_422(
    api_client, memory, monkeypatch, supports_vision_val, expected_detail
):
    bank_id = f"dryrun-vlm-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(type(memory._retain_llm_config), "supports_vision", lambda self: supports_vision_val)

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": [_text_block("Here is an invoice:"), _image_block()]},
    )
    assert resp.status_code == 422
    assert expected_detail in resp.json()["detail"]


@pytest.mark.asyncio
async def test_dry_run_multimodal_invalid_blocks(api_client):
    bank_id = "test-bank"
    # Empty content list
    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": []},
    )
    assert resp.status_code == 422

    # Malformed block (missing discriminator type)
    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": [{"text": "hello"}]},
    )
    assert resp.status_code == 422

    # Invalid base64
    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "not-valid-base64!!!"},
                }
            ]
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_dry_run_multimodal_chunk_rendering_and_fact_attribution(api_client, memory):
    bank_id = f"dryrun-mm-{uuid.uuid4().hex[:8]}"
    content = [
        _text_block("Overview of company report:"),
        _image_block(),
        _text_block("Supporting document attached below:"),
        _file_block(filename="audit.pdf"),
    ]

    mock_facts = {
        "facts": [
            {
                "what": "Alice was depicted in the company picture",
                "fact_type": "world",
                "entities": [{"text": "Alice"}],
                "from_attachments": [1],
            },
            {
                "what": "Audit was attached in the financial PDF",
                "fact_type": "world",
                "entities": [{"text": "Audit"}],
                "from_attachments": [2],
            },
            {
                "what": "Both image and document were reviewed together",
                "fact_type": "world",
                "entities": [{"text": "Review"}],
                "from_attachments": [1, 2],
            },
        ]
    }
    memory._retain_llm_config.set_mock_response(mock_facts)

    try:
        resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
            json={"content": content},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        chunks = body["chunks"]
        assert chunks
        # Chunks must have rendered placeholders:
        # block index 1 is image (image/png)
        # block index 3 is file (application/pdf)
        chunk_texts = " ".join(c["text"] for c in chunks)
        assert "[Block #1: image (image/png)]" in chunk_texts
        assert "[Block #3: file (application/pdf)]" in chunk_texts

        facts = body["facts"]
        assert len(facts) == 3

        # Fact 0 attributes to attachment 1 (block index 1, image)
        assert facts[0]["text"] == "Alice was depicted in the company picture"
        assert facts[0]["attachments"] == [{"block_index": 1, "type": "image", "media_type": "image/png"}]

        # Fact 1 attributes to attachment 2 (block index 3, file)
        assert facts[1]["text"] == "Audit was attached in the financial PDF"
        assert facts[1]["attachments"] == [{"block_index": 3, "type": "file", "media_type": "application/pdf"}]

        # Fact 2 attributes to both attachments 1 and 2
        assert facts[2]["text"] == "Both image and document were reviewed together"
        assert facts[2]["attachments"] == [
            {"block_index": 1, "type": "image", "media_type": "image/png"},
            {"block_index": 3, "type": "file", "media_type": "application/pdf"},
        ]
    finally:
        memory._retain_llm_config.set_mock_response(None)


@pytest.mark.asyncio
async def test_dry_run_multimodal_duplicate_identical_attachment_attribution(api_client, memory):
    """When the same attachment data is used across multiple block positions, each occurrence
    is assigned its own block_index in rendered chunks and fact attribution."""
    bank_id = f"dryrun-mm-dup-{uuid.uuid4().hex[:8]}"
    content = [
        _text_block("Before first diagram:"),
        _image_block(PNG_BYTES),
        _text_block("Before second diagram with identical image bytes:"),
        _image_block(PNG_BYTES),
    ]

    mock_facts = {
        "facts": [
            {
                "what": "Fact only from first occurrence",
                "fact_type": "world",
                "entities": [{"text": "First"}],
                "from_attachments": [1],
            },
            {
                "what": "Fact only from second occurrence",
                "fact_type": "world",
                "entities": [{"text": "Second"}],
                "from_attachments": [2],
            },
            {
                "what": "Fact citing both occurrences",
                "fact_type": "world",
                "entities": [{"text": "Both"}],
                "from_attachments": [1, 2],
            },
        ]
    }
    memory._retain_llm_config.set_mock_response(mock_facts)

    try:
        resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
            json={"content": content},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Rendered chunk placeholders must preserve occurrences #1 and #3
        chunk_texts = " ".join(c["text"] for c in body["chunks"])
        assert "[Block #1: image (image/png)]" in chunk_texts
        assert "[Block #3: image (image/png)]" in chunk_texts

        facts = body["facts"]
        assert len(facts) == 3

        # Fact 0 must point specifically to block 1
        assert facts[0]["attachments"] == [{"block_index": 1, "type": "image", "media_type": "image/png"}]
        # Fact 1 must point specifically to block 3 (NOT 1!)
        assert facts[1]["attachments"] == [{"block_index": 3, "type": "image", "media_type": "image/png"}]
        # Fact 2 must point to both block 1 and block 3
        assert facts[2]["attachments"] == [
            {"block_index": 1, "type": "image", "media_type": "image/png"},
            {"block_index": 3, "type": "image", "media_type": "image/png"},
        ]
    finally:
        memory._retain_llm_config.set_mock_response(None)


@pytest.mark.asyncio
async def test_dry_run_multimodal_chunks_mode(api_client, memory):
    bank_id = f"dryrun-mm-chunks-{uuid.uuid4().hex[:8]}"
    content = [
        _text_block("Section 1 header"),
        _image_block(),
        _text_block("Section 2 details"),
        _file_block(filename="spec.pdf"),
    ]

    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/dry-run-extract",
        json={"content": content, "retain_extraction_mode": "chunks"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    chunks = body["chunks"]
    assert chunks
    chunk_texts = " ".join(c["text"] for c in chunks)
    assert "[Block #1: image (image/png)]" in chunk_texts
    assert "[Block #3: file (application/pdf)]" in chunk_texts

    facts = body["facts"]
    assert facts
    # In chunks mode, facts correspond to chunks and carry the chunk's attachments
    all_att_indices = {att["block_index"] for f in facts for att in f["attachments"]}
    assert 1 in all_att_indices
    assert 3 in all_att_indices
