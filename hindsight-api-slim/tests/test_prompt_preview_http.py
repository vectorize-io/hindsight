"""HTTP + unit tests for prompt preview.

POST /banks/{bank_id}/prompts/preview renders the messages an operation would send
WITHOUT calling an LLM. The tests that matter here are the ones a naive
implementation gets wrong: that a mission override lands in the *user* message for
retain and observations (both keep their system prompt bank-agnostic so one
provider-side cache serves every bank), that chunks mode reports no prompt rather
than inventing the concise one, and that the preview is byte-identical to what the
extraction path actually builds.
"""

import uuid
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.prompt_preview import render_prompt_preview
from hindsight_api.engine.retain.fact_extraction import build_chunk_prompt_parts


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank_id(memory):
    bank = f"preview-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank, request_context=RequestContext())
    return bank


async def _preview(api_client, bank_id, **body):
    resp = await api_client.post(f"/v1/default/banks/{bank_id}/prompts/preview", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _message(body, role):
    return next(m for m in body["messages"] if m["role"] == role)


def _text(message):
    """The message as sent — the active blocks partition it exactly."""
    return "".join(b["text"] for b in message["blocks"] if b["active"])


def _block(message, field):
    """The block of the message that `field` is reported to decide."""
    return next(b for b in message["blocks"] if b["field"] == field)


@pytest.mark.asyncio
async def test_messages_are_in_send_order(api_client, bank_id):
    body = await _preview(api_client, bank_id, operation="retain")
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_retain_mission_lands_in_the_user_message_not_the_system_prompt(api_client, bank_id):
    """The bank-agnostic system prefix must stay mission-free — that's the whole reason
    the preview returns both messages."""
    mission = "Only retain facts about pricing and contract terms."
    body = await _preview(api_client, bank_id, operation="retain", retain_mission=mission)

    assert body["operation"] == "retain"
    assert mission in _text(_message(body, "user"))
    assert mission not in _text(_message(body, "system"))


@pytest.mark.asyncio
async def test_observations_mission_lands_in_the_user_message(api_client, bank_id):
    mission = "Track only competitor pricing moves."
    body = await _preview(api_client, bank_id, operation="observations", observations_mission=mission)

    assert mission in _text(_message(body, "user"))
    assert mission not in _text(_message(body, "system"))


@pytest.mark.asyncio
async def test_reflect_mission_lands_in_the_system_prompt(api_client, bank_id):
    """Reflect is the one operation whose mission IS the system prompt's role line."""
    mission = "You are a pricing analyst for the sales team."
    body = await _preview(api_client, bank_id, operation="reflect", reflect_mission=mission)

    assert mission in _text(_message(body, "system"))


@pytest.mark.asyncio
async def test_chunks_mode_reports_no_prompt(api_client, bank_id):
    """Chunks mode returns before any LLM call, so there is no prompt. The prompt
    builder falls through to the concise template for it — showing that would invent
    a prompt retain never sends."""
    body = await _preview(api_client, bank_id, operation="retain", retain_extraction_mode="chunks")

    assert body["messages"] == []
    assert "never calls an LLM" in body["skipped_reason"]


@pytest.mark.asyncio
async def test_a_real_mode_reports_no_skip(api_client, bank_id):
    body = await _preview(api_client, bank_id, operation="retain", retain_extraction_mode="verbose")
    assert body.get("skipped_reason") is None
    assert len(body["messages"]) == 2


@pytest.mark.asyncio
async def test_blocks_attribute_the_mission(api_client, bank_id):
    mission = "Only retain facts about pricing."
    body = await _preview(api_client, bank_id, operation="retain", retain_mission=mission)

    assert mission in _block(_message(body, "user"), "retain_mission")["text"]


@pytest.mark.asyncio
async def test_blocks_attribute_the_output_language_directive(api_client, bank_id):
    body = await _preview(api_client, bank_id, operation="retain", llm_output_language="Italian")

    assert "Italian" in _block(_message(body, "system"), "llm_output_language")["text"]


@pytest.mark.asyncio
async def test_active_blocks_partition_the_message_exactly(api_client, bank_id):
    """The active blocks ARE the message — a client that renders them block by block
    must not drop, duplicate or reorder a single character of what the model gets."""
    body = await _preview(api_client, bank_id, operation="retain", retain_mission="Pricing only.")

    for message in body["messages"]:
        assert message["blocks"], "every message is made of at least one block"
        for block in message["blocks"]:
            assert block["source"] in ("config", "builtin", "runtime")
            # Active means "contributes text"; inactive means "contributes none, and
            # says what it would". Neither may be empty of both.
            assert bool(block["text"]) == block["active"]
            if not block["active"]:
                assert block["source"] == "config" and block["note"] and block["field"]


@pytest.mark.asyncio
async def test_inactive_blocks_mark_unset_settings_in_place(api_client, bank_id):
    """An unset setting produces no text, so without an inactive block it would be
    invisible on the one screen built for changing it."""
    body = await _preview(api_client, bank_id, operation="retain")

    mission = _block(_message(body, "user"), "retain_mission")
    assert mission["active"] is False
    # Null fields are omitted from responses API-wide (#2204), so an unset value is
    # an absent key rather than an explicit null.
    assert mission.get("value") is None
    assert "prepended" in mission["note"].lower()

    # And it becomes a real block once set.
    body = await _preview(api_client, bank_id, operation="retain", retain_mission="Pricing only.")
    assert _block(_message(body, "user"), "retain_mission")["active"] is True


@pytest.mark.asyncio
async def test_custom_instructions_slot_only_exists_in_custom_mode(api_client, bank_id):
    """In any other mode the builder never reads the field, so an off block for it
    would point at a slot this prompt does not have."""
    concise = await _preview(api_client, bank_id, operation="retain", retain_extraction_mode="concise")
    assert not [b for b in _message(concise, "system")["blocks"] if b["field"] == "retain_custom_instructions"]

    custom = await _preview(api_client, bank_id, operation="retain", retain_extraction_mode="custom")
    slot = _block(_message(custom, "system"), "retain_custom_instructions")
    assert slot["active"] is False
    assert "falls back" in slot["note"]

    written = await _preview(
        api_client,
        bank_id,
        operation="retain",
        retain_extraction_mode="custom",
        retain_custom_instructions="Extract only pricing lines.",
    )
    assert _block(_message(written, "system"), "retain_custom_instructions")["active"] is True


@pytest.mark.asyncio
async def test_builtin_blocks_are_named_after_their_own_headings(api_client, bank_id):
    """Numbering the leftovers "(1/2)" told the reader only that the text had been
    cut, which is an artefact of how it is cut, not something they need."""
    body = await _preview(api_client, bank_id, operation="retain")

    labels = [b["label"] for b in _message(body, "system")["blocks"] if b["source"] == "builtin"]
    assert "Selectivity rules" in labels
    assert not any("(1/" in label for label in labels)


@pytest.mark.asyncio
async def test_server_level_fields_are_not_offered_for_editing(api_client, bank_id):
    """`llm_output_language` shapes the prompt but is server-level; offering to edit
    it would only collect a 400 from the bank config API."""
    body = await _preview(api_client, bank_id, operation="retain")

    assert _block(_message(body, "system"), "llm_output_language")["editable"] is False
    assert _block(_message(body, "user"), "retain_mission")["editable"] is True


@pytest.mark.asyncio
async def test_blocks_reassemble_into_the_real_prompt(api_client, bank_id, memory):
    """Byte-identity against what the extraction path itself builds, through the parts."""
    from hindsight_api.engine.retain.fact_extraction import build_chunk_prompt_parts

    # Pinned: with no timestamp both sides default to "now" and would differ by the
    # microseconds between the two calls.
    when = "2021-03-04T10:00:00Z"
    body = await _preview(
        api_client, bank_id, operation="retain", content="Acme raised prices.", context="a call", timestamp=when
    )
    config = await memory._config_resolver.resolve_full_config(bank_id, RequestContext())
    actual = build_chunk_prompt_parts(
        config, chunk="Acme raised prices.", context="a call", event_date=datetime.fromisoformat(when)
    )

    assert _text(_message(body, "system")) == actual.system_prompt
    assert _text(_message(body, "user")) == actual.user_message


@pytest.mark.asyncio
async def test_content_is_rendered_and_placeholdered_when_omitted(api_client, bank_id):
    with_content = await _preview(api_client, bank_id, operation="retain", content="Acme moved to a usage-based plan.")
    assert "Acme moved to a usage-based plan." in _text(_message(with_content, "user"))

    without = await _preview(api_client, bank_id, operation="retain")
    assert "«the text being retained»" in _text(_message(without, "user"))


@pytest.mark.asyncio
async def test_event_date_defaults_to_now_like_retain_does(api_client, bank_id):
    """Retain stamps the current time on an item that carries no timestamp (only an
    explicit null leaves it unset), so a preview showing "Event Date: Unknown" was
    showing a line the model does not get for an ordinary retain."""
    body = await _preview(api_client, bank_id, operation="retain")
    assert "Event Date: Unknown" not in _text(_message(body, "user"))

    body = await _preview(api_client, bank_id, operation="retain", timestamp="2021-03-04T10:00:00Z")
    assert "March 04, 2021" in _text(_message(body, "user"))


@pytest.mark.asyncio
async def test_retain_preview_returns_the_response_schema(api_client, bank_id):
    body = await _preview(api_client, bank_id, operation="retain")
    assert body["response_schema"]["type"] == "object"
    assert "facts" in body["response_schema"]["properties"]


@pytest.mark.asyncio
async def test_preview_persists_nothing(api_client, bank_id, memory):
    before = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    await _preview(api_client, bank_id, operation="retain", content="Alice moved to Berlin in 2021.")
    after = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    assert len(after) == len(before)


@pytest.mark.asyncio
async def test_unknown_operation_is_rejected_by_the_schema(api_client, bank_id):
    resp = await api_client.post(f"/v1/default/banks/{bank_id}/prompts/preview", json={"operation": "teleport"})
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_unsupported_override_is_rejected(api_client, bank_id, memory):
    """Overrides are validated against an allowlist in the engine, not the request model."""
    with pytest.raises(ValueError, match="Unsupported prompt preview override"):
        await memory.preview_prompt(
            bank_id,
            "retain",
            overrides={"database_url": "postgres://nope"},
            request_context=RequestContext(),
        )


def test_preview_matches_what_extraction_actually_builds():
    """The preview must not drift from the real request — both go through the same builder."""
    config = HindsightConfig.from_env()
    config.retain_mission = "Only retain facts about pricing."

    when = datetime(2021, 3, 4, 10, 0, tzinfo=UTC)
    preview = render_prompt_preview(
        "retain", config, {}, content="Acme raised prices.", context="a sales call", event_date=when
    )
    actual = build_chunk_prompt_parts(config, chunk="Acme raised prices.", context="a sales call", event_date=when)

    assert preview.messages[0].text == actual.system_prompt
    assert preview.messages[1].text == actual.user_message
    assert "Only retain facts about pricing." in next(
        b.text for b in preview.messages[1].blocks if b.field == "retain_mission"
    )


def test_unknown_operation_raises():
    with pytest.raises(ValueError, match="Unknown prompt preview operation"):
        render_prompt_preview("teleport", HindsightConfig.from_env(), {})
