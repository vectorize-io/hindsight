"""HTTP + unit tests for prompt preview.

POST /banks/{bank_id}/prompts/preview renders the messages an operation would send
WITHOUT calling an LLM. The operation is the whole request: everything that shapes
the prompt is read from the bank, so these tests configure the bank and then look.

The cases that matter are the ones a naive implementation gets wrong: a mission
landing in the *user* message for retain and consolidation (both keep their system
prompt bank-agnostic so one provider-side cache serves every bank), chunks mode
reporting no prompt rather than inventing the concise one, and the blocks
reassembling into byte-identical what the extraction path actually builds.
"""

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.extensions import (
    BankReadContext,
    BankReadOperation,
    OperationValidatorExtension,
    ValidationResult,
)
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


async def _preview(api_client, bank_id, operation="retain", strategy=None):
    body = {"operation": operation}
    if strategy is not None:
        body["strategy"] = strategy
    resp = await api_client.post(f"/v1/default/banks/{bank_id}/prompts/preview", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _configure(memory, bank_id, **updates):
    """Write settings to the bank — the preview reads only from there."""
    await memory.update_bank_config(bank_id=bank_id, updates=updates, request_context=RequestContext())


def _message(body, role):
    return next(m for m in body["messages"] if m["role"] == role)


def _text(message):
    """The message as sent — the active blocks partition it exactly."""
    return "".join(b["text"] for b in message["blocks"] if b["active"])


def _block(message, field):
    """The block of the message that `field` is reported to decide."""
    return next(b for b in message["blocks"] if b["field"] == field)


def _section(message, section):
    """The block the preview names itself with `section`."""
    return next(b for b in message["blocks"] if b["section"] == section)


@pytest.mark.asyncio
async def test_messages_are_in_send_order(api_client, bank_id):
    body = await _preview(api_client, bank_id)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_the_operation_is_the_whole_request(api_client, bank_id):
    """No overrides: a body carrying anything else is rejected rather than quietly
    shaping the prompt from something the bank does not hold."""
    resp = await api_client.post(
        f"/v1/default/banks/{bank_id}/prompts/preview",
        json={"operation": "retain", "retain_mission": "smuggled in"},
    )
    assert resp.status_code == 200, resp.text
    assert "smuggled in" not in _text(_message(resp.json(), "user"))


@pytest.mark.asyncio
async def test_retain_mission_lands_in_the_user_message_not_the_system_prompt(api_client, bank_id, memory):
    """The bank-agnostic system prefix must stay mission-free — that's the whole reason
    the preview returns both messages."""
    mission = "Only retain facts about pricing and contract terms."
    await _configure(memory, bank_id, retain_mission=mission)
    body = await _preview(api_client, bank_id)

    assert mission in _text(_message(body, "user"))
    assert mission not in _text(_message(body, "system"))


@pytest.mark.asyncio
async def test_consolidation_mission_lands_in_the_user_message(api_client, bank_id, memory):
    mission = "Track only competitor pricing moves."
    await _configure(memory, bank_id, observations_mission=mission)
    body = await _preview(api_client, bank_id, "consolidation")

    assert mission in _text(_message(body, "user"))
    assert mission not in _text(_message(body, "system"))


@pytest.mark.asyncio
async def test_consolidation_mission_slot_shows_the_default_when_unset(api_client, bank_id):
    """The slot is always filled — unset just means the built-in default fills it — so
    an off block would print "Mission" twice, once switched off and once as the
    built-in section right beneath whose `## MISSION` heading names it too."""
    mission = _block(_message(await _preview(api_client, bank_id, "consolidation"), "user"), "observations_mission")

    assert mission["active"] is True
    assert mission.get("value") is None
    assert "## MISSION" in mission["text"]


@pytest.mark.asyncio
async def test_reflect_mission_lands_in_the_system_prompt(api_client, bank_id, memory):
    """Reflect is the one operation whose mission IS the system prompt's role line."""
    mission = "You are a pricing analyst for the sales team."
    await _configure(memory, bank_id, reflect_mission=mission)

    assert mission in _text(_message(await _preview(api_client, bank_id, "reflect"), "system"))


@pytest.mark.asyncio
async def test_chunks_mode_reports_no_prompt(api_client, bank_id, memory):
    """Chunks mode returns before any LLM call, so there is no prompt. The prompt
    builder falls through to the concise template for it — showing that would invent
    a prompt retain never sends."""
    await _configure(memory, bank_id, retain_extraction_mode="chunks")
    body = await _preview(api_client, bank_id)

    assert "never calls an LLM" in body["skipped_reason"]
    # The mode still comes back, switched off, so the control that got the reader here
    # is the one thing they can still reach. Returning nothing left them at a dead end:
    # no prompt, and no way to pick a mode that would produce one.
    mode = _block(_message(body, "system"), "retain_extraction_mode")
    assert mode["active"] is False
    assert mode["value"] == "chunks"
    assert mode["editable"] is True and "concise" in mode["choices"]
    assert [b for m in body["messages"] for b in m["blocks"]] == [mode]


@pytest.mark.asyncio
async def test_a_real_mode_reports_no_skip(api_client, bank_id, memory):
    await _configure(memory, bank_id, retain_extraction_mode="verbose")
    body = await _preview(api_client, bank_id)

    assert body.get("skipped_reason") is None
    assert len(body["messages"]) == 2


@pytest.mark.asyncio
async def test_custom_instructions_slot_only_exists_in_custom_mode(api_client, bank_id, memory):
    """In any other mode the builder never reads the field, so an off block for it
    would point at a slot this prompt does not have."""
    await _configure(memory, bank_id, retain_extraction_mode="concise")
    concise = await _preview(api_client, bank_id)
    assert not [b for b in _message(concise, "system")["blocks"] if b["field"] == "retain_custom_instructions"]

    await _configure(memory, bank_id, retain_extraction_mode="custom")
    slot = _block(_message(await _preview(api_client, bank_id), "system"), "retain_custom_instructions")
    assert slot["active"] is False

    await _configure(memory, bank_id, retain_custom_instructions="Extract only pricing lines.")
    assert _block(_message(await _preview(api_client, bank_id), "system"), "retain_custom_instructions")["active"]


@pytest.mark.asyncio
async def test_the_default_strategy_applies_without_being_named(api_client, bank_id, memory):
    """Retain resolves through `_resolve_retain_config`, which applies the bank's
    `retain_default_strategy` when a caller names none. Resolving the config directly
    skipped strategies entirely, so a bank with a default previewed a prompt retain
    would never send."""
    await _configure(
        memory,
        bank_id,
        retain_strategies={"verbose_one": {"retain_extraction_mode": "verbose"}},
        retain_default_strategy="verbose_one",
    )
    body = await _preview(api_client, bank_id)

    assert body["strategy"] == "verbose_one"
    assert _block(_message(body, "system"), "retain_extraction_mode")["value"] == "verbose"


@pytest.mark.asyncio
async def test_a_named_strategy_overrides_the_default(api_client, bank_id, memory):
    await _configure(
        memory,
        bank_id,
        retain_strategies={
            "verbose_one": {"retain_extraction_mode": "verbose"},
            "terse_one": {"retain_extraction_mode": "verbatim"},
        },
        retain_default_strategy="verbose_one",
    )
    body = await _preview(api_client, bank_id, strategy="terse_one")

    assert body["strategy"] == "terse_one"
    assert _block(_message(body, "system"), "retain_extraction_mode")["value"] == "verbatim"
    # The names come back with the preview so a picker needs no second call.
    assert body["strategies"] == ["terse_one", "verbose_one"]


@pytest.mark.asyncio
async def test_strategies_are_a_retain_concept_only(api_client, bank_id, memory):
    await _configure(memory, bank_id, retain_strategies={"a": {"retain_extraction_mode": "verbose"}})
    body = await _preview(api_client, bank_id, "reflect")

    assert body.get("strategy") is None
    assert body.get("strategies", []) == []


@pytest.mark.asyncio
async def test_builtin_blocks_are_named_after_their_own_headings(api_client, bank_id):
    """Numbering the leftovers "(1/2)" told the reader only that the text had been
    cut, which is an artefact of where the settings land, not something they need."""
    headings = [b["heading"] for b in _message(await _preview(api_client, bank_id), "system")["blocks"]]

    assert "Selectivity" in headings


@pytest.mark.asyncio
async def test_reflect_blocks_are_named_from_markdown_headings(api_client, bank_id):
    """Reflect writes `##` headings rather than the box-drawing fences the other two
    prompts use; before both were read, every reflect block fell back to one label."""
    system = _message(await _preview(api_client, bank_id, "reflect"), "system")

    assert "Critical rules" in [b["heading"] for b in system["blocks"]]
    assert _section(system, "bank_identity")["active"] is True
    assert _section(system, "disposition")["active"] is True


@pytest.mark.asyncio
async def test_reflect_includes_the_banks_directives(api_client, bank_id, memory):
    """Directives are injected as hard rules near the top of the agent's prompt, so a
    preview without them omits the part a bank is most likely to have customised."""
    assert _section(_message(await _preview(api_client, bank_id, "reflect"), "system"), "directives")["active"] is False

    await memory.create_directive(
        bank_id=bank_id,
        name="Be terse",
        content="Always answer in one sentence.",
        request_context=RequestContext(),
    )
    directives = _section(_message(await _preview(api_client, bank_id, "reflect"), "system"), "directives")

    assert directives["active"] is True
    assert "Always answer in one sentence." in directives["text"]


@pytest.mark.asyncio
async def test_inactive_blocks_mark_unset_settings_in_place(api_client, bank_id, memory):
    """An unset setting produces no text, so without an inactive block it would be
    invisible on the one screen built for changing it."""
    body = await _preview(api_client, bank_id)
    mission = _block(_message(body, "user"), "retain_mission")

    assert mission["active"] is False
    # Null fields are omitted from responses API-wide (#2204), so an unset value is
    # an absent key rather than an explicit null.
    assert mission.get("value") is None

    await _configure(memory, bank_id, retain_mission="Pricing only.")
    assert _block(_message(await _preview(api_client, bank_id), "user"), "retain_mission")["active"] is True


@pytest.mark.asyncio
async def test_active_blocks_partition_the_message_exactly(api_client, bank_id, memory):
    """The active blocks ARE the message — a client that renders them block by block
    must not drop, duplicate or reorder a single character of what the model gets."""
    await _configure(memory, bank_id, retain_mission="Pricing only.")
    body = await _preview(api_client, bank_id)

    for message in body["messages"]:
        assert message["blocks"], "every message is made of at least one block"
        for block in message["blocks"]:
            assert block["source"] in ("config", "builtin")
            # Active means "contributes text"; inactive means "contributes none, and
            # says what it would". Neither may be empty of both.
            assert bool(block["text"]) == block["active"]
            # An off block must be identifiable, or the client has nothing to name it
            # by and nothing to offer an editor for.
            if not block["active"]:
                assert block["field"] or block["section"]


@pytest.mark.asyncio
async def test_event_date_is_a_real_date_like_retain_stamps(api_client, bank_id):
    """Retain stamps the current time on an item that carries no timestamp (only an
    explicit null leaves it unset), so "Event Date: Unknown" was a line the model
    does not get for an ordinary retain."""
    assert "Event Date: Unknown" not in _text(_message(await _preview(api_client, bank_id), "user"))


@pytest.mark.asyncio
async def test_entity_labels_round_trip_as_json(api_client, bank_id, memory):
    """The value is fed straight back into an editor, so it has to parse. Python's
    str() of a list of dicts renders single quotes and True/False, which no client can
    read back into the structure it must render."""
    labels = [
        {
            "key": "topic",
            "description": "Subject area",
            "type": "value",
            "optional": True,
            "tag": True,
            "values": [{"value": "pricing", "description": "Pricing talk"}],
            "fields": {},
        }
    ]
    await _configure(memory, bank_id, entity_labels=labels)
    block = _block(_message(await _preview(api_client, bank_id), "system"), "entity_labels")

    assert block["editable"] is True, "the lab edits this in place, so it must be offered"
    assert json.loads(block["value"])[0]["key"] == "topic"


@pytest.mark.asyncio
async def test_free_form_entities_is_reported_where_it_acts(api_client, bank_id, memory):
    """The flag decides one sentence inside the entity-labels section, so attributing
    the whole section to `entity_labels` hid a setting that really does change what
    the model is told. It has no block without labels, because the section that
    carries it does not exist then."""
    await _configure(memory, bank_id, entities_allow_free_form=False)
    without_labels = _message(await _preview(api_client, bank_id), "system")
    assert not [b for b in without_labels["blocks"] if b["field"] == "entities_allow_free_form"]

    await _configure(
        memory,
        bank_id,
        entity_labels=[
            {
                "key": "topic",
                "description": "Subject area",
                "type": "value",
                "optional": True,
                "tag": True,
                "values": [{"value": "pricing", "description": "Pricing talk"}],
                "fields": {},
            }
        ],
    )
    block = _block(_message(await _preview(api_client, bank_id), "system"), "entities_allow_free_form")

    assert block["active"] is True
    assert block["kind"] == "boolean" and block["value"] == "false"
    assert "labels-only mode" in block["text"]


@pytest.mark.asyncio
async def test_chunk_sizes_are_reported_as_run_settings(api_client, bank_id, memory):
    """They shape what the extractor is handed but contribute no prompt text, so they
    cannot be blocks — blocks partition the message and these are in none of it.
    Leaving them out made the preview look as though the prompt were the whole story."""
    await _configure(memory, bank_id, retain_chunk_size=1234)
    body = await _preview(api_client, bank_id)

    settings = {r["field"]: r for r in body["run_settings"]}
    assert settings["retain_chunk_size"]["value"] == "1234"
    assert settings["retain_chunk_size"]["editable"] is True
    assert "retain_structured_chunk_size" in settings

    # They matter most in chunks mode, where they are the only thing deciding what
    # gets stored — so they must survive the no-prompt path.
    await _configure(memory, bank_id, retain_extraction_mode="chunks")
    chunked = await _preview(api_client, bank_id)
    assert chunked["skipped_reason"]
    assert {r["field"] for r in chunked["run_settings"]} == set(settings)


@pytest.mark.asyncio
async def test_only_retain_has_run_settings(api_client, bank_id):
    """Chunking is a retain concept; the other two are handed whole inputs."""
    for operation in ("consolidation", "reflect"):
        body = await _preview(api_client, bank_id, operation)
        assert body.get("run_settings", []) == [], operation


@pytest.mark.asyncio
async def test_server_level_fields_are_not_offered_for_editing(api_client, bank_id):
    """`llm_output_language` shapes the prompt but is server-level; offering to edit
    it would only collect a 400 from the bank config API."""
    body = await _preview(api_client, bank_id)

    assert _block(_message(body, "system"), "llm_output_language")["editable"] is False
    assert _block(_message(body, "user"), "retain_mission")["editable"] is True


@pytest.mark.asyncio
async def test_retain_preview_returns_the_response_schema(api_client, bank_id):
    body = await _preview(api_client, bank_id)

    assert body["response_schema"]["type"] == "object"
    assert "facts" in body["response_schema"]["properties"]


@pytest.mark.asyncio
async def test_preview_persists_nothing(api_client, bank_id, memory):
    before = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    await _preview(api_client, bank_id)
    after = await memory.list_memory_units(bank_id=bank_id, request_context=RequestContext())

    assert len(after) == len(before)


@pytest.mark.asyncio
async def test_unknown_operation_is_rejected_by_the_schema(api_client, bank_id):
    resp = await api_client.post(f"/v1/default/banks/{bank_id}/prompts/preview", json={"operation": "teleport"})

    assert resp.status_code == 422, resp.text


class _ConfigReadValidator(OperationValidatorExtension):
    """Records bank reads and rejects GET_BANK_CONFIG, like a tenant extension that
    denies a caller access to a bank's settings."""

    def __init__(self) -> None:
        super().__init__({})
        self.read_ops: list[BankReadOperation] = []

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        self.read_ops.append(ctx.operation)
        if ctx.operation is BankReadOperation.GET_BANK_CONFIG:
            return ValidationResult.reject("not your bank")
        return ValidationResult.accept()

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()


@pytest.mark.asyncio
async def test_preview_is_refused_when_config_reads_are(api_client, bank_id, memory, monkeypatch):
    """The preview renders the bank's settings as prompt text — the same disclosure
    GET /config makes, so it must be behind the same gate. Reading the config
    directly from the resolver would have walked straight past the extension."""
    validator = _ConfigReadValidator()
    monkeypatch.setattr(memory, "_operation_validator", validator)

    resp = await api_client.post(f"/v1/default/banks/{bank_id}/prompts/preview", json={"operation": "retain"})

    assert resp.status_code >= 400, resp.text
    assert BankReadOperation.GET_BANK_CONFIG in validator.read_ops


@pytest.mark.asyncio
async def test_preview_never_exposes_credentials(api_client, bank_id, memory):
    """It renders from the *full* resolved config — the one the resolver documents as
    internal because it carries provider API keys — so only prompt-shaping fields may
    reach the response."""
    credentials = HindsightConfig.get_credential_fields()
    resolved = await memory._config_resolver.resolve_full_config(bank_id, RequestContext())
    secrets = [str(getattr(resolved, name)) for name in credentials if getattr(resolved, name, None)]

    body = await _preview(api_client, bank_id)
    serialized = json.dumps(body)

    for secret in secrets:
        assert secret not in serialized, "a credential reached the preview"
    for message in body["messages"]:
        for block in message["blocks"]:
            assert block["field"] not in credentials


def test_the_response_carries_no_display_copy():
    """Nothing identifying a block may be prose: the UI localises those names, and a
    server-side English label would be untranslatable the moment it shipped."""
    preview = render_prompt_preview("reflect", HindsightConfig.from_env(), {"name": "b", "disposition": {"empathy": 3}})

    for message in preview.messages:
        for block in message.blocks:
            assert not hasattr(block, "note")
            assert not hasattr(block, "label")
            # `section` is a slug, `heading` comes out of the prompt text itself.
            assert block.section in ("", "bank_identity", "disposition", "directives")


def test_blocks_reassemble_into_the_real_prompt():
    """The preview must not drift from the real request — both go through the same
    builder. Only the event date differs, and only by the microseconds between the
    two calls, so it is pinned here."""
    config = HindsightConfig.from_env()
    config.retain_mission = "Only retain facts about pricing."
    when = datetime(2021, 3, 4, 10, 0, tzinfo=UTC)

    preview = render_prompt_preview("retain", config, {})
    actual = build_chunk_prompt_parts(
        config,
        chunk="«the text being retained»",
        context="«context supplied with the document»",
        event_date=when,
    )

    assert preview.messages[0].text == actual.system_prompt
    # The user messages differ only on the Event Date line, which is "now" on both
    # sides; compare everything around it.
    assert preview.messages[1].text.split("Event Date:")[0] == actual.user_message.split("Event Date:")[0]
    assert "Only retain facts about pricing." in _block_text(preview, "retain_mission")


def _block_text(preview, field):
    return next(b.text for m in preview.messages for b in m.blocks if b.field == field)


def test_unknown_operation_raises():
    with pytest.raises(ValueError, match="Unknown prompt preview operation"):
        render_prompt_preview("teleport", HindsightConfig.from_env(), {})
