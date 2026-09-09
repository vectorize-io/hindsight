"""A bank's configuration can be exported as a template, and inspected as prompts.

These are the two ways a caller gets at "how is this bank set up" without
guessing.

A **template** is the configuration without the memories: what makes one bank
behave like another. Its value is entirely in fidelity — a template that drops
a field produces a bank that looks configured and behaves differently, and the
difference shows up as an agent quietly reasoning the wrong way rather than as
an error.

A **prompt preview** is the same question from the other end: not what the
settings are, but what the model will actually be told. It renders the prompts
for an operation without spending an LLM call, which is the only way to answer
"is my custom instruction actually in there?" before paying to find out.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.api.bank_templates_api import BankTemplatesApi

pytestmark = pytest.mark.asyncio

CUSTOM_INSTRUCTION = "Always record the neighbourhood, not just the city."


@pytest.fixture
async def configured_bank(client, bank_id) -> str:
    await client.acreate_bank(bank_id=bank_id, name="Alice", disposition_skepticism=5, disposition_empathy=1)
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "retain_custom_instructions": CUSTOM_INSTRUCTION,
                # Load-bearing: the instruction is only consulted in `custom`
                # extraction mode. See the test at the bottom of this file.
                "retain_extraction_mode": "custom",
                "enable_reranking": False,
            }
        },
    )
    return bank_id


@pytest.fixture
async def fresh_bank(client) -> AsyncIterator[str]:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    yield bank
    await client.banks.delete_bank(bank)


def _templates(client) -> BankTemplatesApi:
    return BankTemplatesApi(client.banks.api_client)


async def test_a_template_carries_the_settings_that_were_changed(client, configured_bank):
    """Fidelity is the whole feature. A field dropped here produces a bank that
    looks configured and behaves differently."""
    template = await _templates(client).export_bank_template(configured_bank)

    assert template.version == "1"
    assert template.bank.disposition_skepticism == 5
    assert template.bank.disposition_empathy == 1
    assert template.bank.retain_custom_instructions == CUSTOM_INSTRUCTION


async def test_a_template_leaves_untouched_settings_unset(client, configured_bank):
    """Absent, not defaulted. A template that materialised every field as its
    current default would freeze today's defaults into every bank made from it —
    and silently stop those banks tracking a later change to them.
    """
    template = await _templates(client).export_bank_template(configured_bank)

    assert template.bank.disposition_literalism is None
    assert template.bank.retain_mission is None


async def test_a_template_applied_to_a_new_bank_reproduces_the_configuration(client, configured_bank, fresh_bank):
    """The round trip that makes templates worth having: configure once, stamp
    out many.

    Failing today, deliberately (#4232). The import handler reads its body off
    the raw request, so the spec declares no `requestBody` and every generated
    client's `import_bank_template` has no parameter to put the manifest in.

    Not rewritten against raw HTTP: reaching around the client would hide exactly
    the defect a client-driven suite exists to surface — the export half works
    and hands you a typed manifest with nowhere to send it.
    """
    template = await _templates(client).export_bank_template(configured_bank)

    await _templates(client).import_bank_template(fresh_bank, template)

    config = (await client.banks.get_bank_config(fresh_bank)).config
    assert config["retain_custom_instructions"] == CUSTOM_INSTRUCTION
    assert config["enable_reranking"] is False
    assert config["disposition_skepticism"] == 5


async def test_a_template_carries_no_memories(client, llm, configured_bank, fresh_bank, settled):
    """Configuration, not content. A template that dragged the source bank's
    memories along would leak one tenant's data into every bank stamped from it.
    """
    from hindsight_system_tests.payloads import consolidation, extracted, fact

    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=configured_bank, content="Alice moved to Berlin.")
    await settled(configured_bank)

    template = await _templates(client).export_bank_template(configured_bank)
    await _templates(client).import_bank_template(fresh_bank, template)

    memories = await client.memory.list_memories(fresh_bank, limit=100)
    assert memories.items == []


async def test_the_prompt_preview_renders_what_the_model_will_be_told(client, configured_bank):
    """Answers "is my instruction actually in the prompt?" without paying for a
    call to find out — the question a config UI exists to answer."""
    preview = await client.banks.preview_prompt(configured_bank, {"operation": "retain"})

    assert preview.messages, "a preview with no messages tells the caller nothing"
    # A message is a list of blocks, not a flat string — an image-capable prompt
    # is interleaved content, so the text has to be gathered out of the blocks.
    rendered = "\n".join(block.text or "" for message in preview.messages for block in message.blocks)
    assert CUSTOM_INSTRUCTION in rendered


async def test_the_preview_spends_no_llm_call(client, llm, configured_bank):
    """It is a rendering, not a dry run. Nothing is scripted for a retain here,
    so a preview that reached the model would arrive unstubbed and fail."""
    before = len(llm.calls)

    await client.banks.preview_prompt(configured_bank, {"operation": "retain"})

    assert len(llm.calls) == before


async def test_the_preview_matches_the_prompt_a_real_retain_sends(client, llm, configured_bank, settled):
    """The property that makes a preview worth trusting.

    A preview that renders its own approximation is worse than none: someone
    tunes a prompt against it, ships, and the model reads something else. So the
    instruction is looked for in *both* — the rendered preview and the prompt the
    stub actually received from a real retain.
    """
    from hindsight_system_tests.payloads import consolidation, extracted, fact

    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    preview = await client.banks.preview_prompt(configured_bank, {"operation": "retain"})
    rendered = "\n".join(block.text or "" for message in preview.messages for block in message.blocks)

    await client.aretain(bank_id=configured_bank, content="Alice moved to Berlin.")
    await settled(configured_bank)
    sent = llm.prompts_for("extract_facts")

    assert CUSTOM_INSTRUCTION in rendered
    assert any(CUSTOM_INSTRUCTION in prompt for prompt in sent), (
        "the preview showed an instruction the retain never sent"
    )


async def test_a_custom_instruction_is_ignored_outside_custom_mode(client, llm, bank_id, settled):
    """The footgun, pinned in both halves.

    `retain_custom_instructions` is only consulted when `retain_extraction_mode`
    is `custom`. Set the instruction and leave the mode at its default and it is
    stored, returned on read, and never sent — no warning, no error, and a
    caller reasonably concluding their instruction is in force.

    The preview is *faithful* about this, which is the saving grace: it shows the
    instruction missing, so the config UI tells the truth even though the config
    read does not.
    """
    from hindsight_system_tests.payloads import consolidation, extracted, fact

    await client.acreate_bank(bank_id=bank_id, name="Alice")
    await client.banks.update_bank_config(bank_id, {"updates": {"retain_custom_instructions": CUSTOM_INSTRUCTION}})

    # Stored and read back — which is all the config API tells you.
    config = (await client.banks.get_bank_config(bank_id)).config
    assert config["retain_custom_instructions"] == CUSTOM_INSTRUCTION
    assert config["retain_extraction_mode"] != "custom"

    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    assert not any(CUSTOM_INSTRUCTION in prompt for prompt in llm.prompts_for("extract_facts"))

    preview = await client.banks.preview_prompt(bank_id, {"operation": "retain"})
    rendered = "\n".join(block.text or "" for message in preview.messages for block in message.blocks)
    assert CUSTOM_INSTRUCTION not in rendered
