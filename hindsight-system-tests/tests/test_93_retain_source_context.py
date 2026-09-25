"""A decision keeps its own source while earlier options change (#2551).

The stub proves prompt routing and persistence, not language understanding. The
api-slim real-LLM judge test separately evaluates reference resolution/leakage.
"""

from __future__ import annotations

import json
import re

import pytest
from hindsight_client_api.models.content import Content

from hindsight_system_tests.payloads import ExtractedFacts, consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest

pytestmark = pytest.mark.asyncio

TARGET = {"role": "user", "name": "Mira", "content": "Decision marker: I approve option 2 and reject option 1."}


def conversation(choice: str) -> str:
    prior = {
        "role": "assistant",
        "name": "Rowan",
        "content": f"Option 1: Tuesday with Jules. Option 2: {choice}. "
        + "These are proposals awaiting approval. " * 10,
    }
    return json.dumps([prior, TARGET])


def answer_extraction(request: ChatRequest) -> ExtractedFacts:
    content = request.user_text.rsplit("\nContent:\n", 1)[1]
    if "Decision marker" not in content:
        return extracted(fact("Rowan proposed two plans.", who="Rowan"))
    preceding = request.user_text.split("\nContent:\n", 1)[0]
    match = re.search(r"Option 2: ([^.]+)", preceding)
    choice = match.group(1) if match else "unresolved option 2"
    return extracted(fact(f"Mira approved {choice}.", who="Mira"))


async def test_context_edits_append_and_disabling_refresh_the_decision(client, llm, bank_id, settled) -> None:
    await client.acreate_bank(bank_id=bank_id, name="Context regression")
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "retain_chunk_size": 500,
                "retain_structured_chunk_size": 1000,
                "retain_strategies": {"chat": {"retain_context_chars": 1200}},
            }
        },
    )
    llm.on_step("extract_facts").answers_with(answer_extraction)
    llm.on_step("consolidate").returns(consolidation())

    async def retain(text: str, *, strategy: str | None, update_mode: str = "replace") -> None:
        await client.memory.retain_memories(
            bank_id,
            {
                "items": [
                    {
                        "content": Content(actual_instance=text),
                        "document_id": "chat",
                        "update_mode": update_mode,
                        "strategy": strategy,
                    }
                ],
            },
        )
        await settled(bank_id)

    async def decisions() -> list[str]:
        memories = await client.memory.list_memories(bank_id, document_id="chat", limit=100)
        return [m.text for m in memories.items if "Mira approved" in m.text]

    preview = await client.memory.dry_run_extract_memories(
        bank_id, {"content": conversation("Friday with Nia"), "strategy": "chat"}
    )
    preview_decisions = [f for f in preview.facts if f.chunk_index == 1]
    assert len(preview_decisions) == 1
    assert "Friday with Nia" in preview_decisions[0].text
    assert (await client.memory.list_memories(bank_id, limit=100)).total == 0

    for choice in ("Friday with Nia", "Monday with Theo"):
        text = conversation(choice)
        await retain(text, strategy="chat")
        found = await decisions()
        assert len(found) == 1
        assert choice in found[0]
        doc = await client.documents.get_document(bank_id, "chat")
        assert doc.original_text == text
        chunks = await client.documents.list_document_chunks(bank_id, "chat")
        assert chunks.total == 2
        target = next(c for c in chunks.items if c.chunk_index == 1)
        assert json.loads(target.chunk_text) == [TARGET]
        assert "Option 2:" not in target.chunk_text

    # Same source, edited named strategy: force fresh facts under the new setting.
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "retain_strategies": {"chat": {"retain_context_chars": 0}},
            }
        },
    )
    before = len(llm.prompts_for("extract_facts"))
    await retain(conversation("Monday with Theo"), strategy="chat")
    assert len(llm.prompts_for("extract_facts")) == before + 2
    assert "unresolved option 2" in (await decisions())[0]

    # Once the disabled refresh is complete, ordinary unchanged retains still dedup.
    before = len(llm.prompts_for("extract_facts"))
    await retain(conversation("Monday with Theo"), strategy=None)
    assert len(llm.prompts_for("extract_facts")) == before

    # Appending an identical target must see the stored options, including when
    # the options were originally retained with lookback disabled.
    await retain(json.dumps(json.loads(conversation("Friday with Nia"))[:1]), strategy=None)
    await client.banks.update_bank_config(bank_id, {"updates": {"retain_context_chars": 1200}})
    await retain(json.dumps([TARGET]), strategy=None, update_mode="append")
    assert "Friday with Nia" in (await decisions())[0]

    # The next independent document cannot inherit this document's window.
    await client.aretain(bank_id=bank_id, content=json.dumps([TARGET]), document_id="other")
    await settled(bank_id)
    other = await client.memory.list_memories(bank_id, document_id="other", limit=100)
    assert len(other.items) == 1
    assert "unresolved option 2" in other.items[0].text


async def test_json_append_items_share_only_the_stored_base(client, llm, bank_id, settled) -> None:
    await client.acreate_bank(bank_id=bank_id, name="Append isolation regression")
    await client.banks.update_bank_config(bank_id, {"updates": {"retain_context_chars": 1200}})
    llm.on_step("extract_facts").answers_with(answer_extraction)
    llm.on_step("consolidate").returns(consolidation())

    base = [{"role": "assistant", "content": "Option 2: Monday with Theo."}]
    sibling = [{"role": "assistant", "content": "Option 2: Friday with Nia."}]
    await client.aretain(bank_id=bank_id, content=json.dumps(base), document_id="chat")
    await settled(bank_id)
    before = len(llm.prompts_for("extract_facts"))
    await client.memory.retain_memories(
        bank_id,
        {
            "items": [
                {
                    "content": Content(actual_instance=json.dumps(item)),
                    "document_id": "chat",
                    "update_mode": "append",
                }
                for item in (sibling, [TARGET])
            ],
        },
    )
    await settled(bank_id)

    prompts = llm.prompts_for("extract_facts")[before:]
    target_prompts = [p for p in prompts if "Decision marker" in p.rsplit("\nContent:\n", 1)[-1]]
    assert len(target_prompts) == 1
    preceding, target = target_prompts[0].rsplit("\nContent:\n", 1)
    assert "Monday with Theo" in preceding
    assert "Friday with Nia" not in preceding
    assert json.loads(target) == [TARGET]

    memories = await client.memory.list_memories(bank_id, document_id="chat", limit=100)
    decisions = [m.text for m in memories.items if "Mira approved" in m.text]
    assert len(decisions) == 1
    assert "Mira approved Monday with Theo." in decisions[0]
    doc = await client.documents.get_document(bank_id, "chat")
    assert json.loads(doc.original_text) == base + sibling + [TARGET]
    chunks = await client.documents.list_document_chunks(bank_id, "chat")
    assert chunks.total == 3
    assert {c.chunk_index: json.loads(c.chunk_text) for c in chunks.items} == {
        0: base,
        1: sibling,
        2: [TARGET],
    }
