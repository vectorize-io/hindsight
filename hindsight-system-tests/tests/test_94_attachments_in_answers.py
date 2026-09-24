"""A screenshot reaches the caller however the answer was built.

`test_93` pins that an image survives retain and hangs off the fact read from it.
This is the seam after that: a knowledge-base article's screenshot has to reach the
client even when the answer did not come straight from that fact.

Two places it used to drop out. Consolidation folds the fact into an observation,
and reflect prefers observations — but an observation is read off no image, so it
carried none. And reflect's `based_on` rebuilt each cited memory from the agent's
trimmed tool output, with no attachments and no document to trace it to.
"""

from __future__ import annotations

import base64

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

# A 1x1 transparent PNG — the smallest thing that is unambiguously an image.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
ARTICLE = [
    {"type": "text", "text": "KB-1042: SSO is configured on the page shown below."},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
]
OBSERVATION = "SSO is configured on the Admin > Identity > SSO page"
QUERY = "Where do I configure SSO?"


@pytest.fixture
async def kb_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("SSO is configured on the Admin > Identity > SSO page", from_attachments=[1]))
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))
    await client.aretain_batch(bank_id=bank_id, items=[{"content": ARTICLE, "tags": ["kb"]}], document_id="kb-1042")
    await settled(bank_id)
    return bank_id


async def _attachment_id(client, bank: str) -> str:
    document = await client.documents.get_document(bank, "kb-1042")
    assert document.attachments, "the article's screenshot was not kept"
    return document.attachments[0].id


async def test_a_recalled_observation_carries_the_screenshot_behind_it(client, kb_bank):
    screenshot = await _attachment_id(client, kb_bank)

    response = await client.arecall(bank_id=kb_bank, query=QUERY, types=["observation"])

    assert [r.text for r in response.results] == [OBSERVATION]
    assert [a.id for a in response.results[0].attachments or []] == [screenshot]


async def test_reflect_evidence_carries_the_screenshot_and_where_it_came_from(client, llm, kb_bank):
    """Every cited memory — the fact and the observation built on it — comes back
    with the screenshot, and the fact with the document it was retained from."""
    screenshot = await _attachment_id(client, kb_bank)
    reflect_loop(llm, answer="Use the Admin > Identity > SSO page.", query=QUERY)

    response = await client.areflect(bank_id=kb_bank, query=QUERY, include_facts=True)

    cited = {m.type: m for m in response.based_on.memories}
    assert set(cited) == {"world", "observation"}, "reflect should cite both the fact and its observation"
    assert cited["world"].document_id == "kb-1042"
    assert cited["world"].tags == ["kb"]
    for memory in cited.values():
        assert [a.id for a in memory.attachments or []] == [screenshot], f"the {memory.type} lost the screenshot"
    fetched = await client.memory.get_bank_attachment(kb_bank, screenshot)
    assert fetched == PNG
