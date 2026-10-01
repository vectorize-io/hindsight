"""Two people share a bank; a validator confines each to their own tag plus the team's rules.

Kate writes a note under `user:kate`. One sentence in it is a team rule, and the bank's
entity label `kind` (with `tag: true`) projects it into the shared `kind:rule` scope — the
only part of her note meant for anyone else. Dan writes under `user:dan`. The server's
operation validator (`server_extensions/tag_scope_validator.py`) confines each API key to
`user:<key>` OR `kind:rule` (#5032).

This is where the pieces meet: label extraction tags a single fact, the forced scope admits
that fact to Dan, and the note it came from must still stay Kate's — its chunk and its
document are hers, even though one fact extracted from them is not (#5030). Entities count
only what Dan can see (#5031).
"""

from __future__ import annotations

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import NotFoundException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio

KATE_NOTE = (
    "Team sync. Nobody deploys to production on Fridays. Personal: Kate is interviewing at another company next week."
)
DAN_NOTE = "Dan is planning a production deploy this Friday."

RULE = "Nobody deploys to production on Fridays | Involving: Team"
PRIVATE = "Kate is interviewing at another company | Involving: Kate"
DANS = "Dan is planning a production deploy this Friday | Involving: Dan"


@pytest.fixture
async def clients(scoped_server):
    admin = Hindsight(base_url=scoped_server.url)
    dan = Hindsight(base_url=scoped_server.url, api_key="dan")
    yield admin, dan
    await admin.aclose()
    await dan.aclose()


@pytest.fixture
async def shared_bank(clients, llm, bank_id) -> str:
    admin, _ = clients
    await admin.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "entity_labels": [
                    {
                        "key": "kind",
                        "type": "value",
                        "tag": True,
                        "description": "Whether the fact is a team rule",
                        "values": [{"value": "rule", "description": "A rule the whole team must follow"}],
                    }
                ]
            }
        },
    )
    llm.on_step("extract_facts", contains="Fridays").returns(
        extracted(
            fact(
                "Nobody deploys to production on Fridays",
                who="Team",
                entities=["Team", "production"],
                labels={"kind": "rule"},
            ),
            fact(
                "Kate is interviewing at another company", who="Kate", entities=["Kate", "Team"], labels={"kind": None}
            ),
        )
    )
    llm.on_step("extract_facts", contains="this Friday").returns(
        extracted(
            fact(
                "Dan is planning a production deploy this Friday",
                who="Dan",
                entities=["Dan", "production"],
                labels={"kind": None},
            )
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await admin.aretain(bank_id=bank_id, content=KATE_NOTE, document_id="kate-sync", tags=["user:kate"])
    await admin.aretain(bank_id=bank_id, content=DAN_NOTE, document_id="dan-notes", tags=["user:dan"])
    await wait_until_settled(admin, bank_id)
    return bank_id


async def test_the_label_puts_only_the_rule_in_the_shared_scope(clients, shared_bank):
    admin, _ = clients
    listing = await admin.alist_memories(bank_id=shared_bank)

    assert {(m.text, tuple(sorted(m.tags or []))) for m in listing.items} == {
        (RULE, ("kind:rule", "user:kate")),
        (PRIVATE, ("user:kate",)),
        (DANS, ("user:dan",)),
    }


async def test_dan_recalls_his_memories_and_the_rule_but_not_kates_note(clients, shared_bank):
    _, dan = clients
    response = await dan.arecall(bank_id=shared_bank, query="Can I deploy on Friday?", include_chunks=True)

    assert sorted(r.text for r in response.results) == sorted([RULE, DANS])
    # The rule came from Kate's note, but the note is hers: its chunk does not come along.
    chunk_texts = [c.text for c in (response.chunks or {}).values()]
    assert chunk_texts == [DAN_NOTE]


async def test_dan_cannot_reach_kates_document_by_any_door(clients, shared_bank):
    _, dan = clients
    documents = await dan.documents.list_documents(shared_bank)
    assert [d.id for d in documents.items] == ["dan-notes"]

    with pytest.raises(NotFoundException):
        await dan.documents.get_document(shared_bank, "kate-sync")
    with pytest.raises(NotFoundException):
        await dan.documents.list_document_chunks(shared_bank, "kate-sync")


async def test_dan_sees_only_the_entities_of_what_he_can_read(clients, shared_bank):
    admin, dan = clients
    everything = await admin.entities.list_entities(shared_bank)
    dans_view = await dan.entities.list_entities(shared_bank)

    assert "Kate" in {e.canonical_name for e in everything.items}
    assert {e.canonical_name for e in dans_view.items} == {"Team", "production", "kind:rule", "Dan"}

    def mentions(listing, name: str) -> int:
        return next(e.mention_count for e in listing.items if e.canonical_name == name)

    # "Team" is named by the rule and by Kate's private sentence; Dan can only count the rule.
    assert mentions(everything, "Team") == 2
    assert mentions(dans_view, "Team") == 1
    assert mentions(dans_view, "production") == 2, "the rule and Dan's note both mention production"


async def test_dan_cannot_open_kates_private_memory_by_id(clients, shared_bank):
    admin, dan = clients
    listing = await admin.alist_memories(bank_id=shared_bank)
    private = next(m for m in listing.items if m.text == PRIVATE)
    rule = next(m for m in listing.items if m.text == RULE)

    with pytest.raises(NotFoundException):
        await dan.memory.get_memory(shared_bank, private.id)
    assert (await dan.memory.get_memory(shared_bank, rule.id))["text"] == RULE
