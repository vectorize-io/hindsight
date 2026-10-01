"""Entity reads honour a tag filter (#5031).

In a bank where tags separate scopes, a reader asking for ``user:dan`` must not
learn that ``Kate`` exists, nor how often out-of-scope memories mention an
entity they share. So the list, the detail and the graph recompute their counts
from the matching memories instead of returning the stored totals — which these
tests seed with a sentinel value so a leak of them is visible.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.search.tags import TagGroupLeaf, TagGroupNot, build_tag_filter_clause, tag_filter_active

# Seeds `entities` / `unit_entities` / `memory_units` with raw INSERTs, so a store
# that keeps its entity registry outside SQL has nothing here to read.
pytestmark = pytest.mark.memory_backend_incompatible

# The stored counter on every seeded entity. Any filtered read that returns it
# leaked the bank-wide total.
STORED_COUNT = 99

DAN = ["user:dan"]
KATE = ["user:kate"]

JAN = datetime(2026, 1, 10, tzinfo=UTC)
FEB = datetime(2026, 2, 10, tzinfo=UTC)
MAR = datetime(2026, 3, 10, tzinfo=UTC)


async def _entity(conn, bank_id: str, name: str) -> uuid.UUID:
    entity_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, $4, $4, $5)
        """,
        entity_id,
        bank_id,
        name,
        JAN,
        STORED_COUNT,
    )
    return entity_id


async def _unit(conn, bank_id: str, tags: list[str], mentioned_at: datetime, entities: list[uuid.UUID]) -> None:
    unit_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, mentioned_at, tags, created_at, updated_at)
        VALUES ($1, $2, 'fact', 'world', $3, $3, $4, NOW(), NOW())
        """,
        unit_id,
        bank_id,
        mentioned_at,
        tags,
    )
    for entity_id in entities:
        await conn.execute("INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)", unit_id, entity_id)


@pytest.fixture
async def scoped_bank(memory: MemoryEngine, request_context: RequestContext) -> dict[str, uuid.UUID | str]:
    """Dan's memories name Alice (twice) and Bob; Kate's name Alice, Kate and the billing
    service; one untagged memory names Zed."""
    bank_id = f"test-entity-tags-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        alice = await _entity(conn, bank_id, "Alice")
        bob = await _entity(conn, bank_id, "Bob")
        kate = await _entity(conn, bank_id, "Kate")
        billing = await _entity(conn, bank_id, "billing service")
        zed = await _entity(conn, bank_id, "Zed")
        await _unit(conn, bank_id, DAN, JAN, [alice])
        await _unit(conn, bank_id, DAN, FEB, [alice, bob])
        await _unit(conn, bank_id, KATE, MAR, [alice, kate, billing])
        await _unit(conn, bank_id, [], MAR, [zed])
    return {"bank_id": bank_id, "alice": alice, "bob": bob, "kate": kate}


def _counts(listing: dict) -> dict[str, int]:
    return {item["canonical_name"]: item["mention_count"] for item in listing["items"]}


@pytest.mark.asyncio
async def test_list_shows_only_entities_the_matching_memories_mention(
    memory: MemoryEngine, request_context: RequestContext, scoped_bank
):
    bank_id = scoped_bank["bank_id"]

    strict = await memory.list_entities(bank_id, tags=DAN, tags_match="any_strict", request_context=request_context)
    assert _counts(strict) == {"Alice": 2, "Bob": 1}
    assert strict["total"] == 2
    # Ordered by the recomputed count, and dated by Dan's memories only (Kate's is March).
    assert [i["canonical_name"] for i in strict["items"]] == ["Alice", "Bob"]
    alice = strict["items"][0]
    assert alice["first_seen"] == JAN.isoformat()
    assert alice["last_seen"] == FEB.isoformat()

    # "any" keeps untagged memories visible, like every other tag-filtered read.
    loose = await memory.list_entities(bank_id, tags=DAN, tags_match="any", request_context=request_context)
    assert _counts(loose) == {"Alice": 2, "Bob": 1, "Zed": 1}

    # "exact" with no tags is the untagged scope.
    untagged = await memory.list_entities(bank_id, tags=None, tags_match="exact", request_context=request_context)
    assert _counts(untagged) == {"Zed": 1}

    searched = await memory.list_entities(
        bank_id, search="ali", tags=KATE, tags_match="any_strict", request_context=request_context
    )
    assert _counts(searched) == {"Alice": 1}
    assert searched["total"] == 1

    paged = await memory.list_entities(
        bank_id, tags=KATE, tags_match="any_strict", limit=1, offset=0, request_context=request_context
    )
    assert len(paged["items"]) == 1
    assert paged["total"] == 3


@pytest.mark.asyncio
async def test_unfiltered_list_keeps_the_stored_counters(
    memory: MemoryEngine, request_context: RequestContext, scoped_bank
):
    listing = await memory.list_entities(scoped_bank["bank_id"], request_context=request_context)
    assert set(_counts(listing).values()) == {STORED_COUNT}
    assert listing["total"] == 5


@pytest.mark.asyncio
async def test_detail_is_a_miss_outside_the_scope(memory: MemoryEngine, request_context: RequestContext, scoped_bank):
    bank_id = scoped_bank["bank_id"]

    hidden = await memory.get_entity(
        bank_id, str(scoped_bank["kate"]), tags=DAN, tags_match="any_strict", request_context=request_context
    )
    assert hidden is None

    alice = await memory.get_entity(
        bank_id, str(scoped_bank["alice"]), tags=DAN, tags_match="any_strict", request_context=request_context
    )
    assert alice is not None
    assert alice["mention_count"] == 2
    assert alice["first_seen"] == JAN.isoformat()
    assert alice["last_seen"] == FEB.isoformat()

    unfiltered = await memory.get_entity(bank_id, str(scoped_bank["kate"]), request_context=request_context)
    assert unfiltered is not None
    assert unfiltered["mention_count"] == STORED_COUNT


@pytest.mark.asyncio
async def test_graph_is_rebuilt_from_the_matching_memories(
    memory: MemoryEngine, request_context: RequestContext, scoped_bank
):
    graph = await memory.get_entity_graph(
        scoped_bank["bank_id"], tags=DAN, tags_match="any_strict", request_context=request_context
    )
    nodes = {n["data"]["label"]: n["data"]["mentionCount"] for n in graph["nodes"]}
    assert nodes == {"Alice": 2, "Bob": 1}
    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]["data"]
    assert {edge["source"], edge["target"]} == {str(scoped_bank["alice"]), str(scoped_bank["bob"])}
    assert edge["weight"] == 1
    assert edge["lastCooccurred"] == FEB.isoformat()

    kate_graph = await memory.get_entity_graph(
        scoped_bank["bank_id"], tags=KATE, tags_match="any_strict", min_count=2, request_context=request_context
    )
    # Kate's memory makes three pairs, each seen once: all below min_count.
    assert kate_graph["edges"] == []


@pytest.mark.asyncio
async def test_tag_groups_filter_the_same_three_reads(
    memory: MemoryEngine, request_context: RequestContext, scoped_bank
):
    """A compound filter goes through the same recomputation as plain tags."""
    bank_id = scoped_bank["bank_id"]
    not_kate = [TagGroupNot(filter=TagGroupLeaf(tags=KATE, match="any_strict")), TagGroupLeaf(tags=DAN + KATE)]

    listing = await memory.list_entities(bank_id, tag_groups=not_kate, request_context=request_context)
    assert _counts(listing) == {"Alice": 2, "Bob": 1}

    hidden = await memory.get_entity(
        bank_id, str(scoped_bank["kate"]), tag_groups=not_kate, request_context=request_context
    )
    assert hidden is None

    graph = await memory.get_entity_graph(bank_id, tag_groups=not_kate, request_context=request_context)
    assert {n["data"]["label"] for n in graph["nodes"]} == {"Alice", "Bob"}

    # A fuzzy leaf resolves against the bank's tags first: `user:kat` reaches `user:kate`.
    fuzzy = [TagGroupLeaf(tags=["user:kat"], match="any_strict", resolve="fuzzy")]
    kate_scope = await memory.list_entities(bank_id, tag_groups=fuzzy, request_context=request_context)
    assert _counts(kate_scope) == {"Alice": 1, "Kate": 1, "billing service": 1}


@pytest.mark.asyncio
async def test_http_takes_tag_groups_as_json(api_client, scoped_bank):
    bank_id = scoped_bank["bank_id"]
    groups = json.dumps([{"tags": DAN, "match": "any_strict"}])

    response = await api_client.get(f"/v1/default/banks/{bank_id}/entities", params={"tag_groups": groups})
    assert response.status_code == 200
    assert {i["canonical_name"]: i["mention_count"] for i in response.json()["items"]} == {"Alice": 2, "Bob": 1}

    kate = await api_client.get(
        f"/v1/default/banks/{bank_id}/entities/{scoped_bank['kate']}", params={"tag_groups": groups}
    )
    assert kate.status_code == 404

    graph = await api_client.get(f"/v1/default/banks/{bank_id}/entities/graph", params={"tag_groups": groups})
    assert graph.status_code == 200
    assert {n["data"]["label"] for n in graph.json()["nodes"]} == {"Alice", "Bob"}

    both = await api_client.get(
        f"/v1/default/banks/{bank_id}/entities", params={"tag_groups": groups, "tags": "user:dan"}
    )
    assert both.status_code == 400

    for bad in ("not json", json.dumps([{"match": "any"}])):
        invalid = await api_client.get(f"/v1/default/banks/{bank_id}/entities", params={"tag_groups": bad})
        assert invalid.status_code == 400, bad


def test_tags_and_tag_groups_combine_into_one_clause_with_ordered_binds():
    """Both filters at once (the engine allows it; only HTTP forbids it): AND-ed, with
    the plain tags bound first and each group leaf numbered after them."""
    groups = [TagGroupNot(filter=TagGroupLeaf(tags=KATE, match="any_strict"))]

    built = build_tag_filter_clause(DAN, "any_strict", groups, param_offset=2)

    assert built.sql == (
        "AND tags IS NOT NULL AND tags != '{}' AND tags && $2 "
        "AND NOT (tags IS NOT NULL AND tags != '{}' AND tags && $3)"
    )
    assert built.params == [DAN, KATE]
    assert built.next_param_offset == 4

    assert build_tag_filter_clause(None, "any", None, param_offset=2).sql == ""
    assert not tag_filter_active(None, "any")
    assert tag_filter_active(None, "exact")
    assert tag_filter_active(None, "any", groups)
