"""created_after / created_before must also bound the graph-expansion arm.

The semantic, BM25 and temporal arms filter on ``updated_at`` in SQL, but graph
retrieval only filtered its *seeds*: once a fresh fact was picked as a seed, link
expansion pulled in its whole neighbourhood — entity co-occurrence, semantic kNN
links and causal links — regardless of how old those neighbours were.  Mental-model
delta refresh recalls with ``created_after=<last refresh>`` expecting only what
changed since, so stale neighbours silently re-entered every refresh.

Each test seeds one in-window fact (the seed the query matches) plus one
out-of-window neighbour reachable only through the graph, and asserts recall never
returns the neighbour.

No LLM required — uses the mock provider.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.retain import embedding_utils

# Each graph fixture seeds its out-of-window neighbour by INSERTing straight into
# memory_links, so the neighbourhood the assertions turn on only exists in SQL.
pytestmark = [
    pytest.mark.xdist_group("recall_time_range_graph"),
    pytest.mark.memory_backend_incompatible,
]

T_OLD = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
T_CUTOFF = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
T_NEW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

RC = RequestContext(tenant_id="default")

# The seed and its neighbours are deliberately about unrelated subjects so the
# neighbours cannot be retrieved by the semantic or BM25 arms — the only path to
# them is graph expansion.
SEED_TEXT = "the tabby cat sat on the woven mat"
QUERY = "tabby cat on a mat"
NEIGHBOUR_TEXTS = {
    "entity": "quarterly logistics invoices were reconciled in the ledger",
    "semantic": "the harbour crane was repainted before the monsoon season",
    "causal": "shipping container seals are audited by the port authority",
}

# The temporal arm needs a parseable date in the query and dated facts to match it.
TEMPORAL_QUERY = "what did the tabby cat do in March 2025?"
TEMPORAL_SEED_TEXT = "the tabby cat sat on the woven mat"
TEMPORAL_NEIGHBOUR_TEXT = "the tabby cat knocked over the milk jug"
IN_WINDOW_DATE = datetime(2025, 3, 15, tzinfo=timezone.utc)
OUT_OF_WINDOW_DATE = datetime(2024, 1, 10, tzinfo=timezone.utc)


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


async def _insert_fact(
    conn,
    *,
    fact_id: uuid.UUID,
    bank_id: str,
    text: str,
    embedding_str: str,
    updated_at: datetime,
    fact_type: str = "world",
    source_memory_ids: list[uuid.UUID] | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (
            id, bank_id, text, fact_type, embedding, created_at, updated_at,
            source_memory_ids, proof_count
        )
        VALUES ($1, $2, $3, $4, $5::vector, $6, $6, $7::uuid[], 1)
        """,
        fact_id,
        bank_id,
        text,
        fact_type,
        embedding_str,
        updated_at,
        source_memory_ids,
    )


async def _insert_entity(conn, bank_id: str, name: str) -> uuid.UUID:
    entity_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, NOW(), NOW(), 1)
        """,
        entity_id,
        bank_id,
        name,
    )
    return entity_id


async def _link_unit_entity(conn, unit_id: uuid.UUID, entity_id: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
        unit_id,
        entity_id,
    )


async def _insert_link(conn, bank_id: str, from_id: uuid.UUID, to_id: uuid.UUID, link_type: str) -> None:
    await conn.execute(
        """
        INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
        VALUES ($1, $2, $3, 0.9, $4)
        """,
        from_id,
        to_id,
        link_type,
        bank_id,
    )


async def _seed_neighbourhood(conn, engine, bank_id: str, *, fact_type: str) -> dict[str, uuid.UUID]:
    """Seed one in-window fact plus three out-of-window neighbours.

    The neighbours hang off the seed by each of link expansion's three signals
    (shared entity, semantic kNN link, causal link) so a single recall exercises
    all of them.  For observations the entity signal runs through
    ``source_memory_ids`` rather than the observation's own entity rows, so the
    shared entity is attached to the sources.
    """
    texts = [SEED_TEXT, *NEIGHBOUR_TEXTS.values()]
    embeddings = await embedding_utils.generate_embeddings_batch(engine.embeddings, texts)

    ids = {"seed": uuid.uuid4(), **{k: uuid.uuid4() for k in NEIGHBOUR_TEXTS}}

    sources: dict[str, uuid.UUID] = {}
    if fact_type == "observation":
        # Observations reach entities through their sources; give the seed and the
        # entity-neighbour one source each, both carrying the shared entity.
        for key in ("seed", "entity"):
            source_id = uuid.uuid4()
            sources[key] = source_id
            await _insert_fact(
                conn,
                fact_id=source_id,
                bank_id=bank_id,
                text=f"source fact for {key}",
                embedding_str=_to_str(embeddings[0]),
                updated_at=T_NEW if key == "seed" else T_OLD,
            )

    for key, text in [("seed", SEED_TEXT), *NEIGHBOUR_TEXTS.items()]:
        await _insert_fact(
            conn,
            fact_id=ids[key],
            bank_id=bank_id,
            text=text,
            embedding_str=_to_str(embeddings[texts.index(text)]),
            updated_at=T_NEW if key == "seed" else T_OLD,
            fact_type=fact_type,
            source_memory_ids=[sources[key]] if key in sources else None,
        )

    shared_entity = await _insert_entity(conn, bank_id, "Harbour Logistics")
    for key in ("seed", "entity"):
        await _link_unit_entity(conn, sources.get(key, ids[key]), shared_entity)

    await _insert_link(conn, bank_id, ids["seed"], ids["semantic"], "semantic")
    await _insert_link(conn, bank_id, ids["seed"], ids["causal"], "causes")

    return ids


@pytest_asyncio.fixture
async def world_graph(memory_no_llm_verify: MemoryEngine):
    engine = memory_no_llm_verify
    bank_id = f"test-tr-graph-world-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)
    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        ids = await _seed_neighbourhood(conn, engine, bank_id, fact_type="world")
    yield engine, bank_id, ids
    await engine.delete_bank(bank_id, request_context=RC)


@pytest_asyncio.fixture
async def observation_graph(memory_no_llm_verify: MemoryEngine):
    engine = memory_no_llm_verify
    bank_id = f"test-tr-graph-obs-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)
    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        ids = await _seed_neighbourhood(conn, engine, bank_id, fact_type="observation")
    yield engine, bank_id, ids
    await engine.delete_bank(bank_id, request_context=RC)


@pytest_asyncio.fixture
async def temporal_graph(memory_no_llm_verify: MemoryEngine):
    """A dated seed inside the query's month, plus a temporally-linked older fact.

    The neighbour sits outside the query's date window, so the temporal arm's
    entry-point query can never return it — only the multi-hop spread that walks
    ``memory_links`` from the entry points can.
    """
    engine = memory_no_llm_verify
    bank_id = f"test-tr-graph-temporal-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)

    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings, [TEMPORAL_SEED_TEXT, TEMPORAL_NEIGHBOUR_TEXT]
    )
    seed_id, neighbour_id = uuid.uuid4(), uuid.uuid4()

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        for fact_id, text, embedding, mentioned_at, updated_at in (
            (seed_id, TEMPORAL_SEED_TEXT, embeddings[0], IN_WINDOW_DATE, T_NEW),
            (neighbour_id, TEMPORAL_NEIGHBOUR_TEXT, embeddings[1], OUT_OF_WINDOW_DATE, T_OLD),
        ):
            await conn.execute(
                """
                INSERT INTO memory_units (
                    id, bank_id, text, fact_type, embedding, mentioned_at,
                    created_at, updated_at, proof_count
                )
                VALUES ($1, $2, $3, 'world', $4::vector, $5, $6, $6, 1)
                """,
                fact_id,
                bank_id,
                text,
                _to_str(embedding),
                mentioned_at,
                updated_at,
            )
        await _insert_link(conn, bank_id, seed_id, neighbour_id, "temporal")

    yield engine, bank_id, seed_id, neighbour_id
    await engine.delete_bank(bank_id, request_context=RC)


def _result_ids(result) -> set[str]:
    return {str(r.id) for r in result.results}


class TestGraphExpansionTimeRange:
    async def test_unfiltered_recall_reaches_neighbours(self, world_graph):
        """Without a window, graph expansion does surface all three neighbours.

        This is the control: it proves the fixture's links are live, so the
        filtered assertions below can only fail because of the window.
        """
        engine, bank_id, ids = world_graph
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            max_tokens=10000,
        )
        found = _result_ids(result)
        assert str(ids["seed"]) in found
        for key in NEIGHBOUR_TEXTS:
            assert str(ids[key]) in found, f"{key} neighbour unreachable — fixture links are not wired"

    async def test_created_after_excludes_graph_neighbours(self, world_graph):
        engine, bank_id, ids = world_graph
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            max_tokens=10000,
            created_after=T_CUTOFF,
        )
        found = _result_ids(result)
        assert str(ids["seed"]) in found, "the in-window seed must still be returned"
        for key in NEIGHBOUR_TEXTS:
            assert str(ids[key]) not in found, (
                f"{key}-linked neighbour (updated_at={T_OLD}) leaked past created_after={T_CUTOFF}"
            )

    async def test_created_before_excludes_graph_neighbours(self, world_graph):
        """The mirror bound: an old seed must not drag in newer neighbours."""
        engine, bank_id, ids = world_graph
        pool = await engine._get_pool()
        async with pool.acquire() as conn:
            # Flip the timestamps so the seed is the old one and its neighbours new.
            await conn.execute(
                "UPDATE memory_units SET updated_at = $1 WHERE bank_id = $2",
                T_NEW,
                bank_id,
            )
            await conn.execute(
                "UPDATE memory_units SET updated_at = $1 WHERE id = $2",
                T_OLD,
                ids["seed"],
            )

        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            max_tokens=10000,
            created_before=T_CUTOFF,
        )
        found = _result_ids(result)
        assert str(ids["seed"]) in found
        for key in NEIGHBOUR_TEXTS:
            assert str(ids[key]) not in found, (
                f"{key}-linked neighbour (updated_at={T_NEW}) leaked past created_before={T_CUTOFF}"
            )

    async def test_created_after_excludes_temporal_spreading_neighbours(self, temporal_graph):
        """The temporal arm spreads outward too, and had the same hole.

        Its entry-point query filtered on the window, but the multi-hop spread that
        walks temporal/causal links from those entry points did not.
        """
        engine, bank_id, seed_id, neighbour_id = temporal_graph

        unfiltered = await engine.recall_async(
            bank_id=bank_id,
            query=TEMPORAL_QUERY,
            request_context=RC,
            max_tokens=10000,
        )
        assert str(neighbour_id) in _result_ids(unfiltered), (
            "control: the out-of-window-by-date neighbour is reachable only by temporal "
            "spreading — if it is absent here the test proves nothing"
        )

        result = await engine.recall_async(
            bank_id=bank_id,
            query=TEMPORAL_QUERY,
            request_context=RC,
            max_tokens=10000,
            created_after=T_CUTOFF,
        )
        found = _result_ids(result)
        assert str(seed_id) in found
        assert str(neighbour_id) not in found, (
            f"temporally-linked neighbour (updated_at={T_OLD}) leaked past created_after={T_CUTOFF}"
        )

    async def test_created_after_excludes_observation_neighbours(self, observation_graph):
        engine, bank_id, ids = observation_graph
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            max_tokens=10000,
            created_after=T_CUTOFF,
        )
        found = _result_ids(result)
        for key in NEIGHBOUR_TEXTS:
            assert str(ids[key]) not in found, (
                f"{key}-linked observation (updated_at={T_OLD}) leaked past created_after={T_CUTOFF}"
            )
