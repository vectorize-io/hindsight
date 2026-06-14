"""get_entity must return the observations that mention the requested entity.

The entity-detail endpoint (`GET /v1/default/banks/{bank_id}/entities/{entity_id}`)
backs an entity-detail UI panel that lists observations attached to the entity.
Observations don't carry direct rows in `unit_entities` — the consolidator
documents this and points at the transitive path observation → `source_memory_ids`
→ `unit_entities` of the source memories.

`get_entity` used to return `observations: []` hardcoded, so every panel was
empty regardless of how many observations actually referenced the entity. This
test seeds the inverse of the recall fixture in
`test_recall_observation_entities.py`: one entity, two observations (one linked
through `source_memory_ids` inheritance, one with a direct `unit_entities`
row), and asserts both surface through `get_entity`.

No LLM required.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.retain import embedding_utils

# Tests in this file insert memory_units with shared hardcoded UUIDs and
# memory_units.id is a global PK; share an xdist group so parallel workers
# don't collide on the same row IDs.
pytestmark = pytest.mark.xdist_group("get_entity_observations")

ID_FACT = "22222222-0000-0000-0000-000000000001"
ID_OBS_INHERITED = "22222222-0000-0000-0000-000000000002"
ID_OBS_DIRECT = "22222222-0000-0000-0000-000000000003"
ID_OBS_UNRELATED = "22222222-0000-0000-0000-000000000004"

RC = RequestContext(tenant_id="default")


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


@pytest_asyncio.fixture
async def seeded(memory_no_llm_verify: MemoryEngine):
    engine = memory_no_llm_verify
    bank_id = f"test-get-entity-obs-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)

    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings,
        [
            "Acme released a new product line",
            "Acme is expanding into Europe",
            "Acme acquired BetaCorp",
            "Unrelated observation about weather",
        ],
    )

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memory_units WHERE id IN ($1, $2, $3, $4)",
            ID_FACT,
            ID_OBS_INHERITED,
            ID_OBS_DIRECT,
            ID_OBS_UNRELATED,
        )
        await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)

        acme_id = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, mention_count, first_seen, last_seen)
            VALUES ($1, 'Acme', 1, now(), now()) RETURNING id
            """,
            bank_id,
        )

        # Source fact carrying the entity link.
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, mentioned_at)
            VALUES ($1, $2, $3, 'world', $4::vector, now(), now() - interval '7 days')
            """,
            ID_FACT,
            bank_id,
            "Acme released a new product line",
            _to_str(embeddings[0]),
        )
        await conn.execute(
            "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
            ID_FACT,
            acme_id,
        )

        # Observation that inherits Acme transitively through source_memory_ids.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date,
                source_memory_ids, proof_count, mentioned_at
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, now(), $5::uuid[], 1, now() - interval '2 days')
            """,
            ID_OBS_INHERITED,
            bank_id,
            "Acme is expanding into Europe",
            _to_str(embeddings[1]),
            [ID_FACT],
        )

        # Observation that carries a direct unit_entities row instead of the
        # inherited path. Rare in practice but should still be returned.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date,
                source_memory_ids, proof_count, mentioned_at
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, now(), NULL, 1, now() - interval '1 day')
            """,
            ID_OBS_DIRECT,
            bank_id,
            "Acme acquired BetaCorp",
            _to_str(embeddings[2]),
        )
        await conn.execute(
            "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
            ID_OBS_DIRECT,
            acme_id,
        )

        # Unrelated observation — must NOT come back when asking for the Acme entity.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date,
                source_memory_ids, proof_count, mentioned_at
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, now(), NULL, 1, now())
            """,
            ID_OBS_UNRELATED,
            bank_id,
            "Unrelated observation about weather",
            _to_str(embeddings[3]),
        )

    yield engine, bank_id, str(acme_id)

    await engine.delete_bank(bank_id, request_context=RC)


@pytest.mark.asyncio
async def test_get_entity_returns_inherited_and_direct_observations(seeded):
    """get_entity must walk observation → source_memory_ids → unit_entities AND
    pick up direct unit_entities rows; observations unrelated to the entity must
    NOT appear.
    """
    engine, bank_id, acme_id = seeded

    result = await engine.get_entity(bank_id, acme_id, request_context=RC)

    assert result is not None
    texts = {obs.text for obs in result["observations"]}

    assert "Acme is expanding into Europe" in texts, (
        "Observation linked via source_memory_ids inheritance must surface; "
        f"got {texts}"
    )
    assert "Acme acquired BetaCorp" in texts, (
        f"Observation with a direct unit_entities row must surface; got {texts}"
    )
    assert "Unrelated observation about weather" not in texts, (
        f"Observation with no link to this entity must not surface; got {texts}"
    )


@pytest.mark.asyncio
async def test_get_entity_observations_ordered_recent_first(seeded):
    """Observations must come back ordered by mentioned_at DESC (most recent
    first) so the UI panel renders the freshest material at the top.
    """
    engine, bank_id, acme_id = seeded

    result = await engine.get_entity(bank_id, acme_id, request_context=RC)
    assert result is not None

    texts_in_order = [obs.text for obs in result["observations"]]
    # Fixture timestamps: direct (1 day ago) is newer than inherited (2 days ago).
    assert texts_in_order.index("Acme acquired BetaCorp") < texts_in_order.index(
        "Acme is expanding into Europe"
    ), f"Expected most-recent observation first; got {texts_in_order}"


@pytest.mark.asyncio
async def test_get_entity_respects_max_observations_cap(seeded):
    """max_observations is the load-bearing brake for hot entities (the panel
    must not be allowed to pull thousands). With max_observations=1 we expect
    exactly one row, and it should be the most recent.
    """
    engine, bank_id, acme_id = seeded

    result = await engine.get_entity(
        bank_id, acme_id, request_context=RC, max_observations=1
    )
    assert result is not None
    assert len(result["observations"]) == 1
    assert result["observations"][0].text == "Acme acquired BetaCorp"


@pytest.mark.asyncio
async def test_get_entity_missing_entity_returns_none(seeded):
    """Sanity: an entity_id that doesn't exist returns None (was the only
    correct behavior the previous stub got right, so guard against regression).
    """
    engine, bank_id, _acme_id = seeded
    missing = str(uuid.uuid4())

    result = await engine.get_entity(bank_id, missing, request_context=RC)
    assert result is None
