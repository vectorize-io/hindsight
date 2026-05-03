"""Recall projects entities for observations through source_memory_ids.

Observations don't carry rows in `unit_entities`; their entity association
lives transitively via `memory_units.source_memory_ids`. The per-memory
endpoint (`get_memory_unit`) follows that chain, but recall used to query
`unit_entities` directly and silently dropped entities for every observation
result, even when `include_entities=True` was set.

This test seeds an observation linked through `source_memory_ids` to a fact
with entities, runs an observation-only recall, and asserts both the
per-result `entities` field and the top-level aggregate map carry the
inherited entities.

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
pytestmark = pytest.mark.xdist_group("recall_observation_entities")

ID_FACT = "11111111-0000-0000-0000-000000000001"
ID_OBS_INHERITED = "11111111-0000-0000-0000-000000000002"
ID_OBS_DIRECT = "11111111-0000-0000-0000-000000000003"

RC = RequestContext(tenant_id="default")


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


@pytest_asyncio.fixture
async def seeded(memory_no_llm_verify: MemoryEngine):
    engine = memory_no_llm_verify
    bank_id = f"test-recall-obs-ent-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)

    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings,
        [
            "HeadClaw waitlist tracked in Google Sheets",
            "HeadClaw users sign up via the waitlist",
            "Reddit thread mentions HeadClaw waitlist signups",
        ],
    )

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memory_units WHERE id IN ($1, $2, $3)",
            ID_FACT,
            ID_OBS_INHERITED,
            ID_OBS_DIRECT,
        )
        await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)

        # Source fact with two entities. The observation that lacks direct
        # entity rows must inherit both through source_memory_ids.
        headclaw_id = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, mention_count)
            VALUES ($1, $2, 1) RETURNING id
            """,
            bank_id,
            "HeadClaw",
        )
        waitlist_id = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, mention_count)
            VALUES ($1, $2, 1) RETURNING id
            """,
            bank_id,
            "waitlist users",
        )
        # Independent entity attached directly to the second observation —
        # exercises the existing direct-link path so we don't regress it.
        reddit_id = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, mention_count)
            VALUES ($1, $2, 1) RETURNING id
            """,
            bank_id,
            "Reddit",
        )

        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date)
            VALUES ($1, $2, $3, 'world', $4::vector, now())
            """,
            ID_FACT,
            bank_id,
            "HeadClaw waitlist tracked in Google Sheets",
            _to_str(embeddings[0]),
        )
        await conn.execute(
            """
            INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2), ($1, $3)
            """,
            ID_FACT,
            headclaw_id,
            waitlist_id,
        )

        # Observation with NO direct unit_entities — must inherit HeadClaw +
        # waitlist users from source_memory_ids.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date,
                source_memory_ids, history, proof_count
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, now(), $5::uuid[], '[]'::jsonb, 1)
            """,
            ID_OBS_INHERITED,
            bank_id,
            "HeadClaw users sign up via the waitlist",
            _to_str(embeddings[1]),
            [ID_FACT],
        )

        # Observation with a DIRECT unit_entities link — must keep its own entity.
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date,
                source_memory_ids, history, proof_count
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, now(), NULL, '[]'::jsonb, 1)
            """,
            ID_OBS_DIRECT,
            bank_id,
            "Reddit thread mentions HeadClaw waitlist signups",
            _to_str(embeddings[2]),
        )
        await conn.execute(
            "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
            ID_OBS_DIRECT,
            reddit_id,
        )

    yield engine, bank_id

    await engine.delete_bank(bank_id, request_context=RC)


@pytest.mark.asyncio
async def test_recall_includes_inherited_entities_for_observations(seeded):
    """Observation-only recall must surface entities inherited from source memories."""
    engine, bank_id = seeded

    result = await engine.recall_async(
        bank_id=bank_id,
        query="HeadClaw waitlist users",
        fact_type=["observation"],
        request_context=RC,
        max_tokens=4000,
        include_entities=True,
        max_entity_tokens=2000,
    )

    by_id = {str(r.id): r for r in result.results}
    assert ID_OBS_INHERITED in by_id, f"Expected inherited-entity observation in results, got {list(by_id)}"
    assert ID_OBS_DIRECT in by_id, f"Expected direct-entity observation in results, got {list(by_id)}"

    inherited = by_id[ID_OBS_INHERITED].entities or []
    assert "HeadClaw" in inherited, f"Observation entities must inherit through source_memory_ids; got {inherited}"
    assert "waitlist users" in inherited, f"All source entities should propagate; got {inherited}"

    direct = by_id[ID_OBS_DIRECT].entities or []
    assert "Reddit" in direct, f"Direct unit_entities link must still project on observation results; got {direct}"

    assert result.entities is not None, "Top-level entities map must populate when include_entities=True"
    aggregate_names = set(result.entities.keys())
    assert {"HeadClaw", "waitlist users", "Reddit"}.issubset(aggregate_names), (
        f"Top-level entities map must include inherited + direct entities; got {aggregate_names}"
    )


@pytest.mark.asyncio
async def test_get_memory_unit_inherits_observation_entities(seeded):
    """get_memory_unit shares the recall helper, so observation inheritance
    must keep working through the per-memory endpoint as well.
    """
    engine, bank_id = seeded

    inherited = await engine.get_memory_unit(
        memory_id=ID_OBS_INHERITED,
        bank_id=bank_id,
        request_context=RC,
    )
    assert inherited is not None
    assert set(inherited["entities"]) >= {"HeadClaw", "waitlist users"}, (
        f"Observation must inherit source-memory entities; got {inherited['entities']}"
    )

    direct = await engine.get_memory_unit(
        memory_id=ID_OBS_DIRECT,
        bank_id=bank_id,
        request_context=RC,
    )
    assert direct is not None
    assert "Reddit" in direct["entities"], (
        f"Direct unit_entities link must still resolve via get_memory_unit; got {direct['entities']}"
    )
