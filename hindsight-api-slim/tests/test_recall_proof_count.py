"""Recall returns how many facts back each observation (`proof_count`, #6966).

Recall already reads `proof_count` for ranking; this checks it reaches the HTTP response on
observations, and stays unset on raw facts, where "sources" means nothing.

Seeded through SQL (no LLM) so the count is exact.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.retain import embedding_utils

# Seeds `memory_units` directly: an alternative MEMORIES store would never see the rows.
pytestmark = pytest.mark.memory_backend_incompatible

FACT_TEXT = "Alice migrated the billing service to the new cluster"
OBS_TEXT = "Alice handles infrastructure migrations"


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


@pytest_asyncio.fixture
async def seeded(memory, request_context):
    bank_id = f"test-proof-count-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    fact_id, obs_id = str(uuid.uuid4()), str(uuid.uuid4())
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [FACT_TEXT, OBS_TEXT])
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding) VALUES ($1, $2, $3, 'world', $4::vector)",
            fact_id,
            bank_id,
            FACT_TEXT,
            _to_str(embeddings[0]),
        )
        # proof_count (3) deliberately differs from the one listed source, so the assertion shows
        # recall returns the stored count rather than recounting `source_memory_ids`.
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, source_memory_ids, proof_count) "
            "VALUES ($1, $2, $3, 'observation', $4::vector, $5::uuid[], 3)",
            obs_id,
            bank_id,
            OBS_TEXT,
            _to_str(embeddings[1]),
            [uuid.UUID(fact_id)],
        )
    yield bank_id, fact_id, obs_id
    await memory.delete_bank(bank_id, request_context=request_context)


async def test_recall_returns_proof_count_on_observations_only(memory, seeded):
    bank_id, fact_id, obs_id = seeded
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            f"/v1/default/banks/{bank_id}/memories/recall",
            json={"query": "Alice infrastructure migration", "types": ["world", "observation"], "max_tokens": 4000},
        )
    assert r.status_code == 200, r.text
    by_id = {x["id"]: x for x in r.json()["results"]}
    assert by_id[obs_id]["proof_count"] == 3
    assert by_id[fact_id]["proof_count"] is None
