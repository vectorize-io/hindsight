"""GET /memory-tree: a bank's observations come back as topic files, every one exactly once."""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine

# Seeds rows straight into the Postgres tables: the tree is built from observations and the
# entities of their source facts, and no LLM-free path produces observations.
pytestmark = pytest.mark.memory_backend_incompatible


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _seed(conn, bank_id: str, subject: str, entities: list[str], count: int) -> None:
    """``count`` observations about ``subject``, each backed by one fact naming ``entities``."""
    entity_ids = []
    for name in entities:
        entity_ids.append(
            await conn.fetchval(
                """
                INSERT INTO entities (id, canonical_name, bank_id) VALUES ($1, $2, $3)
                ON CONFLICT (bank_id, lower(canonical_name)) DO UPDATE SET canonical_name = EXCLUDED.canonical_name
                RETURNING id
                """,
                uuid.uuid4(),
                name,
                bank_id,
            )
        )
    for i in range(count):
        fact_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type) VALUES ($1, $2, $3, 'world')",
            fact_id,
            bank_id,
            f"{subject} fact {i}",
        )
        for eid in entity_ids:
            await conn.execute("INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)", fact_id, eid)
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, source_memory_ids, proof_count, mentioned_at)
            VALUES ($1, $2, $3, 'observation', $4, 1, '2026-10-05T12:00:00Z')
            """,
            uuid.uuid4(),
            bank_id,
            f"{subject} observation {i}",
            [fact_id],
        )


def _files(folder: dict) -> list[dict]:
    return folder["files"] + [f for sub in folder["folders"] for f in _files(sub)]


@pytest.mark.asyncio
async def test_memory_tree_groups_observations_into_topic_files(
    api_client: httpx.AsyncClient, memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-memory-tree-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    # Four subjects so no entity is on more than a third of the bank (that counts as "everywhere",
    # names nothing and is ignored), and more than one file's worth of observations in total.
    subjects = {
        "redis": ["Redis", "TTL"],
        "postgres": ["Postgres", "locking"],
        "gpu": ["GPU", "vLLM"],
        "s3": ["S3", "bucket"],
    }
    async with pool.acquire() as conn:
        for subject, entities in subjects.items():
            await _seed(conn, bank_id, subject, entities, 40)

    response = await api_client.get(f"/v1/default/banks/{bank_id}/memory-tree")
    assert response.status_code == 200, response.text
    tree = response.json()

    assert tree["observation_count"] == 160
    topics = [f for f in _files(tree) if f["kind"] == "topic"]
    placed = [o["text"] for f in topics for o in f["observations"]]
    assert sorted(placed) == sorted(f"{s} observation {i}" for s in subjects for i in range(40))
    assert len(topics) == 4
    for f in topics:
        assert len({o["text"].split()[0] for o in f["observations"]}) == 1, f"{f['path']} mixes subjects"
        assert "(2026-10-05 · 1 sources)" in f["content"]
    assert {f["path"] for f in _files(tree) if f["kind"] == "readme"} == {"README.md"}


@pytest.mark.asyncio
async def test_memory_tree_of_an_empty_bank(api_client: httpx.AsyncClient, memory: MemoryEngine, request_context):
    bank_id = f"test-memory-tree-empty-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    response = await api_client.get(f"/v1/default/banks/{bank_id}/memory-tree")
    assert response.status_code == 200, response.text
    assert response.json()["observation_count"] == 0
