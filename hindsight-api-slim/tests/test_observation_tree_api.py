"""GET /observation-tree: a bank's observations come back as topic files, every one exactly once."""

import json
import threading
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


async def _seed(
    conn, bank_id: str, subject: str, entities: list[str], count: int, tags: list[str] | None = None
) -> None:
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
            INSERT INTO memory_units (id, bank_id, text, fact_type, source_memory_ids, proof_count, mentioned_at, tags)
            VALUES ($1, $2, $3, 'observation', $4, 1, '2026-10-05T12:00:00Z', $5)
            """,
            uuid.uuid4(),
            bank_id,
            f"{subject} observation {i}",
            [fact_id],
            tags or [],
        )


def _files(folder: dict) -> list[dict]:
    return folder["files"] + [f for sub in folder["folders"] for f in _files(sub)]


@pytest.mark.asyncio
async def test_observation_tree_groups_observations_into_topic_files(
    api_client: httpx.AsyncClient, memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-observation-tree-{uuid.uuid4().hex[:8]}"
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

    response = await api_client.get(f"/v1/default/banks/{bank_id}/observations/tree")
    assert response.status_code == 200, response.text
    tree = response.json()

    assert tree["observation_count"] == 160
    topics = [f for f in _files(tree) if f["kind"] == "topic"]
    placed = [o["text"] for f in topics for o in f["observations"]]
    assert sorted(placed) == sorted(f"{s} observation {i}" for s in subjects for i in range(40))
    assert len(topics) == 4
    for f in topics:
        assert len({o["text"].split()[0] for o in f["observations"]}) == 1, f"{f['path']} mixes subjects"
        assert "(2026-10-05 12:00 · 1 source)" in f["content"]
    assert {f["path"] for f in _files(tree) if f["kind"] == "index"} == {"index.md"}


@pytest.mark.asyncio
async def test_observation_tree_of_an_empty_bank(api_client: httpx.AsyncClient, memory: MemoryEngine, request_context):
    bank_id = f"test-observation-tree-empty-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    response = await api_client.get(f"/v1/default/banks/{bank_id}/observations/tree")
    assert response.status_code == 200, response.text
    tree = response.json()
    assert tree["observation_count"] == 0
    assert [f["path"] for f in _files(tree)] == ["index.md"]


@pytest.mark.asyncio
async def test_observation_tree_for_one_scope(api_client: httpx.AsyncClient, memory: MemoryEngine, request_context):
    """tags / tags_match / tag_groups scope the tree as they scope recall: only matching
    observations are grouped."""
    bank_id = f"test-observation-tree-scope-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await _seed(conn, bank_id, "redis", ["Redis"], 5, tags=["team:a"])
        await _seed(conn, bank_id, "gpu", ["GPU"], 4, tags=["team:b"])
        await _seed(conn, bank_id, "s3", ["S3"], 3, tags=["team:a", "team:b"])
        await _seed(conn, bank_id, "misc", ["Misc"], 2)

    def placed(tree: dict) -> set[str]:
        return {o["text"].split()[0] for f in _files(tree) if f["kind"] == "topic" for o in f["observations"]}

    url = f"/v1/default/banks/{bank_id}/observations/tree"
    scoped = (await api_client.get(url, params={"tags": "team:a", "tags_match": "any_strict"})).json()
    assert scoped["observation_count"] == 8
    assert placed(scoped) == {"redis", "s3"}

    both = (await api_client.get(url, params={"tags": ["team:a", "team:b"], "tags_match": "all_strict"})).json()
    assert placed(both) == {"s3"}

    groups = json.dumps([{"tags": ["team:b"], "match": "any_strict"}, {"not": {"tags": ["team:a"]}}])
    assert placed((await api_client.get(url, params={"tag_groups": groups})).json()) == {"gpu"}

    rejected = await api_client.get(url, params={"tags": "team:a", "tag_groups": groups})
    assert rejected.status_code == 400


@pytest.mark.asyncio
async def test_observation_tree_is_serialised_off_the_event_loop(
    api_client: httpx.AsyncClient, memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A large bank's tree is seconds of CPU to validate and encode: on the event loop that
    would hold every other request on the worker."""
    from hindsight_api.api import http

    bank_id = f"test-observation-tree-loop-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await _seed(conn, bank_id, "redis", ["Redis"], 3)

    loop_thread = threading.get_ident()
    threads = []
    serialise = http._observation_tree_json

    def recording(tree):
        threads.append(threading.get_ident())
        return serialise(tree)

    monkeypatch.setattr(http, "_observation_tree_json", recording)
    response = await api_client.get(f"/v1/default/banks/{bank_id}/observations/tree")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.json()["observation_count"] == 3
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_observation_tree_of_a_store_owned_bank_reads_each_observation_once(
    api_client: httpx.AsyncClient, memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A store that owns its memories returns each observation with its entity ids, so the tree
    resolves names only and never re-reads the observations, and comes out the same."""
    from hindsight_api.engine.memories import get_memories

    bank_id = f"test-observation-tree-owned-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for subject, entity in [("redis", "Redis"), ("gpu", "GPU"), ("s3", "S3"), ("kafka", "Kafka")]:
            await _seed(conn, bank_id, subject, [entity], 4)
    url = f"/v1/default/banks/{bank_id}/observations/tree"
    expected = (await api_client.get(url)).json()

    # Stand the SQL store in for one that owns its memories: scanned observations carry their
    # (inherited) entity ids, as such a store writes them.
    # The engine hands a store-owned bank no Postgres connection, so the stand-in opens its own.
    store = get_memories()
    scan_all = store.scan_all_memories
    entity_map = store.entity_map_for_units
    resolve_names = store.resolve_entity_names

    async def scan_with_ids(**kw):
        async with pool.acquire() as conn:
            units = await scan_all(**{**kw, "conn": conn})
            ids = await entity_map(
                conn=conn, fq_table=kw["fq_table"], bank_id=kw["bank_id"], unit_ids=[u.unit_id for u in units]
            )
        for u in units:
            u.entity_ids = [e["entity_id"] for e in ids.get(u.unit_id, [])]
        return units

    async def names_on_own_conn(**kw):
        async with pool.acquire() as conn:
            return await resolve_names(**{**kw, "conn": conn})

    async def no_reread(**kw):
        raise AssertionError("the tree re-read the observations for their entity ids")

    monkeypatch.setattr(store, "scan_all_memories", scan_with_ids)
    monkeypatch.setattr(store, "entity_map_for_units", no_reread)
    monkeypatch.setattr(store, "resolve_entity_names", names_on_own_conn)
    monkeypatch.setattr(store, "store_owned_for", lambda _bank_id: True)

    response = await api_client.get(url)
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert any(f["about"] for f in _files(expected) if f["kind"] == "topic")
