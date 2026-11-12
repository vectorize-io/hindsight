"""`hindsight-admin reembed`: a schema's vectors move to a new model (and dimension) in place."""

import asyncio
import hashlib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.admin.reembed import _vector_index_names, fact_embedding_text, finish_schema, reembed_schema
from hindsight_api.engine.embeddings import EMBEDDING_MODEL_SETTING, Embeddings, embedding_model_id
from hindsight_api.migrations import run_migrations

pytestmark = pytest.mark.memory_backend_incompatible

DIM = 8


class HashEmbeddings(Embeddings):
    """Deterministic 8-dim vectors from the text, so the stored vector proves which text was embedded."""

    model_name = "hash-8"

    def __init__(self, on_encode=None):
        self.on_encode = on_encode
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def dimension(self) -> int:
        return DIM

    async def initialize(self) -> None:
        pass

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.on_encode:
            hook, self.on_encode = self.on_encode, None
            await hook()
        # Answer slowly, like a remote model, so batches sent concurrently are seen overlapping.
        await asyncio.sleep(0.2)
        self.in_flight -= 1
        return [vector(t) for t in texts]


def vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 + 0.01 for b in digest[:DIM]]


@pytest_asyncio.fixture
async def schema(pg0_db_url):
    name = f"reembed_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    await conn.execute(f"CREATE SCHEMA {name}")
    await conn.close()
    run_migrations(pg0_db_url, schema=name)
    yield name
    conn = await asyncpg.connect(pg0_db_url)
    await conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    await conn.close()


async def _seed(conn, s: str) -> dict[str, str]:
    old = "[" + ",".join(["0.1"] * 384) + "]"
    await conn.execute(f"INSERT INTO {s}.banks (bank_id) VALUES ('b1')")
    fact = await conn.fetchval(
        f"""INSERT INTO {s}.memory_units (bank_id, text, fact_type, embedding, event_date, occurred_start)
            VALUES ('b1', 'Bob met Alice', 'world', $1::vector, now(), '2024-06-03T00:00:00Z') RETURNING id""",
        old,
    )
    for name in ("Bob", "Alice"):
        entity = await conn.fetchval(
            f"INSERT INTO {s}.entities (bank_id, canonical_name) VALUES ('b1', $1) RETURNING id", name
        )
        await conn.execute(f"INSERT INTO {s}.unit_entities (unit_id, entity_id) VALUES ($1, $2)", fact, entity)
    obs = await conn.fetchval(
        f"""INSERT INTO {s}.memory_units (bank_id, text, fact_type, embedding, event_date)
            VALUES ('b1', 'Bob and Alice are friends', 'observation', $1::vector, now()) RETURNING id""",
        old,
    )
    other = await conn.fetchval(
        f"""INSERT INTO {s}.memory_units (bank_id, text, fact_type, embedding, event_date)
            VALUES ('b1', 'The sky is blue', 'world', $1::vector, now()) RETURNING id""",
        old,
    )
    # Another bank with a mental model of the same id (ids are unique per bank only), written in the
    # same statement so both rows share an xmin: matching on id alone would cross banks.
    await conn.execute(f"INSERT INTO {s}.banks (bank_id) VALUES ('b2')")
    await conn.execute(
        f"""INSERT INTO {s}.mental_models (id, bank_id, subtype, name, source_query, content, embedding)
            VALUES ('mm1', 'b1', 'pinned', 'Friends', 'who?', 'Bob and Alice', $1::vector),
                   ('mm1', 'b2', 'pinned', 'Weather', 'sky?', 'It rains', $1::vector)""",
        old,
    )
    return {"fact": fact, "obs": obs, "other": other}


async def _stored(conn, s: str, table: str, row_id, bank_id: str = "b1") -> list[float]:
    # Raw SQL on purpose: no engine method returns stored vectors, and the vector itself is what
    # proves which text the re-embed embedded.
    text = await conn.fetchval(
        f"SELECT embedding::text FROM {s}.{table} WHERE id = $1 AND bank_id = $2", row_id, bank_id
    )
    return [float(x) for x in text.strip("[]").split(",")]


@pytest.mark.asyncio
async def test_reembed_rebuilds_every_vector_and_guards_old_model_writes(pg0_db_url, schema):
    conn = await asyncpg.connect(pg0_db_url)
    side = await asyncpg.connect(pg0_db_url)
    try:
        ids = await _seed(conn, schema)

        async def concurrent_edit():
            # A server rewrites a fact while the fill runs: its new text must be what ends up embedded.
            await side.execute(f"UPDATE {schema}.memory_units SET text = 'The sky is grey' WHERE id = $1", ids["other"])

        embeddings = HashEmbeddings(on_encode=concurrent_edit)
        result = await reembed_schema(conn, pg0_db_url, schema, embeddings, batch_size=1, concurrency=3)
        assert result.status == "swapped"
        # One row per batch, three batches at the model at once.
        assert embeddings.max_in_flight == 3

        dim = await conn.fetchval(
            f"""SELECT atttypmod FROM pg_attribute WHERE attrelid = '{schema}.memory_units'::regclass
                AND attname = 'embedding'"""
        )
        assert dim == DIM
        expected_fact = fact_embedding_text(
            fact_type="world",
            text="Bob met Alice",
            entities=["Bob", "Alice"],
            occurred_start=datetime(2024, 6, 3, tzinfo=UTC),
            occurred_end=None,
            mentioned_at=None,
        )
        assert expected_fact == "Bob met Alice (happened in June 2024) [Alice, Bob]"
        assert await _stored(conn, schema, "memory_units", ids["fact"]) == pytest.approx(vector(expected_fact))
        assert await _stored(conn, schema, "memory_units", ids["obs"]) == pytest.approx(
            vector("Bob and Alice are friends")
        )
        assert await _stored(conn, schema, "memory_units", ids["other"]) == pytest.approx(vector("The sky is grey"))
        assert await _stored(conn, schema, "mental_models", "mm1") == pytest.approx(vector("Friends Bob and Alice"))
        assert await _stored(conn, schema, "mental_models", "mm1", "b2") == pytest.approx(vector("Weather It rains"))

        # The vector indexes exist again, on the new column.
        # Not pg_indexes: it renders every index in the database, and another xdist worker dropping
        # its schema mid-scan fails the query with "could not open relation with OID".
        assert await _vector_index_names(conn, schema, "memory_units", "embedding")

        new_vec = "[" + ",".join(["0.5"] * DIM) + "]"
        # A connection on the old model (untagged) may not write vectors...
        with pytest.raises(asyncpg.PostgresError) as refused:
            await side.execute(
                f"UPDATE {schema}.memory_units SET embedding = $1::vector WHERE id = $2", new_vec, ids["fact"]
            )
        assert refused.value.sqlstate == "HS503"
        assert "writes are shed" in str(refused.value)
        # ...but may still write everything else.
        await side.execute(f"UPDATE {schema}.memory_units SET text = 'Bob met Alice twice' WHERE id = $1", ids["fact"])
        # A connection on the new model writes normally.
        await side.execute("SELECT set_config($1, $2, false)", EMBEDDING_MODEL_SETTING, embedding_model_id(embeddings))
        await side.execute(
            f"UPDATE {schema}.memory_units SET embedding = $1::vector WHERE id = $2", new_vec, ids["fact"]
        )

        # Re-running after the swap is a no-op; --finish lifts the guard.
        assert (await reembed_schema(conn, pg0_db_url, schema, embeddings)).status == "already swapped"
        await finish_schema(conn, schema)
        await side.execute("RESET ALL")
        await side.execute(
            f"UPDATE {schema}.memory_units SET embedding = $1::vector WHERE id = $2", new_vec, ids["fact"]
        )
    finally:
        await side.close()
        await conn.close()


class InterruptedEmbeddings(HashEmbeddings):
    """Embeds ``batches`` batches, then fails: a run stopped half way."""

    def __init__(self, batches: int):
        super().__init__()
        self.batches = batches
        self.calls = 0

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > self.batches:
            raise RuntimeError("run stopped")
        return await super().encode(texts)


class SameSizeEmbeddings(HashEmbeddings):
    """Another model with the same dimension: its vectors differ for the same text."""

    model_name = "hash-8-v2"

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [vector(f"v2 {t}") for t in texts]


async def _assert_every_vector_from(
    conn: asyncpg.Connection, s: str, ids: dict[str, str], vector_of: Callable[[str], list[float]]
) -> None:
    fact = fact_embedding_text(
        fact_type="world",
        text="Bob met Alice",
        entities=["Bob", "Alice"],
        occurred_start=datetime(2024, 6, 3, tzinfo=UTC),
        occurred_end=None,
        mentioned_at=None,
    )
    assert await _stored(conn, s, "memory_units", ids["fact"]) == pytest.approx(vector_of(fact))
    assert await _stored(conn, s, "memory_units", ids["obs"]) == pytest.approx(vector_of("Bob and Alice are friends"))
    assert await _stored(conn, s, "memory_units", ids["other"]) == pytest.approx(vector_of("The sky is blue"))
    assert await _stored(conn, s, "mental_models", "mm1") == pytest.approx(vector_of("Friends Bob and Alice"))
    assert await _stored(conn, s, "mental_models", "mm1", "b2") == pytest.approx(vector_of("Weather It rains"))


async def _filled(conn: asyncpg.Connection, s: str, table: str) -> int:
    return await conn.fetchval(f"SELECT count(*) FROM {s}.{table} WHERE embedding_next IS NOT NULL")


@pytest.mark.asyncio
# One batch is one row: stopped in the memory units, or after them in the mental models.
@pytest.mark.parametrize(("batches", "units", "models"), [(1, 1, 0), (4, 3, 1)])
async def test_a_resumed_run_with_another_model_of_the_same_size_drops_the_first_models_vectors(
    pg0_db_url, schema, batches, units, models
):
    conn = await asyncpg.connect(pg0_db_url)
    try:
        ids = await _seed(conn, schema)
        stopping = InterruptedEmbeddings(batches)
        with pytest.raises(Exception) as stopped:
            await reembed_schema(conn, pg0_db_url, schema, stopping, batch_size=1, concurrency=1)
        assert stopped.group_contains(Exception, match="run stopped")
        assert (await _filled(conn, schema, "memory_units"), await _filled(conn, schema, "mental_models")) == (
            units,
            models,
        )

        # The operator picks another model of the same size and re-runs.
        result = await reembed_schema(conn, pg0_db_url, schema, SameSizeEmbeddings(), batch_size=1, concurrency=1)
        assert result.status == "swapped"
        await _assert_every_vector_from(conn, schema, ids, lambda t: vector(f"v2 {t}"))
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_a_rerun_after_the_swap_with_another_model_is_refused_until_finish(pg0_db_url, schema):
    conn = await asyncpg.connect(pg0_db_url)
    try:
        ids = await _seed(conn, schema)
        first = HashEmbeddings()
        assert (await reembed_schema(conn, pg0_db_url, schema, first)).status == "swapped"

        # Before --finish, reembed runs again with another model (changed, or a stale environment).
        other = SameSizeEmbeddings()
        with pytest.raises(RuntimeError, match="already swapped to test:hash-8:8") as refused:
            await reembed_schema(conn, pg0_db_url, schema, other)
        assert "test:hash-8-v2:8" in str(refused.value)
        # Nothing moved: the vectors are still the first model's.
        await _assert_every_vector_from(conn, schema, ids, vector)
        assert (await reembed_schema(conn, pg0_db_url, schema, first)).status == "already swapped"

        # As the error says: --finish, then the other model re-embeds from scratch.
        await finish_schema(conn, schema)
        assert (await reembed_schema(conn, pg0_db_url, schema, other)).status == "swapped"
        await _assert_every_vector_from(conn, schema, ids, lambda t: vector(f"v2 {t}"))
    finally:
        await conn.close()


def test_model_mismatch_is_recognised_in_both_shapes():
    """The guard's refusal and pgvector's dimension errors mean "retry on a new-model server"; nothing else does."""
    from hindsight_api.engine.embeddings import is_embedding_model_mismatch

    class Guard(Exception):
        sqlstate = "HS503"

    assert is_embedding_model_mismatch(Guard("writes are shed"))
    assert is_embedding_model_mismatch(Exception("expected 768 dimensions, not 384"))
    assert is_embedding_model_mismatch(Exception("different vector dimensions 768 and 384"))
    assert not is_embedding_model_mismatch(Exception("relation does not exist"))
