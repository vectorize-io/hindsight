"""Entropy gate for entity resolution (B2): low-information names never fuzzy-merge.

Pure-function coverage of the Shannon-entropy gate plus integration tests on
the real scoring path: the Bob/Rob false merge is prevented, the
Alice→"Alice Chen" partial-name alias keeps merging, and disabling the gate
restores the previous behavior exactly (the rollout escape hatch).
"""

import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.entity_resolver import EntityResolver, has_high_entropy, name_entropy
from hindsight_api.pg0 import resolve_database_url

EVENT_DATE = datetime(2024, 1, 15, tzinfo=UTC)


class TestEntropyFunctions:
    def test_low_information_names_gated(self):
        # "bob": p(b)=2/3, p(o)=1/3 -> ~0.918; "ai" -> 1.0; repetition -> 0.
        assert not has_high_entropy("Bob")
        assert not has_high_entropy("AI")
        assert not has_high_entropy("aaaa")
        assert not has_high_entropy("")

    def test_ordinary_names_pass(self):
        # Five distinct letters -> log2(5) ~= 2.32. The Graphiti original's
        # length rule (<6 chars and single token) would gate "Alice" and break
        # the partial-name alias contract — entropy alone must let it through.
        assert has_high_entropy("Alice")
        assert has_high_entropy("Alice Chen")
        assert has_high_entropy("Kubernetes")

    def test_entropy_values(self):
        assert name_entropy("bob") == pytest.approx(0.918, abs=0.001)
        assert name_entropy("ai") == pytest.approx(1.0)
        assert name_entropy("alice") == pytest.approx(2.3219, abs=0.001)
        # Whitespace is ignored: "alice chen" counts characters only.
        assert name_entropy("a a a") == 0.0


async def _make_resolver_bank(pg0_db_url):
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-entropy-{uuid.uuid4().hex[:8]}"
    return backend, bank_id, EntityResolver(pool=backend, entity_lookup="full")


async def _insert_entity(conn, bank_id: str, name: str) -> str:
    return await conn.fetchval(
        """
        INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, $3, 1)
        RETURNING id
        """,
        bank_id,
        name,
        EVENT_DATE,
    )


async def _insert_cooccurrence(conn, id1, id2) -> None:
    # The table enforces entity_id_1 < entity_id_2 (entity_cooccurrence_order_check).
    low, high = sorted((id1, id2), key=str)
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, 5, $3)
        """,
        low,
        high,
        EVENT_DATE,
    )


async def _resolve_one(resolver, conn, bank_id: str, text: str, nearby: list[str], **kwargs) -> str:
    ids = await resolver.resolve_entities_batch(
        bank_id=bank_id,
        entities_data=[
            {"text": text, "nearby_entities": [{"text": n} for n in nearby], "event_date": EVENT_DATE}
        ],
        context="",
        unit_event_date=EVENT_DATE,
        conn=conn,
        **kwargs,
    )
    return str(ids[0]) if ids[0] is not None else ""


async def test_gate_blocks_low_entropy_fuzzy_merge(pg0_db_url):
    """A low-entropy partial name must not fuzzy-merge, however strong the context.

    Scenario constructible under the "full" lookup (whose candidate generation
    is exact/substring; the Bob↔Rob pairing only arises on the trigram path,
    but the gate lives in the shared scoring so this covers every strategy):
    input "Ann" vs existing "Ann Lee" — name 0.6*0.5 + co-occurrence 0.3 +
    same-day temporal 0.2 = 0.80 merges at baseline; the gate caps it at 0.5.
    """
    backend, bank_id, resolver = await _make_resolver_bank(pg0_db_url)
    try:
        async with backend.acquire() as conn:
            lee_id = await _insert_entity(conn, bank_id, "Ann Lee")
            corp_id = await _insert_entity(conn, bank_id, "TechCorp")
            await _insert_cooccurrence(conn, lee_id, corp_id)

            merged = await _resolve_one(resolver, conn, bank_id, "Ann", ["TechCorp"], entropy_gate=False)
            assert merged == str(lee_id), "without the gate the blended score must merge (baseline)"

            # Gate on (default): "ann" entropy ≈ 0.92 < 1.5 — name signal zeroed,
            # max achievable 0.5 < 0.6 threshold — stays a distinct entity.
            resolver.discard_pending_stats()
            distinct = await _resolve_one(resolver, conn, bank_id, "Ann", ["TechCorp"])
            assert distinct != str(lee_id)
            ann_name = await conn.fetchval("SELECT canonical_name FROM entities WHERE id = $1", uuid.UUID(distinct))
            assert ann_name == "Ann"
    finally:
        await backend.shutdown()


async def test_gate_preserves_partial_name_alias(pg0_db_url):
    """'Alice' (high entropy) must still merge into 'Alice Chen' via name+context."""
    backend, bank_id, resolver = await _make_resolver_bank(pg0_db_url)
    try:
        async with backend.acquire() as conn:
            chen_id = await _insert_entity(conn, bank_id, "Alice Chen")
            corp_id = await _insert_entity(conn, bank_id, "TechCorp")
            await _insert_cooccurrence(conn, chen_id, corp_id)

            resolved = await _resolve_one(resolver, conn, bank_id, "Alice", ["TechCorp"])
            assert resolved == str(chen_id), "high-entropy partial names keep the alias-merge behavior"
    finally:
        await backend.shutdown()


async def test_gated_exact_match_still_resolves(pg0_db_url):
    """The gate only kills FUZZY merging — an exact low-entropy name resolves to
    the existing row via the (bank_id, LOWER(name)) unique index path."""
    backend, bank_id, resolver = await _make_resolver_bank(pg0_db_url)
    try:
        async with backend.acquire() as conn:
            bob_id = await _insert_entity(conn, bank_id, "Bob")
            resolved = await _resolve_one(resolver, conn, bank_id, "bob", [])
            assert resolved == str(bob_id)
    finally:
        await backend.shutdown()
