"""Gray-band LLM arbitration for entity resolution (B2 tier 3).

Deterministic coverage: the verdict parser's defensive bounds, routing (which
mentions defer, that the LLM is never called when disabled), and end-to-end
application of stubbed verdicts on the real scoring path. Prompt quality is
judged in ``test_entity_gray_band_judge.py`` (``hs_llm_core``).
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.entity_arbitration import (
    GrayBandCandidate,
    GrayBandCase,
    GrayBandChoice,
    GrayBandVerdict,
    apply_verdict,
)
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.pg0 import resolve_database_url

EVENT_DATE = datetime(2024, 1, 15, tzinfo=UTC)


def _case(mention: str = "Ann", candidate_ids: tuple[str, ...] = ("e1", "e2")) -> GrayBandCase:
    return GrayBandCase(
        mention=mention,
        nearby_names=("TechCorp",),
        candidates=tuple(
            GrayBandCandidate(entity_id=cid, canonical_name=f"name-{cid}", cooccurring_names=("TechCorp",))
            for cid in candidate_ids
        ),
    )


class TestApplyVerdict:
    def test_choice_maps_to_entity_id(self):
        verdict = GrayBandVerdict(choices=[GrayBandChoice(case_index=0, candidate_index=1)])
        assert apply_verdict(verdict, [_case()]) == {0: "e2"}

    def test_minus_one_and_out_of_range_dropped(self):
        verdict = GrayBandVerdict(
            choices=[
                GrayBandChoice(case_index=0, candidate_index=-1),  # explicit "none"
                GrayBandChoice(case_index=5, candidate_index=0),  # case out of range
                GrayBandChoice(case_index=1, candidate_index=9),  # candidate out of range
            ]
        )
        assert apply_verdict(verdict, [_case(), _case("Lee")]) == {}

    def test_duplicate_case_answers_keep_first(self):
        verdict = GrayBandVerdict(
            choices=[
                GrayBandChoice(case_index=0, candidate_index=0),
                GrayBandChoice(case_index=0, candidate_index=1),
            ]
        )
        assert apply_verdict(verdict, [_case()]) == {0: "e1"}


class _StubLLMConfig:
    """Stands in for LLMConfig: records calls, returns a canned verdict."""

    def __init__(self, choices: list[dict] | None = None):
        self.calls: list[str] = []
        self._choices = choices or []

    async def call(self, *, messages, **_kwargs):
        self.calls.append(messages[1]["content"])
        return json.dumps({"choices": self._choices}), None


async def _make_bank(pg0_db_url, llm_config):
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-grayband-{uuid.uuid4().hex[:8]}"
    resolver = EntityResolver(pool=backend, entity_lookup="full", arbitration_llm_config=llm_config)
    return backend, bank_id, resolver


async def _seed_ann_lee(conn, bank_id: str) -> str:
    """Existing 'Ann Lee' co-occurring with TechCorp; last_seen far in the past
    so the temporal signal stays out of the blend (keeps scores predictable)."""
    old_date = EVENT_DATE - timedelta(days=400)
    lee_id = await conn.fetchval(
        """
        INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, 'Ann Lee', $2, $2, 1) RETURNING id
        """,
        bank_id,
        old_date,
    )
    corp_id = await conn.fetchval(
        """
        INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, 'TechCorp', $2, $2, 1) RETURNING id
        """,
        bank_id,
        old_date,
    )
    low, high = sorted((lee_id, corp_id), key=str)
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, 5, $3)
        """,
        low,
        high,
        old_date,
    )
    return str(lee_id)


async def _resolve_ann(resolver, conn, bank_id: str, **kwargs) -> str:
    ids = await resolver.resolve_entities_batch(
        bank_id=bank_id,
        entities_data=[{"text": "Ann", "nearby_entities": [{"text": "TechCorp"}], "event_date": EVENT_DATE}],
        context="",
        unit_event_date=EVENT_DATE,
        conn=conn,
        **kwargs,
    )
    return str(ids[0]) if ids[0] is not None else ""


async def test_arbitration_merges_when_llm_picks_candidate(pg0_db_url):
    # 'Ann' is low-entropy: the gate suppresses its name signal (score 0.3 from
    # co-occurrence only), routing it to arbitration instead of silent creation.
    stub = _StubLLMConfig(choices=[{"case_index": 0, "candidate_index": 0}])
    backend, bank_id, resolver = await _make_bank(pg0_db_url, stub)
    try:
        async with backend.acquire() as conn:
            lee_id = await _seed_ann_lee(conn, bank_id)
            resolved = await _resolve_ann(resolver, conn, bank_id, gray_band_lower=0.25)
            assert resolved == lee_id, "the arbiter's pick must be applied"
            assert len(stub.calls) == 1
            # Context names ride along lowercased (the resolver's co-occurrence
            # maps are lowercase); candidate canonical names keep their casing.
            assert "Ann Lee" in stub.calls[0] and "techcorp" in stub.calls[0]
    finally:
        await backend.shutdown()


async def test_arbitration_rejection_creates_new_entity(pg0_db_url):
    stub = _StubLLMConfig(choices=[{"case_index": 0, "candidate_index": -1}])
    backend, bank_id, resolver = await _make_bank(pg0_db_url, stub)
    try:
        async with backend.acquire() as conn:
            lee_id = await _seed_ann_lee(conn, bank_id)
            resolved = await _resolve_ann(resolver, conn, bank_id, gray_band_lower=0.25)
            assert resolved != lee_id
            name = await conn.fetchval("SELECT canonical_name FROM entities WHERE id = $1", uuid.UUID(resolved))
            assert name == "Ann"
    finally:
        await backend.shutdown()


async def test_arbitration_disabled_never_calls_llm(pg0_db_url):
    stub = _StubLLMConfig()
    backend, bank_id, resolver = await _make_bank(pg0_db_url, stub)
    try:
        async with backend.acquire() as conn:
            await _seed_ann_lee(conn, bank_id)
            # gray_band_lower=None (default) — tier 3 off even though the
            # resolver holds an LLM config.
            resolved = await _resolve_ann(resolver, conn, bank_id)
            assert resolved
            assert stub.calls == []
    finally:
        await backend.shutdown()


async def test_arbitration_failure_falls_back_to_create(pg0_db_url):
    class _BrokenLLM:
        async def call(self, **_kwargs):
            raise RuntimeError("provider down")

    backend, bank_id, resolver = await _make_bank(pg0_db_url, _BrokenLLM())
    try:
        async with backend.acquire() as conn:
            lee_id = await _seed_ann_lee(conn, bank_id)
            resolved = await _resolve_ann(resolver, conn, bank_id, gray_band_lower=0.25)
            assert resolved and resolved != lee_id, "LLM failure must degrade to the deterministic outcome"
    finally:
        await backend.shutdown()
