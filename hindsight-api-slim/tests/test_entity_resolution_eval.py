"""Dataset-driven eval of entity resolution, end to end against Postgres.

The cases live in ``entity_resolution_cases.jsonl`` next to this file, one JSON object per
line. Each seeds real ``entities`` and ``entity_cooccurrences`` rows into a bank, runs the
real ``resolve_entities_batch`` over them, and asserts what every mention resolved to.
Nothing is stubbed: the pg_trgm probe, the partial index that excludes labels, the candidate
cap, the scoring pass and the insert path all run.

That is the point. The unit tests around this module pin individual pieces — a similarity
number, one scoring branch, one strategy's SQL — and a resolution bug is usually an
*interaction* between them. #3751 was: every piece behaved as designed and the combination
attributed a new person's facts to an unrelated country.

Two properties are asserted per case:

* what each mention resolved to — an existing name means it was reused, its own name means a
  new entity was created;
* that ``trigram`` and ``full`` agree, unless the case declares ``expect_full``. They build
  candidate sets differently (trigram similarity vs exact-or-substring), so a divergence is a
  real behavioural difference between two settings documented as a performance choice, and
  belongs written down rather than discovered.

Time is fixed: ``last_seen`` is seeded relative to ``EVENT_DATE`` and every mention carries
``EVENT_DATE``, so the recency term is exact rather than depending on when the suite runs.

Case fields
-----------
``id``            unique, and the pytest parameter id.
``pins``          what property the case exists to hold. Shown on failure.
``existing``      entities already in the bank: ``name``, ``last_seen_days_ago`` (default
                  3650, i.e. outside the recency window), ``cooccurs_with`` (names of other
                  seeded entities), ``hub_degree`` (filler partners, to make an entity as
                  indiscriminate as ``user`` is on a real bank), ``kind``.
``other_bank``    entities seeded into a *different* bank, which must never be reachable.
``mentions``      the entities extracted from ONE fact, so they are each other's
                  ``nearby_entities`` — which is what retain passes (``retain/link_utils.py``).
                  A bystander that is not itself a mention would not be in the trigram
                  strategy's candidate set, so its co-occurrences would not count and the
                  case would measure something production never does.
``expect``        what each mention must resolve to, positionally.
``expect_full``   the same for ``retain_entity_lookup="full"``, when it legitimately differs.
``labels``        ``entity_labels`` config for the batch.
``known_limitation``  set when the case records something resolution gets *wrong*. Run as
                  xfail rather than quietly given whatever answer the code produces.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.pg0 import resolve_database_url

EVENT_DATE = datetime(2026, 6, 1, tzinfo=UTC)
STRATEGIES = ("trigram", "full")
CASES_FILE = Path(__file__).parent / "entity_resolution_cases.jsonl"
# The dataset is the test surface; a truncated or half-migrated file would still "pass".
MIN_PASSING_CASES = 50


@dataclass(frozen=True)
class Existing:
    name: str
    kind: str = "regular"
    # How stale the entity is at EVENT_DATE. Recency is worth up to 0.2 and decays to zero at
    # 7 days, so this decides whether history gets a vote at all.
    last_seen_days_ago: int = 3650
    cooccurs_with: tuple[str, ...] = ()
    hub_degree: int = 0


@dataclass(frozen=True)
class Mention:
    text: str
    # False = the caller authored this name and it must be taken literally (#3479).
    resolve: bool = True


@dataclass(frozen=True)
class Case:
    id: str
    pins: str
    mentions: tuple[Mention, ...]
    expect: tuple[str, ...]
    existing: tuple[Existing, ...] = ()
    other_bank: tuple[Existing, ...] = ()
    expect_full: tuple[str, ...] | None = None
    labels: tuple[dict, ...] = ()
    known_limitation: str = ""

    def expected_for(self, strategy: str) -> tuple[str, ...]:
        if strategy == "full" and self.expect_full is not None:
            return self.expect_full
        return self.expect


def _existing(row: dict) -> Existing:
    return Existing(
        name=row["name"],
        kind=row.get("kind", "regular"),
        last_seen_days_ago=row.get("last_seen_days_ago", 3650),
        cooccurs_with=tuple(row.get("cooccurs_with", ())),
        hub_degree=row.get("hub_degree", 0),
    )


def _load_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for lineno, raw in enumerate(CASES_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - a broken dataset must be loud
            raise AssertionError(f"{CASES_FILE.name}:{lineno} is not valid JSON: {exc}") from exc
        cases.append(
            Case(
                id=row["id"],
                pins=row["pins"],
                mentions=tuple(Mention(text=m["text"], resolve=m.get("resolve", True)) for m in row["mentions"]),
                expect=tuple(row["expect"]),
                existing=tuple(_existing(e) for e in row.get("existing", ())),
                other_bank=tuple(_existing(e) for e in row.get("other_bank", ())),
                expect_full=tuple(row["expect_full"]) if row.get("expect_full") is not None else None,
                labels=tuple(row.get("labels", ())),
                known_limitation=row.get("known_limitation", ""),
            )
        )
    return tuple(cases)


CASES = _load_cases()


def test_the_dataset_is_well_formed():
    """A case that references a name it never seeded, or expects the wrong number of answers,
    would pass while measuring nothing — which is exactly how this eval nearly shipped blind."""
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    for case in CASES:
        seeded = {e.name for e in case.existing}
        assert len(case.expect) == len(case.mentions), f"{case.id}: expect must have one entry per mention"
        if case.expect_full is not None:
            assert len(case.expect_full) == len(case.mentions), f"{case.id}: expect_full length"
        for ent in case.existing:
            missing = set(ent.cooccurs_with) - seeded
            assert not missing, f"{case.id}: {ent.name} co-occurs with unseeded {sorted(missing)}"
        allowed = seeded | {m.text for m in case.mentions}
        unknown = set(case.expect) - allowed
        assert not unknown, f"{case.id}: expects {sorted(unknown)}, which is neither seeded nor mentioned"
        assert case.pins, f"{case.id}: every case must say what it pins"

    passing = [c for c in CASES if not c.known_limitation]
    assert len(passing) >= MIN_PASSING_CASES, f"only {len(passing)} passing cases in the dataset"


async def _seed(conn, bank_id: str, existing: tuple[Existing, ...]) -> None:
    ids: dict[str, str] = {}
    for ent in existing:
        ids[ent.name] = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count, entity_kind)
            VALUES ($1, $2, $3, $3, 5, $4)
            RETURNING id
            """,
            bank_id,
            ent.name,
            EVENT_DATE - timedelta(days=ent.last_seen_days_ago),
            ent.kind,
        )
    for ent in existing:
        # Filler partners exist only to give an entity a degree. Their names are deliberately
        # nothing like any name under test, so they never turn up as candidates themselves.
        for i in range(ent.hub_degree):
            filler = await conn.fetchval(
                """
                INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count, entity_kind)
                VALUES ($1, $2, $3, $3, 1, 'regular')
                RETURNING id
                """,
                bank_id,
                f"zzz-filler-{i:04d}",
                EVENT_DATE,
            )
            await _pair(conn, ids[ent.name], filler)
        for partner in ent.cooccurs_with:
            await _pair(conn, ids[ent.name], ids[partner])


async def _pair(conn, a: str, b: str) -> None:
    first, second = sorted((str(a), str(b)))
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, 3, $3)
        ON CONFLICT (entity_id_1, entity_id_2) DO NOTHING
        """,
        first,
        second,
        EVENT_DATE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_entity_resolution_case(case: Case, strategy: str, pg0_db_url):
    if case.known_limitation:
        pytest.xfail(f"known limitation: {case.known_limitation} — {case.pins}")

    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"eval-entities-{uuid.uuid4().hex[:8]}"
    other_bank_id = f"eval-other-{uuid.uuid4().hex[:8]}"
    # Built from the shipped configuration, not the constructor defaults, so the dataset
    # measures what a deployment actually runs — and so moving a threshold shows up here.
    config = get_config()
    resolver = EntityResolver(
        pool=backend,
        entity_lookup=strategy,
        intrabatch_merge_similarity=config.entity_intrabatch_merge_similarity,
        entity_resolution_max_candidates=config.retain_entity_resolution_max_candidates,
        merge_min_similarity=config.entity_merge_min_similarity,
    )

    try:
        async with backend.acquire() as conn:
            # The pool MemoryEngine builds applies this to every connection; a backend created
            # directly would otherwise probe at the Postgres default of 0.3 and admit far
            # fewer candidates than production does.
            await conn.execute(
                "SELECT set_config('pg_trgm.similarity_threshold', $1, false)",
                str(config.entity_trgm_similarity_threshold),
            )
            await _seed(conn, bank_id, case.existing)
            await _seed(conn, other_bank_id, case.other_bank)

            resolved = await resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=[
                    {
                        "text": m.text,
                        "nearby_entities": [{"text": other.text} for other in case.mentions],
                        "resolve": m.resolve,
                        "event_date": EVENT_DATE,
                    }
                    for m in case.mentions
                ],
                context=case.pins,
                unit_event_date=EVENT_DATE,
                conn=conn,
                entity_labels=[dict(cfg) for cfg in case.labels] or None,
            )

        assert [r.canonical_name for r in resolved] == list(case.expected_for(strategy)), case.pins
    finally:
        resolver.discard_pending_stats()
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = ANY($1::text[])", [bank_id, other_bank_id])
        await backend.shutdown()
