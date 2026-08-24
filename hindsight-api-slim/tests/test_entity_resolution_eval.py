"""Dataset-driven eval of entity resolution, end to end against Postgres.

Every case seeds a bank with real ``entities`` and ``entity_cooccurrences`` rows, runs the
real ``resolve_entities_batch`` over it, and asserts what each mention resolved to. Nothing
is stubbed: the pg_trgm probe, the partial index that excludes labels, the candidate cap, the
scoring pass and the insert path all run. That is the point — the unit tests around this
module pin individual pieces (a similarity number, one scoring branch), and a resolution bug
is usually an *interaction* between them. #3751 was: each piece behaved as designed and the
combination attributed a new person's facts to an unrelated country.

Two properties are asserted per case:

* what each mention resolved to — an existing name means it was reused, its own name means a
  new entity was created;
* that ``trigram`` and ``full`` agree, unless the case says they don't. They pick candidates
  differently (trigram similarity vs exact-or-substring), so a case where they diverge is a
  real behavioural difference between two settings documented as a performance choice, and
  has to be written down rather than discovered.

Time is fixed: ``last_seen`` is seeded relative to ``EVENT_DATE`` and the mention carries
``EVENT_DATE``, so the recency term is exact instead of depending on when the suite runs.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.pg0 import resolve_database_url

EVENT_DATE = datetime(2026, 6, 1, tzinfo=UTC)
STRATEGIES = ("trigram", "full")


@dataclass(frozen=True)
class Existing:
    """An entity already in the bank when the mention arrives."""

    name: str
    kind: str = "regular"
    # How stale the entity is at EVENT_DATE. The recency term is worth up to 0.2 and decays to
    # zero at 7 days, so this is what decides whether history gets a vote at all.
    last_seen_days_ago: int = 3650
    # Other seeded entities this one has been seen with.
    cooccurs_with: tuple[str, ...] = ()
    # Filler partners, to make an entity indiscriminate the way `user` is on a real bank.
    hub_degree: int = 0


@dataclass(frozen=True)
class Mention:
    """One extracted name being resolved."""

    text: str
    # False = the caller authored this name and it must be taken literally (#3479).
    resolve: bool = True


@dataclass(frozen=True)
class Case:
    id: str
    pins: str
    # The entities extracted from ONE fact. They are each other's ``nearby_entities``, which is
    # what retain passes (``retain/link_utils.py``) — a bystander that is not itself a mention
    # would not be in the trigram strategy's candidate set, so its co-occurrences would not
    # count and the case would be measuring something production never does.
    mentions: tuple[Mention, ...]
    # What each mention must resolve to, positionally. An existing name = reused, the
    # mention's own text = a new entity.
    expect: tuple[str, ...]
    existing: tuple[Existing, ...] = ()
    # Only when `full` legitimately differs from `trigram` — see the module docstring.
    expect_full: tuple[str, ...] | None = None
    labels: tuple[dict, ...] = ()
    # Set when the case records something resolution gets *wrong* and we have not fixed yet.
    known_limitation: str = ""


def _m(text: str, resolve: bool = True) -> Mention:
    return Mention(text=text, resolve=resolve)


CASES: tuple[Case, ...] = (
    # ---------------------------------------------------------------- identity
    Case(
        id="exact-name-is-reused",
        pins="The floor must never get between a name and its own entity.",
        existing=(Existing("Alice Chen", last_seen_days_ago=400),),
        mentions=(_m("Alice Chen"),),
        expect=("Alice Chen",),
    ),
    Case(
        id="case-differences-are-the-same-entity",
        pins="Canonical names are matched case-insensitively.",
        existing=(Existing("Alice Chen"),),
        mentions=(_m("alice chen"),),
        expect=("Alice Chen",),
    ),
    Case(
        id="turkish-dotted-i-is-the-same-entity",
        pins="Postgres and Python disagree on lowercasing İ; resolution must not split on it.",
        existing=(Existing("İstanbul"),),
        mentions=(_m("istanbul"),),
        expect=("İstanbul",),
    ),
    # ------------------------------------------------- variants that should merge
    Case(
        id="added-surname-merges-with-cooccurrence",
        pins="The canonical merge this resolver exists for: alice/alice chen is 0.55 by trigram.",
        existing=(Existing("Alice Chen", last_seen_days_ago=2, cooccurs_with=("Google",)), Existing("Google")),
        mentions=(_m("Alice"), _m("Google")),
        expect=("Alice Chen", "Google"),
    ),
    Case(
        id="typo-merges-on-name-and-recency-alone",
        pins="A typo variant arriving with no co-occurrence context at all (#3479). Trigram 0.55.",
        existing=(Existing("Dr Wall", last_seen_days_ago=0),),
        mentions=(_m("Dr Waler"),),
        # `full` never even considers it: neither name contains the other.
        expect=("Dr Wall",),
        expect_full=("Dr Waler",),
    ),
    Case(
        id="corporate-suffix-merges",
        pins="Google/Google Inc is 0.64 by trigram and a substring, so both strategies see it.",
        existing=(
            Existing("Google Inc", last_seen_days_ago=1, cooccurs_with=("Sundar Pichai",)),
            Existing("Sundar Pichai"),
        ),
        mentions=(_m("Google"), _m("Sundar Pichai")),
        expect=("Google Inc", "Sundar Pichai"),
    ),
    # ------------------------------------------- coincidental names that must not
    Case(
        id="new-person-is-not-absorbed-by-a-similar-country",
        pins="#3751 exactly: 0.20 by trigram, 0.80 by sequence ratio, every other signal maxed.",
        existing=(
            Existing("Iran", last_seen_days_ago=0, cooccurs_with=("user", "topic:finance")),
            Existing("user", hub_degree=60),
            Existing("topic:finance"),
        ),
        mentions=(_m("Tigran"), _m("user"), _m("topic:finance")),
        expect=("Tigran", "user", "topic:finance"),
    ),
    Case(
        id="cyrillic-form-of-the-same-collision",
        pins="The reporting bank was Cyrillic; pg_trgm scores it identically to the transliteration.",
        existing=(
            Existing("Иран", last_seen_days_ago=0, cooccurs_with=("user",)),
            Existing("user", hub_degree=60),
        ),
        mentions=(_m("Тигран"), _m("user")),
        expect=("Тигран", "user"),
    ),
    Case(
        id="city-is-not-absorbed-by-its-country",
        pins="Tehran/Iran is 0.20 — a shared suffix, and two genuinely different entities.",
        existing=(Existing("Iran", last_seen_days_ago=0, cooccurs_with=("user",)), Existing("user", hub_degree=60)),
        mentions=(_m("Tehran"), _m("user")),
        expect=("Tehran", "user"),
    ),
    Case(
        id="unrelated-short-name-is-not-absorbed",
        pins="Ivan/Iran is 0.25 — above the probe's 0.15, below the merge floor.",
        existing=(Existing("Iran", last_seen_days_ago=0, cooccurs_with=("user",)), Existing("user", hub_degree=60)),
        mentions=(_m("Ivan"), _m("user")),
        expect=("Ivan", "user"),
    ),
    Case(
        id="different-given-names-stay-separate",
        pins="Ahmed/Mohammed is 0.15 by trigram but 0.62 by sequence ratio.",
        existing=(Existing("Mohammed", last_seen_days_ago=0, cooccurs_with=("user",)), Existing("user", hub_degree=60)),
        mentions=(_m("Ahmed"), _m("user")),
        expect=("Ahmed", "user"),
    ),
    # ------------------------------------------------------- co-occurrence quality
    Case(
        id="an-indiscriminate-partner-cannot-carry-a-merge",
        pins="Iran/Iraq clears the floor at 0.43; sharing only `user` must not be enough.",
        existing=(
            Existing("Iraq", last_seen_days_ago=30, cooccurs_with=("user",)),
            Existing("user", hub_degree=60),
        ),
        mentions=(_m("Iran"), _m("user")),
        expect=("Iran", "user"),
    ),
    Case(
        id="a-selective-partner-is-real-evidence",
        pins="The same weak name merges when the shared entity is one that means something.",
        existing=(
            Existing("Alise", last_seen_days_ago=30, cooccurs_with=("Bletchley Park",)),
            Existing("Bletchley Park"),
        ),
        mentions=(_m("Alice"), _m("Bletchley Park")),
        expect=("Alise", "Bletchley Park"),
        # `full` builds candidates by exact-or-substring, and "Alise" contains no "Alice".
        expect_full=("Alice", "Bletchley Park"),
    ),
    # ------------------------------------------------------------------- recency
    Case(
        id="a-stale-entity-does-not-win-on-name-alone",
        pins="Outside the 7-day window the recency term is zero, so the name must stand alone.",
        existing=(Existing("Dr Wall", last_seen_days_ago=30),),
        mentions=(_m("Dr Waler"),),
        expect=("Dr Waler",),
        expect_full=("Dr Waler",),
    ),
    # -------------------------------------------------------------------- labels
    Case(
        id="a-label-value-is-reused-exactly",
        pins="Labels resolve by exact match; the same value must land on the same entity.",
        existing=(Existing("topic:finance", kind="label"),),
        mentions=(_m("topic:finance"),),
        expect=("topic:finance",),
        labels=({"key": "topic", "type": "text"},),
    ),
    Case(
        id="near-identical-label-values-stay-distinct",
        pins="A controlled vocabulary must not collapse; topic:finance/topic:finances is 0.78.",
        existing=(Existing("topic:finance", kind="label", last_seen_days_ago=0),),
        mentions=(_m("topic:finances"),),
        expect=("topic:finances",),
        labels=({"key": "topic", "type": "text"},),
    ),
    Case(
        id="regular-text-never-merges-into-a-label-row",
        pins="Label rows are excluded from fuzzy matching in SQL and again while scoring (#1558).",
        existing=(Existing("topic:empathy", kind="label", last_seen_days_ago=0),),
        mentions=(_m("topic empathy"),),
        expect=("topic empathy",),
        labels=({"key": "topic", "type": "text"},),
    ),
    # ------------------------------------------------------ caller-authored names
    Case(
        id="a-literal-name-is-not-re-resolved",
        pins="resolve=False means the caller authored it; the graph must not overrule them (#3479).",
        existing=(Existing("Dr Wall", last_seen_days_ago=0),),
        mentions=(_m("Dr. Waller", resolve=False),),
        expect=("Dr. Waller",),
    ),
    Case(
        id="opting-out-is-per-mention",
        pins="One retain carries both the caller's names and the extractor's; only the caller's are literal.",
        existing=(Existing("Dr Wall", last_seen_days_ago=0),),
        mentions=(_m("Dr. Waller", resolve=False), _m("Dr Waler")),
        expect=("Dr. Waller", "Dr Wall"),
        expect_full=("Dr. Waller", "Dr Waler"),
    ),
    # ---------------------------------------------------------------- in-batch
    Case(
        id="same-batch-variants-collapse-to-one-entity",
        pins="Both names are new, so only the in-batch pass can unify them (#3107).",
        mentions=(_m("Wren"), _m("Wren 🎵")),
        expect=("Wren", "Wren"),
    ),
    Case(
        id="same-batch-literal-names-stay-apart",
        pins="Two names the caller wrote as two must stay two, however similar (#3479).",
        mentions=(_m("Alice", resolve=False), _m("Alice Smith", resolve=False)),
        expect=("Alice", "Alice Smith"),
    ),
    Case(
        id="same-batch-unrelated-names-stay-apart",
        pins="The in-batch cutoff is 0.5; unrelated names are nowhere near it.",
        mentions=(_m("Alice"), _m("Bogotá")),
        expect=("Alice", "Bogotá"),
    ),
    # --------------------------------------------------------------- containment
    Case(
        id="a-short-name-is-not-swallowed-by-a-long-one",
        pins="`full` admits any substring match, so containment alone must not merge.",
        existing=(
            Existing("Alice Smith Holdings Ltd", last_seen_days_ago=0, cooccurs_with=("user",)),
            Existing("user", hub_degree=60),
        ),
        mentions=(_m("Alice"), _m("user")),
        expect=("Alice", "user"),
    ),
)

KNOWN_LIMITATIONS: tuple[Case, ...] = (
    Case(
        id="emoji-decoration-forks-off-a-stale-entity",
        pins=(
            "A decorated form of a stored name has *identical* trigram sets to it — the strongest "
            "name evidence there is — but the score reads the sequence ratio (0.80 -> 0.40), so it "
            "needs history to reach 0.6 and forks into a second entity once the stored one is a day "
            "stale. The same two surface forms DO unify when both are new, because the in-batch pass "
            "merges at 0.5 trigram (#3107 fixed that half only). So whether 'Wren' and 'Wren [emoji]' "
            "are one entity depends on whether they arrived in the same retain. Closing it means "
            "letting strong name evidence stand on its own, which is a policy change with its own "
            "blast radius (it would also merge 'Alice'/'Alice Chen' at 0.55 with no other signal), "
            "not another threshold."
        ),
        existing=(Existing("Wren", last_seen_days_ago=2),),
        mentions=(_m("Wren 🎵"),),
        expect=("Wren",),
        known_limitation="a decorated variant forks off a stale entity despite 1.0 trigram similarity",
    ),
    Case(
        id="selective-partner-still-merges-two-real-countries",
        pins=(
            "Iran/Iraq clears the floor at 0.43, so a genuinely selective shared partner still "
            "merges two distinct entities. Nothing in the current signals can tell 'same entity, "
            "different spelling' from 'different entity, similar spelling' — it would need the "
            "extractor's type, or an LLM adjudication step, not another threshold."
        ),
        existing=(Existing("Iraq", last_seen_days_ago=1, cooccurs_with=("Basra",)), Existing("Basra")),
        mentions=(_m("Iran"), _m("Basra")),
        expect=("Iran", "Basra"),
        known_limitation="resolution merges Iran onto Iraq",
    ),
)

ALL_CASES = CASES + KNOWN_LIMITATIONS


async def _seed(conn, bank_id: str, existing: tuple[Existing, ...]) -> None:
    ids: dict[str, str] = {}
    for ent in existing:
        last_seen = EVENT_DATE - timedelta(days=ent.last_seen_days_ago)
        ids[ent.name] = await conn.fetchval(
            """
            INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count, entity_kind)
            VALUES ($1, $2, $3, $3, 5, $4)
            RETURNING id
            """,
            bank_id,
            ent.name,
            last_seen,
            ent.kind,
        )
    for ent in existing:
        # Filler partners exist only to give an entity a degree; their names are deliberately
        # nothing like any name under test so they never turn up as candidates themselves.
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
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.id)
async def test_entity_resolution_case(case: Case, strategy: str, pg0_db_url):
    if case.known_limitation:
        pytest.xfail(f"known limitation: {case.known_limitation} — {case.pins}")

    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"eval-entities-{uuid.uuid4().hex[:8]}"
    # Built from the shipped configuration, not the constructor defaults, so the dataset
    # measures what a deployment actually runs — and so changing a threshold shows up here.
    config = get_config()
    resolver = EntityResolver(
        pool=backend,
        entity_lookup=strategy,
        intrabatch_merge_similarity=config.entity_intrabatch_merge_similarity,
        entity_resolution_max_candidates=config.retain_entity_resolution_max_candidates,
        merge_min_similarity=config.entity_merge_min_similarity,
    )
    expected = case.expect_full if (strategy == "full" and case.expect_full is not None) else case.expect

    try:
        async with backend.acquire() as conn:
            # The pool MemoryEngine builds applies this on every connection; a backend created
            # directly would otherwise probe at the Postgres default of 0.3 and admit far fewer
            # candidates than production does.
            await conn.execute(
                "SELECT set_config('pg_trgm.similarity_threshold', $1, false)",
                str(config.entity_trgm_similarity_threshold),
            )
            await _seed(conn, bank_id, case.existing)

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

        assert [r.canonical_name for r in resolved] == list(expected), case.pins
    finally:
        resolver.discard_pending_stats()
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()
