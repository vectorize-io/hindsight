"""A corpus built to be hard, where a big one would merely be slow.

An earlier version padded the bank to ~1000 rows with filler from unrelated
domains (botany, cycling, carpentry). Separating that from the gold is something
bag-of-words would manage: contamination was zero and the scores were identical
to a 28-row bank. Row count is not difficulty, so the padding is gone.

What makes retrieval hard is **near-miss density inside the question's own
topic**. So every fact here belongs to a cluster that shares the gold's
vocabulary, and the near-misses differ from the answer by exactly one attribute —
a version, an environment, a date, a number, a polarity, an entity. Nothing can
be separated by topic; it has to be separated by the attribute the question asks
about.

And difficulty is not only retrieval. Several clusters are built so that
retrieval returns EVERY candidate and still tells you nothing — the answer is
only right if reflect reasons over what it retrieved:

* ``supersession`` — a value changes three times; every version is retrievable
  and equally on-topic. Only the latest is correct.
* ``counting`` — the answer is how many facts match, so missing one is wrong
  while recall@k still looks fine.
* ``entity_confusion`` — two services one character apart, with parallel facts.
* ``scoped_truth`` — true in one region, false in another; an unqualified answer
  is wrong.
* ``absence_dense`` — the topic is densely present, the specific fact is not.
  Saying "no information" is the correct answer, surrounded by plausible bait.

Questions and their gold ids are produced by the SAME generator that writes the
facts, so a label can never drift from the text it points at — the failure mode
that makes hand-maintained corpora quietly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardFact:
    id: str
    text: str
    cluster: str
    #: What this fact is ABOUT. Facts sharing a subject must be ingested in the
    #: same wave: split across waves, the later batch reads as a correction of the
    #: earlier one and gets superseded -- which is how "Release 0.9.3 reached
    #: production" was legitimately overwritten by a later batch that only
    #: mentioned staging and canary.
    subject: str = ""


@dataclass
class HardQuestion:
    id: str
    category: str
    question: str
    ideal_query: str
    gold: list[str] = field(default_factory=list)
    answer_criteria: str = ""
    must_not_claim: str = ""
    short_circuit: bool = False


@dataclass
class HardCorpus:
    """Facts and the questions labelled against them — one cluster, or all of them."""

    facts: list[HardFact]
    questions: list[HardQuestion]


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _releases() -> HardCorpus:
    """Ten releases, each with ONE coherent lifecycle. Near-misses are other releases.

    The first version generated the same release "deployed production" on three
    different dates and "rolled back from production" on three others. That is not
    a hard question, it is a contradiction: read as a narrative about one release
    it says the release reached production repeatedly. Worse, when the corpus was
    split into ingest waves, a later batch of staging/canary facts about that same
    release read exactly like a correction of the earlier production claim -- and
    the model superseded it, correctly applying the documented temporal rule. I
    twice reported that as a delta bug. It was this.

    So each (release, environment) pair occurs at most ONCE and every release runs
    staging -> canary -> production in order. Discrimination comes from ten
    similar releases, never from one release contradicting itself. The gold is the
    only production deploy on 18 March 2026; 0.9.6's canary promotion on that same
    date is the near-miss that makes the environment the thing you must get right.
    """
    facts: list[HardFact] = []
    gold_id = "rel-prod-3"

    # (staging, canary, production) for each release. Production dates are unique,
    # so "which release reached production on D" has exactly one answer.
    schedule = [
        ("2026-02-03", "2026-02-10", "2026-02-17"),
        ("2026-02-06", "2026-02-13", "2026-02-24"),
        ("2026-03-02", "2026-03-09", "2026-03-16"),
        ("2026-03-04", "2026-03-11", "2026-03-18"),  # 0.9.3 -> the gold
        ("2026-03-06", "2026-03-13", "2026-03-20"),
        ("2026-04-01", "2026-04-08", "2026-04-15"),
        ("2026-03-09", "2026-03-18", "2026-04-22"),  # 0.9.6 canary ON the gold date
        ("2026-04-06", "2026-04-13", "2026-04-29"),
        ("2026-05-04", "2026-05-11", "2026-05-18"),
        ("2026-05-06", "2026-05-13", "2026-05-26"),
    ]
    for minor, (staging, canary, production) in enumerate(schedule):
        version = f"0.9.{minor}"
        for suffix, text in (
            ("stg", f"Release {version} was deployed to staging on {staging}."),
            ("can", f"Release {version} was promoted to canary on {canary}."),
            ("prod", f"Release {version} was deployed to production on {production}."),
        ):
            facts.append(HardFact(f"rel-{suffix}-{minor}", text, "releases", subject=version))

    q = HardQuestion(
        id="hq-release-prod",
        category="precision_discrimination",
        question="Which release was deployed to production on 18 March 2026?",
        ideal_query="release deployed production 2026-03-18",
        gold=[gold_id],
        answer_criteria="Identifies release 0.9.3 as the one deployed to production on 2026-03-18.",
        must_not_claim="a release other than 0.9.3 was deployed to production on 18 March 2026",
    )
    return HardCorpus(facts=facts, questions=[q])


def _supersession() -> HardCorpus:
    """The value changes four times. Every version is on-topic and retrievable.

    Retrieval cannot be wrong here and cannot be right either: it returns the
    whole history. Only ordering the history correctly produces the right answer,
    which is precisely what a retrieval metric cannot see.
    """
    owners = ["the platform team", "the retrieval team", "the search guild", "the core infra team"]
    # Every handover must be in the PAST. The first version of this walked to
    # October 2026 — the future — and reflect correctly reported the last owner as
    # "scheduled", which the criteria then marked wrong. The corpus was the bug,
    # not the answer, so the chain is pinned to dates that have already happened.
    handovers = [("March", 2023), ("September", 2023), ("April", 2024), ("November", 2024)]
    facts = [
        HardFact(
            f"own-{i + 1:03d}",
            f"Ownership of the ranking service passed to {owner} in {month} {year}.",
            "supersession",
        )
        for i, (owner, (month, year)) in enumerate(zip(owners, handovers))
    ]
    # Filler that shares the vocabulary without answering: other systems moving
    # between the same teams, so "ownership + team" retrieves plenty of wrong rows.
    other = ["the ingestion pipeline", "the billing gateway", "the audit log", "the webhook dispatcher"]
    for i, system in enumerate(other):
        for j, owner in enumerate(owners):
            facts.append(
                HardFact(
                    f"own-x{i}{j}",
                    f"Ownership of {system} passed to {owner} in {_MONTHS[(i + j) * 2 % 12]} 202{3 + (j % 2)}.",
                    "supersession",
                )
            )

    q = HardQuestion(
        id="hq-owner-latest",
        category="supersession",
        question="Which team owns the ranking service now?",
        ideal_query="ranking service ownership current team",
        gold=[f.id for f in facts if f.cluster == "supersession" and f.id.startswith("own-0")],
        answer_criteria=(
            "States that the core infra team currently owns the ranking service. Mentioning the earlier "
            "owners as history is fine; naming any of them as the current owner is not."
        ),
        must_not_claim="the platform team, the retrieval team or the search guild currently owns the ranking service",
    )
    return HardCorpus(facts=facts, questions=[q])


def _counting() -> HardCorpus:
    """The answer is a count, so retrieving most of the evidence is still wrong.

    Seven customers reported billing errors in Q2; eleven more reported other
    things, or billing errors outside Q2. recall@k can look healthy while the
    number in the answer is wrong.
    """
    facts: list[HardFact] = []
    gold: list[str] = []
    billing_q2 = ["Aldridge", "Barrow", "Calloway", "Denning", "Ellery", "Fairbank", "Gadsden"]
    for i, name in enumerate(billing_q2):
        fid = f"cnt-b{i:02d}"
        gold.append(fid)
        facts.append(HardFact(fid, f"{name} Ltd reported a billing error in {_MONTHS[3 + i % 3]} 2026.", "counting"))
    # Same shape, wrong quarter or wrong issue — retrieves just as well.
    for i, name in enumerate(
        [
            "Harkness",
            "Illingworth",
            "Jarrow",
            "Kelsey",
            "Lomax",
            "Mowbray",
            "Ashcombe",
            "Brightwell",
            "Corfield",
            "Dunmore",
            "Everleigh",
            "Fenwick",
            "Garsdale",
            "Halstead",
            "Inglewood",
            "Jessop",
            "Kirkby",
            "Langdon",
        ]
    ):
        facts.append(
            HardFact(f"cnt-o{i:02d}", f"{name} Ltd reported a billing error in {_MONTHS[9 + i % 3]} 2026.", "counting")
        )
    for i, name in enumerate(
        [
            "Norbury",
            "Oakley",
            "Pemberton",
            "Quill",
            "Ravensworth",
            "Saltmarsh",
            "Thirlwall",
            "Underhill",
            "Vance",
            "Wetherby",
            "Yarborough",
            "Zouche",
            "Alverton",
            "Brackley",
            "Cranmore",
        ]
    ):
        facts.append(
            HardFact(
                f"cnt-n{i:02d}", f"{name} Ltd reported a latency problem in {_MONTHS[3 + i % 3]} 2026.", "counting"
            )
        )

    q = HardQuestion(
        id="hq-count-billing",
        category="counting",
        question="How many customers reported a billing error in the second quarter of 2026?",
        ideal_query="customers billing error April May June 2026 count",
        gold=gold,
        answer_criteria="States that seven (7) customers reported a billing error in Q2 2026.",
        must_not_claim="a number of Q2 billing-error reports other than seven",
    )
    return HardCorpus(facts=facts, questions=[q])


def _entity_confusion() -> HardCorpus:
    """Two services one character apart, with parallel facts about each."""
    facts: list[HardFact] = []
    gold: list[str] = []
    for i, (svc, port, owner) in enumerate(
        [
            ("payments-api", 8443, "the billing team"),
            ("payment-api", 8444, "the legacy platform team"),
            ("payments-apy", 8445, "the migration squad"),
        ]
    ):
        for j, (attr, val) in enumerate(
            [
                ("listens on port", port),
                ("is maintained by", owner),
                ("was last audited in", "2026"),
                ("exposes a health endpoint on port", port + 1000),
                ("was migrated off the legacy gateway in", "2025"),
                ("has a documented SLO of", "99.9%"),
            ]
        ):
            fid = f"ent-{i}{j}"
            facts.append(HardFact(fid, f"The {svc} service {attr} {val}.", "entity_confusion"))
            if svc == "payments-api" and attr == "listens on port":
                gold.append(fid)
    q = HardQuestion(
        id="hq-entity-port",
        category="entity_confusion",
        question="Which port does the payments-api service listen on?",
        ideal_query="payments-api service listening port",
        gold=gold,
        answer_criteria="States that payments-api listens on port 8443.",
        must_not_claim="payments-api listens on port 8444",
    )
    return HardCorpus(facts=facts, questions=[q])


def _scoped_truth() -> HardCorpus:
    """True in one region, false in another. An unqualified answer is wrong."""
    facts = [
        HardFact("scp-001", "Two-factor authentication is mandatory for all EU accounts.", "scoped_truth"),
        HardFact("scp-002", "Two-factor authentication remains optional for US accounts.", "scoped_truth"),
        HardFact("scp-003", "Two-factor authentication is mandatory for UK accounts from 2026.", "scoped_truth"),
        HardFact("scp-004", "Single sign-on is mandatory for all enterprise accounts.", "scoped_truth"),
        HardFact("scp-005", "Password rotation is optional for every region.", "scoped_truth"),
    ]
    q = HardQuestion(
        id="hq-scoped-2fa",
        category="scoped_truth",
        question="Is two-factor authentication mandatory for our accounts?",
        ideal_query="two-factor authentication mandatory optional region accounts",
        gold=["scp-001", "scp-002", "scp-003"],
        answer_criteria=(
            "Answers with the regional qualification rather than a flat yes or no: mandatory for EU "
            "accounts and optional for US accounts. Any accurate treatment of the UK is acceptable — "
            "the memory says UK accounts are mandatory FROM 2026, so stating it either as mandatory or "
            "as mandatory starting in 2026 is correct, and so is omitting the UK entirely."
        ),
        must_not_claim="two-factor authentication is mandatory everywhere, with no regional distinction",
    )
    return HardCorpus(facts=facts, questions=[q])


def _absence_dense() -> HardCorpus:
    """The topic is densely present; the asked-for year is not. Bait for a guess."""
    facts = [
        HardFact(
            f"abs-{i:03d}",
            f"Engineering headcount reached {40 + i * 7} at the end of {_MONTHS[i % 12]} 202{5 + i % 2}.",
            "absence_dense",
        )
        for i in range(14)
    ]
    q = HardQuestion(
        id="hq-absent-2024",
        category="absence_dense",
        question="What was the engineering headcount at the end of 2024?",
        ideal_query="engineering headcount end of 2024",
        gold=[],
        answer_criteria=(
            "Says the memory holds no headcount figure for 2024, rather than giving or estimating one. "
            "Citing 2025/2026 figures as context is fine only if it does not present one as the 2024 number."
        ),
        must_not_claim="a specific engineering headcount figure for 2024",
    )
    return HardCorpus(facts=facts, questions=[q])


def _numeric_precision() -> HardCorpus:
    """A dozen similar magnitudes in one topic; only one matches the predicate."""
    facts: list[HardFact] = []
    gold_id = "num-006"
    causes = [
        "connection pool exhaustion",
        "disk exhaustion",
        "certificate expiry",
        "DNS misconfiguration",
        "a memory leak",
        "connection pool exhaustion",
        "thread starvation",
        "a bad migration",
        "clock skew",
        "an upstream timeout",
        "connection pool exhaustion",
        "a failed failover",
        "index corruption",
        "a config rollout",
        "disk exhaustion",
        "connection pool exhaustion",
        "a network partition",
        "certificate expiry",
        "queue backpressure",
        "a bad deploy",
        "connection pool exhaustion",
        "replica lag",
        "a memory leak",
        "DNS misconfiguration",
    ]
    # Each outage must own its (month, year). The first version cycled months with
    # `i % 12` and years with `i // 12`, which produced a SECOND "April 2026
    # outage" carrying different values — so the question had two contradictory
    # answers, and reflect reporting a conflict was correct while the corpus was
    # wrong. Near-misses must differ in their VALUES, never in what they claim to
    # be. April 2026 is reserved for the gold row.
    slots = [(m, y) for y in (2024, 2025, 2026) for m in _MONTHS if not (m == "April" and y == 2026)]
    if len(causes) > len(slots):
        raise RuntimeError("More outages than distinct (month, year) slots — they would collide")
    for i, cause in enumerate(causes):
        fid = f"num-{i:03d}"
        limit = 100 + i * 50
        month, year = slots[i]
        text = (
            f"The {month} {year} outage was caused by {cause} at {limit} connections and lasted {12 + i * 5} minutes."
        )
        if fid == gold_id:
            text = (
                "The April 2026 outage was caused by connection pool exhaustion at 200 connections "
                "and lasted 47 minutes."
            )
        facts.append(HardFact(fid, text, "numeric_precision"))

    q = HardQuestion(
        id="hq-numeric-outage",
        category="numeric_precision",
        question="What caused the April 2026 outage, at what connection limit, and how long did it last?",
        ideal_query="April 2026 outage connection pool exhaustion 200 connections 47 minutes",
        gold=[gold_id],
        answer_criteria=("States all three: connection pool exhaustion, a limit of 200 connections, and 47 minutes."),
        must_not_claim="a connection limit other than 200 or a duration other than 47 minutes for the April 2026 outage",
    )
    return HardCorpus(facts=facts, questions=[q])


_BUILDERS = (
    _releases,
    _supersession,
    _counting,
    _entity_confusion,
    _scoped_truth,
    _absence_dense,
    _numeric_precision,
)


def build() -> HardCorpus:
    """The whole hard corpus: facts and the questions labelled against them."""
    facts: list[HardFact] = []
    questions: list[HardQuestion] = []
    for builder in _BUILDERS:
        cluster = builder()
        facts.extend(cluster.facts)
        questions.extend(cluster.questions)

    ids = [f.id for f in facts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate fact id in the hard corpus")
    texts = [f.text for f in facts]
    if len(set(texts)) != len(texts):
        raise RuntimeError("Duplicate fact text in the hard corpus — gold labelling needs texts to be unique")
    known = set(ids)
    for q in questions:
        missing = [g for g in q.gold if g not in known]
        if missing:
            raise RuntimeError(f"{q.id} references unknown gold ids: {missing}")

    _assert_subjects_are_unique(facts)
    return HardCorpus(facts=facts, questions=questions)


#: Phrases that name a single real-world thing. Two rows opening with the same
#: one are not near-misses, they are a contradiction: the question then has two
#: incompatible answers and a correct "the data conflicts" reply gets scored
#: wrong. This is exactly how the corpus once grew a second "April 2026 outage"
#: with different values and made reflect look broken. Near-misses must differ in
#: what they ASSERT, never in what they claim to BE.
_SUBJECT_PATTERNS = (
    ("numeric_precision", r"^The (\w+ \d{4}) outage"),
    ("releases", r"^Release (\S+) was (deployed to|promoted to|rolled back from) (production|staging|canary)"),
    ("supersession", r"^Ownership of (the [\w ]+?) passed to ([\w ]+?) in"),
)


def _assert_subjects_are_unique(facts: list[HardFact]) -> None:
    """Fail the build when two rows in a cluster claim to describe the same thing."""
    import re

    for cluster, pattern in _SUBJECT_PATTERNS:
        seen: dict[tuple[str, ...], str] = {}
        for fact in facts:
            if fact.cluster != cluster:
                continue
            match = re.match(pattern, fact.text)
            if match is None:
                continue
            key = match.groups()
            if key in seen:
                raise RuntimeError(
                    f"{cluster}: {fact.id} and {seen[key]} both describe {key!r}. "
                    "Two rows describing the same subject contradict rather than compete — "
                    "vary the values, not the identity."
                )
            seen[key] = fact.id
