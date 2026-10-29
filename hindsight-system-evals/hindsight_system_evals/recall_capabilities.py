"""Recall's capabilities, one case each, graded on what a query brings back.

The scenario, end to end through the public API:

    import the case's bank -> add the haystack -> recall one query -> grade the results

Recall itself calls no LLM by default: dates in a query are read by a rule-based parser
(``engine/query_analyzer.py``), and ranking is the embedder plus a reranker. What a model
changes is what recall has to search: the facts, dates and entities retain wrote. Those
are frozen: each case's memories were extracted once by a real model and exported to
``fixtures/recall-cases/<case id>.zip``, and a run imports that archive, so the same code
ranks the same facts every time and a run needs no LLM. ``HINDSIGHT_EVAL_RECALL_RECORD=1`` re-extracts with the configured model and
rewrites the archives — after a case's memories change, or to measure another model's
extraction as recall finds it. ``manifest.json`` beside them names the model.

Graded on the returned facts' text, with no judge:

* ``found`` — a word must appear in one of the top ``within`` results (``"a|b"`` accepts
  either, since extraction may reword);
* ``ranked`` — the first result naming A must come before the first one naming B
  without A (a fact that names both, "switched from MySQL to Postgres", counts as A).

Only what the embedder and the ranking decide is graded here. Filters, tags, token
budgets, ``min_scores`` floors and observations are deterministic, and the stub-model
system tests pin them. Each case also stores decoys: wrong answers built so one search
arm alone would pick them (dated inside the asked window, sharing the query's words,
about someone else), so a case passes only when ranking weighs the arms well.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hindsight_client import Hindsight
from hindsight_client_api.models.recall_result import RecallResult

from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.recall_haystack import haystack_items


#: One archive per case: its own memories as extraction (and consolidation) wrote them.
CASE_BANKS = Path(__file__).resolve().parents[1] / "fixtures" / "recall-cases"
CASE_BANKS_MANIFEST = CASE_BANKS / "manifest.json"
RECORD = os.getenv("HINDSIGHT_EVAL_RECALL_RECORD") == "1"


@dataclass(frozen=True)
class Memory:
    content: str
    #: The event date extraction resolves relative dates against (ISO).
    timestamp: str | None = None


@dataclass(frozen=True)
class Case:
    id: str
    #: The group the per-category tables report it under.
    category: str
    #: Where it comes from: issues, PRs, docs.
    origin: str
    memories: tuple[Memory, ...]
    query: str
    #: The "now" a relative date in the query resolves against (ISO).
    query_timestamp: str | None = None
    #: Wrong answers built to win one search arm: dated inside the asked window, sharing
    #: the query's words, about someone else. Stored as written, beside the haystack.
    decoys: tuple[Memory, ...] = ()
    found: tuple[str, ...] = ()
    within: int = 3
    ranked: tuple[tuple[str, str], ...] = ()
    minimum: bool = False


@dataclass
class CaseOutcome:
    case: Case
    bank_id: str
    results: list[RecallResult]
    failed: list[str] = field(default_factory=list)

    @property
    def rendered(self) -> str:
        if not self.results:
            return "(no results)"
        return "\n".join(f"{i + 1}. [{r.type}] {r.text}" for i, r in enumerate(self.results))


def _names(text: str, word: str) -> bool:
    return any(option.lower() in text.lower() for option in word.split("|"))


def _first(results: list[RecallResult], word: str, *, without: str | None = None) -> int | None:
    for i, result in enumerate(results):
        if _names(result.text, word) and not (without and _names(result.text, without)):
            return i
    return None


async def run_case(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> CaseOutcome:
    if RECORD:
        await _record(client, bank_id, settled, case)
    else:
        archive = CASE_BANKS / f"{case.id}.zip"
        if not archive.exists():
            raise RuntimeError(f"missing {archive} — record it with HINDSIGHT_EVAL_RECALL_RECORD=1")
        # Off before the import, so nothing consolidates the frozen facts again.
        await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
        await client.document_transfer.import_documents(bank_id, archive.read_bytes())
        await settled(bank_id)
    # Stored as written, so the same every run without an archive of its own.
    await client.aupdate_bank_config(bank_id, retain_extraction_mode="chunks")
    await client.aretain_batch(
        bank_id=bank_id,
        items=haystack_items()
        + [
            {"content": m.content, "document_id": f"decoy-{i}", **({"timestamp": m.timestamp} if m.timestamp else {})}
            for i, m in enumerate(case.decoys)
        ],
    )
    await settled(bank_id)

    options = {"query_timestamp": case.query_timestamp} if case.query_timestamp else {}
    response = await client.arecall(bank_id, case.query, **options)
    outcome = CaseOutcome(case=case, bank_id=bank_id, results=list(response.results))
    results = outcome.results

    top = results[: case.within]
    for word in case.found:
        if not any(_names(r.text, word) for r in top):
            outcome.failed.append(f"{word!r} not in the top {case.within}")
    for better, worse in case.ranked:
        a, b = _first(results, better), _first(results, worse, without=better)
        if a is None:
            outcome.failed.append(f"nothing naming {better!r} came back")
        elif b is not None and b < a:
            outcome.failed.append(f"{worse!r} (#{b + 1}) ranked above {better!r} (#{a + 1})")
    return outcome


async def _record(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    """Retain the case's memories through the real extraction and freeze them as its archive."""
    await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
    # One document per memory, retained together: each is its own source, as separate
    # conversations would be.
    await client.aretain_batch(
        bank_id=bank_id,
        items=[
            {
                "content": m.content,
                "document_id": f"{case.id}-{i}",
                **({"timestamp": m.timestamp} if m.timestamp else {}),
            }
            for i, m in enumerate(case.memories)
        ],
    )
    await settled(bank_id)
    archive = await client.aexport_documents(bank_id)
    CASE_BANKS.mkdir(parents=True, exist_ok=True)
    (CASE_BANKS / f"{case.id}.zip").write_bytes(archive)
    manifest = json.loads(CASE_BANKS_MANIFEST.read_text()) if CASE_BANKS_MANIFEST.exists() else {}
    manifest[case.id] = {
        "llm": f"{os.getenv('HINDSIGHT_EVAL_LLM_PROVIDER')}/{os.getenv('HINDSIGHT_EVAL_LLM_MODEL')}",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    CASE_BANKS_MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")


# ── Cases ───────────────────────────────────────────────────────────────────────────

_CHATTER = (
    Memory("I had pasta for lunch today, it was fine."),
    Memory("The weather has been grey all week."),
)

CASES: tuple[Case, ...] = (
    # ── Semantic: meaning, not words ─────────────────────────────────────────────
    Case(
        id="paraphrase-dislike",
        category="semantic",
        origin="the embedder's job: 'can't stand' is a dislike though no word says so",
        memories=(
            Memory("I can't stand cilantro, it tastes like soap to me."),
            Memory("I love dark chocolate, the more bitter the better."),
            *_CHATTER,
        ),
        query="Which foods does the user dislike?",
        decoys=(
            Memory("My brother dislikes olives and won't eat them."),
            Memory("The user's favourite foods are ramen and dumplings."),
            Memory("An article listed the foods most kids dislike: broccoli and liver."),
        ),
        found=("cilantro",),
        minimum=True,
    ),
    Case(
        id="no-shared-words",
        category="semantic",
        origin="a query sharing no content word with the memory it needs",
        memories=(
            Memory("My flight to Denver got cancelled because of a snowstorm.", timestamp="2024-01-15T09:00:00"),
            Memory("I renewed my passport at the consulate.", timestamp="2024-01-10T09:00:00"),
            *_CHATTER,
        ),
        query="travel disruption caused by bad weather",
        decoys=(
            Memory("The travel budget was disrupted by the reorg."),
            Memory("Bad weather is forecast for the weekend, so the picnic is off."),
            Memory("The deploy failure was caused by a bad config value."),
            Memory("A rail strike could cause travel disruption next month."),
        ),
        found=("Denver",),
    ),
    Case(
        id="intent-over-keywords",
        category="semantic",
        origin="two memories share 'password'; only one is about the test environment's credentials",
        memories=(
            Memory("The staging database password is rotated every 30 days by the platform team."),
            Memory("The office wifi password is printed on the fridge in the kitchen."),
        ),
        query="How often do the credentials for the test environment change?",
        decoys=(
            Memory("The credentials for the production environment change every 90 days."),
            Memory("The test environment is rebuilt from scratch every night."),
            Memory("How often should passwords change? The security team says when compromised."),
        ),
        ranked=(("staging", "wifi"),),
        found=("30 days|30-day|thirty",),
    ),
    # ── Keyword: exact identifiers the embedder blurs ─────────────────────────────
    Case(
        id="error-code",
        category="keyword",
        origin="BM25 arm: three error codes an embedder sees as near-identical",
        memories=(
            Memory("Checkout fails with ERR_7731_TIMEOUT when the payment gateway is slow."),
            Memory("Login fails with ERR_7713_AUTH when the session cookie has expired."),
            Memory("Search fails with ERR_1377_INDEX while the index is rebuilding."),
        ),
        query="ERR_7731_TIMEOUT",
        decoys=(
            Memory("Checkout TIMEOUT errors spiked on Friday after the sale started."),
            Memory("The mobile app shows ERR_1773_TIMEOUT when the network drops."),
            Memory("ERR_TIMEOUT errors are logged to the payments dashboard."),
        ),
        found=("7731",),
        within=1,
        minimum=True,
    ),
    Case(
        id="order-id",
        category="keyword",
        origin="BM25 arm: two order ids one transposition apart",
        memories=(
            Memory("Order A9-44-ZQ shipped from the Rotterdam warehouse on Monday."),
            Memory("Order A9-44-QZ was returned by the customer as damaged."),
        ),
        query="A9-44-ZQ",
        decoys=(
            Memory("Order A9-44-ZR shipped from the Hamburg warehouse on Tuesday."),
            Memory("Order A9-45-ZQ was cancelled before it shipped."),
        ),
        ranked=(("Rotterdam", "damaged"),),
    ),
    Case(
        id="rare-surname",
        category="keyword",
        origin="a rare name the embedder has never seen, among common ones",
        memories=(
            Memory("Dr. Oyelaran-Whitfield reviewed the safety report and asked for two more trials."),
            Memory("Dr. Smith reviewed the budget report and approved it."),
            Memory("Dr. Jones reviewed the hiring plan."),
        ),
        query="Oyelaran-Whitfield",
        decoys=(
            Memory("Dr. Olaniran-Whitfield approved the two extra trials."),
            Memory("Dr. Whitfield-Oyelowo reviewed the safety report last year."),
        ),
        found=("Oyelaran",),
        within=1,
    ),
    # ── Temporal: dates in the query, resolved against query_timestamp ────────────
    # With a cross-encoder the date or recency is a ±10% multiplier on its score, too small
    # to reorder (#6132); a decision model is shown each memory's date and weighs it itself.
    Case(
        id="last-month",
        category="temporal",
        origin="query_analyzer: 'last month' resolved against query_timestamp (#4356 made the anchor explicit)",
        memories=(
            Memory("I went to Lisbon for a conference.", timestamp="2024-03-02T10:00:00"),
            Memory("I went to Rome to see my family.", timestamp="2024-01-10T10:00:00"),
            Memory("I went to Berlin for a friend's wedding.", timestamp="2023-11-12T10:00:00"),
        ),
        query="Where did I travel last month?",
        decoys=(
            Memory("My colleague travelled to Prague for a sales meeting.", timestamp="2024-03-15T10:00:00"),
            Memory("I planned a trip to Tokyo for the summer.", timestamp="2024-03-12T10:00:00"),
            Memory("I fixed the leaking kitchen sink.", timestamp="2024-03-20T10:00:00"),
            Memory("I went to Porto for a long weekend.", timestamp="2024-04-02T10:00:00"),
        ),
        query_timestamp="2024-04-05T12:00:00",
        found=("Lisbon",),
        within=1,
        minimum=True,
    ),
    Case(
        id="yesterday",
        category="temporal",
        origin="'yesterday' against the query date, not the server's clock (#4356)",
        memories=(
            Memory("I went to a jazz concert at the Blue Note.", timestamp="2024-06-15T21:00:00"),
            Memory("I had a dentist appointment for a filling.", timestamp="2024-06-12T09:00:00"),
            Memory("I went to the gym for leg day.", timestamp="2024-06-10T18:00:00"),
        ),
        query="What did I do yesterday?",
        decoys=(
            Memory("My partner went to a football match with friends.", timestamp="2024-06-15T15:00:00"),
            Memory("The neighbours had a loud party until 2am.", timestamp="2024-06-15T23:00:00"),
            Memory("I went to a pottery class.", timestamp="2024-06-14T18:00:00"),
        ),
        query_timestamp="2024-06-16T10:00:00",
        found=("concert|Blue Note|jazz",),
        within=1,
    ),
    Case(
        id="year-only-event",
        category="temporal",
        origin="#3933: a year-only date is scored over the whole year, not its first day",
        memories=(
            Memory("The 2026 climate summit will be held in Nairobi.", timestamp="2025-01-05T10:00:00"),
            Memory("The 2024 climate summit was held in Lima.", timestamp="2024-12-01T10:00:00"),
            Memory("The 2025 climate summit was held in Hanoi.", timestamp="2025-11-20T10:00:00"),
        ),
        query="Where is the climate summit in 2026?",
        decoys=(
            Memory("The 2026 football World Cup will be held in North America.", timestamp="2025-02-01T10:00:00"),
            Memory("The climate summit dinner was held at the Sheraton.", timestamp="2025-11-21T10:00:00"),
            Memory("In 2026 I want to visit Kenya.", timestamp="2025-12-30T10:00:00"),
        ),
        query_timestamp="2026-02-01T10:00:00",
        found=("Nairobi",),
        within=1,
    ),
    Case(
        id="named-week",
        category="temporal",
        origin="an explicit date range in the query narrows the temporal arm",
        memories=(
            Memory("We deployed version 2 of the billing service.", timestamp="2024-03-05T15:00:00"),
            Memory("We hired Priya as a backend engineer.", timestamp="2024-03-20T10:00:00"),
            Memory("The office moved to the third floor.", timestamp="2024-02-26T10:00:00"),
        ),
        query="What happened in the first week of March 2024?",
        decoys=(
            Memory("We planned the March offsite in the first week of February.", timestamp="2024-02-05T10:00:00"),
            Memory("In the last week of March we hired two interns.", timestamp="2024-03-28T10:00:00"),
            Memory("The first week of April was the quarterly planning week.", timestamp="2024-04-03T10:00:00"),
        ),
        query_timestamp="2024-04-10T10:00:00",
        found=("billing",),
        within=1,
    ),
    # ── Currency: what is true now ranks above what was ──────────────────────────
    Case(
        id="current-job",
        category="currency",
        origin="recency boost: a newer job ranks above the old one (reranking.py, ±10% over 365 days)",
        memories=(
            Memory("I work at Globex as a data analyst.", timestamp="2022-03-01T10:00:00"),
            Memory("I just started a new job at Initech as a staff engineer.", timestamp="2024-05-01T10:00:00"),
        ),
        query="Where does the user work now?",
        decoys=(
            Memory("My friend just started a new job at Hooli as a product manager.", timestamp="2024-05-20T10:00:00"),
            Memory("I worked at Vandelay as an intern.", timestamp="2019-06-01T10:00:00"),
            Memory("Where does the user want to work next? Maybe a startup.", timestamp="2024-05-25T10:00:00"),
        ),
        query_timestamp="2024-06-01T10:00:00",
        ranked=(("Initech", "Globex"),),
        minimum=True,
    ),
    Case(
        id="latest-decision",
        category="currency",
        origin="a later decision supersedes an earlier one; the later must rank first",
        memories=(
            Memory("We decided to use MySQL for the orders service.", timestamp="2024-01-10T10:00:00"),
            Memory(
                "We switched the orders service from MySQL to Postgres because we needed JSONB columns.",
                timestamp="2024-04-02T10:00:00",
            ),
        ),
        query="Which database does the orders service use?",
        decoys=(
            Memory("The inventory service uses MongoDB.", timestamp="2024-04-20T10:00:00"),
            Memory("The orders team discussed which database to use for analytics.", timestamp="2024-04-25T10:00:00"),
        ),
        query_timestamp="2024-05-01T10:00:00",
        ranked=(("Postgres", "MySQL"),),
    ),
    # ── Multi-hop: the answer is one link away ────────────────────────────────────
    Case(
        id="employer-headquarters",
        category="multi_hop",
        origin="graph arm: Alice -> Acme Robotics -> Oslo through a shared entity",
        memories=(
            Memory("Alice Moreau works at Acme Robotics as a firmware engineer."),
            Memory("Acme Robotics has its headquarters in Oslo."),
            Memory("Bob Tanaka works at Initech."),
            Memory("Initech has its headquarters in Austin."),
        ),
        query="In which city is Alice Moreau's employer headquartered?",
        decoys=(
            Memory("Alice Moreau visited a client's headquarters in Copenhagen."),
            Memory("Alice Moreau's brother is a teacher in Bergen."),
        ),
        found=("Oslo",),
        within=4,
        minimum=True,
    ),
    Case(
        id="cause-chain",
        category="multi_hop",
        origin="graph arm, causal links: the cause of the cause comes back too",
        memories=(
            Memory("The checkout outage on 3 May was caused by an expired TLS certificate."),
            Memory("The TLS certificate expired because the renewal cron job had been disabled during the migration."),
            Memory("The search outage on 9 May was caused by a full disk on the index node."),
        ),
        query="Why did the checkout outage happen?",
        decoys=(
            Memory("The checkout outage post-mortem meeting is on Friday."),
            Memory("Why is checkout slow? Mostly large product images."),
            Memory("The checkout page was redesigned in April."),
        ),
        found=("cron|renewal",),
        within=4,
    ),
    Case(
        id="relative-and-pet",
        category="multi_hop",
        origin="'my sister' -> Hana -> her dog: the question never names Hana",
        memories=(
            Memory("My sister Hana lives in Kyoto and teaches piano."),
            Memory("Hana adopted a shiba inu called Mochi last year."),
            Memory("My neighbour has a cat called Biscuit."),
        ),
        query="What's the name of my sister's dog?",
        decoys=(
            Memory("My sister's friend has a dog called Rex."),
            Memory("My colleague's sister has a beagle called Pepper."),
        ),
        found=("Mochi",),
        within=4,
    ),
    # ── Language: queries in the language of their memories ──────────────────────
    Case(
        id="italian",
        category="language",
        origin="#4410/#4432 kept non-English memories in their language; recall has to find them in it",
        memories=(
            Memory("Ho comprato una bicicletta rossa a Torino la settimana scorsa."),
            Memory("Mia sorella lavora in un ospedale a Milano."),
            Memory("Domenica abbiamo mangiato la pizza da Gino."),
        ),
        query="Che bicicletta ho comprato?",
        decoys=(
            Memory("Ho comprato un casco nuovo per andare al lavoro."),
            Memory("Mio fratello ha comprato una moto rossa."),
        ),
        # Either language: extraction may translate the memory, and recall is what is graded.
        found=("bicicletta|bicycle",),
        within=1,
    ),
    Case(
        id="chinese-last-week",
        category="language",
        origin="query_analyzer's Chinese rules: 上周 (last week) against the query date",
        memories=(
            Memory("我在北京见了王老师，讨论了论文的修改。", timestamp="2024-05-08T10:00:00"),
            Memory("我在上海参加了一个设计会议。", timestamp="2024-04-10T10:00:00"),
            Memory("我在家里休息，看了一部电影。", timestamp="2024-05-15T10:00:00"),
        ),
        query="我上周见了谁？",
        decoys=(
            Memory("我妹妹在医院见了李医生。", timestamp="2024-05-09T10:00:00"),
            Memory("我在公司开了一个很长的会。", timestamp="2024-05-10T10:00:00"),
        ),
        query_timestamp="2024-05-16T10:00:00",
        found=("王|Wang",),
        within=2,
    ),
    # ── Precision: the right value among near-identical ones ─────────────────────
    Case(
        id="number-among-numbers",
        category="precision",
        origin="#5497-like: the same sentence with a different number or quarter",
        memories=(
            Memory("The marketing budget for Q3 2024 is $12,500."),
            Memory("The marketing budget for Q2 2024 was $1,250."),
            Memory("The design budget for Q3 2024 is $8,000."),
        ),
        query="What is the Q3 marketing budget?",
        decoys=(
            Memory("The marketing budget for Q3 2023 was $11,000."),
            Memory("Q3 marketing plan: three webinars and a newsletter."),
        ),
        found=("12,500|12500|12.5",),
        within=1,
        minimum=True,
    ),
    Case(
        id="negation-kept",
        category="precision",
        origin="a negated fact must come back as negated, not as its opposite",
        memories=(
            Memory("I don't use Docker at all; I run every container with Podman."),
            Memory("My colleague Sam uses Docker Desktop on his Mac."),
        ),
        query="Does the user use Docker?",
        decoys=(
            Memory("Docker images for the billing service are built in CI."),
            Memory("Does the user use Linux? Yes, Fedora on a ThinkPad."),
        ),
        # The user's "don't use Docker" outranks Sam's "uses Docker Desktop", which shares more
        # words with the query. That the stored fact keeps the "don't" is extraction's
        # job, checked once when the archive is recorded.
        found=("Podman",),
        ranked=(("Podman", "Docker Desktop"),),
    ),
)
