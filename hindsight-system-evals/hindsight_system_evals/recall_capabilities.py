"""Recall's capabilities, one case each, graded on what a query brings back.

The scenario, end to end through the public API:

    retain the memories (real extraction) -> wait -> recall one query -> grade the results

Recall itself calls no LLM by default: dates in a query are read by a rule-based parser
(``engine/query_analyzer.py``), and ranking is the embedder plus a cross-encoder. What a
model changes is what recall has to search: the facts, dates and entities retain wrote
(and the observations consolidation wrote). So every case retains through the real
extraction, and a model comparison here measures that model's memories as recall finds
them — the end-to-end path a user's question takes.

Graded deterministically wherever the API allows, on the returned facts' text:

* ``found`` — a word must appear in one of the top ``within`` results (``"a|b"`` accepts
  either, since extraction may reword);
* ``ranked`` — the first result naming A must come before the first one naming B
  without A (a fact that names both, "switched from MySQL to Postgres", counts as A);
* ``absent`` — a word that must not come back at all;
* ``types`` — every result is one of these fact types;
* ``check`` — anything else (token budgets, chunks, observations and their sources).

and with the judge only where wording decides it (``must`` / ``traps``, on the results).

What the stub-model system tests already pin is left to them: every tag mode, the
``min_scores`` floors, the budget levels, query-length capping. A case here exists
where the real embedder, reranker or extraction is the thing that could fail.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.recall_result import RecallResult

from hindsight_system_evals.judge import evaluate
from hindsight_system_evals.pages import SettleFn


@dataclass(frozen=True)
class Memory:
    content: str
    #: The event date extraction resolves relative dates against (ISO).
    timestamp: str | None = None
    context: str | None = None
    tags: tuple[str, ...] = ()


#: Extra checks a case needs beyond the declarative ones: given the client, the bank and
#: the response, return what failed.
Check = Callable[[Hindsight, str, RecallResponse], Awaitable[list[str]]]


@dataclass(frozen=True)
class Recall:
    """The ``arecall`` options a case sets; anything left None keeps the client's default."""

    types: list[str] | None = None
    tags: list[str] | None = None
    tags_match: Literal["any", "all", "any_strict", "all_strict", "exact"] | None = None
    query_timestamp: str | None = None
    max_tokens: int | None = None
    include_chunks: bool | None = None
    prefer_observations: bool | None = None

    def options(self) -> dict[str, object]:
        return {name: value for name, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Case:
    id: str
    #: The group the per-category tables report it under.
    category: str
    #: Where it comes from: issues, PRs, docs.
    origin: str
    memories: tuple[Memory, ...]
    query: str
    recall: Recall = Recall()
    found: tuple[str, ...] = ()
    within: int = 3
    ranked: tuple[tuple[str, str], ...] = ()
    absent: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    must: tuple[str, ...] = ()
    traps: tuple[str, ...] = ()
    check: Check | None = None
    #: Consolidation on, for the cases about observations.
    observations: bool = False
    minimum: bool = False


@dataclass
class CaseOutcome:
    case: Case
    bank_id: str
    results: list[RecallResult]
    failed: list[str] = field(default_factory=list)
    trapped: list[str] = field(default_factory=list)

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
    await client.aupdate_bank_config(
        bank_id, enable_observations=case.observations, enable_auto_consolidation=case.observations
    )
    # One document per memory, retained together: each is its own source, as separate
    # conversations would be.
    await client.aretain_batch(
        bank_id=bank_id,
        items=[
            {
                "content": m.content,
                "document_id": f"{case.id}-{i}",
                **({"timestamp": m.timestamp} if m.timestamp else {}),
                **({"context": m.context} if m.context else {}),
                **({"tags": list(m.tags)} if m.tags else {}),
            }
            for i, m in enumerate(case.memories)
        ],
    )
    await settled(bank_id)

    response = await client.arecall(bank_id, case.query, **case.recall.options())
    outcome = CaseOutcome(case=case, bank_id=bank_id, results=list(response.results))
    results = outcome.results

    top = results[: case.within]
    for word in case.found:
        if not any(_names(r.text, word) for r in top):
            outcome.failed.append(f"{word!r} not in the top {case.within}")
    for word in case.absent:
        if any(_names(r.text, word) for r in results):
            outcome.trapped.append(f"{word!r} came back")
    for better, worse in case.ranked:
        a, b = _first(results, better), _first(results, worse, without=better)
        if a is None:
            outcome.failed.append(f"nothing naming {better!r} came back")
        elif b is not None and b < a:
            outcome.failed.append(f"{worse!r} (#{b + 1}) ranked above {better!r} (#{a + 1})")
    if case.types:
        wrong = sorted({r.type for r in results} - set(case.types))
        if wrong:
            outcome.failed.append(f"results of type {wrong}, asked for {list(case.types)}")
    if case.check is not None:
        outcome.failed.extend(await case.check(client, bank_id, response))

    if case.must or case.traps:
        context = f"These are the memories a search returned, best first, for the query: {case.query!r}"
        for trap in case.traps:
            verdict = await evaluate(outcome.rendered, trap, context=context, confirm_met=True)
            if verdict.meets_criteria:
                outcome.trapped.append(f"{trap} — {verdict.reasoning}")
        for criterion in case.must:
            verdict = await evaluate(outcome.rendered, criterion, context=context)
            if not verdict.meets_criteria:
                outcome.failed.append(f"{criterion} — {verdict.reasoning}")
    return outcome


# ── Checks a declarative field cannot express ───────────────────────────────────────


async def _budget_is_respected(client: Hindsight, bank_id: str, response: RecallResponse) -> list[str]:
    """A small ``max_tokens`` returns fewer facts than the default, but never none (#3688)."""
    uncapped = await client.arecall(bank_id, "What has the team been working on?")
    if not response.results:
        return ["max_tokens=60 returned nothing — a small budget must still return the facts that fit"]
    if len(response.results) >= len(uncapped.results):
        return [f"max_tokens=60 returned {len(response.results)} facts, as many as the default budget"]
    return []


async def _chunks_without_facts(client: Hindsight, bank_id: str, response: RecallResponse) -> list[str]:
    """``max_tokens=0`` is the documented way to ask for the source text only (#364)."""
    failed = []
    if response.results:
        failed.append(f"max_tokens=0 still returned {len(response.results)} facts")
    if not response.chunks:
        failed.append("include_chunks with max_tokens=0 returned no chunks")
    elif not any("Juniper" in chunk.text for chunk in response.chunks.values()):
        failed.append("no returned chunk holds the Juniper release note")
    return failed


async def _observations_supersede_their_facts(client: Hindsight, bank_id: str, response: RecallResponse) -> list[str]:
    """With ``prefer_observations``, no raw fact comes back beside the observation built from it (#5617)."""
    observations = [r for r in response.results if r.type == "observation"]
    if not observations:
        return ["no observation came back"]
    sources = {source for o in observations for source in (o.source_fact_ids or [])}
    doubled = [r.text for r in response.results if r.type != "observation" and r.id in sources]
    return [f"raw facts returned beside their observation: {doubled}"] if doubled else []


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
        found=("Oyelaran",),
        within=1,
    ),
    # ── Temporal: dates in the query, resolved against query_timestamp ────────────
    # These and the currency cases fail on every model today: the date or recency is a
    # ±10% multiplier on the cross-encoder's score, too small to reorder (#6132).
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
        recall=Recall(query_timestamp="2024-04-05T12:00:00"),
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
        recall=Recall(query_timestamp="2024-06-16T10:00:00"),
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
        recall=Recall(query_timestamp="2026-02-01T10:00:00"),
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
        recall=Recall(query_timestamp="2024-04-10T10:00:00"),
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
        recall=Recall(query_timestamp="2024-06-01T10:00:00"),
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
        recall=Recall(query_timestamp="2024-05-01T10:00:00"),
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
        found=("Mochi",),
        within=4,
    ),
    # ── Filters: the fact types and tags a caller asks for ───────────────────────
    Case(
        id="experience-only",
        category="filters",
        origin="types=[experience]: what the assistant did, and nothing the user said (#775, #2440)",
        memories=(
            Memory(
                "user: the payments test is flaky again, I prefer we fix tests rather than skip them.\n"
                "assistant: I fixed the flaky payments test in test_payments.py by adding a retry around the "
                "network call.",
                context="conversation between the user and a coding assistant",
            ),
        ),
        query="What did the assistant change?",
        recall=Recall(types=["experience"]),
        types=("experience",),
        found=("retry|flaky",),
    ),
    Case(
        id="world-only",
        category="filters",
        origin="types=[world]: the user's stated preference, without the assistant's actions",
        memories=(
            Memory(
                "user: I prefer tabs over spaces, and please never reformat files I didn't touch.\n"
                "assistant: Understood. I reformatted only main.py and committed it as 'fix lint'.",
                context="conversation between the user and a coding assistant",
            ),
        ),
        query="What are the user's formatting preferences?",
        recall=Recall(types=["world"]),
        types=("world",),
        found=("tabs",),
    ),
    Case(
        id="tag-scoped",
        category="filters",
        origin="tags any_strict: one user's memories, not another's, not the untagged ones (#2295)",
        memories=(
            Memory("Ana's favourite editor is Helix.", tags=("user:ana",)),
            Memory("Ben's favourite editor is Zed.", tags=("user:ben",)),
            Memory(
                "The team's standard editor is VS Code.",
            ),
        ),
        query="Which editor is the favourite?",
        recall=Recall(tags=["user:ana"], tags_match="any_strict"),
        found=("Helix",),
        absent=("Zed", "VS Code"),
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
        found=("bicicletta",),
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
        recall=Recall(query_timestamp="2024-05-16T10:00:00"),
        found=("王",),
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
        must=("A result says the user does not use Docker (they use Podman).",),
        traps=("A result states that the user (not Sam) uses Docker.",),
    ),
    # ── Budget: how much comes back ──────────────────────────────────────────────
    Case(
        id="small-token-budget",
        category="budget",
        origin="#3688/#3704: max_tokens cuts in rank order and never returns nothing",
        memories=tuple(
            Memory(text)
            for text in (
                "The team migrated the billing service to Go.",
                "The team added rate limiting to the public API.",
                "The team rewrote the onboarding emails.",
                "The team moved CI from Jenkins to GitHub Actions.",
                "The team shipped dark mode in the dashboard.",
                "The team cut the Docker image size in half.",
            )
        ),
        query="What has the team been working on?",
        recall=Recall(max_tokens=60),
        check=_budget_is_respected,
    ),
    Case(
        id="chunks-only",
        category="budget",
        origin="#364: max_tokens=0 with include_chunks returns the source text and no facts",
        memories=(
            Memory("Release Juniper adds offline mode and fixes the sync crash on Android."),
            Memory("Release Hawthorn added the export to CSV."),
        ),
        query="What does the Juniper release add?",
        recall=Recall(max_tokens=0, include_chunks=True),
        check=_chunks_without_facts,
    ),
    # ── Observations: what consolidation wrote, found by recall ──────────────────
    Case(
        id="observation-type",
        category="observations",
        origin="types=[observation]: the bank's consolidated belief, not the raw facts",
        observations=True,
        memories=(
            Memory("I always drink green tea in the morning.", timestamp="2024-02-01T08:00:00"),
            Memory("Green tea is the first thing I have every morning.", timestamp="2024-02-08T08:00:00"),
            Memory("I don't drink coffee, it makes me jittery.", timestamp="2024-02-15T08:00:00"),
        ),
        query="What does the user drink in the morning?",
        recall=Recall(types=["observation"]),
        types=("observation",),
        found=("tea",),
    ),
    Case(
        id="prefer-observations",
        category="observations",
        origin="#5617: prefer_observations drops the raw facts an observation was built from",
        observations=True,
        memories=(
            Memory("Marta leads the payments team.", timestamp="2024-03-01T10:00:00"),
            Memory("Marta has led the payments team since 2022.", timestamp="2024-03-10T10:00:00"),
            Memory("The payments team is led by Marta.", timestamp="2024-03-20T10:00:00"),
        ),
        query="Who leads the payments team?",
        recall=Recall(types=["world", "observation"], prefer_observations=True),
        found=("Marta",),
        check=_observations_supersede_their_facts,
    ),
)
