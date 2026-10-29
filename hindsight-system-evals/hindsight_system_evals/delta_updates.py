"""Updates a delta refresh has to fold into a page it already wrote.

The convergence cases grade a page that is *built* across waves: most of what the second
wave carries is new to the page. This grades the other half of a delta's job —
an established page, then a wave that changes what it says. Each case is one
way an update reaches a page (its category in brackets):

* ``du-oncall-departed`` (update) — the new memory never names the topic; only the page
  connects them ("Dana left the company", on a page naming Dana as on-call);
* ``du-renamed-service`` (update) — the update names a service by its new name, and the rename
  arrives in the same wave;
* ``du-backup-correction`` (update) — a memory saying an earlier figure was wrong;
* ``du-enterprise-plan`` (aggregation) — one item joins a list and another leaves it;
* ``du-launch-risk`` (aggregation) — something the topic covers that the page has never mentioned;
* ``du-quiet-wave`` (stability) — nothing in the wave bears on the page, which must survive it;
* ``du-rerecorded-history`` (stability) — a release history grown over five deltas, 20 releases
  to 61, each wave re-recording in other words every release the page already lists. A prompt
  that called every fact "genuinely new" got them re-added: on replays a model re-added 27
  listed releases per reply, and one reply in ten wrote until its output cap, the cut-off reply
  failing the refresh;
* ``du-incident-table`` (aggregation) — an incident table grown over four deltas, 10 rows to 45,
  every wave's rows dated between rows already there. A prompt that said "re-emit the table with
  the row added" got the table re-typed, and re-sorted, on every delta, dropping rows as it grew.

The last two also count the items the page lists, a fact about the text a judge reads poorly over
sixty items: nothing the wave-1 page listed may vanish, every item a later wave brings must land,
and at most one may appear twice (a "latest" line may restate the newest one).

Behind it: the decision-model gate in front of a delta refresh asks, memory by
memory, "could this change the page?" and hands only the yeses to the LLM. A
memory it drops is never seen again — the next refresh starts after it. On one
cost run a decision model dropped 20 of 20 on-topic memories and the page kept
reading as it had. A wrong "no" leaves no trace but a page that stopped
updating, so this suite looks at the page. The first four cases are the ones a
yes/no on one memory at a time finds hardest, and every wave also carries the
hard corpus, so the gate has plenty to say no to.

Every case is graded twice, like the convergence cases: ``correct`` and the trap, which
here is the page still asserting the value the wave replaced. The wave-1 page
must state that old value first, or the update has nothing to replace.
"""

from __future__ import annotations

from dataclasses import dataclass

from hindsight_client import Hindsight

from hindsight_system_evals.corpus import HardQuestion
from hindsight_system_evals.pages import (
    PageOutcome,
    SettleFn,
    WaveSnapshot,
    create_page,
    facts,
    prepare_bank,
    read_page,
    split_into_waves,
)

#: The page is written from wave 1 and updated by wave 2; the dates make the
#: second wave unambiguously the later one.
_BASE_TIME = "2026-03-02T12:00:00Z"
_UPDATE_TIME = "2026-07-01T12:00:00Z"


@dataclass(frozen=True)
class DeltaUpdate:
    question: HardQuestion
    #: Wave 1: what the page is first written from.
    base: tuple[str, ...]
    #: Wave 2: what has to change it.
    update: tuple[str, ...]
    #: What the wave-1 page must already say, or the update replaces nothing.
    base_criteria: str
    #: Items the page lists, counted in its text: none the wave-1 page had may vanish,
    #: every one a later wave brings must land, at most one may appear twice.
    listed: tuple[str, ...] = ()
    #: Waves after the update, each retained and refreshed in turn, for a page that
    #: has to grow over several deltas to reach the size where a failure shows.
    later: tuple[tuple[str, ...], ...] = ()

    @property
    def id(self) -> str:
        return self.question.id

    @property
    def category(self) -> str:
        return self.question.category


def _q(question_id: str, category: str, question: str, criteria: str, trap: str) -> HardQuestion:
    return HardQuestion(
        id=question_id,
        category=category,
        question=question,
        ideal_query=question,
        answer_criteria=criteria,
        must_not_claim=trap,
    )


@dataclass(frozen=True)
class _Release:
    version: str
    date: str
    feature: str

    def shipped(self) -> str:
        return f"Corvid {self.version} shipped on {self.date} with {self.feature}."

    def rerecorded(self) -> str:
        """The same release, recorded again in other words."""
        return f"Changelog: Corvid release {self.version} ({self.date}) added {self.feature}."


@dataclass(frozen=True)
class _Incident:
    id: str
    date: str
    service: str
    duration: str
    region: str
    cause: str
    mitigation: str

    def text(self) -> str:
        return (
            f"Incident {self.id} on {self.date}: {self.service} was down for {self.duration} in {self.region}; "
            f"root cause was {self.cause}, mitigated with {self.mitigation}."
        )


_FEATURES = (
    "offline mode", "SAML single sign-on", "audit log export", "a dark theme", "bulk CSV import",
    "signed webhooks", "per-tenant rate limits", "scheduled exports", "keyboard shortcuts", "usage alerts",
    "a GraphQL endpoint", "field-level encryption",
)  # fmt: skip
_COMPONENTS = ("the auth service", "the billing service", "the web console", "the CLI", "the search indexer")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October")
#: Sixty releases, January to October 2026, six a month: version 3.<n>.0.
_RELEASES = tuple(
    _Release(f"3.{n}.0", f"{2 + (n % 6) * 5} {_MONTHS[n // 6]} 2026", f"{_FEATURES[n % 12]} in {_COMPONENTS[n % 5]}")
    for n in range(60)
)
_NEW_RELEASE = _Release("3.60.0", "30 October 2026", "scheduled reports in the web console")
#: Each later wave re-records every release the page already lists and adds ten new
#: ones: 20 on the first page, 60 by the last wave, then the one new release.
_HISTORY_WAVES = tuple(
    tuple(r.rerecorded() for r in _RELEASES[:end]) + tuple(r.shipped() for r in _RELEASES[end : end + 10])
    for end in (20, 30, 40, 50)
) + (tuple(r.rerecorded() for r in _RELEASES) + (_NEW_RELEASE.shipped(),),)

_SERVICES = ("billing-api", "search", "ingest", "auth", "notifier", "scheduler", "web console")
_REGIONS = ("us-east-1", "eu-west-1", "ap-southeast-2")
_CAUSES = (
    "an expired TLS certificate", "a full disk on the primary", "a bad config push", "a connection-pool leak",
    "a runaway cron job", "a DNS misconfiguration", "a schema migration lock",
)  # fmt: skip
_MITIGATIONS = (
    "certificate expiry alerts", "disk usage alerts at 80%", "staged config rollouts", "pool size limits",
    "cron concurrency caps", "a secondary resolver", "online migrations with a shadow table",
)  # fmt: skip


def _incident(n: int) -> _Incident:
    # Dates scattered over the year, so every wave's rows fall *between* rows already
    # in the table — the shape that tempted a model to re-sort it.
    day = (n * 37) % 300
    return _Incident(
        id=f"INC-{301 + n}",
        date=f"{1 + day % 30} {_MONTHS[min(day // 30, 9)]} 2026",
        service=_SERVICES[n % 7],
        duration=f"{15 + (n * 7) % 140} minutes",
        region=_REGIONS[n % 3],
        cause=_CAUSES[(n * 3) % 7],
        mitigation=_MITIGATIONS[(n * 3) % 7],
    )


_INCIDENTS = tuple(_incident(n) for n in range(45))


def _incident_wave(start: int, end: int) -> tuple[str, ...]:
    return tuple(i.text() for i in _INCIDENTS[start:end])


CASES: tuple[DeltaUpdate, ...] = (
    DeltaUpdate(
        question=_q(
            "du-oncall-departed",
            "update",
            "Who should be paged for a payments-api incident?",
            "Does not present Dana Reyes as someone to page now: it says she has left the company, or "
            "drops her, or names Marco Silva as the one to page.",
            "Dana Reyes is currently the person to page for payments-api incidents, with no mention that she has left",
        ),
        base=(
            "Dana Reyes is the primary on-call engineer for payments-api incidents.",
            "Unanswered payments-api pages escalate to Marco Silva after 15 minutes.",
        ),
        update=("Dana Reyes left the company on 30 June 2026 and her accounts were deactivated.",),
        base_criteria="Names Dana Reyes as the primary on-call engineer for payments-api.",
    ),
    DeltaUpdate(
        question=_q(
            "du-renamed-service",
            "update",
            "What platform does the search-indexer service run on?",
            "States that the service (now called lexicon) runs on AWS Lambda since June 2026. Mentioning "
            "Kubernetes as where it ran before is fine.",
            "the search-indexer service currently runs on Kubernetes",
        ),
        base=("The search-indexer service runs on the shared Kubernetes cluster in us-east-1.",),
        update=(
            "The search-indexer service was renamed to lexicon on 2 May 2026.",
            "lexicon was moved off Kubernetes onto AWS Lambda on 10 June 2026.",
        ),
        base_criteria="States that the search-indexer service runs on Kubernetes.",
    ),
    DeltaUpdate(
        question=_q(
            "du-backup-correction",
            "update",
            "How long are database backups retained?",
            "States that database backups are retained for 14 days. Mentioning that 30 days was an "
            "earlier, wrong figure is fine.",
            "database backups are retained for 30 days",
        ),
        base=("Database backups are retained for 30 days.",),
        update=(
            "The infra review found the 30-day backup figure was never true: backups have always been "
            "kept for 14 days.",
        ),
        base_criteria="States that database backups are retained for 30 days.",
    ),
    DeltaUpdate(
        question=_q(
            "du-enterprise-plan",
            "aggregation",
            "Which customers are on the Enterprise plan?",
            "Lists exactly Harbor Mills, Quanta Rail and Orbital Freight as the Enterprise customers. "
            "Kestrel Labs may appear only as a former Enterprise customer.",
            "Kestrel Labs is currently on the Enterprise plan",
        ),
        base=(
            "Harbor Mills is on the Enterprise plan.",
            "Quanta Rail is on the Enterprise plan.",
            "Kestrel Labs is on the Enterprise plan.",
            "Pinewood Dental is on the Team plan.",
        ),
        update=(
            "Orbital Freight signed an Enterprise plan contract on 4 June 2026.",
            "Kestrel Labs moved down to the Team plan on 12 June 2026.",
        ),
        base_criteria="Lists Harbor Mills, Quanta Rail and Kestrel Labs as Enterprise customers.",
    ),
    DeltaUpdate(
        question=_q(
            "du-launch-risk",
            "aggregation",
            "What are the known risks to the September 2026 launch?",
            "Includes, among the risks, that the payment SDK vendor ends support for the SDK on 1 September "
            "2026, before the launch. Also keeps the load-test risk.",
            "",
        ),
        base=(
            "The launch is scheduled for 15 September 2026.",
            "A known risk for the September launch: the load tests have only reached 60% of the expected peak traffic.",
        ),
        update=(
            "The vendor of the payment SDK we ship with announced it ends support for that SDK on 1 September 2026.",
        ),
        base_criteria="Names the load tests reaching only 60% of peak traffic as a launch risk.",
    ),
    DeltaUpdate(
        question=_q(
            "du-quiet-wave",
            "stability",
            "Which region hosts the primary analytics warehouse?",
            "States that the primary analytics warehouse is in eu-west-1.",
            "the primary analytics warehouse is in a region other than eu-west-1",
        ),
        base=("The primary analytics warehouse is hosted in eu-west-1.",),
        update=(),
        base_criteria="States that the primary analytics warehouse is in eu-west-1.",
    ),
    DeltaUpdate(
        question=_q(
            "du-rerecorded-history",
            "stability",
            "Which releases of Corvid have shipped, when, and what did each add?",
            f"Lists release {_NEW_RELEASE.version} ({_NEW_RELEASE.date}, {_NEW_RELEASE.feature}) and still "
            "lists the earlier 3.x releases.",
            "",
        ),
        base=tuple(r.shipped() for r in _RELEASES[:20]),
        update=_HISTORY_WAVES[0],
        later=_HISTORY_WAVES[1:],
        base_criteria="Lists Corvid 3.x releases with their dates.",
        listed=tuple(r.version for r in (*_RELEASES, _NEW_RELEASE)),
    ),
    DeltaUpdate(
        question=_q(
            "du-incident-table",
            "aggregation",
            "List every Corvid production incident in one table: id, date, service, duration, region, root "
            "cause and mitigation.",
            "The incident table includes INC-341, INC-342, INC-343, INC-344 and INC-345 alongside the "
            "earlier incidents.",
            "",
        ),
        base=_incident_wave(0, 10),
        update=_incident_wave(10, 20),
        later=(_incident_wave(20, 30), _incident_wave(30, 40), _incident_wave(40, 45)),
        base_criteria="Has a markdown table of Corvid incidents.",
        listed=tuple(i.id for i in _INCIDENTS),
    ),
)

MINIMUM_ACCEPTANCE = ("du-oncall-departed",)


def _waves(case: DeltaUpdate) -> list[list[dict[str, str]]]:
    """The one-wave cases' facts plus the hard corpus, so each page shares its bank
    with plenty that has nothing to do with it — and a growing case's own facts.

    A growing case (one with ``later`` waves) keeps its facts to its own bank. Its
    waves carry dozens of facts, and shared they put every case's update wave over
    the delta gate's memory cap, so the gate stood down and each page was refreshed
    by a search instead: the cases stopped measuring what they were written for.
    """
    noise = split_into_waves(facts())
    shared = [c for c in CASES if not c.later]
    own = [case] if case.later else []
    waves: list[list[dict[str, str]]] = []
    for index, timestamp in enumerate((_BASE_TIME, _UPDATE_TIME)):
        texts = [t for c in (*shared, *own) for t in (c.base if index == 0 else c.update)]
        texts += [f.text for f in noise[index]]
        waves.append([{"content": t, "timestamp": timestamp} for t in texts])
    return waves


async def build(client: Hindsight, bank_id: str, case: DeltaUpdate, settle: SettleFn) -> PageOutcome:
    """Write the page from wave 1, update it from wave 2, and snapshot each."""
    outcome = PageOutcome(question_id=case.question.id, category=case.question.category, bank_id=bank_id)
    await prepare_bank(client, bank_id)
    created = await create_page(client, bank_id, case.question)
    outcome.page_id, outcome.mental_model_id = created.page_id, created.mental_model_id
    await settle(bank_id)

    later = [[{"content": t, "timestamp": _UPDATE_TIME} for t in texts] for texts in case.later]
    for index, wave in enumerate(_waves(case) + later, start=1):
        await client.aretain_batch(bank_id=bank_id, items=wave)
        await settle(bank_id)
        await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=outcome.mental_model_id)
        await settle(bank_id)
        content = await read_page(client, bank_id, outcome.mental_model_id)
        outcome.waves.append(WaveSnapshot(wave=index, facts_ingested=len(wave), content=content))

    outcome.final_content = outcome.waves[-1].content
    return outcome
