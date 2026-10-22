"""Drive consolidation in waves and check the observations it leaves behind.

The scenario, end to end through the public API:

    retain wave 1 (as written) -> wait for consolidation -> snapshot observations
    retain wave 2              -> wait for consolidation -> snapshot observations
    ...
    grade the FINAL observations, and check every snapshot

Facts go in with ``retain_extraction_mode=chunks``: each item is stored as one
fact, verbatim. That isolates the step under test. Real extraction would
paraphrase, split or merge the input, and a wrong observation could then be
extraction's fault, which ``test_03``/``test_04`` already grade. The facts are
written the way extraction writes them (third person, one claim each, entities
attached), so consolidation sees the shape it sees in production.

Two kinds of check:

* **deterministic**, on every snapshot: an observation cites evidence that
  exists, its ``proof_count`` matches that evidence, its dates come from that
  evidence, its text carries no prompt metadata, every durable fact of a wave
  ends up cited, and (where the scenario says so) a wave of restatements adds
  no new observation;
* **judged**, on the final snapshot: a *trap* (the observations assert the
  specific wrong thing) and a *must* (they record the right thing). The trap is
  graded first and is what matters most: a wrong observation is a stored
  falsehood that recall and reflect read back as true.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from hindsight_client import Hindsight
from hindsight_client_api.api.llm_traces_api import LLMTracesApi
from hindsight_client_api.models.memory_unit_list_item import MemoryUnitListItem

from hindsight_system_evals.judge import evaluate
from hindsight_system_evals.pages import SettleFn


@dataclass(frozen=True)
class Fact:
    text: str
    #: When the statement was made: an ISO date (read as noon UTC), or a full ISO
    #: timestamp when the time of day matters. Becomes the fact's ``mentioned_at``,
    #: which is what consolidation is told to read as recency.
    on: str
    entities: tuple[str, ...] = ()
    #: An ephemeral fact (chatter, screen state) may legitimately be dropped, so
    #: it is exempt from the coverage check.
    durable: bool = True
    #: Terms that ONE existing observation must contain, among those the model was
    #: SHOWN when this fact was consolidated: the recall surfaced the fact's twin.
    #: Read from the public LLM trace, so it separates a recall miss from a model miss.
    twin: tuple[str, ...] = ()


@dataclass(frozen=True)
class Wave:
    facts: tuple[Fact, ...]
    tags: tuple[str, ...] = ()
    #: Restatements only: this wave must update existing observations, never add
    #: one. The bug it catches is the near-duplicate sibling (#1566, #3249).
    no_new_observations: bool = False


@dataclass(frozen=True)
class Scenario:
    id: str
    #: Where the scenario comes from: issues, PRs, prompt rules.
    origin: str
    waves: tuple[Wave, ...]
    must: tuple[str, ...] = ()
    traps: tuple[str, ...] = ()
    mission: str | None = None
    observation_scopes: str | None = None
    #: Grade only observations mentioning one of these (case-insensitive). For
    #: the large bank, where the full listing would bury the few that matter.
    focus: tuple[str, ...] = ()


@dataclass
class Observation:
    id: str
    text: str
    proof_count: int
    sources: list[str]
    mentioned_at: str | None
    occurred_start: str | None
    occurred_end: str | None


@dataclass
class ConsolidationCall:
    """One consolidation LLM call from the server's trace: what the model was shown and answered."""

    scope: str
    started_at: str | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    duration_ms: int
    input: Any
    output: Any


@dataclass
class ConsolidationCost:
    """What consolidating the scenario cost, from the server's own LLM trace."""

    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    #: Summed call durations: what the provider was busy for.
    llm_seconds: float = 0.0
    #: Retain to settled, summed over waves: what a caller waits for.
    wall_seconds: float = 0.0
    #: Decision-model triage requests (scope ``decision_consolidation_triage``), priced apart.
    triage_calls: int = 0
    triage_input_tokens: int = 0


@dataclass
class ConsolidationOutcome:
    scenario_id: str
    bank_id: str
    snapshots: list[list[Observation]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    traps_hit: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    cost: ConsolidationCost = field(default_factory=ConsolidationCost)

    @property
    def passed(self) -> bool:
        return not (self.traps_hit or self.missed or self.problems)

    @property
    def final(self) -> list[Observation]:
        return self.snapshots[-1] if self.snapshots else []

    def report(self) -> str:
        lines = [f"{self.scenario_id} — bank {self.bank_id}"]
        lines += [f"  TRAP: {t}" for t in self.traps_hit]
        lines += [f"  MISSING: {m}" for m in self.missed]
        lines += [f"  CHECK: {p}" for p in self.problems]
        c = self.cost
        lines.append(
            f"  cost: {c.calls} calls, {c.input_tokens} in ({c.cached_tokens} cached), {c.output_tokens} out, "
            f"llm {c.llm_seconds:.1f}s, wall {c.wall_seconds:.1f}s"
            + (f", triage {c.triage_calls} calls, {c.triage_input_tokens} in" if c.triage_calls else "")
        )
        for i, snapshot in enumerate(self.snapshots, 1):
            lines.append(f"  observations after wave {i} ({len(snapshot)}):")
            lines += [f"    - {o.text} [proof={o.proof_count}]" for o in snapshot]
        return "\n".join(lines)


#: Scope the server traces consolidation's decision-model triage requests under.
TRIAGE_SCOPE = "decision_consolidation_triage"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
#: Prompt scaffolding that must never reach an observation's text (#2544: leaked
#: JSON and the model's own "(Note: ...)" asides; the prompt's "never copy raw
#: fact lines or their metadata" rule).
_LEAKS = (
    "Involving:",
    "occurred_start",
    "occurred_end",
    "mentioned_at",
    "proof_count",
    "source_fact",
    '"text"',
    "(Note",
)


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


async def _list(client: Hindsight, bank_id: str, kind: str) -> list[MemoryUnitListItem]:
    items: list[MemoryUnitListItem] = []
    offset = 0
    while True:
        page = await client.alist_memories(bank_id, type=kind, limit=500, offset=offset)
        items += page.items
        offset += len(page.items)
        if not page.items or offset >= page.total:
            return items


async def _observations(client: Hindsight, bank_id: str) -> list[Observation]:
    return [
        Observation(
            id=o.id,
            text=o.text or "",
            proof_count=o.proof_count or 0,
            sources=list(o.source_memory_ids or []),
            mentioned_at=o.mentioned_at,
            occurred_start=o.occurred_start,
            occurred_end=o.occurred_end,
        )
        for o in await _list(client, bank_id, "observation")
    ]


def _check_snapshot(
    wave_no: int,
    observations: list[Observation],
    facts_by_id: dict[str, MemoryUnitListItem],
    outcome: ConsolidationOutcome,
) -> None:
    for o in observations:
        where = f"wave {wave_no}, {o.text[:80]!r}"
        if _UUID.search(o.text) or any(leak in o.text for leak in _LEAKS):
            outcome.problems.append(f"{where}: prompt metadata leaked into the text")
        if not o.text.strip(" .\n"):
            outcome.problems.append(f"{where}: empty observation (#2520)")
        if not o.sources:
            outcome.problems.append(f"{where}: no evidence")
            continue
        dangling = [s for s in o.sources if s not in facts_by_id]
        if dangling:
            outcome.problems.append(f"{where}: cites {len(dangling)} fact(s) that do not exist")
            continue
        # proof_count is the evidence count, not 1 for a fresh multi-fact create (#4955).
        if o.proof_count != len(set(o.sources)):
            outcome.problems.append(f"{where}: proof_count {o.proof_count} != {len(set(o.sources))} sources")
        sources = [facts_by_id[s] for s in o.sources]
        # Dates come from the evidence: latest mention, earliest start (#3477), and
        # nothing invented for undated evidence (#501).
        latest = max((_parse(f.mentioned_at) for f in sources if f.mentioned_at), default=None)
        if latest and _parse(o.mentioned_at) != latest:
            outcome.problems.append(f"{where}: mentioned_at {o.mentioned_at} is not its evidence's latest {latest}")
        starts = [_parse(f.occurred_start) for f in sources if f.occurred_start]
        if not starts and o.occurred_start:
            outcome.problems.append(f"{where}: occurred_start {o.occurred_start} invented for undated evidence")
        elif starts and _parse(o.occurred_start) != min(starts):
            outcome.problems.append(f"{where}: occurred_start {o.occurred_start} is not its evidence's earliest")


def _render(observations: list[Observation], focus: tuple[str, ...]) -> str:
    shown = [o for o in observations if not focus or any(f.lower() in o.text.lower() for f in focus)]
    if not shown:
        return "(no observations)"
    return "\n".join(f"- {o.text}" for o in shown)


def _timeline(scenario: Scenario) -> str:
    lines = []
    for i, wave in enumerate(scenario.waves, 1):
        lines += [f"[wave {i}, stated {f.on}] {f.text}" for f in wave.facts if _relevant(f, scenario.focus)]
    return "\n".join(lines)


async def _consolidation_calls(client: Hindsight, bank_id: str) -> list[ConsolidationCall]:
    """Every consolidation call the server traced for the bank, oldest first: what the
    model was shown and what it answered."""
    api = LLMTracesApi(client._api_client)
    calls: list[ConsolidationCall] = []
    offset = 0
    while True:
        page = await api.list_llm_requests(bank_id, limit=500, offset=offset)
        calls += [
            ConsolidationCall(
                scope=c.scope or "",
                started_at=c.started_at,
                input_tokens=c.input_tokens or 0,
                cached_tokens=c.cached_tokens or 0,
                output_tokens=c.output_tokens or 0,
                duration_ms=c.duration_ms or 0,
                input=c.input,
                output=c.output,
            )
            for c in page.items
            # The consolidation LLM calls plus its decision-model triage. Not the
            # embeddings a consolidation run also makes: they are not the spend priced here.
            if (c.scope or "").startswith("consolidation") or c.scope == TRIAGE_SCOPE
        ]
        offset += len(page.items)
        if not page.items or offset >= page.total:
            return sorted(calls, key=lambda c: c.started_at or "")


def _check_twins(
    scenario: Scenario,
    calls: list[ConsolidationCall],
    outcome: ConsolidationOutcome,
    facts: list[MemoryUnitListItem],
) -> None:
    """Every fact with a ``twin`` must have been shown an observation containing it.

    A fact the LLM never saw passes only if it was settled without it — attached as
    evidence to an observation that contains the twin (a server may decide plain
    restatements before the LLM call).
    """
    #: Per traced call: the batch's new-facts section, and one chunk per observation shown.
    shown: list[tuple[str, list[str]]] = []
    for call in calls:
        if call.scope != "consolidation":
            continue
        text = json.dumps(call.input, ensure_ascii=False)
        # The system prompt carries both headings too, as format notes; the last pair is the batch.
        split = text.rfind("### Existing observations")
        # One chunk per observation entry (its text and its evidence).
        shown.append((text[text.rfind("### New facts", 0, split) : split], text[split:].lower().split('"id')))
    for wave in scenario.waves:
        for fact in wave.facts:
            if not fact.twin:
                continue
            seen = [entries for batch, entries in shown if fact.text in batch]
            if not seen:
                ids = {f.id for f in facts if (f.text or "").startswith(fact.text)}
                holders = [o.text.lower() for o in outcome.final if ids.intersection(o.sources)]
                if not any(all(t.lower() in h for t in fact.twin) for h in holders):
                    outcome.problems.append(f"never consolidated: {fact.text[:80]!r}")
            elif not any(all(t.lower() in entry for t in fact.twin) for entries in seen for entry in entries):
                outcome.problems.append(f"recall showed no observation with {fact.twin} for {fact.text[:80]!r}")


def _relevant(fact: Fact, focus: tuple[str, ...]) -> bool:
    return not focus or any(term.lower() in fact.text.lower() for term in focus)


async def run_scenario(client: Hindsight, bank_id: str, settled: SettleFn, scenario: Scenario) -> ConsolidationOutcome:
    outcome = ConsolidationOutcome(scenario_id=scenario.id, bank_id=bank_id)
    await client.aupdate_bank_config(
        bank_id,
        retain_extraction_mode="chunks",
        enable_observations=True,
        enable_auto_consolidation=True,
        **({"observations_mission": scenario.mission} if scenario.mission else {}),
    )

    for wave_no, wave in enumerate(scenario.waves, 1):
        items = []
        for j, fact in enumerate(wave.facts):
            item: dict[str, Any] = {
                "content": fact.text,
                "timestamp": fact.on if "T" in fact.on else f"{fact.on}T12:00:00Z",
                "document_id": f"{scenario.id}-w{wave_no}-{j}",
                "entities": [{"text": e} for e in fact.entities],
            }
            if wave.tags:
                item["tags"] = list(wave.tags)
            if scenario.observation_scopes:
                item["observation_scopes"] = scenario.observation_scopes
            items.append(item)
        # The client sends one request per call; a wave of hundreds is split so a
        # single retain stays an ordinary size.
        started = time.monotonic()
        for start in range(0, len(items), 100):
            await client.aretain_batch(bank_id=bank_id, items=items[start : start + 100])
        await settled(bank_id)
        outcome.cost.wall_seconds += time.monotonic() - started

        facts = await _list(client, bank_id, "world") + await _list(client, bank_id, "experience")
        facts_by_id = {f.id: f for f in facts}
        observations = await _observations(client, bank_id)
        _check_snapshot(wave_no, observations, facts_by_id, outcome)

        # Every durable fact of this wave must land in some observation: a fact the
        # model skipped is stamped consolidated all the same and never retried (#5054).
        cited = {s for o in observations for s in o.sources}
        for j, fact in enumerate(wave.facts):
            if not fact.durable:
                continue
            ids = [f.id for f in facts if f.document_id == f"{scenario.id}-w{wave_no}-{j}"]
            if ids and not cited.intersection(ids):
                outcome.problems.append(f"wave {wave_no}: no observation cites {fact.text[:90]!r}")

        if wave.no_new_observations and outcome.snapshots:
            before = len(outcome.snapshots[-1])
            if len(observations) > before:
                outcome.problems.append(
                    f"wave {wave_no} only restated known facts, yet observations grew {before} -> {len(observations)}"
                )
        outcome.snapshots.append(observations)

    traced = await _consolidation_calls(client, bank_id)
    triage = [c for c in traced if c.scope == TRIAGE_SCOPE]
    calls = [c for c in traced if c.scope != TRIAGE_SCOPE]
    outcome.cost.triage_calls = len(triage)
    outcome.cost.triage_input_tokens = sum(c.input_tokens for c in triage)
    outcome.cost.calls = len(calls)
    outcome.cost.input_tokens = sum(c.input_tokens for c in calls)
    outcome.cost.cached_tokens = sum(c.cached_tokens for c in calls)
    outcome.cost.output_tokens = sum(c.output_tokens for c in calls)
    outcome.cost.llm_seconds = sum(c.duration_ms for c in calls) / 1000
    _check_twins(scenario, calls, outcome, facts)
    if dump_dir := os.getenv("HINDSIGHT_EVAL_CONSOLIDATION_DUMP"):
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"{scenario.id}.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in traced], f, indent=1, ensure_ascii=False, default=str)

    rendered = _render(outcome.final, scenario.focus)
    context = (
        "These are the OBSERVATIONS a memory system consolidated from the facts below. Facts arrived in "
        "waves; 'stated' is when each was said, and a later statement is the more recent truth. Grade the "
        "observations, not the facts.\n\nFacts:\n" + _timeline(scenario)
    )
    for trap in scenario.traps:
        verdict = await evaluate(rendered, trap, context=context, confirm_met=True)
        if verdict.meets_criteria:
            outcome.traps_hit.append(f"{trap} — {verdict.reasoning}")
    for must in scenario.must:
        verdict = await evaluate(rendered, must, context=context)
        if not verdict.meets_criteria:
            outcome.missed.append(f"{must} — {verdict.reasoning}")
    return outcome


#: Confusable on purpose: ``ledger-sync`` and ``ledger-sink`` embed almost the
#: same, so the observation an update needs is one of several near-identical
#: candidates — the shape where the true twin used to be ranked out of the
#: model's view and a duplicate created instead (#1907).
_PREFIXES = ("ledger", "billing", "invoice", "payout", "refund", "settle")
_SUFFIXES = ("api", "worker", "sync", "sink", "gateway")
_TEAMS = ("Atlas", "Nimbus", "Orion", "Vega", "Lyra", "Draco")
_LANGUAGES = ("Go", "Rust", "Kotlin", "Python")
_CHATTER = (
    "The user said they were grabbing a coffee and would be back in five minutes.",
    "The user mentioned the office wifi was slow this morning.",
    "The user said thanks and that the last answer was helpful.",
    "The user's screen showed the Grafana home dashboard.",
    "The user asked to pause for a quick stand-up call.",
)


@dataclass(frozen=True)
class _Service:
    team: str
    language: str
    port: int


@dataclass(frozen=True)
class _Person:
    relation: str
    city: str
    job: str
    birthday: str


def noisy_service_bank() -> Scenario:
    """A platform team's service catalogue: 30 services, 120 facts, plus chatter.

    Wave 1 seeds the catalogue. Wave 2 is what a real bank receives later: two
    changes to services that have a near-twin, and three paraphrased restatements.
    For each, the recall must put the observation it belongs to in front of the
    model (``twin``), the change must land as current (no stale port or owner left
    standing), and the confusable neighbours must not move with it.

    Measured on gemini-3.1-flash-lite: the twin is found every time, and 9 of 9 runs
    pass with the server's defaults. Growth in wave 2 is reported rather than gated —
    whether a paraphrase updates or duplicates is the model decision the dedicated
    restatement scenarios gate.
    """
    rng = random.Random(4715)
    services = [f"{p}-{s}" for p in _PREFIXES for s in _SUFFIXES]
    # The fixed ports below are kept out of the draw: a second service on 8211 makes
    # "billing-api listens on 8211" ambiguous, and the eval would grade a coin flip.
    fixed_ports = {8210, 8211, 8443, 8444}
    ports = rng.sample([p for p in range(8100, 8999) if p not in fixed_ports], len(services))
    spec = {
        name: _Service(rng.choice(_TEAMS), rng.choice(_LANGUAGES), port)
        for name, port in zip(services, ports, strict=True)
    }
    # Fixed so the wave-2 sentences below can name the before and after values.
    spec["ledger-sync"] = _Service("Orion", "Go", 8443)
    spec["ledger-sink"] = _Service("Orion", "Go", 8444)
    spec["billing-gateway"] = _Service("Atlas", "Kotlin", 8210)
    spec["billing-api"] = _Service("Atlas", "Kotlin", 8211)

    seed: list[Fact] = []
    for i, (name, service) in enumerate(spec.items()):
        team, language, port = service.team, service.language, service.port
        day = f"2026-08-{1 + i % 28:02d}"
        seed += [
            Fact(f"The {name} service is owned by team {team}.", day, (name, f"team {team}")),
            Fact(f"The {name} service is written in {language}.", day, (name,)),
            Fact(f"The {name} service listens on port {port}.", day, (name,)),
            Fact(f"Pages for {name} go to the #oncall-{team.lower()} Slack channel.", day, (name, f"team {team}")),
        ]
    seed += [Fact(text, "2026-08-15", durable=False) for text in _CHATTER for _ in range(2)]
    rng.shuffle(seed)

    later = "2026-09-10"
    update = (
        Fact(
            "The ledger-sync service moved from port 8443 to port 9443.",
            later,
            ("ledger-sync",),
            twin=("ledger-sync", "8443"),
        ),
        Fact(
            "Ownership of the billing-gateway service moved from team Atlas to team Nimbus.",
            later,
            ("billing-gateway", "team Nimbus", "team Atlas"),
            twin=("billing-gateway", "atlas"),
        ),
        Fact(
            f"Team {spec['payout-worker'].team} owns the payout-worker service.",
            later,
            ("payout-worker", f"team {spec['payout-worker'].team}"),
            twin=("payout-worker", spec["payout-worker"].team),
        ),
        Fact(
            f"payout-worker is a {spec['payout-worker'].language} service.",
            later,
            ("payout-worker",),
            twin=("payout-worker", spec["payout-worker"].language),
        ),
        Fact(
            f"The invoice-api service accepts traffic on port {spec['invoice-api'].port}.",
            later,
            ("invoice-api",),
            twin=("invoice-api", str(spec["invoice-api"].port)),
        ),
    )
    return Scenario(
        id="noisy-service-catalogue",
        origin="#1907 (twin ranked out of view), #1977/#2000 (update drifts into a duplicate), #1566",
        # Growth is reported, not gated, here: whether a restatement updates rather than
        # duplicates is a model decision, graded strictly by `restatements-merge` and
        # `shared-scope-merges-across-tags`. What scale adds is whether the twin is FOUND
        # among 30 near-identical services — the `twin` check — and whether the
        # neighbours survive the edits (judged below).
        waves=(Wave(tuple(seed)), Wave(update)),
        focus=("ledger-sync", "ledger-sink", "billing-gateway", "billing-api", "payout-worker", "invoice-api"),
        must=(
            "An observation states that ledger-sync currently listens on port 9443.",
            "An observation states that billing-gateway is currently owned by team Nimbus.",
            "ledger-sink is still recorded as listening on port 8444, and billing-api as owned by team Atlas.",
        ),
        traps=(
            "Some observation presents ledger-sync as currently listening on port 8443, without saying it moved.",
            "Some observation presents billing-gateway as currently owned by team Atlas, without saying ownership "
            "moved to Nimbus.",
            "Some observation attributes the port change to ledger-sink (or gives ledger-sink port 9443), or moves "
            "billing-api to team Nimbus.",
        ),
    )


#: Twenty people whose names come in lookalike pairs, the way a real contact list does.
_PEOPLE = (
    ("Maria", "sister"),
    ("Mario", "neighbour"),
    ("Paolo", "cousin"),
    ("Paola", "colleague"),
    ("Lucia", "aunt"),
    ("Luca", "friend"),
    ("Marco", "brother"),
    ("Marta", "colleague"),
    ("Anna", "friend"),
    ("Hanna", "manager"),
    ("Franco", "uncle"),
    ("Franca", "neighbour"),
    ("Giulio", "friend"),
    ("Giulia", "niece"),
    ("Carlo", "dentist"),
    ("Carla", "colleague"),
    ("Sara", "friend"),
    ("Sergio", "cousin"),
    ("Elena", "mentor"),
    ("Ilaria", "friend"),
)
_CITIES = ("Turin", "Milan", "Genoa", "Verona", "Padua", "Trieste", "Parma", "Bergamo")
_JOBS = ("nurse", "teacher", "architect", "pharmacist", "baker", "accountant", "engineer", "translator")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October")
_PERSONAL_CHATTER = (
    "The user said they were heading out for a run.",
    "The user asked for a reminder to water the plants tonight.",
    "The user said good morning.",
    "The user mentioned they were tired after a long day.",
    "The user said thanks for the help with the email.",
)


def noisy_personal_bank() -> Scenario:
    """A personal assistant's memory of the people in a user's life: 80 facts and chatter.

    The second large bank, because a service catalogue is one shape of bank: here every
    name has a lookalike (Maria / Mario, Paolo / Paola) and the facts are about people.
    Wave 2 brings two life changes and two paraphrased restatements; the right person's
    observation must be found and changed, and the lookalike's left alone.
    """
    rng = random.Random(5497)
    spec = {
        name: _Person(relation, rng.choice(_CITIES), rng.choice(_JOBS), f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}")
        for name, relation in _PEOPLE
    }
    # Fixed so wave 2 can name the before and after values, and the lookalikes differ.
    spec["Maria"] = _Person("sister", "Turin", "nurse", "March 3")
    spec["Mario"] = _Person("neighbour", "Milan", "baker", "June 12")
    spec["Paolo"] = _Person("cousin", "Genoa", "teacher", "May 9")
    spec["Paola"] = _Person("colleague", "Verona", "architect", "July 21")

    seed: list[Fact] = []
    for i, (name, person) in enumerate(spec.items()):
        relation, city, job, birthday = person.relation, person.city, person.job, person.birthday
        day = f"2026-07-{1 + i % 28:02d}"
        seed += [
            Fact(f"{name} is the user's {relation}.", day, (name, "user")),
            Fact(f"{name} lives in {city}.", day, (name, city)),
            Fact(f"{name} works as a {job}.", day, (name,)),
            Fact(f"{name}'s birthday is on {birthday}.", day, (name,)),
        ]
    seed += [Fact(text, "2026-07-15", ("user",), durable=False) for text in _PERSONAL_CHATTER for _ in range(2)]
    rng.shuffle(seed)

    later = "2026-09-20"
    update = (
        Fact("Maria moved from Turin to Bologna.", later, ("Maria", "Turin", "Bologna"), twin=("maria", "turin")),
        Fact(
            "Paolo left his job as a teacher and now works as a pilot.",
            later,
            ("Paolo",),
            twin=("paolo", "teacher"),
        ),
        Fact(
            f"Lucia celebrates her birthday on {spec['Lucia'].birthday}.",
            later,
            ("Lucia",),
            twin=("lucia", spec["Lucia"].birthday.split()[0].lower()),
        ),
        Fact(
            f"Franco is based in {spec['Franco'].city}.", later, ("Franco",), twin=("franco", spec["Franco"].city.lower())
        ),
    )
    return Scenario(
        id="noisy-personal-contacts",
        origin="#1907 (finding the twin at scale), rule 9 (lookalike people), #1151 (newer fact supersedes)",
        # Growth is reported, not gated — see noisy_service_bank.
        waves=(Wave(tuple(seed)), Wave(update)),
        focus=("Maria", "Mario", "Paolo", "Paola"),
        must=(
            "An observation states that Maria now lives in Bologna.",
            "An observation states that Paolo now works as a pilot.",
            "Mario is still recorded as living in Milan, and Paola as working as an architect.",
        ),
        traps=(
            "Some observation presents Maria as currently living in Turin, without saying she moved.",
            "Some observation presents Paolo as currently working as a teacher, without saying he changed jobs.",
            "Some observation says Mario moved to Bologna or Paola works as a pilot.",
        ),
    )
