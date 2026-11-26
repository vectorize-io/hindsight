"""Hermes Agent with the Hindsight plugin, end to end: does it remember the user?

The plugin's own tests (``hindsight-integrations/hermes/tests``) run against a
stubbed Hermes and check the calls the plugin makes. This drives the real thing —
a real Hermes and its model, the plugin from this checkout, every setting on its
default, and a real Hindsight server — and grades what Hermes tells the user.

Each scenario:

1. earlier sessions, where the user mentions things in passing;
2. the gateway stops (the plugin flushes what it still holds) and the bank settles
   (fact extraction, consolidation; a page is created and refreshed if the scenario
   has one);
3. a **fresh Hermes profile on the same bank** — a user on a new machine. Hermes'
   own memory files and its past-chat search start empty, so only Hindsight can
   carry anything across;
4. one question, judged twice: the trap first (the answer ignores or contradicts
   what the user said), then whether it uses the memory.

Needs ``hermes`` on PATH with a working provider login (its ``auth.json`` is
copied into the sandbox), Docker running, and the usual server and judge
credentials. Skips without them. Run it on its own:

    uv run pytest evals/hermes

See ``hindsight_system_evals/hermes.py`` for the sandbox, and the README for the
first-run cost.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import Target, evaluate
from hindsight_system_evals.hermes import (
    MODEL,
    PLUGIN_OVERRIDES,
    PROVIDER,
    HermesSandbox,
    Turn,
    build_sandbox,
    builtin_memory,
    keep_logs,
    missing_prerequisite,
    run_session,
    stop_gateway,
)
from hindsight_system_evals.pages import SettleFn, read_page

log = logging.getLogger(__name__)

_MISSING = missing_prerequisite()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=_MISSING)

#: How long the plugin's retain may take to reach the bank after the gateway stopped.
_FIRST_FACT_TIMEOUT_SECONDS = 180
_PAGE_TOOLS = frozenset({"hindsight_search_pages", "hindsight_read_page", "hindsight_list_pages"})


@dataclass(frozen=True)
class PageSeed:
    """A page created on the bank once the earlier sessions have settled, as a user would."""

    name: str
    source_query: str


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    #: Earlier sessions, each a list of user turns. Each one is a separate Hermes session.
    sessions: list[list[str]]
    #: The first message on the fresh profile; answering it well needs the earlier sessions.
    question: str
    #: What a good answer does.
    criteria: str
    #: The failure that matters: the answer behaves as if the earlier sessions never happened.
    trap: str
    page: PageSeed | None = None
    #: The question turn must call at least one of these. Empty: any route to the answer counts.
    expect_tools: frozenset[str] = frozenset()
    #: Hermes' built-in memory must have been written, and copied into the bank.
    mirror: bool = False


SCENARIOS = [
    Scenario(
        id="remembers-me-next-time",
        category="cross_session",
        sessions=[
            [
                "hey! planning a dinner party next saturday, about 8 people. any ideas for a main course? "
                "fyi I'm vegetarian, have been for ten years",
                "sounds good. one thing: my partner Tamsin has a serious nut allergy, so nothing with nuts at all",
                "perfect. it's kind of a goodbye party actually, we're moving from Berlin to Lisbon next week",
                "thanks, that's all for now!",
            ]
        ],
        question="we just landed in our new city 🎉 what could I cook for the two of us tonight?",
        criteria=(
            "The answer suggests vegetarian food (no meat or fish) that is free of nuts, or explicitly says it "
            "avoids nuts for the partner, AND it shows it knows the user's situation from before: it mentions "
            "Lisbon or Portugal, Tamsin, or the nut allergy."
        ),
        trap=(
            "The answer suggests a dish containing meat, fish or nuts, OR asks the user where they are or what "
            "they can eat as if it knew nothing about them."
        ),
    ),
    # What the user said is spread over two sessions, and the question needs both.
    Scenario(
        id="remembers-across-two-sessions",
        category="cross_session",
        sessions=[
            [
                "ugh, long day. our team finally dropped the ORM, we write plain SQL now with sqlx. "
                "anyway, what's a good way to unwind after work?",
                "ok thanks, I'll try that",
            ],
            [
                "quick q: how do I make my terminal prompt show the git branch? I use zsh",
                "cool. btw our service is in Rust, on Postgres 16, in case it matters later",
            ],
        ],
        question="I need to add a table of user invites to our service and look one up by email. how would I write it?",
        criteria=(
            "The answer writes the code in Rust against Postgres using sqlx with plain SQL (a CREATE TABLE and "
            "a SELECT ... WHERE email = ... query), not an ORM."
        ),
        trap=(
            "The answer uses an ORM (Diesel, SeaORM, Prisma, SQLAlchemy, ...) or a language other than Rust for "
            "the service code, OR asks which language, database or library the user's service uses."
        ),
    ),
    # #6901: Hermes is told to store a correction when the user says a memory is wrong.
    Scenario(
        id="a-correction-wins",
        category="update",
        sessions=[
            [
                "my sister Anna lives in Turin, I'm visiting her for Christmas. any tips for Turin in winter?",
                "thanks!",
            ],
            [
                "hey, what do you remember about my sister?",
                "no, that's out of date: she moved to Milan last month, she doesn't live in Turin anymore",
                "yep, that's it",
            ],
        ],
        question="planning my christmas trip to see my sister. what should we do while I'm there?",
        criteria="The answer plans the trip in Milan, where the sister lives now.",
        trap="The answer plans the trip in Turin or says the sister lives in Turin.",
    ),
    # #6886: recalled lines carry their dates, so the newer of two states can win.
    Scenario(
        id="a-new-job-supersedes-the-old",
        category="update",
        sessions=[
            [
                "I'm a data engineer at Acme Corp, been there 4 years. can you suggest a good book on data modeling?",
                "thanks",
            ],
            [
                "big news: I quit Acme! I start at Globex next monday as a machine learning engineer",
                "yeah, a bit nervous but excited. thanks!",
            ],
        ],
        question="write me a one-line LinkedIn headline for my current role",
        criteria="The headline presents the user as a machine learning engineer at Globex.",
        trap=(
            "The headline presents Acme or the data engineer job as the user's current role, OR the answer asks "
            "where the user works as if it did not know."
        ),
    ),
    # #6897: page tools are on by default; a question about what was decided should reach the page.
    Scenario(
        id="renovation-decisions-from-the-page",
        category="pages",
        sessions=[
            [
                "we're renovating the flat. we decided on oak floors instead of tiles, tiles are too slippery for "
                "our dog Pepe",
                "the budget is capped at 40k euros total, no exceptions",
                "the contractor is Marco Rossi, he starts next month",
                "ok thanks",
            ]
        ],
        page=PageSeed(
            name="Flat renovation",
            source_query="What has been decided about the flat renovation: flooring, budget, contractor, and why?",
        ),
        question="where did we land on the flat renovation? summarize what's decided",
        criteria=(
            "The answer says oak floors (not tiles, because of the dog), a 40k euro budget cap, and Marco Rossi as "
            "the contractor."
        ),
        trap=(
            "The answer says tiles were chosen, gives a different budget or contractor, or says it knows nothing "
            "about the renovation."
        ),
        expect_tools=_PAGE_TOOLS,
    ),
    Scenario(
        id="training-plan-from-the-page",
        category="pages",
        sessions=[
            [
                "I'm training for a marathon in about 12 weeks, aiming for under 3:30",
                "my left knee flares up if I run more than 3 days a week, so 3 runs max",
                "long runs have to be on sundays, I work saturdays",
                "cool, thanks",
            ]
        ],
        page=PageSeed(
            name="Marathon training",
            source_query="What are the goals and constraints of the user's marathon training?",
        ),
        question="what does my running week look like? lay out this week's plan",
        criteria=(
            "The plan has at most 3 runs, the long run on Sunday, no run on Saturday, and it mentions the sub-3:30 "
            "goal or the knee."
        ),
        trap="The plan has 4 or more runs, a run on Saturday, or the long run on a day other than Sunday.",
        expect_tools=_PAGE_TOOLS,
    ),
    # #6889: with built-in memory on (Hermes' default), what Hermes writes to MEMORY.md / USER.md is also
    # retained, so a profile without those files still knows it.
    Scenario(
        id="an-allergy-told-to-remember",
        category="builtin_memory",
        sessions=[
            [
                "add this to your notes about me, your user profile: I'm allergic to penicillin, bad reaction as a kid",
                "thanks",
            ]
        ],
        # Amoxicillin, not "antibiotics": penicillin is the usual first choice, so a generic "mention any
        # allergies, especially penicillin" passed the first version without any memory at all.
        question="my doctor wants to put me on amoxicillin for a sinus infection tomorrow. any reason not to?",
        criteria=(
            "The answer warns that amoxicillin is a penicillin and the user is allergic to penicillin, stated as a "
            "known fact about this user."
        ),
        trap=(
            "The answer does not say that this user is allergic to penicillin. A generic reminder to mention any "
            "allergies the user might have does not count as saying it."
        ),
        mirror=True,
    ),
    Scenario(
        id="a-birthday-told-to-remember",
        category="builtin_memory",
        sessions=[
            [
                "put this in your notes about me, your user profile: my daughter Elif's birthday is June 3 and "
                "she's obsessed with dinosaurs",
                "thanks!",
            ]
        ],
        question="help me pick a birthday present for my daughter, and when do I need it by?",
        criteria="The answer suggests something dinosaur-themed and says it is needed by June 3.",
        trap=(
            "The answer asks the daughter's interests or birthday as if it did not know them, without mentioning "
            "dinosaurs or June 3."
        ),
        mirror=True,
    ),
]


async def _wait_for_first_fact(client: Hindsight, bank_id: str, log_path: Path) -> None:
    """The bank has something from the plugin, or the plugin never wrote at all."""
    deadline = time.monotonic() + _FIRST_FACT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if (await client.alist_memories(bank_id, limit=1)).total:
            return
        await asyncio.sleep(2)
    raise AssertionError(
        f"bank {bank_id} is still empty {_FIRST_FACT_TIMEOUT_SECONDS}s after the sessions ended: the plugin "
        f"never retained. Check that Hermes loaded it: {log_path}"
    )


async def _build_page(client: Hindsight, bank_id: str, seed: PageSeed, settled: SettleFn) -> str:
    """Create the page with its default trigger and refresh it now, rather than on the next consolidation."""
    created = await client.pages.create_page(bank_id, {"name": seed.name, "source_query": seed.source_query})
    await settled(bank_id)
    await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=created.mental_model_id)
    await settled(bank_id)
    content = await read_page(client, bank_id, created.mental_model_id)
    assert content.strip(), f"page {seed.name!r} is still empty after its refresh — bank {bank_id}"
    return content


def _transcript(turns: list[Turn]) -> str:
    return "\n".join(f"user: {t.prompt}\nhermes [{', '.join(t.tools) or 'no tools'}]: {t.reply}" for t in turns)


def _readable(turns: list[Turn]) -> str:
    """The conversation as a person reads it, with each tool call and its arguments."""
    lines: list[str] = []
    for t in turns:
        lines.append(f"user: {t.prompt}")
        lines += [f"  -> {call.name} {call.input}" for call in t.calls]
        lines.append(f"hermes: {t.reply or '(no reply: ' + t.error + ')'}\n")
    return "\n".join(lines)


def _note(path: Path, title: str, body: str) -> None:
    """Add a section to the scenario's transcript, as it happens: a failed run keeps what came before."""
    with path.open("a", encoding="utf-8") as f:
        f.write(f"===== {title} =====\n{body.strip() or '(empty)'}\n\n")


def _sessions(sandbox: HermesSandbox, scenario: Scenario, log_path: Path, transcript: Path) -> list[Turn]:
    before: list[Turn] = []
    for n, prompts in enumerate(scenario.sessions, 1):
        turns = run_session(sandbox, prompts, log_path=log_path)
        before += turns
        _note(transcript, f"Session {n} of {len(scenario.sessions)}", _readable(turns))
        failed = [t for t in turns if t.error]
        assert not failed, f"Hermes failed a turn: {failed[0].error}\nlog: {log_path}"
    return before


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"{s.category}/{s.id}")
async def test_hermes(
    target: Target, client: Hindsight, bank_id: str, settled: SettleFn, scenario: Scenario, tmp_path: Path
) -> None:
    def sandbox() -> HermesSandbox:
        return build_sandbox(api_url=target.url, api_key=target.api_key, bank_id=bank_id)

    log_path = tmp_path / "hermes-sessions.log"
    transcript = tmp_path / "transcript.txt"
    log.info(
        "hermes %s/%s, plugin %s, bank %s, transcript %s",
        PROVIDER,
        MODEL,
        PLUGIN_OVERRIDES or "defaults",
        bank_id,
        transcript,
    )
    earlier = sandbox()
    try:
        before = _sessions(earlier, scenario, log_path, transcript)
        builtin = builtin_memory(earlier)
        _note(transcript, "Hermes' own memory files (MEMORY.md, USER.md) after those sessions", builtin)
    finally:
        # Stopped before the bank is read: the plugin's shutdown sends what its writer still holds.
        stop_gateway(earlier.home)
        keep_logs(earlier, tmp_path / "hermes-logs-earlier")

    await _wait_for_first_fact(client, bank_id, log_path)
    await settled(bank_id)
    held = await client.alist_memories(bank_id, limit=200)
    _note(
        transcript,
        f"What the Hindsight bank {bank_id} held before the question ({held.total} memories)",
        "\n".join(f"[{m.fact_type}] {m.text}" + (f"  tags={m.tags}" if m.tags else "") for m in held.items),
    )
    # In every scenario, not only the mirror ones: Hermes writes its built-in memory when it sees fit
    # (a new job was patched into USER.md unasked), and each such write must reach the bank.
    if builtin:
        mirrored = await client.alist_memories(bank_id, tags=["builtin-memory"], limit=10)
        assert mirrored.total, f"Hermes wrote built-in memory but none of it reached the bank:\n{builtin}"
    elif scenario.mirror:
        raise AssertionError(
            "Hermes never wrote its built-in memory, so there was nothing to mirror. Told to remember "
            "something, it reaches for hindsight_retain; the scenario has to ask for its own notes."
        )
    page = await _build_page(client, bank_id, scenario.page, settled) if scenario.page else ""
    if scenario.page:
        _note(transcript, f"Page {scenario.page.name!r}", page)

    fresh = sandbox()
    try:
        [answer] = run_session(fresh, [scenario.question], log_path=log_path)
    finally:
        stop_gateway(fresh.home)
        keep_logs(fresh, tmp_path / "hermes-logs-fresh")
    _note(transcript, "A new device (empty Hermes profile, same bank): the question", _readable([answer]))
    assert not answer.error, f"Hermes failed the question: {answer.error}\nlog: {log_path}"

    context = (
        f"Earlier sessions with the assistant:\n{_transcript(before)}\n\n"
        f"Then, on a new device, the user asked: {scenario.question}"
    )
    evidence = (
        f"bank {bank_id}, tools used: {answer.tools or 'none'}, transcript {transcript}\nanswer: {answer.reply[:600]}"
    )
    log.info("%s: tools=%s builtin_memory=%r answer=%s", scenario.id, answer.tools, builtin[:200], answer.reply[:300])
    if page:
        log.info("%s: page=%s", scenario.id, page[:300])

    trap = await evaluate(answer.reply, scenario.trap, context=context, confirm_met=True)
    _note(transcript, f"Judge: did it fall in the trap? {'YES' if trap.meets_criteria else 'no'}", trap.reasoning)
    assert not trap.meets_criteria, f"{scenario.id}: Hermes forgot the user — {trap.reasoning}\n{evidence}"
    verdict = await evaluate(answer.reply, scenario.criteria, context=context)
    _note(
        transcript, f"Judge: did it meet the criteria? {'yes' if verdict.meets_criteria else 'NO'}", verdict.reasoning
    )
    assert verdict.meets_criteria, f"{scenario.id}: {verdict.reasoning}\n{evidence}"
    if scenario.expect_tools:
        assert scenario.expect_tools & set(answer.tools), (
            f"{scenario.id}: answered without any of {sorted(scenario.expect_tools)}\n{evidence}"
        )
