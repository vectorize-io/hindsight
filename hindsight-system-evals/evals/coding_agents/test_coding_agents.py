"""The coding-agents plugin end to end, on a real Claude Code: does the agent use its memory?

The plugin's own tests (``hindsight-integrations/coding-agents``) stub the host and check
the calls the plugin makes. The search-rate harness (``hindsight_system_evals.coding_agents``)
counts how often a turn searches. This grades what the agent actually *does* with memory,
the way ``evals/hermes`` does for Hermes:

* **Memory scenarios.** Earlier sessions on machine A, where the user says things in passing.
  The bank settles. Then a **new machine** (a fresh HOME, a fresh clone of the repo, the same
  bank) gets one request. The fresh clone holds none of machine A's edits, so only the bank
  can carry anything across. The reply and the diff are judged twice: the trap first, then
  the criteria.
* **Initiatives.** Whether the agent records a feature as one tracked page, keeps that page
  on the current plan, and leaves bug fixes alone. Page counts are asserted exactly; what the
  page says is judged.

Claude Code is the harness. Scenarios only touch it through ``_machine`` (a sandbox with the
plugin installed) and ``_session`` (prompts in, turns with their tool calls out). Another agent,
Codex for example, is another pair of those two.

Needs ``claude`` on PATH with a login (macOS: it is copied out of the Keychain), node with
``npm install`` done in the plugin, and the usual server and judge credentials. Skips without
them. Run it on its own:

    uv run pytest evals/coding_agents
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.models.page_node import PageNode

from hindsight_system_evals import Target, evaluate
from hindsight_system_evals.coding_agents.sandbox import (
    Sandbox,
    build_plugin,
    build_sandbox,
    missing_prerequisite,
    refresh_credentials,
)
from hindsight_system_evals.coding_agents.scenario import write_fixture_repo
from hindsight_system_evals.coding_agents.session import ToolCall, Turn, run_session
from hindsight_system_evals.pages import SettleFn, read_page

log = logging.getLogger(__name__)

_MISSING = missing_prerequisite()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=_MISSING)

#: The Claude Code model under test. Any `claude --model` value.
MODEL = os.getenv("HINDSIGHT_EVAL_CODING_MODEL", "opus")
#: How long the plugin's per-turn saves may take to reach the bank after a session.
_FIRST_FACT_TIMEOUT_SECONDS = 180
CAPTURE = "hindsight_capture_initiative"


@pytest.fixture(scope="module", autouse=True)
def _plugin_built() -> None:
    """The sandbox installs `dist/` of this checkout, so it must hold the code under test."""
    build_plugin()


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    #: Earlier sessions on machine A, each a list of user turns.
    sessions: list[list[str]]
    #: The one request on the new machine; doing it well needs the earlier sessions.
    request: str
    #: What a good reply and diff do.
    criteria: str
    #: The failure that matters: the work goes ahead as if the earlier sessions never happened.
    trap: str


SCENARIOS = [
    # A decision the user states in chat and nothing in the code or history records.
    Scenario(
        id="a-decision-told-in-passing",
        category="cross_session",
        sessions=[
            [
                "fyi, a cart line can never be more than 20 units. anything bigger is wholesale and goes "
                "through the B2B system, not this service - product decided that last week. for now just add a "
                "test that quote prices two lines of the same sku correctly",
                "thanks, that's it",
            ]
        ],
        request="a customer wants to order 50 of SKU-1 in one go. make post_quote support that",
        criteria=(
            "The work does not simply let a 50-unit line through: it rejects or redirects lines over 20 units "
            "(or stops and points out that orders that big are wholesale and go through the B2B system), "
            "citing the 20-unit limit."
        ),
        trap=(
            "The change makes a 50-unit line quote normally, or the reply never mentions the 20-unit limit "
            "or the B2B/wholesale system."
        ),
    ),
    # What the user said is spread over two sessions, and the request needs both.
    # Red until #7012: the first-prompt reflect answers the request's own words, and the rules for errors
    # never come up although the bank holds both.
    Scenario(
        id="two-rules-from-two-sessions",
        category="cross_session",
        sessions=[
            [
                "team rule, keep it in mind: every error response we add carries a machine-readable `code` field "
                "next to `error`, the mobile client switches on it. nothing to change right now",
                "ok",
            ],
            [
                "another rule from review: every new error has to be listed in the README, in a section for "
                "errors. no change needed now either",
                "thanks",
            ],
        ],
        request="post_reserve should reject a quantity of 0 with a clear error. do it",
        criteria=(
            "The new error response for a zero quantity has a machine-readable `code` field next to `error`, "
            "AND the README lists the new error."
        ),
        trap=(
            "The new error response has no `code` field, or the README is not updated with the new error, or the "
            "reply asks how errors should look as if it did not know the team's rules."
        ),
    ),
    # A correction in a later session must beat the original statement.
    Scenario(
        id="a-correction-wins",
        category="update",
        sessions=[
            [
                "context for later: we ship every order from our Rotterdam warehouse. no code change needed now, "
                "just keep it in mind",
                "ok",
            ],
            [
                "correction on what I told you about shipping: the Rotterdam warehouse closed last month, all "
                "fulfilment moved to Antwerp. Rotterdam is gone",
                "yep",
            ],
        ],
        request="add a ships_from field to the quote response naming the warehouse the order ships from",
        criteria="The ships_from value names Antwerp ('Antwerp', 'Antwerp, Belgium' and the like all count).",
        trap="The ships_from value is Rotterdam, or the reply asks which warehouse to use as if it did not know.",
    ),
    # A newer state, said in passing, supersedes the older one.
    Scenario(
        id="a-newer-address-supersedes",
        category="update",
        sessions=[
            [
                "for later: customers reach us at help@orderbook.test, that's our support inbox. no change now",
                "ok",
            ],
            [
                "fyi support moved to the Shopdesk tool last week, the new address is support@shopdesk.test and "
                "the old help@ inbox is closed",
                "cool",
            ],
        ],
        request="add a support_contact field to every error response so customers know where to write",
        criteria="The support_contact value is support@shopdesk.test.",
        trap=(
            "The support_contact value is help@orderbook.test, or the reply asks which address to use as if it did "
            "not know."
        ),
    ),
]


def _machine(root: Path, target: Target, bank_id: str) -> Sandbox:
    """A computer with the plugin installed and a fresh clone of the repo, on the shared bank."""
    # Resolved: on macOS /tmp is a symlink, and the plugin's MCP server exits silently when its
    # own path does not match (see coding_agents/run.py).
    root = root.resolve()
    sandbox = build_sandbox(
        root,
        api_url=target.url,
        bank_id=bank_id,
        workdir=root / "repo",
        config_overrides={"apiToken": target.api_key} if target.api_key else None,
    )
    write_fixture_repo(sandbox.workdir)
    return sandbox


def _session(sandbox: Sandbox, prompts: list[str], transcript: Path, title: str) -> list[Turn]:
    refresh_credentials(sandbox)
    n = len(list(sandbox.root.glob("session-*.jsonl"))) + 1
    result = run_session(sandbox, prompts, model=MODEL, log_path=sandbox.root / f"session-{n}.jsonl")
    _note(transcript, title, _readable(result.turns))
    assert not result.error, f"Claude Code failed a turn: {result.error}\nlog: {sandbox.root}/session-{n}.jsonl"
    # Without this, "the agent did not capture" also passes when the plugin's tools never loaded.
    assert CAPTURE in result.offered, f"the session was never offered the plugin's tools: {sandbox.mcp_config}"
    return result.turns


def _readable(turns: list[Turn]) -> str:
    lines: list[str] = []
    for t in turns:
        lines.append(f"user: {t.prompt}")
        lines += [f"  -> {call.name} {call.input}" for call in t.calls]
        lines.append(f"agent: {t.reply}\n")
    return "\n".join(lines)


def _note(path: Path, title: str, body: str) -> None:
    """Add a section to the scenario's transcript as it happens, so a failed run keeps what came before."""
    with path.open("a", encoding="utf-8") as f:
        f.write(f"===== {title} =====\n{body.strip() or '(empty)'}\n\n")


def _changes(workdir: Path) -> str:
    """What the agent changed in the repo, new files included."""
    subprocess.run(["git", "add", "-A", "--intent-to-add"], cwd=workdir, check=True, capture_output=True)
    return subprocess.run(["git", "diff"], cwd=workdir, check=True, capture_output=True, text=True).stdout


async def _wait_for_first_fact(client: Hindsight, bank_id: str, sandbox: Sandbox) -> None:
    deadline = time.monotonic() + _FIRST_FACT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if (await client.alist_memories(bank_id, limit=1)).total:
            return
        await asyncio.sleep(2)
    raise AssertionError(
        f"bank {bank_id} is still empty {_FIRST_FACT_TIMEOUT_SECONDS}s after the sessions ended: the plugin "
        f"never retained. Its log: {sandbox.plugin_log}"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"{s.category}/{s.id}")
async def test_remembers_on_a_new_machine(
    target: Target, client: Hindsight, bank_id: str, settled: SettleFn, scenario: Scenario, tmp_path: Path
) -> None:
    transcript = tmp_path / "transcript.txt"
    log.info("coding-agents %s, model %s, bank %s, transcript %s", scenario.id, MODEL, bank_id, transcript)
    earlier = _machine(tmp_path / "machine-a", target, bank_id)
    before: list[Turn] = []
    for n, prompts in enumerate(scenario.sessions, 1):
        before += _session(earlier, prompts, transcript, f"Machine A, session {n} of {len(scenario.sessions)}")

    await _wait_for_first_fact(client, bank_id, earlier)
    await settled(bank_id)
    held = await client.alist_memories(bank_id, limit=200)
    _note(
        transcript,
        f"What the bank {bank_id} held before the request ({held.total} memories)",
        "\n".join(f"[{m.fact_type}] {m.text}" for m in held.items),
    )

    fresh = _machine(tmp_path / "machine-b", target, bank_id)
    [answer] = _session(fresh, [scenario.request], transcript, "Machine B (fresh HOME and clone, same bank)")
    diff = _changes(fresh.workdir)
    _note(transcript, "Machine B: the diff", diff)

    response = f"Reply:\n{answer.reply}\n\nDiff of the repository:\n{diff or '(no changes)'}"
    context = (
        "Earlier sessions with the coding agent, on another machine:\n"
        + "\n".join(f"user: {t.prompt}\nagent: {t.reply}" for t in before)
        + f"\n\nThen, on a new machine with a fresh clone of the repository, the user asked: {scenario.request}"
    )
    evidence = f"bank {bank_id}, tools used: {answer.tools or 'none'}, transcript {transcript}"
    trap = await evaluate(response, scenario.trap, context=context, confirm_met=True)
    _note(transcript, f"Judge: did it fall in the trap? {'YES' if trap.meets_criteria else 'no'}", trap.reasoning)
    assert not trap.meets_criteria, f"{scenario.id}: the agent forgot — {trap.reasoning}\n{evidence}"
    verdict = await evaluate(response, scenario.criteria, context=context)
    _note(
        transcript, f"Judge: did it meet the criteria? {'yes' if verdict.meets_criteria else 'NO'}", verdict.reasoning
    )
    assert verdict.meets_criteria, f"{scenario.id}: {verdict.reasoning}\n{evidence}"


def _initiatives_folder(roots: list[PageNode]) -> PageNode | None:
    return next((n for n in roots if n.kind == "folder" and n.name.lower() == "initiatives"), None)


def _pages(nodes: list[PageNode]) -> list[PageNode]:
    """Every page under ``nodes``, at any depth."""
    out: list[PageNode] = []
    for n in nodes:
        if n.kind == "page":
            out.append(n)
        out += _pages(n.children or [])
    return out


async def _initiative_pages(client: Hindsight, bank_id: str) -> list[PageNode]:
    folder = _initiatives_folder((await client.aget_page_tree(bank_id)).roots)
    return _pages(folder.children or []) if folder else []


def _captures(turns: list[Turn]) -> list[ToolCall]:
    return [call for t in turns for call in t.calls if call.name == CAPTURE]


async def test_initiative_stays_one_page_through_a_plan_change(
    target: Target, client: Hindsight, bank_id: str, settled: SettleFn, tmp_path: Path
) -> None:
    """A feature is captured once, recaptured when its plan changes, and finished from another machine.

    Ends with the page moved into a subfolder and the agent on a machine that never saw the page id:
    the recapture that made a second page in #6919.
    """
    transcript = tmp_path / "transcript.txt"
    log.info("coding-agents initiative, model %s, bank %s, transcript %s", MODEL, bank_id, transcript)
    machine_a = _machine(tmp_path / "machine-a", target, bank_id)

    planned = _session(
        machine_a,
        [
            "I want gift cards at checkout: the customer passes a code with the quote, and the card's remaining "
            "balance comes off the total. a card can be used across several orders until it's empty. plan it "
            "first, don't write code yet",
            "plan approved, build it",
        ],
        transcript,
        "Machine A, session 1: the plan",
    )
    assert _captures(planned), f"approving a new feature did not capture an initiative — {transcript}"
    pages = await _initiative_pages(client, bank_id)
    assert len(pages) == 1, f"expected one initiative page, got {[p.name for p in pages]} — {transcript}"
    page = pages[0]

    changed = _session(
        machine_a,
        [
            "change of plan on the gift cards: no more balances. they're single-use vouchers now, a fixed amount, "
            "and the code is burned on first use even if the order costs less. update it"
        ],
        transcript,
        "Machine A, session 2: the plan changes",
    )
    assert _captures(changed), f"a changed plan was not recaptured — {transcript}"
    pages = await _initiative_pages(client, bank_id)
    assert [p.id for p in pages] == [page.id], f"the recapture made another page: {[p.name for p in pages]}"

    # Filed by hand into a subfolder, where the server's unique name per folder no longer stops a duplicate.
    initiatives = _initiatives_folder((await client.aget_page_tree(bank_id)).roots)
    assert initiatives is not None
    payments = await client.acreate_page_folder(bank_id, "Payments", parent_id=initiatives.id)
    await client.aupdate_page_node(bank_id, page.id, parent_id=payments.id)

    machine_b = _machine(tmp_path / "machine-b", target, bank_id)
    finished = _session(
        machine_b,
        ["gift cards shipped to prod today. record on the gift cards initiative that it's done and released"],
        transcript,
        "Machine B: the initiative is finished",
    )
    assert _captures(finished), f"finishing the initiative was not recorded — {transcript}"
    pages = await _initiative_pages(client, bank_id)
    assert [p.id for p in pages] == [page.id], f"recording it made another page: {[p.name for p in pages]}"

    await settled(bank_id)
    assert page.mental_model_id, f"page {page.id} has no mental model"
    await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=page.mental_model_id)
    await settled(bank_id)
    content = await read_page(client, bank_id, page.mental_model_id)
    _note(transcript, f"The initiative page {page.name!r}", content)

    context = "A coding agent tracked a gift-card feature on one page while the user changed its plan, then shipped it."
    trap = await evaluate(
        content,
        "The page presents a balance that carries across several orders as the current design.",
        context=context,
        confirm_met=True,
    )
    _note(transcript, f"Judge: stale plan? {'YES' if trap.meets_criteria else 'no'}", trap.reasoning)
    assert not trap.meets_criteria, f"the page still shows the dropped plan — {trap.reasoning}\n{transcript}"
    verdict = await evaluate(
        content,
        "The page describes gift cards as single-use, fixed-value vouchers burned on first use, and says the "
        "feature is done or released.",
        context=context,
    )
    _note(transcript, f"Judge: current plan? {'yes' if verdict.meets_criteria else 'NO'}", verdict.reasoning)
    assert verdict.meets_criteria, f"{verdict.reasoning}\n{transcript}"


async def test_small_work_is_not_an_initiative(target: Target, client: Hindsight, bank_id: str, tmp_path: Path) -> None:
    """A bug fix and a rename are work, not initiatives: no capture, no page."""
    transcript = tmp_path / "transcript.txt"
    machine = _machine(tmp_path / "machine-a", target, bank_id)
    turns = _session(
        machine,
        [
            "quote computes tax on the unrounded subtotal, so it can be off by a cent. round the subtotal first",
            "also rename the `lines` param of quote to `cart_lines`",
        ],
        transcript,
        "A bug fix and a rename",
    )
    assert not _captures(turns), f"small work was captured as an initiative: {_captures(turns)} — {transcript}"
    pages = await _initiative_pages(client, bank_id)
    assert not pages, f"small work made initiative pages: {[p.name for p in pages]} — {transcript}"
