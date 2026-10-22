"""A live session's notes must carry the project's decisions, and only them.

A live session checks each request, and each action the agent takes, against the bank's memory, and
may send the agent a note. Every case here is a kind of note seen in real use, rewritten in neutral
terms. Two failure families, and the behaviour each must not cost:

**Local context turned into a project rule** (``expect_note=False``). Something that belonged to one
past session came back as a rule for the project:

* one scheduled run's own choice ("no new article this run") became "never generate articles", and
  told the next run to delete the article its job asked for;
* the folder a past session ran a command in became "use that checkout";
* a past status ("not shipped yet") became a contradiction with the session shipping it;
* a remark about where one task's work lived became "work in that repo";
* the agent's own earlier implementation choice became a rule against the user changing it;
* the user's previous, similar request came back as a reason the new one was wrong;
* a decision the request itself just replaced ("ok let's switch to python then") came back as a rule.

**A heads-up for a request that changes nothing** (``expect_note=False``): a question or an estimate,
a scheduled job told to follow its own instruction file, and a request that was only an issue link,
each got "Before you change ..." anyway.

**A real decision the agent must hear** (``expect_note=True``), so that a check that says nothing
fails: a decided value, a house rule recorded in a commit message, a rule about who bumps versions,
and an amended rule, which must reach the agent in its amended form.

Each case seeds the history, opens a live session with the request, then performs the agent's action
(writing the file, running the command), because the conflict check that action triggers is where the
wrong notes came from. The judge grades the notes against the case's criteria.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.models.session_append_request import SessionAppendRequest
from hindsight_client_api.models.session_message import SessionMessage
from hindsight_client_api.models.session_options import SessionOptions

from hindsight_system_evals import evaluate
from hindsight_system_evals.pages import SettleFn


@dataclass(frozen=True)
class Memory:
    """One document in the bank's history; ``at`` dates it when the order matters."""

    text: str
    at: str | None = None


@dataclass(frozen=True)
class Action:
    """What the agent then does: a tool call, its result, and the change so far."""

    tool: str
    input: dict[str, str]
    diff: str = ""


@dataclass(frozen=True)
class Case:
    id: str
    history: tuple[Memory, ...]
    request: str
    action: Action
    criteria: str
    expect_note: bool


def _chat(*turns: tuple[str, str]) -> str:
    """A saved conversation, one JSON turn per line, as a live session writes it."""
    return "\n".join(json.dumps({"role": role, "content": content}) for role, content in turns)


NO_NOTE_OK = " No note at all also meets the criteria."

CASES = (
    Case(
        id="a-decision-the-request-replaces",
        history=(
            Memory(
                _chat(
                    ("user", "the report service: write it in Go, one binary, serving SQL over the Parquet files"),
                    ("assistant", "Decided: a Go report service, a single binary, SQL over Parquet."),
                ),
                at="2026-04-01T09:00:00Z",
            ),
        ),
        request="would python be simpler here? reads are rare. ok let's switch to python then",
        action=Action(
            "Bash",
            {"command": "git rm -rq report/go.mod report/main.go && mkdir -p report/src && touch report/pyproject.toml"},
            "+++ b/report/pyproject.toml\n+[project]\n+name = \"report\"",
        ),
        criteria=(
            "No note says replacing the Go service with Python breaks a rule or contradicts the recorded "
            "decision: the request itself replaced that decision." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    # ---------------------------------------------------------------- a request that changes nothing
    Case(
        id="a-question-only",
        history=(
            Memory(
                _chat(
                    ("user", "the rate limiter allows 120 requests per minute per key, keep it there, never raise it"),
                    ("assistant", "Set the per-key limit to 120/min in gateway/limits.py."),
                )
            ),
        ),
        request="How much would it cost per month to serve 10x today's traffic on the current gateway? Just estimate.",
        action=Action("Read", {"file_path": "gateway/limits.py"}, ""),
        criteria=(
            "No note warns the agent about changing the gateway or its rate limit (no 'Before you change ...'): "
            "the request only asks for an estimate and changes nothing." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-request-that-is-only-a-link",
        history=(
            Memory(
                _chat(
                    ("user", "in the install docs for the Lumen plugin, always pin lumen-plugin@^2.1 and use `lumen add`"),
                    ("assistant", "Updated docs/plugins/lumen.md: `lumen add`, pinned to ^2.1."),
                )
            ),
        ),
        request="can you add this to the docs as the reporter asked? https://github.com/acme/app/issues/5842",
        action=Action("Read", {"file_path": "docs/configuration.md"}, ""),
        criteria=(
            "No note tells the agent how to document the Lumen plugin or to pin lumen-plugin: nothing in the "
            "request says the Lumen docs are involved." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-job-run-by-its-own-file",
        history=(
            Memory(
                _chat(
                    ("user", "write jobs/disk-check/JOB.md: keep at least 60 GB free; prune at most 15 cache folders a run"),
                    ("assistant", "Wrote JOB.md with the 60 GB floor and the 15-folder limit."),
                )
            ),
        ),
        request="You are running the scheduled job 'disk-check'. Read jobs/disk-check/JOB.md and do what it says.",
        action=Action("Read", {"file_path": "jobs/disk-check/JOB.md"}, ""),
        criteria=(
            "No note repeats the job's own rules to it as a warning (the 60 GB floor or the 15-folder limit as "
            "'Before you change ...'): the job's file already states them." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    # ---------------------------------------------------------------- local context, not a rule
    Case(
        id="one-run-choice",
        history=(
            Memory(
                "Curation log, run 1 of the scheduled job 'site-curator' (2026-03-02): improved the caching "
                "guide with a new benchmark; no new article this run — nothing was good enough. When no topic "
                "qualifies, the job improves the strongest remaining pages instead of generating articles."
            ),
        ),
        request=(
            "You are running the scheduled job 'site-curator'. Its instructions say: pick the ONE best topic and "
            "write a new article when it qualifies. Today's topic qualifies: write content/articles/edge-caching.md."
        ),
        action=Action(
            "Write",
            {"file_path": "content/articles/edge-caching.md"},
            "+++ b/content/articles/edge-caching.md\n+# Edge caching\n+When to cache at the edge.",
        ),
        criteria=(
            "No note tells the agent not to write the new article, to remove it, or that writing new articles "
            "breaks a project rule. A note that only mentions how a past run went, without presenting it as a "
            "rule, is acceptable." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="where-a-session-ran",
        history=(
            Memory(
                _chat(
                    ("user", "commit the docs fix"),
                    ("action", "Bash cd /home/dev/infra-2/services/api && git add -A && git diff --cached --stat"),
                    ("assistant", "Committed the docs fix in services/api."),
                )
            ),
        ),
        request="In /home/dev/infra-5, fix the typo 'recieve' in services/api/README.md.",
        action=Action(
            "Bash",
            {"command": "cd /home/dev/infra-5/services/api && git add -A && git commit -m 'docs: typo'"},
            "+++ b/README.md\n-recieve\n+receive",
        ),
        criteria=(
            "No note tells the agent to use /home/dev/infra-2 (or any other checkout) instead of the one the "
            "request names, or says working in infra-5 goes against a rule." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-past-status",
        history=(
            Memory(
                "2026-03-10: filed issue #812 for the scheduler test scenario 'retry-after-clock-skew'. The "
                "scenario was not shipped yet; it is waiting for someone to add it to tests/test_scheduler.py."
            ),
        ),
        request="Add the 'retry-after-clock-skew' scenario from issue #812 to tests/test_scheduler.py and ship it.",
        action=Action(
            "Edit",
            {"file_path": "tests/test_scheduler.py"},
            '+++ b/tests/test_scheduler.py\n+def test_retry_after_clock_skew():\n+    """Issue #812."""',
        ),
        criteria=(
            "No note says that shipping or adding the scenario contradicts the record that it 'was not shipped', "
            "or tells the agent not to ship it. A note restating what the scenario should test is acceptable."
            + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="where-one-task-lived",
        history=(
            Memory(
                _chat(
                    ("user", "port the invoice exporter to the new schema"),
                    (
                        "assistant",
                        "Note that this migration work lives in the separate billing-tools repo, "
                        "not in this one. Opened the PR there.",
                    ),
                )
            ),
        ),
        request="Add a retry with backoff to the HTTP client in app/net/client.py.",
        action=Action(
            "Edit",
            {"file_path": "app/net/client.py"},
            "+++ b/app/net/client.py\n+    for attempt in range(3):\n+        await asyncio.sleep(2 ** attempt)",
        ),
        criteria=(
            "No note tells the agent to do this work in the billing-tools repo, or says changing app/net/client.py "
            "in this repository goes against a decision." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="the-agents-own-earlier-choice",
        history=(
            Memory(
                _chat(
                    ("user", "add a cache in front of the pricing lookup"),
                    ("assistant", "Done: I added an in-memory cache with a 60-second TTL in pricing/cache.py."),
                )
            ),
        ),
        request="Raise the pricing cache TTL in pricing/cache.py to 5 minutes; prices change rarely.",
        action=Action(
            "Edit",
            {"file_path": "pricing/cache.py"},
            "+++ b/pricing/cache.py\n-TTL_SECONDS = 60\n+TTL_SECONDS = 300",
        ),
        criteria=(
            "No note tells the agent the TTL must stay 60 seconds, or presents the earlier 60-second TTL as a "
            "decision the change goes against: it was an implementation choice, and the user is changing it."
            + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="the-users-previous-request",
        history=(
            Memory(
                _chat(
                    ("user", "embed the seven demo clips from media/demo/ into the landing page"),
                    ("assistant", "Embedded the seven clips in web/landing.html."),
                )
            ),
        ),
        request="Embed the eight demo clips from media/demo/ into the landing page; one new clip shows the login flow.",
        action=Action(
            "Edit",
            {"file_path": "web/landing.html"},
            '+++ b/web/landing.html\n+  <video src="media/demo/08-login.mp4" controls></video>',
        ),
        criteria=(
            "No note says embedding eight clips, or the new login clip, contradicts the earlier request for seven, "
            "and no note just restates the user's own request back as a requirement." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    # ---------------------------------------------------------------- a decision the agent must hear
    Case(
        id="a-decided-value",
        history=(
            Memory(
                _chat(
                    (
                        "user",
                        "Decision for parse_quantity: memory suffixes are binary, K means 1024, never 1000. "
                        "Disk sizes stay decimal.",
                    ),
                    ("assistant", "Understood: K = 1024 for memory, decimal for disk."),
                )
            ),
        ),
        request="Fix parse_quantity in infraconf/units.py so pods get the memory their config asks for.",
        action=Action(
            "Edit",
            {"file_path": "infraconf/units.py"},
            '+++ b/infraconf/units.py\n-    "K": 1000,\n+    "K": 1000,  # kilo',
        ),
        criteria="A note tells the agent that memory suffixes are binary: K means 1024, not 1000.",
        expect_note=True,
    ),
    Case(
        id="a-house-rule-in-a-commit",
        history=(
            Memory(
                "commit 3f2a91c: docs: each key number in an article stands in its own sentence with its source "
                "and date, so readers and crawlers that quote one passage keep the number's provenance"
            ),
        ),
        request="Update content/guides/latency.md with the new p95 and p99 numbers from the March benchmark.",
        action=Action(
            "Edit",
            {"file_path": "content/guides/latency.md"},
            "+++ b/content/guides/latency.md\n+Latency improved: p95 fell to 41 ms and p99 to 88 ms, "
            "while throughput rose 12%.",
        ),
        criteria=("A note tells the agent that each key number goes in its own sentence with its source and date."),
        expect_note=True,
    ),
    Case(
        id="who-bumps-versions",
        history=(
            Memory(
                "CONTRIBUTING.md: version numbers are bumped only by the release script (scripts/release.sh). "
                "Never change a package version by hand in a pull request."
            ),
        ),
        request="Fix the crash in cli/args.py when --config is given twice.",
        action=Action(
            "Edit",
            {"file_path": "pyproject.toml"},
            '+++ b/pyproject.toml\n-version = "1.4.2"\n+version = "1.4.3"',
        ),
        criteria=(
            "A note tells the agent not to change the package version by hand: versions are bumped only by the "
            "release script. Grade the notes, not the agent's edit: a note naming the hand edit as the problem "
            "meets the criteria."
        ),
        expect_note=True,
    ),
    Case(
        id="an-amended-rule",
        history=(
            Memory(
                _chat(
                    ("user", "Overage policy: overage bills in 50-unit blocks rounded up, no free units."),
                    ("assistant", "Noted: 50-unit blocks, rounded up, no free units."),
                ),
                at="2026-03-01T09:00:00Z",
            ),
            Memory(
                _chat(
                    (
                        "user",
                        "Amendment to the overage policy: the first 25 overage units each month are free, "
                        "applied before the block rounding: billable = max(0, overage - 25).",
                    ),
                    ("assistant", "Updated: 25 free units per month, before rounding into 50-unit blocks."),
                ),
                at="2026-03-08T09:00:00Z",
            ),
        ),
        request="Fix overage_charge in billing/overage.py: customers just over their allowance are billed a block.",
        action=Action(
            "Edit",
            {"file_path": "billing/overage.py"},
            "+++ b/billing/overage.py\n+    blocks = -(-overage // 50)\n+    return blocks * plan.block_fee",
        ),
        criteria=(
            "A note tells the agent the current rule: the first 25 overage units each month are free, applied "
            "before rounding into 50-unit blocks. No note says there are no free units as the current rule."
        ),
        expect_note=True,
    ),
)


async def _settle_checks(client: Hindsight, bank_id: str, session: str, notes: list[str], timeout: float) -> None:
    """Poll the session until no check is running, collecting the notes it hands back."""
    deadline = time.monotonic() + timeout
    while True:
        r = await client.sessions.append_session_events(bank_id, session, SessionAppendRequest())
        notes += [n.text for n in r.notes if n.text not in notes]
        if not r.checking:
            return
        assert time.monotonic() < deadline, f"a check never finished ({bank_id}/{session})"
        await asyncio.sleep(1.0)


async def _notes(client: Hindsight, bank_id: str, case: Case, timeout: float = 120.0) -> list[str]:
    """Open a live session with the request as its first prompt, then do the case's action; the notes
    of both checks (the heads-up, and the conflict check of the action)."""
    session = f"eval-{uuid.uuid4().hex[:8]}"
    await client.sessions.append_session_events(
        bank_id,
        session,
        SessionAppendRequest(
            session=SessionOptions(live=True, retain="off"),
            messages=[SessionMessage.from_dict({"id": "1", "role": "user", "content": case.request})],
        ),
    )
    notes: list[str] = []
    await _settle_checks(client, bank_id, session, notes, timeout)
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": case.action.tool, "arguments": json.dumps(case.action.input)},
    }
    await client.sessions.append_session_events(
        bank_id,
        session,
        SessionAppendRequest.from_dict(
            {
                "messages": [
                    {"id": "2", "role": "assistant", "content": "", "tool_calls": [call]},
                    {"id": "3", "role": "tool", "tool_call_id": "c", "content": "ok"},
                ],
                "state": {"diff": case.action.diff} if case.action.diff else None,
            }
        ),
    )
    await _settle_checks(client, bank_id, session, notes, timeout)
    return notes


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_a_note_carries_the_projects_decisions_and_only_them(
    client: Hindsight, bank_id: str, settled: SettleFn, case: Case
) -> None:
    # One memory at a time, in order: an amendment must be retained after the rule it amends.
    for memory in case.history:
        item = {"content": memory.text, "context": "developer conversation"}
        if memory.at:
            item["timestamp"] = memory.at
        await client.aretain_batch(bank_id=bank_id, items=[item])
        await settled(bank_id)
    notes = await _notes(client, bank_id, case)
    if case.expect_note:
        assert notes, f"{case.id}: no note, though memory records a decision the work must respect"
    shown = "\n".join(f"- {n}" for n in notes) or "(no note)"
    verdict = await evaluate(shown, case.criteria, context=f"The agent's request: {case.request}")
    assert verdict.meets_criteria, f"{case.id}: {verdict.reasoning}\nnotes:\n{shown}"
