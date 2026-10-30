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
* the assistant's own suggestion, which the user turned down, came back as a rule.
* a rule about the code the request is about was raised as a conflict against an edit to other code.
* a request asked for the very behaviour an earlier rule ruled out, without naming it, and got the rule back.
* a scheduled job was told its own instruction file's rule, ignoring the exception that same file makes.
* a rule about one component was applied to a similar component that shares its words.
* an older memory of a job's rule was cited against the job's own current file.
* the last recorded pin of a dependency was held up as a rule against bumping it.

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
    # A file the request points to, read by the agent before its action: (path, contents).
    job_file: tuple[str, str] | None = None
    # A short message the user sends after the request, before the action ("ok, go on"). It comes after
    # a stretch of routine work, as in a real session: the request is long out of the recent steps.
    follow_up: str | None = None


def _chat(*turns: tuple[str, str]) -> str:
    """A saved conversation, one JSON turn per line, as a live session writes it."""
    return "\n".join(json.dumps({"role": role, "content": content}) for role, content in turns)


NO_NOTE_OK = " No note at all also meets the criteria."

CASES = (
    Case(
        id="a-pin-is-moved",
        history=(
            Memory(
                _chat(
                    ("user", "bump the parser submodule and ship it"),
                    ("assistant", "Bumped vendor/parser to 3a91c0e and merged; the pin is now 3a91c0e."),
                ),
                at="2026-05-10T09:00:00Z",
            ),
        ),
        request="Bump vendor/parser to the latest main and open a PR.",
        action=Action(
            "Bash",
            {"command": "git -C vendor/parser checkout 7be20d4 && git add vendor/parser"},
            "+++ b/vendor/parser\n-Subproject commit 3a91c0e\n+Subproject commit 7be20d4",
        ),
        criteria=(
            "No note says moving the pin from 3a91c0e to 7be20d4 goes against a decision or tells the agent to "
            "keep 3a91c0e: the old pin was the state at the time, and the request is to bump it." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="the-jobs-file-is-newer-than-memory",
        history=(
            Memory(
                _chat(
                    ("user", "for the backup job: investigate the cause only when free space drops 10 GiB or more"),
                    ("assistant", "Set: the backup job looks for a cause only after a 10 GiB drop."),
                ),
                at="2026-02-01T09:00:00Z",
            ),
        ),
        request="You are running the scheduled job 'backup'. Read jobs/backup/JOB.md and do what it says.",
        action=Action("Bash", {"command": "du -sh ~/backups/* | sort -h | tail -5"}, ""),
        criteria=(
            "No note says investigating this 7 GiB drop goes against a 10 GiB rule or tells the agent to skip the "
            "scan: the job's own file, which is current, sets the trigger at 5 GiB." + NO_NOTE_OK
        ),
        expect_note=False,
        job_file=(
            "jobs/backup/JOB.md",
            "# backup\nFind the cause when space drops fast: if free space fell 5 GiB or more since the previous "
            "run, list the five biggest backup folders. This run: free space fell 7 GiB since the previous run.",
        ),
    ),
    Case(
        id="a-rule-for-a-similar-component",
        history=(
            Memory(
                "commit 9a2f1c: search: the result ranker falls back to the backup model on errors, so a failing "
                "ranker never empties the results page.",
                at="2026-03-12T09:00:00Z",
            ),
        ),
        request="The spam classifier should have no fallback model: if it fails, mark the message 'unchecked'.",
        action=Action(
            "Edit",
            {"file_path": "mail/spam_classifier.py"},
            "+++ b/mail/spam_classifier.py\n-    return backup_model.classify(msg)\n+    return Verdict.UNCHECKED",
        ),
        criteria=(
            "No note says removing the classifier's fallback goes against the ranker's fallback rule or tells the "
            "agent to keep a fallback model: the rule is about the search ranker, a different component." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="the-jobs-own-file-makes-an-exception",
        history=(
            Memory(
                _chat(
                    ("user", "the nightly docs job improves at most 2 pages per run, never more"),
                    ("assistant", "Noted: the nightly job improves at most 2 pages per run."),
                ),
                at="2026-04-01T09:00:00Z",
            ),
        ),
        request="You are running the scheduled job 'nightly-docs'. Read jobs/nightly-docs/JOB.md and do what it says.",
        action=Action(
            "Edit",
            {"file_path": "docs/guides/caching.md"},
            "+++ b/docs/guides/caching.md\n+See also the [eviction guide](eviction.md).",
        ),
        criteria=(
            "No note says this run goes over the 2-page limit or tells the agent to drop a page: the job's own "
            "file says added links don't count toward the limit, and the edit only adds a link." + NO_NOTE_OK
        ),
        expect_note=False,
        job_file=(
            "jobs/nightly-docs/JOB.md",
            "# nightly-docs\nImprove at most 2 existing pages. Adding a link from another page to one you improved "
            "does not count as an improvement. This run already improved: docs/guides/eviction.md, docs/guides/ttl.md.",
        ),
    ),
    Case(
        id="a-request-for-what-a-rule-ruled-out",
        history=(
            Memory(
                _chat(
                    ("user", "for now, confirmation dialogs open in a separate window, not inline in the page"),
                    ("assistant", "Done: confirmations open in their own window; the page never shows them inline."),
                ),
                at="2026-02-10T09:00:00Z",
            ),
        ),
        request="Can you show the confirmation dialogs inline in the page? I really need them there, test it end to end.",
        action=Action(
            "Edit",
            {"file_path": "web/confirm.ts"},
            "+++ b/web/confirm.ts\n-openWindow(dialog)\n+page.mount(dialog, { inline: true })",
        ),
        criteria=(
            "No note says showing confirmations inline goes against a rule, or tells the agent to keep them in a "
            "separate window: the request asks for exactly that change. A note that only mentions the earlier "
            "choice as background, without calling the change a conflict, is acceptable." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-rule-about-other-code",
        history=(
            Memory(
                "commit 4e1c2a: auth: refresh the session token 5 minutes before it expires, never on a 401 "
                "retry; decided after the login loop of March.",
                at="2026-03-20T09:00:00Z",
            ),
        ),
        request="the app is stuck in a login loop again, look into it. Also the Save button says 'Save file', make it 'Save'.",
        action=Action(
            "Edit",
            {"file_path": "ui/toolbar.html"},
            '+++ b/ui/toolbar.html\n-<button id="save">Save file</button>\n+<button id="save">Save</button>',
        ),
        criteria=(
            "No note says this edit conflicts with the token refresh rule or tells the agent to change token "
            "refresh: the edit only changes a button label. A note that only mentions the rule as background "
            "for the login loop, without calling the label edit a conflict, is acceptable." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="an-agents-suggestion",
        history=(
            Memory(
                _chat(
                    ("user", "the export button loses its settings after every rebuild, why?"),
                    (
                        "assistant",
                        "Each unsigned build gets a new identity, so saved settings don't carry over. The real fix "
                        "is to sign the app with a fixed certificate of its own.",
                    ),
                    ("user", "no, let's just store the settings in a file next to the app instead"),
                ),
                at="2026-05-02T10:00:00Z",
            ),
        ),
        request="Store the export settings in settings.json next to the app and load them at startup.",
        action=Action(
            "Edit",
            {"file_path": "app/settings.py"},
            "+++ b/app/settings.py\n+SETTINGS = APP_DIR / 'settings.json'\n+def load(): return json.loads(SETTINGS.read_text())",
        ),
        criteria=(
            "No note says the fix must be signing the app with a certificate, or that storing settings in a "
            "file goes against a rule: signing was only the assistant's suggestion, and the user chose the file."
            + NO_NOTE_OK
        ),
        expect_note=False,
    ),
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
            {
                "command": "git rm -rq report/go.mod report/main.go && mkdir -p report/src && touch report/pyproject.toml"
            },
            '+++ b/report/pyproject.toml\n+[project]\n+name = "report"',
        ),
        criteria=(
            "No note says replacing the Go service with Python breaks a rule or contradicts the recorded "
            "decision: the request itself replaced that decision." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-follow-up-after-the-request",
        history=(
            Memory(
                _chat(
                    ("user", "the export endpoint writes CSV with our own writer; keep that writer, no Arrow there"),
                    ("assistant", "Kept the CSV writer in export/writer.py; Arrow stays out of the export endpoint."),
                ),
                at="2026-03-02T09:00:00Z",
            ),
        ),
        request="make the export endpoint stream Arrow record batches like the batch job does, and drop the CSV writer",
        follow_up="ok go on as planned",
        action=Action(
            "Bash",
            {"command": "git rm -q export/writer.py && git commit -qam 'export: stream Arrow record batches'"},
            "",
        ),
        criteria=(
            "No note says moving the export endpoint to Arrow or dropping the CSV writer breaks a rule or "
            "contradicts the recorded decision: the user's request in this session replaced it." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-limit-the-code-still-has",
        history=(
            Memory(
                _chat(
                    ("user", "where are we on the search work?"),
                    (
                        "assistant",
                        "Status:\n- Title search now uses the index.\n- **Body search** still reads the store page by "
                        "page and stops at 10k rows.",
                    ),
                ),
                at="2026-05-04T09:00:00Z",
            ),
        ),
        request=(
            "1. on body search, can we reuse the same trick we did with title search? it's really debugging "
            "stuff 2. yeah clean up the old issues as you suggested"
        ),
        action=Action(
            "Bash",
            {
                "command": "python3 - <<'EOF'\np='search/reads.py'\ns=open(p).read()\n"
                "a='With body search it reports what the scan saw.'\n"
                "b='Body search is answered from the index, not a scan: see _body_search.'\n"
                "assert a in s; s=s.replace(a,b); open(p,'w').write(s)\nEOF"
            },
            "",
        ),
        criteria=(
            "No note says moving body search to the index breaks a rule or contradicts a recorded decision: the "
            "memory only described what the code did then, and the request asks to change it." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-rule-for-another-environment",
        history=(
            Memory(
                _chat(
                    ("user", "turn the result cache off in staging, it hides bugs there"),
                    ("assistant", "Done: staging now runs with `result_cache = false` in envs/staging/app.toml."),
                ),
                at="2026-06-02T09:00:00Z",
            ),
        ),
        request="prod is slow on repeated queries, raise the result cache size there to 2 GB",
        action=Action(
            "Edit",
            {
                "file_path": "envs/prod/app.toml",
                "old_string": "result_cache_mb = 512",
                "new_string": "result_cache_mb = 2048",
            },
            "--- a/envs/prod/app.toml\n+++ b/envs/prod/app.toml\n-result_cache_mb = 512\n+result_cache_mb = 2048",
        ),
        criteria=(
            "No note says the prod change breaks the staging rule or that the cache must be off: that rule was "
            "made for staging only." + NO_NOTE_OK
        ),
        expect_note=False,
    ),
    Case(
        id="a-value-for-another-dataset",
        history=(
            Memory(
                _chat(
                    ("user", "what's our reproducible score on the 100k support-tickets set?"),
                    (
                        "assistant",
                        "On the 100k support-tickets set the reproducible score is 0.677 across 400 queries, "
                        "answered by model-a and judged by judge-b. Keep this as the baseline for that set.",
                    ),
                ),
                at="2026-05-10T09:00:00Z",
            ),
        ),
        request=(
            "run the retrieval eval on the 10k product-reviews set and change the context builder to add chunk "
            "metadata"
        ),
        action=Action(
            "Edit",
            {"file_path": "evals/context.py", "old_string": "parts = [text]", "new_string": "parts = [text, meta]"},
            "--- a/evals/context.py\n+++ b/evals/context.py\n-    parts = [text]\n+    parts = [text, meta]",
        ),
        criteria=(
            "No note presents the support-tickets baseline (0.677, model-a, judge-b) as a rule or target for the "
            "product-reviews work: that value belongs to a different dataset." + NO_NOTE_OK
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
                    (
                        "user",
                        "in the install docs for the Lumen plugin, always pin lumen-plugin@^2.1 and use `lumen add`",
                    ),
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
                    (
                        "user",
                        "write jobs/disk-check/JOB.md: keep at least 60 GB free; prune at most 15 cache folders a run",
                    ),
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
    if case.follow_up:
        # Short remarks: each is a recent step, none long enough to be checked as a plan.
        work = [{"id": f"w{i}", "role": "assistant", "content": f"Looked at part {i}, fine."} for i in range(10)]
        work.append({"id": "f", "role": "user", "content": case.follow_up})
        await client.sessions.append_session_events(
            bank_id, session, SessionAppendRequest.from_dict({"messages": work})
        )
        await _settle_checks(client, bank_id, session, notes, timeout)
    if case.job_file:
        path, text = case.job_file
        read = {
            "id": "r",
            "type": "function",
            "function": {"name": "Read", "arguments": json.dumps({"file_path": path})},
        }
        await client.sessions.append_session_events(
            bank_id,
            session,
            SessionAppendRequest.from_dict(
                {
                    "messages": [
                        {"id": "r1", "role": "assistant", "content": "", "tool_calls": [read]},
                        {"id": "r2", "role": "tool", "tool_call_id": "r", "content": text},
                    ]
                }
            ),
        )
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
    request = case.request + (f"\nThen the user said: {case.follow_up}" if case.follow_up else "")
    verdict = await evaluate(shown, case.criteria, context=f"The agent's request: {request}")
    assert verdict.meets_criteria, f"{case.id}: {verdict.reasoning}\nnotes:\n{shown}"
