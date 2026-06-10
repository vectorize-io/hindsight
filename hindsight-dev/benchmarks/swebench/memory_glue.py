"""Hindsight memory glue for the SWE-bench study.

One :class:`MemoryGlue` instance owns a single Hindsight bank that persists across a
*consecutive* sequence of tasks on one repository. Before each task the agent recalls
durable codebase knowledge; after each task we distil the trajectory into a few durable
facts and retain them. The control arm uses ``enabled=False`` — zero recall/retain calls,
so the only difference between arms is the memory content.

Talks to the Hindsight Cloud dev instance over the HTTP API via the local
``hindsight_client`` SDK. The API token is read from the environment only
(``HINDSIGHT_API_TOKEN``) — never hard-coded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import litellm
from hindsight_client import Hindsight


@dataclass
class MemoryOpStats:
    """Accumulated cost of the memory layer itself (so total-cost numbers stay honest)."""

    recall_calls: int = 0
    retain_calls: int = 0
    reflect_calls: int = 0
    recall_seconds: float = 0.0
    retain_seconds: float = 0.0
    reflect_seconds: float = 0.0
    recall_hits: int = 0
    # Tokens spent by the *summariser* LLM that distils trajectories before retain (client-side,
    # runs on a cheap model).
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    # Tokens Hindsight spends server-side extracting facts during retain (cheap model).
    retain_input_tokens: int = 0
    retain_output_tokens: int = 0
    # Tokens reported by Hindsight's reflect synthesis (server-side LLM, cheap model).
    # NOTE: recall uses ZERO LLM tokens — it is pure retrieval (embeddings/BM25/rerank).
    # NOTE: consolidation runs async server-side and is NOT captured here (a known blind spot).
    reflect_input_tokens: int = 0
    reflect_output_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "recall_calls": self.recall_calls,
            "retain_calls": self.retain_calls,
            "reflect_calls": self.reflect_calls,
            "recall_seconds": round(self.recall_seconds, 3),
            "retain_seconds": round(self.retain_seconds, 3),
            "reflect_seconds": round(self.reflect_seconds, 3),
            "recall_hits": self.recall_hits,
            "summary_input_tokens": self.summary_input_tokens,
            "summary_output_tokens": self.summary_output_tokens,
            "retain_input_tokens": self.retain_input_tokens,
            "retain_output_tokens": self.retain_output_tokens,
            "reflect_input_tokens": self.reflect_input_tokens,
            "reflect_output_tokens": self.reflect_output_tokens,
            # Total Hindsight-side LLM tokens (write path + reflect; recall=0, consolidation N/A).
            "hindsight_llm_tokens": (
                self.summary_input_tokens
                + self.summary_output_tokens
                + self.retain_input_tokens
                + self.retain_output_tokens
                + self.reflect_input_tokens
                + self.reflect_output_tokens
            ),
        }


# Distil a trajectory into REUSABLE DEBUGGING INSIGHT — the transferable engineering lessons a
# senior dev carries from one bug to the next in the same area. Two failure modes to avoid:
# (1) a generic codebase MAP ("X is in file.py") is worthless — the agent re-derives it with a
# grep in seconds; (2) the VERBATIM fix to THIS issue mis-leads a different issue and is
# leak-adjacent. The valuable middle layer is the PATTERN: the class of bug, the root-cause
# mechanism (why this kind of thing breaks), how to diagnose it, and the area a fix of this kind
# belongs — generalisable to SIBLING bugs, not this exact one.
_SUMMARY_SYSTEM = (
    "You are a senior engineer writing a short postmortem note after fixing ONE bug, so that a "
    "teammate solving a DIFFERENT but related bug in the same subsystem is faster and avoids the "
    "same traps. You'll see a transcript of an agent solving one issue. Write 4-8 REUSABLE "
    "lessons — transferable insight, not a map and not this exact patch.\n\n"
    "For each lesson, prefer this kind of content (highest value first):\n"
    "- Root-cause MECHANISM: why this CLASS of bug happens here (e.g. 'ordering state on a "
    "subquery leaks into combined/UNION queries because it isn't reset before combination'). "
    "State the principle, applicable to other bugs of the same class.\n"
    "- DIAGNOSIS approach that worked: what to inspect/print to localise this kind of bug "
    "(e.g. 'print str(queryset.query) to see the generated SQL; set a breakpoint in "
    "Compiler.as_sql to see where a clause is dropped'), and the exact test command that "
    "exercises this area (`python tests/runtests.py <label>`).\n"
    "- GOTCHAS / non-obvious invariants in this subsystem that wasted time or are easy to break "
    "(edge cases, ordering of operations, interactions between features).\n"
    "- WHERE a fix of this kind belongs: the method/abstraction responsible for this behaviour, "
    "so a related fix starts in the right place — without prescribing the literal change.\n\n"
    "Rules: write each lesson as a GENERAL principle about how this subsystem behaves, not as "
    "'the fix for this issue was…'. Do NOT include the verbatim diff/patch. Be concrete (name "
    "the class/method/symptom). One lesson per line, plain text, no numbering, no preamble. "
    "If nothing reusable was learned, output the single word NONE."
)

_SUMMARY_RULES = (
    "Rules: one lesson per line, plain text, no numbering, no preamble, no markdown. Never copy "
    "harness commands, submission markers, or tool-call syntax from the transcript into a lesson. "
    "Be concrete (name the tool, file, method, or symptom). If nothing reusable was learned, "
    "output the single word NONE."
)

# Procedural retention, resolved task: the fix PASSED the repo's test suite, so both the process
# that produced it and the subsystem insight behind it are trustworthy. Process lessons come
# first — trajectory analysis showed the failures that memory could realistically flip were
# caused by execution discipline (blind edits, no verification, empty diffs), not by missing
# domain facts the agent couldn't re-derive from the issue text.
_SUMMARY_SYSTEM_PROCEDURAL_RESOLVED = (
    "You are a senior engineer writing a short postmortem after a bug fix that was VERIFIED "
    "CORRECT by the repository's test suite. Your audience is an AI engineer who will work on "
    "DIFFERENT issues in this same repository and environment. You'll see the transcript of the "
    "successful attempt. Write 4-8 short, reusable lessons, in two groups — working practices "
    "first, they transfer to EVERY future task:\n\n"
    "WORKING PRACTICES (how to operate in this environment — highest value):\n"
    "- Environment facts: which editing/inspection tools exist or are missing in this shell, "
    "safe ways to edit files, output-truncation pitfalls, exact test invocations that work "
    "(e.g. `python tests/runtests.py <label>`).\n"
    "- Verification habits that paid off or would have caught mistakes earlier: re-reading the "
    "whole function after an edit, importing the edited module to catch syntax errors, running "
    "the relevant test module before submitting, checking the final diff is non-empty and "
    "minimal.\n"
    "- Process: anything that wasted steps and how to avoid it next time.\n\n"
    "SUBSYSTEM INSIGHT (verified by the passing tests):\n"
    "- The root-cause MECHANISM for this class of bug, where fixes of this kind belong, and how "
    "to diagnose this area — as general principles, not the verbatim patch.\n\n" + _SUMMARY_RULES
)

# Procedural retention, failed task: the attempt FAILED the test suite, so its theory of the bug
# must NOT be stored as knowledge — storing a failed approach as confident expertise is how the
# bank gets poisoned (observed: task 1's file-corrupting approach was recalled as fact on every
# later task). Failures still teach process: traps, missing tools, and skipped verification are
# real, transferable, and safe to store.
#
# SCOPING (the lesson-shape law, learned from the retry replications): the SAME failure distilled
# as a scoped fact ("deep cloning combined_queries in Query.clone() failed to resolve the union
# DatabaseError") steered a later task TO the winning clone()-based fix, while distilled as an
# over-generalized directive ("avoid modifying Query.clone() for state issues; trace compiler.py
# instead") it fenced off exactly where the next fix lived and the task failed. A failure proves
# only that ONE application of a mechanism at ONE location didn't pass ONE set of tests — never
# that the mechanism is wrong, and never where the real cause lives.
_SUMMARY_SYSTEM_PROCEDURAL_FAILED = (
    "You are a senior engineer writing a short postmortem after a bug-fix attempt that FAILED "
    "the repository's test suite. The attempt's theory of the bug may be wrong — do NOT record "
    "any claim about the root cause or the correct fix as if it were true. Your audience is an "
    "AI engineer who will work on DIFFERENT issues in this same repository and environment. "
    "You'll see the transcript of the failed attempt. Write 3-8 short lessons:\n\n"
    "PROCESS TRAPS (highest value — write as imperative warnings about PROCESS only):\n"
    "- Verification that was skipped and would have caught the failure: re-read the whole "
    "function after an edit, import the edited module to catch syntax errors, write a small "
    "reproduction script, run the relevant test module before submitting, check the final diff "
    "is non-empty and contains only the intended change.\n"
    "- Editing mistakes made: e.g. line-number-based edits applied to wrong ranges, edits that "
    "corrupted code, rewrites whose match patterns silently matched nothing.\n"
    "- Environment facts learned: tools that are missing in this shell, commands that work, "
    "output-truncation pitfalls, exact test invocations.\n\n"
    "FAILED APPROACH (1-2 lines, explicitly tagged, STRICTLY SCOPED): a failure proves only "
    "that one specific change, at one specific place, did not pass specific tests — nothing "
    "more. Each line MUST bind the mechanism to the exact method/file where it was applied and "
    "name the tests it failed: 'FAILED APPROACH: <change> applied in <method, file> did not "
    "pass <test names>.' The same mechanism may be the CORRECT fix at a different location — "
    "do not bias the reader against it.\n\n"
    "FORBIDDEN GENERALIZATIONS (these poison future work — never write them):\n"
    "- Prescriptive directives extrapolated from this one failure: 'avoid X', 'never X', "
    "'use Y instead of X', 'prefer/prioritize Y', 'X is insufficient/wrong'.\n"
    "- Mechanism-level condemnation: claiming an API, method, or technique 'does not work' or "
    "'fails' in general when only one application of it at one location failed.\n"
    "- Unsupported redirection: 'the issue lies in <other place>' or 'the real cause is <Z>' "
    "unless the test evidence directly demonstrates it. Locating the cause is the NEXT "
    "engineer's job; your job is to report only what was tried, where, and what the tests "
    "said.\n\n"
    "OFFICIAL TEST EVIDENCE (only if an '=== OFFICIAL TEST EVALUATION RESULT ===' section is "
    "present): unlike the attempt's own theory, this section IS ground truth from the test "
    "harness. Record it concretely — which required tests still fail and their exact failure "
    "mode (test name + assertion/exception, observed vs expected), and any previously-passing "
    "tests this patch broke. These are the highest-value lessons for a re-attempt: someone "
    "retrying this exact issue must know precisely what the tests expect and what the last "
    "patch got wrong.\n\n" + _SUMMARY_RULES
)

# Placebo notes: the same GENRE as real recalled memories (engineering postmortem lessons,
# same "- lesson | Involving:" formatting) but with ZERO content relevant to these tasks —
# no Python, no Django/ORM/SQL, no testing/editing/verification advice (generic "run tests"
# advice IS treatment content). Domains: infra, frontend, docs, ops, release engineering.
# Purpose: distinguish content-driven steering from any-text perturbation steering at temp 0.
_PLACEBO_NOTES = [
    "Rotating log files daily keeps disk usage predictable on long-lived services. | Involving: assistant",
    "Prefer exponential backoff with jitter when retrying HTTP requests against rate-limited APIs.",
    "Stale assets after a deploy usually point to CDN cache invalidation; version asset filenames to bypass it.",
    "CSS specificity conflicts are easier to trace in the computed-styles panel than in the source. | Involving: assistant",
    "Office Wi-Fi captive portals break VPN auto-reconnect; wired docks avoid the morning login dance.",
    "A Kubernetes readiness probe gates traffic; a liveness probe restarts the pod — confusing the two causes flapping.",
    "README quickstarts go stale fastest; pin them to a released version, not the main branch.",
    "Terraform state locking prevents two applies from corrupting state; always back the lock with a real store.",
    "Long-lived WebSocket connections need keepalive pings or intermediate proxies will drop them silently. | Involving: assistant",
    "Order Dockerfile COPY of source after dependency installation so layer caching survives code-only changes.",
    "Extracting i18n strings early avoids retrofitting string concatenation that can't be translated.",
    "Email deliverability depends on SPF, DKIM, and DMARC alignment more than message content.",
    "Feature flags need a removal date at creation time or they accumulate as permanent dead config.",
    "WCAG contrast ratios: 4.5:1 for body text, 3:1 for large text — check both themes.",
    "At-least-once delivery queues require idempotent consumers; dedupe on a message key, not timing. | Involving: assistant",
    "Automate TLS certificate renewal; calendar reminders fail exactly once, which is enough.",
    "CSV exports for spreadsheet users need a UTF-8 BOM or accented characters render as mojibake.",
    "Touch targets below 44px cause mis-taps on mobile regardless of how precise the design looks.",
    "Sampled analytics undercount rare events; disable sampling before debugging a conversion funnel.",
    "Webhook receivers should return 2xx immediately and process async; slow handlers get retried as duplicates.",
    "Blue-green deploys need session draining or users lose in-flight work at cutover. | Involving: assistant",
    "Monorepo CI should build only affected packages; full rebuilds hide which change broke what.",
    "Pagination cursors beat offsets for large collections; offsets skip or duplicate rows under concurrent writes.",
    "Browser autofill ignores styled divs; real input elements with autocomplete attributes are required.",
    "Service dashboards should show saturation, not just averages — p99 latency hides behind a calm mean.",
    "Changelog entries written at release time are guesses; write them in the PR while context is fresh.",
    "A favicon 404 on every page view is harmless but pollutes error-rate metrics; ship a real one.",
    "Graceful shutdown means stop accepting work, finish in-flight work, then exit — most services skip step two.",
]


class PlaceboGlue:
    """Drop-in stand-in for :class:`MemoryGlue` that injects task-IRRELEVANT notes of matched
    length instead of real memories, and retains nothing.

    The placebo arm answers: do the treatment arm's resolve flips come from the memory
    CONTENT, or would any same-sized block of system-prompt text deterministically perturb
    the temp-0 agent onto different (sometimes successful) paths? Lengths are matched per
    (seq, attempt) to a real run's injected blocks via ``lengths``.
    """

    def __init__(self, lengths: dict[tuple[int, int], int] | None = None) -> None:
        self.stats = MemoryOpStats()
        self.last_retained_summary: str | None = None
        self.lengths = lengths or {}
        self.current_key: tuple[int, int] = (1, 1)  # set by the orchestrator per attempt

    def reset_bank(self) -> None:
        pass

    def retain_after_task(self, *args, **kwargs) -> None:
        pass

    def context_for_task(self, problem_statement: str) -> str:
        return self._block()

    def context_for_step(self, problem_statement: str, recent_context: str) -> str:
        return self._block()

    def _target_len(self) -> int:
        if self.current_key in self.lengths:
            return self.lengths[self.current_key]
        fallback = (self.current_key[0], 1)  # attempt-1 length of the same task
        if fallback in self.lengths:
            return self.lengths[fallback]
        vals = sorted(self.lengths.values())
        return vals[len(vals) // 2] if vals else 2500

    def _block(self) -> str:
        target = self._target_len()
        if target <= 0:
            return ""
        lines: list[str] = []
        i = 0
        while len("\n".join(lines)) < target:
            note = _PLACEBO_NOTES[i % len(_PLACEBO_NOTES)]
            if i >= len(_PLACEBO_NOTES):
                note = f"{note} This held across {2 + i // len(_PLACEBO_NOTES)} separate incidents."
            lines.append(f"- {note}")
            i += 1
        block = "\n".join(lines)
        self.stats.recall_hits += len(lines)
        return block


class ReplayGlue:
    """Drop-in stand-in for :class:`MemoryGlue` that re-injects RECORDED recall blocks
    verbatim (from a prior run's memory_debug.json) and retains nothing.

    Purpose: the pipeline is near-deterministic for identical prompts, so replaying the
    exact block that preceded a flip isolates the final variable — if the flip reproduces,
    it is fully content-determined and all run-to-run variance lives upstream in bank
    generation; if it doesn't, residual trajectory chance exists even given the content.
    Blocks are keyed by (instance_id, attempt); a missing attempt falls back to attempt 1.
    """

    def __init__(self, blocks: dict[tuple[str, int], str]) -> None:
        self.stats = MemoryOpStats()
        self.last_retained_summary: str | None = None
        self.blocks = blocks
        self.current_key: tuple[str, int] = ("", 1)  # set by the orchestrator per attempt

    def reset_bank(self) -> None:
        pass

    def retain_after_task(self, *args, **kwargs) -> None:
        pass

    def context_for_task(self, problem_statement: str) -> str:
        return self._block()

    def context_for_step(self, problem_statement: str, recent_context: str) -> str:
        return self._block()

    def _block(self) -> str:
        block = self.blocks.get(self.current_key) or self.blocks.get((self.current_key[0], 1)) or ""
        if block:
            self.stats.recall_hits += 1
        return block


# Lines containing these markers are transcript artifacts, not lessons — the summariser
# occasionally copies harness commands verbatim (observed in pilot memory_debug dumps).
_ARTIFACT_MARKERS = (
    "COMPLETE_TASK_AND_SUBMIT",
    "MSWEA_",
    "[tool]",
    "<bash>",
    "</bash>",
    "```",
    "cat patch.txt",
)


class MemoryGlue:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        bank_id: str,
        enabled: bool,
        repo: str,
        summary_model: str,
        context_mode: str = "recall",
        recall_max_tokens: int = 1024,
        recall_budget: str = "low",
        recall_types: list[str] | None = None,
        include_chunks: bool = False,
        max_chunk_tokens: int = 4096,
        orientation_enabled: bool = True,
        orientation_query: str | None = None,
        retain_style: str = "insight",
        summary_max_chars: int = 24000,
    ) -> None:
        self.enabled = enabled
        self.bank_id = bank_id
        self.repo = repo
        self.summary_model = summary_model
        # How the agent gets memory context per task:
        #   "recall"  — retrieval only; raw matching facts are pasted in (more, rawer).
        #   "reflect" — retrieval + Hindsight's server-side LLM synthesis; a focused answer to
        #               the task is pasted in (less, distilled, higher signal-to-noise).
        self.context_mode = context_mode
        self.recall_max_tokens = recall_max_tokens
        self.recall_budget = recall_budget
        # An always-on broad recall for facts useful to ANY task (repo layout, how to run
        # tests, conventions). Without it, recall only hits when a later task happens to share
        # a subsystem with an earlier one — sparse at small N. Runs alongside the task-specific
        # recall and results are merged/deduped.
        self.orientation_enabled = orientation_enabled
        self.orientation_query = orientation_query or (
            f"Overview of the {repo} codebase and how to work in it: repository layout and where "
            f"the key modules and subsystems live, how to run the test suite (and a single test), "
            f"build/setup and development conventions, working practices and verification habits "
            f"(how to edit files safely, how to validate a patch before submitting), environment "
            f"gotchas (which tools are available or missing), and common pitfalls and process "
            f"traps to avoid."
        )
        # Fact types for recall/reflect context. None = all types (world/experience/observation).
        # ["observation"] uses ONLY Hindsight's consolidated+deduped layer — cleaner, less
        # redundant context. Consolidation is fast (~12s on dev) and tasks are minutes apart, so
        # the only knowledge an observations-only run can miss is the immediately-preceding task's
        # (older tasks are always consolidated by the time they're recalled).
        self.recall_types = recall_types
        # When True, recall also returns the RAW chunks each fact was distilled from, and we
        # inject the fact paired with its source chunk. For coding, the raw chunk preserves exact
        # paths/signatures the distilled fact may lose; the chunk text is also deterministic (the
        # summariser is temp-0), unlike the LLM-extracted fact.
        self.include_chunks = include_chunks
        self.max_chunk_tokens = max_chunk_tokens
        # What retain distils from a trajectory:
        #   "insight"    — reusable debugging insight (root-cause patterns, diagnosis, gotchas),
        #                  outcome-blind (the original behaviour).
        #   "procedural" — outcome-aware: resolved tasks store working practices + test-verified
        #                  subsystem insight; failed tasks store ONLY process traps/environment
        #                  facts plus the attempted approach explicitly tagged FAILED.
        self.retain_style = retain_style
        self.summary_max_chars = summary_max_chars
        self.stats = MemoryOpStats()
        self.last_retained_summary: str | None = None  # for content analysis / debug dumps
        self._client = Hindsight(base_url=base_url, api_key=api_token) if enabled else None

    # -- bank lifecycle ----------------------------------------------------------------

    def reset_bank(self) -> None:
        """Delete + recreate the bank for a clean cold start. No-op when disabled.

        Retries transient network errors so a brief blip doesn't crash a multi-hour run.
        """
        if not self.enabled or self._client is None:
            return
        try:
            self._client.delete_bank(self.bank_id)
        except Exception:
            pass  # 404 on first run is fine
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                self._create_bank()
                return
            except Exception as e:  # transient connectivity → back off and retry
                last_exc = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"reset_bank failed after retries for {self.bank_id}: {last_exc}")

    def _create_bank(self) -> None:
        if self.retain_style == "procedural":
            mission = (
                "Capture reusable engineering lessons: working practices, verification habits, "
                "environment gotchas (available tools, safe editing, test invocations), and — "
                "only when verified by passing tests — subsystem insight. Never store an "
                "unverified or failed fix as established fact."
            )
        else:
            mission = (
                "Capture durable, reusable facts about the codebase (file locations, how to "
                "run tests, conventions, pitfalls). Do not capture issue-specific fixes."
            )
        self._client.create_bank(
            self.bank_id,
            background=(
                f"Durable engineering knowledge about the {self.repo} codebase, accumulated "
                "by an AI software engineer solving issues over time. Used to navigate and "
                "fix new issues in the same repo faster."
            ),
            retain_mission=mission,
        )

    # -- recall (before a task) --------------------------------------------------------

    # The per-task query is derived ENTIRELY from the task's own problem statement (plus the
    # neutral repo name) — never a hand-authored "how to fix this" prompt, which would be
    # cheating. recall and reflect use the identical query so the only variable is the mode.
    def _task_query(self, problem_statement: str) -> str:
        return f"Working in the {self.repo} codebase. {problem_statement}"

    def context_for_task(self, problem_statement: str) -> str:
        """Return the memory block to inject (or "" if disabled). Dispatches on context_mode."""
        if not self.enabled or self._client is None:
            return ""
        if self.context_mode == "reflect":
            return self._reflect_context(problem_statement)
        return self._recall_context(problem_statement)

    def _reflect_context(self, problem_statement: str) -> str:
        """Ask Hindsight to reflect on the task using its memories; inject the synthesis."""
        t0 = time.time()
        try:
            resp = self._client.reflect(
                self.bank_id,
                query=self._task_query(problem_statement),
                budget=self.recall_budget,
                max_tokens=self.recall_max_tokens,
                fact_types=self.recall_types,  # e.g. ["observation"] to use only the consolidated layer
            )
        except Exception:
            return ""
        finally:
            self.stats.reflect_seconds += time.time() - t0
            self.stats.reflect_calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.reflect_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.stats.reflect_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            self.stats.recall_hits += 1
        return text

    def _recall_context(self, problem_statement: str) -> str:
        """Return a formatted notes block to inject, or "" if no hits.

        Merges a broad "orientation" recall (facts useful to any task) with a task-specific
        recall (the problem statement), deduped. How much memory each recall returns is
        governed by Hindsight's ``max_tokens``/``budget`` (Hindsight has no top-K) — we don't
        impose a client-side cap, so the agent gets everything that fit the token budget.
        """
        queries: list[str] = []
        if self.orientation_enabled:
            queries.append(self.orientation_query)
        queries.append(self._task_query(problem_statement))
        return self._recall_for_queries(queries)

    def context_for_step(self, problem_statement: str, recent_context: str) -> str:
        """Adaptive per-step recall: blend what the agent is CURRENTLY looking at (the latest
        observation) with the task, so the refreshed memory tracks the agent's current focus.
        Returns a formatted notes block (no <codebase_memory> wrapper — caller wraps it)."""
        queries: list[str] = []
        if recent_context.strip():
            queries.append(recent_context[-1800:])  # current focus first
        queries.append(self._task_query(problem_statement))
        return self._recall_for_queries(queries)

    def _recall_for_queries(self, queries: list[str]) -> str:
        """Run a list of recall queries, dedupe results, return a formatted notes block."""
        if self.include_chunks:
            return self._recall_context_with_chunks(queries)

        seen: set[str] = set()
        lines: list[str] = []
        for query in queries:
            for text in self._recall_texts(query):
                key = text.lower()
                if not text or key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {text}")
        self.stats.recall_hits += len(lines)
        return "\n".join(lines)

    def _recall_context_with_chunks(self, queries: list[str]) -> str:
        """Recall facts AND the raw chunk each was distilled from, grouped by chunk.

        Linkage: a world/experience result carries ``chunk_id``; an observation carries
        ``source_fact_ids`` → those source facts carry ``chunk_id``. Both resolve into the
        ``chunks`` map. We group facts under their shared source chunk so the agent sees the
        exact raw detail and what was learned from it.
        """
        chunk_to_facts: dict[str, list[str]] = {}
        chunk_order: list[str] = []
        unchunked: list[str] = []  # facts with no resolvable source chunk
        seen_facts: set[str] = set()

        for query in queries:
            resp = self._recall_results(query)
            if resp is None:
                continue
            chunks = getattr(resp, "chunks", None) or {}
            chunk_text = {cid: (getattr(cd, "text", None) or "").strip() for cid, cd in chunks.items()}
            sfacts = getattr(resp, "source_facts", None) or {}
            sf_chunk = {fid: getattr(sf, "chunk_id", None) for fid, sf in sfacts.items()}

            for r in getattr(resp, "results", None) or []:
                ftext = (getattr(r, "text", None) or "").strip()
                if not ftext or ftext.lower() in seen_facts:
                    continue
                seen_facts.add(ftext.lower())
                cids: list[str] = []
                if getattr(r, "chunk_id", None):
                    cids.append(r.chunk_id)
                sfids = getattr(r, "source_fact_ids", None)
                if isinstance(sfids, str):
                    import ast

                    try:
                        sfids = ast.literal_eval(sfids)
                    except Exception:
                        sfids = []
                for fid in sfids or []:
                    if sf_chunk.get(fid):
                        cids.append(sf_chunk[fid])
                ctexts = [chunk_text[c] for c in cids if chunk_text.get(c)]
                if not ctexts:
                    unchunked.append(ftext)
                    continue
                for ct in ctexts:
                    if ct not in chunk_to_facts:
                        chunk_to_facts[ct] = []
                        chunk_order.append(ct)
                    if ftext not in chunk_to_facts[ct]:
                        chunk_to_facts[ct].append(ftext)

        if not chunk_to_facts and not unchunked:
            return ""
        self.stats.recall_hits += len(seen_facts)

        out = [
            "Each item below is a durable note paired with the RAW DETAIL it was distilled from. "
            "Trust the raw detail for exact file paths, names, and signatures; the note summarises it."
        ]
        for i, ct in enumerate(chunk_order, 1):
            facts = "\n".join(f"  - {f}" for f in chunk_to_facts[ct])
            out.append(f"[detail {i}]\n{ct}\nlearned from the above:\n{facts}")
        if unchunked:
            out.append("[other notes]\n" + "\n".join(f"  - {f}" for f in unchunked))
        return "\n\n".join(out)

    def _recall_results(self, query: str):
        """One recall call returning the raw response (with chunks + source facts)."""
        t0 = time.time()
        try:
            return self._client.recall(
                self.bank_id,
                query=query,
                max_tokens=self.recall_max_tokens,
                budget=self.recall_budget,
                types=self.recall_types,
                include_chunks=True,
                max_chunk_tokens=self.max_chunk_tokens,
                include_source_facts=True,
                max_source_facts_tokens=self.recall_max_tokens,
            )
        except Exception:
            return None
        finally:
            self.stats.recall_seconds += time.time() - t0
            self.stats.recall_calls += 1

    def _recall_texts(self, query: str) -> list[str]:
        """One recall call → list of result texts (accounting for time/calls)."""
        t0 = time.time()
        try:
            resp = self._client.recall(
                self.bank_id,
                query=query,
                max_tokens=self.recall_max_tokens,
                budget=self.recall_budget,
                types=self.recall_types,
            )
        except Exception:
            return []
        finally:
            self.stats.recall_seconds += time.time() - t0
            self.stats.recall_calls += 1
        return [(getattr(r, "text", None) or "").strip() for r in (getattr(resp, "results", None) or [])]

    # -- retain (after a task) ---------------------------------------------------------

    def _summary_system(self, resolved: bool | None) -> str:
        """The summariser prompt for this retain, by style and task outcome.

        For "procedural", an UNKNOWN outcome (scoring skipped) is treated like a failure —
        without a passing test run the attempt's theory of the bug is unverified, so only
        process lessons are safe to store.
        """
        if self.retain_style != "procedural":
            return _SUMMARY_SYSTEM
        if resolved:
            return _SUMMARY_SYSTEM_PROCEDURAL_RESOLVED
        return _SUMMARY_SYSTEM_PROCEDURAL_FAILED

    @staticmethod
    def _strip_artifacts(summary: str) -> str:
        """Drop transcript artifacts the summariser copied verbatim.

        The observed failure shape (pilot memory_debug dumps) is the summariser *continuing
        the transcript* after the lessons — trailing harness commands / tool-call blocks — so
        everything from the first artifact-marker line onward is dropped, not just that line.
        """
        kept: list[str] = []
        for line in summary.splitlines():
            if any(m in line for m in _ARTIFACT_MARKERS):
                break
            if line.strip():
                kept.append(line)
        return "\n".join(kept)

    def retain_after_task(
        self,
        instance_id: str,
        transcript: str,
        resolved: bool | None = None,
        eval_feedback: str | None = None,
        attempt: int | None = None,
    ) -> None:
        """Distil the trajectory into durable facts and store them. No-op when disabled.

        ``resolved`` is the official test outcome for this task (None = not scored yet). It
        selects the summariser prompt for the "procedural" retain style; the "insight" style
        ignores it (outcome-blind, the original behaviour).

        ``eval_feedback`` is CI-style ground truth from the official harness (failing test
        names, assertion output, regressions). It is appended to the summariser input so
        failure lessons are grounded in verified evidence rather than the attempt's own
        theory — the raw material for learning from mistakes across attempts.
        """
        if not self.enabled or self._client is None:
            return
        summariser_input = transcript
        if eval_feedback:
            summariser_input = (
                transcript
                + "\n\n=== OFFICIAL TEST EVALUATION RESULT (ground truth from the test harness) ===\n"
                + eval_feedback
            )
        summary = self._strip_artifacts(self._summarise(summariser_input, self._summary_system(resolved)))
        self.last_retained_summary = summary
        if not summary or summary.strip().upper() == "NONE":
            return
        outcome = "outcome unknown" if resolved is None else ("resolved" if resolved else "failed the tests")
        attempt_note = f", attempt {attempt}" if attempt is not None else ""
        t0 = time.time()
        resp = None
        try:
            resp = self._client.retain(
                self.bank_id,
                content=summary,
                document_id=instance_id,
                update_mode="append",
                tags=[self.repo, "codebase-knowledge"],
                context=(f"Lessons from working on {instance_id} in {self.repo} ({outcome}{attempt_note})."),
            )
        finally:
            self.stats.retain_seconds += time.time() - t0
            self.stats.retain_calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.retain_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.stats.retain_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def _summarise(self, transcript: str, system_prompt: str) -> str:
        clipped = transcript[-self.summary_max_chars :]
        try:
            resp = litellm.completion(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clipped},
                ],
                temperature=0.0,
                drop_params=True,
            )
        except Exception:
            return ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.summary_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.stats.summary_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""
