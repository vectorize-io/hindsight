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


# Distil a trajectory into a durable MAP of the codebase — where things live and how this area
# works — NOT a changelog of how this particular issue was fixed. Storing the specific fix is
# both leak-adjacent and actively harmful: it makes a later agent confidently apply a prior
# fix to a different issue where it doesn't belong (observed: it broke an otherwise-solvable
# task). We keep only navigation/structural knowledge, which transfers safely and helps the
# agent locate and understand code faster.
_SUMMARY_SYSTEM = (
    "You are mapping a large codebase for a teammate who will work on DIFFERENT future issues "
    "in the same area. You'll see a transcript of an agent working on ONE issue. Extract 4-10 "
    "SHORT, DURABLE, STRUCTURAL facts about the codebase — a map, not a changelog.\n\n"
    "CAPTURE ONLY:\n"
    "- Where things live: which file/module/class/function owns a given responsibility "
    "(e.g. 'X is implemented in path/to/file.py in the Foo.bar method').\n"
    "- What key abstractions/classes do and how the area is structured (data/control flow).\n"
    "- How to run the relevant tests (exact `python tests/runtests.py <label>` invocations) and "
    "any repo setup/test gotchas.\n"
    "- Stable naming/structural conventions.\n\n"
    "STRICTLY EXCLUDE (do not write these, they mislead on other issues):\n"
    "- The fix, patch, diff, or solution to THIS issue — anything of the form 'the fix is…', "
    "'to fix this…', 'I changed/optimized/modified…', 'X must be cleared/added/stripped…', "
    "'the bug was…', or the root cause of this specific issue.\n"
    "- Anything phrased as an action you took or a change to make.\n\n"
    "Write each fact as a neutral statement of how the codebase IS (present tense, descriptive), "
    "never as something that was changed. One fact per line, plain text, no numbering, no "
    "preamble. If nothing durable and structural was learned, output the single word NONE."
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
        orientation_enabled: bool = True,
        orientation_query: str | None = None,
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
            f"Overview of the {repo} codebase: repository layout and where the key modules and "
            f"subsystems live, how to run the test suite (and a single test), build/setup and "
            f"development conventions, and common pitfalls."
        )
        # Fact types for recall/reflect context. None = all types (world/experience/observation).
        # ["observation"] uses ONLY Hindsight's consolidated+deduped layer — cleaner, less
        # redundant context. Consolidation is fast (~12s on dev) and tasks are minutes apart, so
        # the only knowledge an observations-only run can miss is the immediately-preceding task's
        # (older tasks are always consolidated by the time they're recalled).
        self.recall_types = recall_types
        self.summary_max_chars = summary_max_chars
        self.stats = MemoryOpStats()
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
        self._client.create_bank(
            self.bank_id,
            background=(
                f"Durable engineering knowledge about the {self.repo} codebase, accumulated "
                "by an AI software engineer solving issues over time. Used to navigate and "
                "fix new issues in the same repo faster."
            ),
            retain_mission=(
                "Capture durable, reusable facts about the codebase (file locations, how to "
                "run tests, conventions, pitfalls). Do not capture issue-specific fixes."
            ),
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

    def retain_after_task(self, instance_id: str, transcript: str) -> None:
        """Distil the trajectory into durable facts and store them. No-op when disabled."""
        if not self.enabled or self._client is None:
            return
        summary = self._summarise(transcript)
        if not summary or summary.strip().upper() == "NONE":
            return
        t0 = time.time()
        resp = None
        try:
            resp = self._client.retain(
                self.bank_id,
                content=summary,
                document_id=instance_id,
                update_mode="append",
                tags=[self.repo, "codebase-knowledge"],
                context=f"Durable knowledge learned while solving {instance_id} in {self.repo}.",
            )
        finally:
            self.stats.retain_seconds += time.time() - t0
            self.stats.retain_calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.retain_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.stats.retain_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def _summarise(self, transcript: str) -> str:
        clipped = transcript[-self.summary_max_chars :]
        try:
            resp = litellm.completion(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
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
