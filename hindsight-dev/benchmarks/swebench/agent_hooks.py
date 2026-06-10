"""mini-swe-agent subclasses for the memory study.

``MeteredAgent`` adds per-task token accounting on top of the stock ``DefaultAgent`` and is
used by *both* arms, so input/output token counts are measured identically. ``MemoryAgent``
adds the Hindsight recall-before / retain-after behaviour for the treatment arm.

Recall is injected via ``extra_template_vars["recalled_memories"]`` plus a ``{% if
recalled_memories %}`` block appended to the stock instance template (see
``run_study.build_agent_config``). Both arms render the *same* template; the control arm just
leaves the variable empty, so prompts are identical except for the injected memory block.
"""

from __future__ import annotations

from minisweagent.agents.default import DefaultAgent

from .memory_glue import MemoryGlue


def _usage_from_message(message: dict) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a litellm model message."""
    usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


class MeteredAgent(DefaultAgent):
    """DefaultAgent that accumulates input/output token usage across the trajectory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_tokens = 0
        self.output_tokens = 0
        # Always define the var so the shared template renders for both arms.
        self.extra_template_vars["recalled_memories"] = ""

    def query(self) -> dict:
        message = super().query()
        pt, ct = _usage_from_message(message)
        self.input_tokens += pt
        self.output_tokens += ct
        return message

    def transcript_text(self) -> str:
        """Plain-text view of the trajectory, for the retain summariser."""
        parts: list[str] = []
        for m in self.messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):  # multimodal blocks
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)


# Wrapper for per-step adaptive memory, refreshed in place in the system message.
_STEP_MEMORY_HEADER = (
    "\n\n<codebase_memory>\n"
    "You have worked in this repository before. The notes below are debugging knowledge YOU "
    "accumulated solving RELATED issues here — root-cause patterns, gotchas, where the relevant "
    "logic lives, and how to verify it. They are refreshed to match what you are working on right "
    "now. Treat them as knowledge you already have and use them actively.\n"
    "{block}\n"
    "</codebase_memory>"
)


class MemoryAgent(MeteredAgent):
    """MeteredAgent that recalls before the task and retains durable knowledge after it.

    With ``step_reinject``, the memory is additionally refreshed every ``reinject_every`` model
    calls: re-recalled against what the agent is *currently* looking at (the latest observation)
    and rewritten in place into the system message, so it stays at the front of context and
    tracks the agent's focus instead of being frozen at step 1.
    """

    def __init__(
        self,
        *args,
        glue: MemoryGlue,
        instance_id: str,
        step_reinject: bool = False,
        reinject_every: int = 1,
        defer_retain: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.glue = glue
        self.instance_id = instance_id
        self.step_reinject = step_reinject
        self.reinject_every = max(1, reinject_every)
        # When True, run() does NOT retain — the orchestrator retains after scoring the task,
        # so the summariser knows whether the attempt actually passed the tests.
        self.defer_retain = defer_retain
        self._task = ""
        self._base_system: str | None = None  # pristine system content, captured on first query

    def run(self, task: str = "", **kwargs) -> dict:
        self._task = task
        # Static injection at task start (renders into the template block). For step_reinject,
        # the template block is absent, so this seeds the very first call's memory via the
        # query() refresh below instead.
        self.extra_template_vars["recalled_memories"] = "" if self.step_reinject else self.glue.context_for_task(task)
        info = super().run(task, **kwargs)
        if not self.defer_retain:
            self.glue.retain_after_task(self.instance_id, self.transcript_text())
        return info

    def _recent_observation(self) -> str:
        """The most recent non-system message content — what the agent is currently looking at."""
        for m in reversed(self.messages):
            if m.get("role") in ("user", "tool"):
                c = m.get("content", "")
                return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return ""

    def query(self) -> dict:
        if self.step_reinject and self.messages and self.messages[0].get("role") == "system":
            if self._base_system is None:
                self._base_system = self.messages[0]["content"]
            if self.n_calls % self.reinject_every == 0:
                block = self.glue.context_for_step(self._task, self._recent_observation())
                self.messages[0]["content"] = (
                    self._base_system + _STEP_MEMORY_HEADER.format(block=block) if block else self._base_system
                )
                # Record last injected block so the recalled_chars metric is non-zero.
                self.extra_template_vars["recalled_memories"] = block
        return super().query()
