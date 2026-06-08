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


class MemoryAgent(MeteredAgent):
    """MeteredAgent that recalls before the task and retains durable knowledge after it."""

    def __init__(self, *args, glue: MemoryGlue, instance_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.glue = glue
        self.instance_id = instance_id

    def run(self, task: str = "", **kwargs) -> dict:
        # Fetch memory context BEFORE messages are built so it renders into the prompt.
        # (recall or reflect, per the glue's context_mode.)
        self.extra_template_vars["recalled_memories"] = self.glue.context_for_task(task)
        info = super().run(task, **kwargs)
        # Retain AFTER the task using the completed trajectory.
        self.glue.retain_after_task(self.instance_id, self.transcript_text())
        return info
