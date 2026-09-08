"""Driving the reflect loop, which every mental-model and reflect story needs.

Reflect is agentic: the server offers one search tool per turn and will not
advance until the model calls it, then finally asks for an answer under a
different system prompt. A story that only cares what reflect *concludes* would
otherwise have to spell out every rung of that ladder, and would break the day a
rung is added.

`reflect_loop` walks it generically — call whatever tool this turn offers, then
give the answer — so a story declares its subject matter and nothing else. Where
a story *is* about a particular rung, it registers its own rule before calling
this: rules match in registration order, so the specific one wins.
"""

from __future__ import annotations

from .rulebook import LLMStub


def reflect_loop(llm: LLMStub, *, answer: str, query: str = "system test") -> None:
    """Climb every search turn, then answer with ``answer``."""
    llm.on_step("reflect").calls_the_offered_tool(query=query)
    llm.on_step("reflect_answer").returns_text(answer)
