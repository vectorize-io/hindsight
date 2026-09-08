"""Driving the reflect loop, which every mental-model and reflect story needs.

Reflect is agentic, and its shape is not fixed: the server walks a ladder of
forced search turns, and — depending on the bank's configuration — may end with a
free choice among every tool plus `done`, or with a separate synthesising turn
under a different system prompt. A story that only cares what reflect *concludes*
should not have to know which, and should not break when the ladder gains a rung.

`reflect_loop` handles all three shapes, in the order the rules must be tried:
finish when finishing is offered, otherwise climb, and answer if asked in prose.
"""

from __future__ import annotations

from .rulebook import LLMStub


def reflect_loop(llm: LLMStub, *, answer: str, query: str = "system test") -> None:
    """Climb every search turn, then answer with ``answer``.

    Rules match in registration order, so `done` is registered first: the final
    turn offers it *alongside* every search tool, and a rule that took the first
    tool offered would search again, be offered the same choice again, and never
    terminate.
    """
    # `done` carries the answer in its own argument — calling it bare ends the
    # loop with "the done tool returned no answer".
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer=answer)
    llm.on_step("reflect").calls_the_offered_tool(query=query)
    # The prose turn, matched by what it *lacks*. A bank with a reflect mission
    # replaces the default role on the answering prompt just as it does on the
    # search prompts, so `reflect_answer`'s anchor disappears exactly when a
    # mission is set. What stays true is that this is the only reflect turn
    # offering no tools — and the two rules above claim every turn that does.
    llm.on_step("reflect").returns_text(answer)
