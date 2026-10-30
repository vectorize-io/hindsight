"""Scripted ``call_with_tools`` replies for reflect agent tests.

The reflect loop writes every answer through its closing step: when a turn calls ``done``, a
second call asks for ``done`` again with the "Answer now" request. A test that scripts the turns
up to that ``done`` would otherwise run out of replies on the closing call.
"""

from collections.abc import Callable, Iterable
from typing import Any

from hindsight_api.engine.reflect.agent import _is_done_tool
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult


def _calls_done(reply: Any) -> bool:
    return isinstance(reply, LLMToolCallResult) and any(_is_done_tool(call.name) for call in reply.tool_calls)


def _offered(kwargs: dict[str, Any]) -> set[str] | None:
    tools = kwargs.get("tools")
    return None if tools is None else {(t.get("function") or t).get("name") for t in tools}


def closing_echo(replies: Iterable[Any]) -> Callable[..., Any]:
    """A ``side_effect`` that plays ``replies`` in order, then answers the closing ``done`` request.

    A scripted ``done`` on a turn that does not offer it is played as ``answer_ready`` first
    (its ``answer_found`` carried over, its ``finish_reason`` not), then handed to the next
    call that asks for ``done`` — ahead of anything still scripted, so a script cannot queue a
    different reply for that closing call. Once the script is used up, a call that names ``done``
    gets the last scripted ``done`` again:
    the model re-emits the answer it was about to give. Any other call past the script raises
    ``StopIteration``, as a plain list would.
    """
    queue = list(replies)
    last_done: list[Any] = []
    held_done: list[Any] = []  # a done played as answer_ready, for the closing call

    def reply(*_args: Any, **kwargs: Any) -> Any:
        choice = kwargs.get("tool_choice")
        asks_for_done = getattr(choice, "function_name", None) == "done"
        if asks_for_done and held_done:
            return held_done.pop()
        if queue:
            item = queue.pop(0)
            if _calls_done(item):
                last_done[:] = [item]
            if isinstance(item, BaseException):
                raise item
            offered = _offered(kwargs)
            if _calls_done(item) and offered is not None and "done" not in offered:
                # Production holds a free turn to the tools it offers, and when the closing step
                # writes the answer that is not done: the model says answer_ready, and the done
                # comes back on the closing call. Played that way, so a script of "search, done"
                # follows the real path rather than one no provider can produce.
                assert "answer_ready" in offered, f"scripted done on a turn offering {sorted(offered)}"
                held_done[:] = [item]
                found = item.tool_calls[0].arguments.get("answer_found", True)
                return LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(id=item.tool_calls[0].id, name="answer_ready", arguments={"answer_found": found})
                    ],
                    finish_reason="tool_calls",
                )
            return item
        if last_done and asks_for_done:
            return last_done[0]
        raise StopIteration

    return reply
