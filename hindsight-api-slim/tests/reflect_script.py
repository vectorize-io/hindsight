"""Scripted ``call_with_tools`` replies for reflect agent tests.

The reflect loop writes every answer through its closing step: when a turn calls ``done``, a
second call asks for ``done`` again with the "Answer now" request. A test that scripts the turns
up to that ``done`` would otherwise run out of replies on the closing call.
"""

from collections.abc import Callable, Iterable
from typing import Any

from hindsight_api.engine.reflect.agent import _is_done_tool
from hindsight_api.engine.response_models import LLMToolCallResult


def _calls_done(reply: Any) -> bool:
    return isinstance(reply, LLMToolCallResult) and any(_is_done_tool(call.name) for call in reply.tool_calls)


def closing_echo(replies: Iterable[Any]) -> Callable[..., Any]:
    """A ``side_effect`` that plays ``replies`` in order, then answers the closing ``done`` request.

    Once the script is used up, a call that names ``done`` gets the last scripted ``done`` again:
    the model re-emits the answer it was about to give. Any other call past the script raises
    ``StopIteration``, as a plain list would.
    """
    queue = list(replies)
    last_done: list[Any] = []

    def reply(*_args: Any, **kwargs: Any) -> Any:
        if queue:
            item = queue.pop(0)
            if _calls_done(item):
                last_done[:] = [item]
            if isinstance(item, BaseException):
                raise item
            return item
        choice = kwargs.get("tool_choice")
        if last_done and getattr(choice, "function_name", None) == "done":
            return last_done[0]
        raise StopIteration

    return reply
