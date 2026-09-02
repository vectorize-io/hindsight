"""Test stub for CodexLLM's streaming POST.

``CodexLLM`` reads its SSE body with ``client.stream()`` under a wall-clock
deadline rather than a buffering ``client.post()`` (issue #3898), so tests that
want to inspect the request or hand back a canned response patch the streaming
call instead. The mock this returns records calls exactly as the old ``post``
mock did — ``mock.call_args.kwargs["json"]`` / ``["headers"]`` still work —
because ``stream()`` is called with the same keyword arguments.
"""

from typing import Any
from unittest.mock import MagicMock, patch


class _StreamContext:
    """Async context manager yielding a canned response, like ``client.stream()``."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self) -> Any:
        return self._response

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


def stub_codex_stream(llm: Any, response: Any) -> Any:
    """Patch ``llm._client.stream`` to yield ``response``; use as a context manager.

    The patched attribute is a plain ``MagicMock`` (not an ``AsyncMock``):
    ``stream()`` is a sync call returning an async context manager.
    """
    return patch.object(llm._client, "stream", MagicMock(return_value=_StreamContext(response)))


def stub_codex_stream_with(llm: Any, handler: Any) -> Any:
    """Like ``stub_codex_stream`` but ``handler(url, **kwargs)`` builds each response.

    For tests that vary the response per attempt (auth-refresh retries). The
    handler runs when the provider calls ``stream()``.
    """
    return patch.object(
        llm._client,
        "stream",
        MagicMock(side_effect=lambda _method, url, **kwargs: _StreamContext(handler(url, **kwargs))),
    )
