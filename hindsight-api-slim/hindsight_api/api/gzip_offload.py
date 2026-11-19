"""Response gzip that keeps large bodies off the event loop.

Starlette's ``GZipMiddleware`` compresses on the loop thread. That is fine for a recall
response, but a full document body of several MB takes ~1 s at its level 9, and every other
request in the process waits for it (#6667). This compresses bodies past
``_OFFLOAD_MIN_BYTES`` in a worker thread; smaller ones stay inline, where a thread hop
would cost more than the compression.

It leans on Starlette's responder internals (``apply_compression`` and the two
pass-through flags); ``tests/test_gzip_offload.py`` fails if an upgrade changes them.
"""

from __future__ import annotations

import asyncio

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import Message, Receive, Scope, Send

# ~3 ms of level-9 compression; below it the thread hop is not worth it.
_OFFLOAD_MIN_BYTES = 64 * 1024


class _OffloadingGZipResponder(GZipResponder):
    """Runs the parent's compression for a large body in a thread, then lets the parent
    send it as usual. The gzip stream is only ever touched by one thread at a time: the
    thread finishes before the parent's (synchronous) ``apply_compression`` reads the result.
    """

    _precompressed: bytes | None = None

    async def send_with_compression(self, message: Message) -> None:
        body = message.get("body", b"")
        if (
            message["type"] == "http.response.body"
            and len(body) >= _OFFLOAD_MIN_BYTES
            # The parent passes these through uncompressed; a body this size always clears
            # minimum_size, so these are the only cases it would not compress.
            and not (self.content_encoding_set or self.content_type_is_excluded)
        ):
            self._precompressed = await asyncio.to_thread(
                super().apply_compression, body, more_body=message.get("more_body", False)
            )
        await super().send_with_compression(message)

    def apply_compression(self, body: bytes, *, more_body: bool) -> bytes:
        if self._precompressed is not None:
            out, self._precompressed = self._precompressed, None
            return out
        return super().apply_compression(body, more_body=more_body)


class OffloadingGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and "gzip" in Headers(scope=scope).get("Accept-Encoding", ""):
            responder = _OffloadingGZipResponder(self.app, self.minimum_size, compresslevel=self.compresslevel)
            await responder(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
