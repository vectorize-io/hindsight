"""Remote embedding providers issue their batches concurrently (#4039).

A retain used to hold exactly one embedding request open at a time: every remote
provider walked its batches in a sequential ``for`` loop. Against TEI that sat the client
at the slowest column of the throughput table (903 texts/s at one in-flight request
against 2,080 at eight), and for hosted providers the longer round trip makes the
serialization cost more, not less.

These tests assert on the shape of what reaches the upstream — how many requests were
open at once, and that the answers still line up with the inputs — rather than on
wall-clock, which would be a flaky proxy for it. The upstream is a real aiohttp server
on the same loop (``tests/aiohttp_stub.py``).
"""

import ast
import asyncio
import contextvars
import inspect
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.embeddings import (
    Embeddings,
    RemoteTEIEmbeddings,
    create_embeddings_from_env,
)
from hindsight_api.engine.retain.embedding_coalescer import resolve_max_batch_size
from tests.aiohttp_stub import Handler, stub_server


class _ConcurrencyProbe:
    """Records the high-water mark of simultaneously open requests."""

    def __init__(self, hold_until: int) -> None:
        self._hold_until = hold_until
        self._reached = asyncio.Event()
        self.open_now = 0
        self.peak = 0
        self.requests = 0

    async def enter(self) -> None:
        self.open_now += 1
        self.requests += 1
        self.peak = max(self.peak, self.open_now)
        if self.open_now >= self._hold_until:
            self._reached.set()
        # Every request parks here until `hold_until` of them are open at once, so the
        # peak below is a real observation and not a lucky interleaving.
        if self._hold_until > 1:
            await asyncio.wait_for(self._reached.wait(), timeout=10)

    def leave(self) -> None:
        self.open_now -= 1


@asynccontextmanager
async def _tei(batch_size: int, concurrency: int, handler: Handler) -> AsyncIterator[RemoteTEIEmbeddings]:
    async with stub_server(handler) as base_url:
        embeddings = RemoteTEIEmbeddings(base_url=base_url, batch_size=batch_size)
        embeddings.max_concurrent_requests = concurrency
        embeddings._initialized = True
        embeddings._dimension = 1
        try:
            yield embeddings
        finally:
            await embeddings._session.close()


async def _inputs(request: web.Request) -> list[str]:
    return (await request.json())["inputs"]


async def test_batches_go_out_concurrently() -> None:
    """Eight batches with eight slots put eight requests on the wire at once."""
    probe = _ConcurrencyProbe(hold_until=8)

    async def handler(request: web.Request) -> web.StreamResponse:
        await probe.enter()
        try:
            return web.json_response([[0.5, 0.5]] * 4)
        finally:
            probe.leave()

    async with _tei(batch_size=4, concurrency=8, handler=handler) as embeddings:
        vectors = await embeddings.encode([f"text {i}" for i in range(32)])

    assert len(vectors) == 32
    assert probe.requests == 8
    assert probe.peak == 8


async def test_results_keep_input_order_when_batches_finish_out_of_order() -> None:
    """A late first batch must not shuffle the vectors behind it."""
    others_answered = 0
    all_others_done = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal others_answered
        (text,) = await _inputs(request)
        index = int(text.split("text ")[1])
        if index == 0:
            # Let the others finish first, then answer last.
            await asyncio.wait_for(all_others_done.wait(), timeout=10)
        else:
            others_answered += 1
            if others_answered == 3:
                all_others_done.set()
        return web.json_response([[float(index)]])

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        vectors = await embeddings.encode([f"text {i}" for i in range(4)])

    assert vectors == [[0.0], [1.0], [2.0], [3.0]]


async def test_single_slot_keeps_requests_strictly_sequential() -> None:
    """max_concurrent_requests=1 is the historical shape, unchanged."""
    probe = _ConcurrencyProbe(hold_until=1)

    async def handler(request: web.Request) -> web.StreamResponse:
        await probe.enter()
        try:
            # Yield so an overlapping request, if any, would get the chance to open.
            await asyncio.sleep(0.01)
            return web.json_response([[0.1]] * 2)
        finally:
            probe.leave()

    async with _tei(batch_size=2, concurrency=1, handler=handler) as embeddings:
        assert len(await embeddings.encode([f"text {i}" for i in range(6)])) == 6

    assert probe.requests == 3
    assert probe.peak == 1


async def test_concurrent_callers_share_one_bound() -> None:
    """max_concurrent_requests bounds the backend, not each caller separately.

    A semaphore per encode() call would let two simultaneous retains put 2x the bound
    on the wire.
    """
    probe = _ConcurrencyProbe(hold_until=1)
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        await probe.enter()
        try:
            # Hold every request open so the peak below counts simultaneous ones.
            await asyncio.wait_for(release.wait(), timeout=10)
            return web.json_response([[0.4]])
        finally:
            probe.leave()

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        # Two callers at once, eight batches each: sixteen requests wanting to go out.
        callers = [
            asyncio.create_task(embeddings.encode([f"a {i}" for i in range(8)])),
            asyncio.create_task(embeddings.encode([f"b {i}" for i in range(8)])),
        ]
        try:
            # Wait until everything the bound allows is open, then a settle window in
            # which an unbounded implementation would open more.
            async with asyncio.timeout(10):
                while probe.open_now < 4:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            peak_while_held = probe.peak
        finally:
            release.set()
            results = await asyncio.gather(*callers)

    assert peak_while_held == 4, f"expected the bound to hold across callers, saw {peak_while_held}"
    assert probe.requests == 16
    assert [len(result) for result in results] == [8, 8]


async def test_the_semaphore_is_created_once_and_reused() -> None:
    """A semaphore per call would make the bound per-caller."""

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response([[0.6]])

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        assert embeddings._request_slots is None

        await embeddings.encode(["one", "two"])
        first = embeddings._request_slots
        assert first is not None
        semaphore = first.get()

        await embeddings.encode(["three", "four"])
        assert embeddings._request_slots is first
        assert first.get() is semaphore


async def test_a_failing_batch_fails_the_call() -> None:
    """One bad batch must not be silently dropped, leaving a short vector list."""

    async def handler(request: web.Request) -> web.StreamResponse:
        if "text 5" in await _inputs(request):
            return web.json_response({"error": "invalid input"}, status=400)
        return web.json_response([[0.2]])

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        with pytest.raises(RuntimeError, match="TEI embedding request failed"):
            await embeddings.encode([f"text {i}" for i in range(8)])


async def test_the_earliest_failing_batch_wins() -> None:
    """When several batches fail, the error does not depend on which one failed first."""

    async def handler(request: web.Request) -> web.StreamResponse:
        (text,) = await _inputs(request)
        if text == "text 1":
            # Fails last in time, but first in input order.
            await asyncio.sleep(0.2)
            return web.json_response({"error": "batch-one"}, status=400)
        if text == "text 3":
            return web.json_response({"error": "batch-three"}, status=400)
        return web.json_response([[0.2]])

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        with pytest.raises(RuntimeError, match="TEI embedding request failed") as exc_info:
            await embeddings.encode([f"text {i}" for i in range(4)])

    assert "batch-one" in str(exc_info.value)


async def test_every_batch_sees_the_callers_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fan-out must not drop the contextvars that carry per-bank attribution."""
    marker: contextvars.ContextVar[str | None] = contextvars.ContextVar("probe_marker", default=None)
    seen: list[str | None] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response([[0.3]])

    async with _tei(batch_size=1, concurrency=4, handler=handler) as embeddings:
        real_embed_batch = embeddings._embed_batch

        async def spy(batch: list[str]) -> list[list[float]]:
            seen.append(marker.get())
            return await real_embed_batch(batch)

        monkeypatch.setattr(embeddings, "_embed_batch", spy)
        token = marker.set("bank-42")
        try:
            await embeddings.encode([f"text {i}" for i in range(4)])
        finally:
            marker.reset(token)

    assert seen == ["bank-42"] * 4


def test_local_backends_stay_sequential_by_default() -> None:
    """The in-process backends have no round trip to overlap."""
    assert Embeddings.max_concurrent_requests == 1


def test_factory_gives_a_remote_provider_the_configured_concurrency() -> None:
    saved = {
        key: os.environ.get(key)
        for key in (
            "HINDSIGHT_API_LLM_PROVIDER",
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
            "HINDSIGHT_API_EMBEDDINGS_TEI_URL",
            "HINDSIGHT_API_EMBEDDINGS_MAX_CONCURRENT_REQUESTS",
        )
    }
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "tei"
    os.environ["HINDSIGHT_API_EMBEDDINGS_TEI_URL"] = "http://localhost:8080"
    os.environ["HINDSIGHT_API_EMBEDDINGS_MAX_CONCURRENT_REQUESTS"] = "5"
    clear_config_cache()
    try:
        assert HindsightConfig.from_env().embeddings_max_concurrent_requests == 5
        assert create_embeddings_from_env().max_concurrent_requests == 5
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()


def test_coalescer_and_backend_together_land_on_the_backends_bound() -> None:
    """The two layers multiply, so the hand-off carries concurrency/slots requests worth.

    Four coalescer slots each handing a backend 2 batches of 8 is 8 requests in flight —
    the backend's own bound, not four times it.
    """
    backend = RemoteTEIEmbeddings(base_url="http://localhost:8080", batch_size=8)
    backend.max_concurrent_requests = 8

    assert resolve_max_batch_size(backend, slots=4) == 16
    assert resolve_max_batch_size(backend, slots=8) == 8
    # More slots than the backend will accept requests: one batch each, never below one.
    assert resolve_max_batch_size(backend, slots=16) == 8

    # A backend that issues one request at a time gets one batch per hand-off.
    backend.max_concurrent_requests = 1
    assert resolve_max_batch_size(backend, slots=4) == 8


# --- Family guards -------------------------------------------------------------------
#
# The serial `for i in range(0, len(texts), self.batch_size)` loop was in every remote
# provider at once (#4039), and the same shape is what a *new* provider naturally gets
# written with. A per-provider test cannot catch the provider nobody wrote a test for,
# so these two assert over the whole family, read straight from the module's source.

# Backends that run the model in-process. They have no round trip to overlap, batch
# internally (SentenceTransformers' own batching / a bounded ONNX forward pass), and must
# keep the sequential default — an exemption, not an oversight.
_IN_PROCESS_BACKENDS = {"LocalSTEmbeddings", "OnnxEmbeddings"}


def _embeddings_module_ast() -> ast.Module:
    import hindsight_api.engine.embeddings as module

    return ast.parse(inspect.getsource(module))


def test_every_remote_provider_batches_through_the_shared_fan_out() -> None:
    """No provider may reintroduce its own sequential batch loop."""
    tree = _embeddings_module_ast()
    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Embeddings"):
            continue
        if node.name in _IN_PROCESS_BACKENDS or node.name == "Embeddings":
            continue
        # A provider that subclasses another provider (CodexOAuthEmbeddings only refreshes
        # an OAuth token and delegates) inherits the batching from its base.
        if any(isinstance(base, ast.Name) and base.id != "Embeddings" for base in node.bases):
            continue
        calls = {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        if "_encode_batched" not in calls:
            offenders.append(node.name)
    assert not offenders, f"remote providers not using Embeddings._encode_batched: {offenders}"


def test_the_factory_gives_every_remote_provider_its_concurrency() -> None:
    """A provider constructed outside _with_request_concurrency silently stays serial."""
    tree = _embeddings_module_ast()
    factory = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_embeddings_from_env"
    )
    unwrapped: list[str] = []
    for node in ast.walk(factory):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or not call.func.id.endswith("Embeddings"):
            continue
        # A bare `return SomeEmbeddings(...)` is only correct for the in-process backends.
        if call.func.id not in _IN_PROCESS_BACKENDS:
            unwrapped.append(call.func.id)
    assert not unwrapped, f"providers returned without _with_request_concurrency: {unwrapped}"
