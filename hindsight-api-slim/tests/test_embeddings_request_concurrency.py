"""Remote embedding providers issue their batches concurrently (#4039).

A retain used to hold exactly one embedding request open at a time: every remote
provider walked its batches in a ``for`` loop inside a single executor thread. Against
TEI that sat the client at the slowest column of the throughput table (903 texts/s at
one in-flight request against 2,080 at eight), and for hosted providers the longer round
trip makes the serialization cost more, not less.

These tests assert on the shape of what reaches the transport — how many requests were
open at once, and that the answers still line up with the inputs — rather than on
wall-clock, which would be a flaky proxy for it.
"""

import ast
import contextvars
import inspect
import os
import threading
import time

import httpx
import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.embeddings import (
    Embeddings,
    RemoteTEIEmbeddings,
    create_embeddings_from_env,
)
from hindsight_api.engine.retain.embedding_coalescer import resolve_max_batch_size


class _ConcurrencyProbe:
    """Records the high-water mark of simultaneously open requests."""

    def __init__(self, hold_until: int) -> None:
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(hold_until, timeout=10) if hold_until > 1 else None
        self.open_now = 0
        self.peak = 0
        self.requests = 0

    def enter(self) -> None:
        with self._lock:
            self.open_now += 1
            self.requests += 1
            self.peak = max(self.peak, self.open_now)
        # Every request parks here until `hold_until` of them are open at once, so the
        # peak below is a real observation and not a lucky interleaving.
        if self._barrier is not None:
            self._barrier.wait()

    def leave(self) -> None:
        with self._lock:
            self.open_now -= 1


def _tei(batch_size: int, concurrency: int, handler) -> RemoteTEIEmbeddings:
    embeddings = RemoteTEIEmbeddings(base_url="http://localhost:8080", batch_size=batch_size)
    embeddings.max_concurrent_requests = concurrency
    embeddings._client = httpx.Client(transport=httpx.MockTransport(handler))
    return embeddings


def test_batches_go_out_concurrently() -> None:
    """Eight batches with eight slots put eight requests on the wire at once."""
    probe = _ConcurrencyProbe(hold_until=8)

    def handler(request: httpx.Request) -> httpx.Response:
        probe.enter()
        try:
            return httpx.Response(200, json=[[0.5, 0.5]] * 4)
        finally:
            probe.leave()

    embeddings = _tei(batch_size=4, concurrency=8, handler=handler)
    vectors = embeddings.encode([f"text {i}" for i in range(32)])

    assert len(vectors) == 32
    assert probe.requests == 8
    assert probe.peak == 8


def test_results_keep_input_order_when_batches_finish_out_of_order() -> None:
    """A late first batch must not shuffle the vectors behind it."""
    started = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        first = "text 0" in payload
        if first:
            # Let the others get ahead, then answer last.
            started.wait(timeout=10)
        else:
            started.set()
        index = int(payload.split("text ")[1].split('"')[0])
        return httpx.Response(200, json=[[float(index)]])

    embeddings = _tei(batch_size=1, concurrency=4, handler=handler)
    vectors = embeddings.encode([f"text {i}" for i in range(4)])

    assert vectors == [[0.0], [1.0], [2.0], [3.0]]


def test_single_slot_keeps_requests_strictly_sequential() -> None:
    """max_concurrent_requests=1 is the historical shape, unchanged."""
    probe = _ConcurrencyProbe(hold_until=1)

    def handler(request: httpx.Request) -> httpx.Response:
        probe.enter()
        try:
            return httpx.Response(200, json=[[0.1]] * 2)
        finally:
            probe.leave()

    embeddings = _tei(batch_size=2, concurrency=1, handler=handler)
    assert len(embeddings.encode([f"text {i}" for i in range(6)])) == 6
    assert probe.requests == 3
    assert probe.peak == 1


def test_concurrent_callers_share_one_bounded_pool() -> None:
    """max_concurrent_requests bounds the backend, not each caller separately.

    A pool per encode() call would let two simultaneous retains put 2x the bound on the
    wire and spawn two pools' worth of threads — which is how the API's several event
    loops in one process (#4067) on a free-threaded build (#4037) would multiply it.
    """
    probe = _ConcurrencyProbe(hold_until=1)
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        probe.enter()
        try:
            # Hold every request open so the peak below counts simultaneous ones.
            release.wait(timeout=10)
            return httpx.Response(200, json=[[0.4]])
        finally:
            probe.leave()

    embeddings = _tei(batch_size=1, concurrency=4, handler=handler)

    # Two callers at once, eight batches each: sixteen requests wanting to go out.
    callers = [
        threading.Thread(target=embeddings.encode, args=([f"a {i}" for i in range(8)],)),
        threading.Thread(target=embeddings.encode, args=([f"b {i}" for i in range(8)],)),
    ]
    for caller in callers:
        caller.start()
    try:
        # Give the pool time to open everything it is willing to open at once, then a
        # settle window in which an unbounded implementation would open more.
        deadline = time.monotonic() + 10
        while probe.open_now < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        peak_while_held = probe.peak
    finally:
        release.set()
        for caller in callers:
            caller.join(timeout=10)

    assert peak_while_held == 4, f"expected the bound to hold across callers, saw {peak_while_held}"
    assert probe.requests == 16


def test_the_pool_is_created_once_and_reused() -> None:
    """A pool per call would leak threads on every encode()."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.6]])

    embeddings = _tei(batch_size=1, concurrency=4, handler=handler)
    assert embeddings._request_pool is None

    embeddings.encode(["one", "two"])
    first = embeddings._request_pool
    assert first is not None

    embeddings.encode(["three", "four"])
    assert embeddings._request_pool is first


def test_a_failing_batch_fails_the_call() -> None:
    """One bad batch must not be silently dropped, leaving a short vector list."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "text 5" in request.read().decode():
            return httpx.Response(400, json={"error": "invalid input"})
        return httpx.Response(200, json=[[0.2]])

    embeddings = _tei(batch_size=1, concurrency=4, handler=handler)
    with pytest.raises(RuntimeError, match="TEI embedding request failed"):
        embeddings.encode([f"text {i}" for i in range(8)])


def test_worker_threads_see_the_callers_context() -> None:
    """Fan-out must not drop the contextvars that carry per-bank attribution."""
    marker: contextvars.ContextVar[str | None] = contextvars.ContextVar("probe_marker", default=None)
    seen: list[str | None] = []
    seen_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with seen_lock:
            seen.append(marker.get())
        return httpx.Response(200, json=[[0.3]])

    embeddings = _tei(batch_size=1, concurrency=4, handler=handler)
    token = marker.set("bank-42")
    try:
        embeddings.encode([f"text {i}" for i in range(4)])
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
