"""Concurrent retains of one bank share a coalescer, and the process bounds its own fan-out.

A coalescer built per retain call cannot merge across calls, so a workload made of many
concurrent retains presents the embedder with many small requests instead of a few full ones --
measured at 42 in flight carrying 6.7 texts each, against an embedder whose optimum is 8 requests
of 32. These pin the two properties that fix: sharing is by BANK (which is the constraint the
per-call scope was protecting), and the total in flight is bounded process-wide rather than per
caller.
"""

import asyncio

import pytest

from hindsight_api.engine.retain import embedding_coalescer as ec


class _Backend:
    """Records how many requests were open at once, which is the thing under test."""

    dimension = 4
    batch_size = 8
    max_concurrent_requests = 4

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.open = 0
        self.peak = 0
        self.sizes: list[int] = []

    def encode_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise AssertionError("the async path must be used")

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        self.open += 1
        self.peak = max(self.peak, self.open)
        self.sizes.append(len(texts))
        try:
            await asyncio.sleep(self.delay)
            return [[0.0] * self.dimension for _ in texts]
        finally:
            self.open -= 1


@pytest.fixture(autouse=True)
def _clean_registry():
    ec._SHARED.clear()
    ec._GLOBAL_SLOTS = None
    ec._GLOBAL_SLOTS_SIZE = 0
    yield
    ec._SHARED.clear()


def test_one_bank_gets_one_coalescer():
    backend = _Backend()
    a = ec.acquire_shared("bank-a", backend)
    b = ec.acquire_shared("bank-a", backend)
    assert a is b, "concurrent retains of one bank must share, or they cannot merge"


def test_different_banks_never_share():
    """The constraint the per-call scope existed to protect: the backends read the ambient bank
    id for cost attribution, so two banks' texts must never land in one request."""
    backend = _Backend()
    assert ec.acquire_shared("bank-a", backend) is not ec.acquire_shared("bank-b", backend)


def test_the_last_user_out_closes_it():
    backend = _Backend()
    first = ec.acquire_shared("bank-a", backend)
    ec.acquire_shared("bank-a", backend)

    ec.release_shared("bank-a")
    assert not first._closed, "a retain finishing must not close a coalescer others are using"

    ec.release_shared("bank-a")
    assert first._closed and "bank-a" not in ec._SHARED


def test_a_closed_entry_is_replaced_rather_than_handed_out():
    backend = _Backend()
    first = ec.acquire_shared("bank-a", backend)
    ec.release_shared("bank-a")
    assert first._closed
    assert ec.acquire_shared("bank-a", backend) is not first
