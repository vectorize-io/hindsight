"""Fixtures for the system suite.

The shape every test takes: a real server process (session-scoped, because
starting one costs seconds), a rulebook reset between tests, and a client. Two
guards run after every test — an unmatched LLM call and a request the stub
rejected as malformed both fail the test that caused them, so neither can pass
silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import (
    LLMStub,
    Stubs,
    start_hindsight_server,
    start_stub_server,
    wait_until_settled,
)

SettleFn = Callable[[str], Awaitable[None]]

BANK_PREFIX = "systest-"


@pytest.fixture(scope="session")
def stubs() -> Stubs:
    return Stubs()


@pytest.fixture(scope="session")
def stub_server(stubs: Stubs) -> Iterator[object]:
    server = start_stub_server(stubs)
    yield server
    server.stop()


@pytest.fixture(scope="session")
def hindsight_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    log_path: Path = tmp_path_factory.mktemp("hindsight-server") / "server.log"
    server = start_hindsight_server(stub_url=stub_server.url, log_path=log_path)
    # Sweep before the first test rather than after the last one: a run killed
    # mid-test leaves banks behind, and only a sweep at startup catches those.
    asyncio.run(_delete_leftover_banks(server.url))
    yield server
    server.stop()


@pytest.fixture
def llm(stubs: Stubs) -> LLMStub:
    """The chat-completions rulebook. Declare what the model says before acting."""
    return stubs.llm


@pytest.fixture(autouse=True)
def _reset_stubs(stubs: Stubs) -> Iterator[None]:
    stubs.reset()
    yield


@pytest.fixture(autouse=True)
def _no_unstubbed_calls(stubs: Stubs, request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a test that let a call through unscripted, or sent a malformed request.

    Both are silent-success hazards. An unmatched call means the test proved less
    than it claims; a rejected request means we sent something a real provider
    would have refused. Neither is allowed to be a warning.
    """
    yield

    if request.node.stash.get(_ALREADY_FAILED, False):
        return

    if stubs.llm.unmatched:
        suggestions = "\n".join(f"    {call.suggestion()}" for call in stubs.llm.unmatched)
        pytest.fail(
            f"{len(stubs.llm.unmatched)} LLM call(s) matched no stub rule. Add:\n{suggestions}",
            pytrace=False,
        )

    if stubs.rejected_requests:
        rejections = "\n".join(f"    {reason}" for reason in stubs.rejected_requests)
        pytest.fail(
            "The server sent request(s) the stub rejected as malformed — a real provider\n"
            f"would have returned 400 too:\n{rejections}",
            pytrace=False,
        )


_ALREADY_FAILED = pytest.StashKey[bool]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Iterator[None]:
    """Remember a failure, so the guards above don't bury the real error."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        item.stash[_ALREADY_FAILED] = True


@pytest.fixture
async def client(hindsight_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=hindsight_server.url)
    yield client
    await client.aclose()


@pytest.fixture
def settled(client: Hindsight) -> SettleFn:
    """Wait for the bank's background work to finish.

    Retain returns before the worker has run consolidation, so a test that asserts
    straight after it is racing. Await this first.
    """

    async def _settled(bank_id: str) -> None:
        await wait_until_settled(client, bank_id)

    return _settled


@pytest.fixture
async def bank_id(client: Hindsight) -> AsyncIterator[str]:
    """A bank of this test's own, deleted when the test ends.

    Isolation is half the reason; the other half is that the database outlives the
    run. A bank left behind with an unfinished consolidation is picked up by the
    worker the *next* time the server starts, and its LLM call arrives during some
    later test's setup — an unmatched call attributed to a test that never asked
    for it. Deleting the bank here, and sweeping leftovers at session start, keeps
    one run's failures out of the next one's results.
    """
    name = f"{BANK_PREFIX}{uuid.uuid4().hex[:12]}"
    yield name
    with contextlib.suppress(Exception):
        await client.banks.delete_bank(name)


async def _delete_leftover_banks(base_url: str) -> None:
    client = Hindsight(base_url=base_url)
    try:
        banks = await client.banks.list_banks()
        for bank in banks.banks:
            if bank.bank_id.startswith(BANK_PREFIX):
                with contextlib.suppress(Exception):
                    await client.banks.delete_bank(bank.bank_id)
    finally:
        await client.aclose()
