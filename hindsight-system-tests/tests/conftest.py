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
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import (
    LLMStub,
    Stubs,
    start_hindsight_server,
    start_hindsight_worker,
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


@pytest.fixture(scope="session")
def typesafe_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A second server whose reranker and decision model are TypeSafe, pointed at the stub.

    The reranker provider is server-level — it is not in `_CONFIGURABLE_FIELDS`, so
    no bank can switch to it — which is why this costs a whole process rather than a
    bank setting. Session-scoped and started lazily, so only the stories that ask
    for it pay for it.
    """
    log_path: Path = tmp_path_factory.mktemp("typesafe-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={
            "HINDSIGHT_API_RERANKER_PROVIDER": "typesafe",
            "HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY": "stub-key",
            "HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL": stub_server.url,
            "HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES": "true",
            # The decision model reflect uses: the same stub endpoint, set up for reflect only. Set up
            # globally it also turned on decision reranking for recall, and the reranker stories
            # stopped seeing the reranker's order.
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_PROVIDER": "typesafe",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY": "stub-key",
            "HINDSIGHT_API_REFLECT_DECISION_MODEL_BASE_URL": stub_server.url,
            # HTTP only. Both servers share one database, so a worker here claims the OTHER
            # server's queued work — every story's consolidation and refreshes — and runs it with
            # this server's reranker, pruning by whatever cut level a TypeSafe story last set. The
            # story then reads a result its own server never produced, and fails one run in a few
            # depending on which worker polled first.
            "HINDSIGHT_API_WORKER_ENABLED": "false",
        },
    )
    yield server
    server.stop()


@pytest.fixture(scope="session")
def decision_model_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A server with a decision model set up, pointed at the stub: recall ranks with it.

    The decision model is server-level, so it costs a process. HTTP only, for the same
    reason as ``typesafe_server``: both share one database, and a worker here would run
    the other server's queued work with this server's ranking.
    """
    log_path: Path = tmp_path_factory.mktemp("decision-model-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={
            "HINDSIGHT_API_DECISION_MODEL_PROVIDER": "typesafe",
            "HINDSIGHT_API_DECISION_MODEL_API_KEY": "stub-key",
            "HINDSIGHT_API_DECISION_MODEL_BASE_URL": stub_server.url,
            "HINDSIGHT_API_WORKER_ENABLED": "false",
        },
    )
    yield server
    server.stop()


#: What the federated server presents at the token exchange — a projected
#: service-account token, in a real deployment.
WIF_IDENTITY_TOKEN = "systest-projected-sa-jwt"
WIF_FEDERATION_RULE_ID = "fdrl_systest"


@pytest.fixture(scope="session")
def wif_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A server on the Anthropic provider with no API key, authenticating through
    Workload Identity Federation against the stub (issue #5502).

    The LLM provider is server-level, hence a process. HTTP only, for the same reason
    as ``typesafe_server``: a worker here would run the other servers' queued work
    with this server's provider.
    """
    run_dir = tmp_path_factory.mktemp("wif-server")
    token_file = run_dir / "identity-token"
    token_file.write_text(WIF_IDENTITY_TOKEN)
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=run_dir / "server.log",
        extra_env={
            "HINDSIGHT_API_LLM_PROVIDER": "anthropic",
            # Empty overrides the stub key `stub_environment` sets: no static key at all.
            "HINDSIGHT_API_LLM_API_KEY": "",
            "HINDSIGHT_API_LLM_BASE_URL": stub_server.url,
            "ANTHROPIC_FEDERATION_RULE_ID": WIF_FEDERATION_RULE_ID,
            "ANTHROPIC_ORGANIZATION_ID": "org_systest",
            "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_systest",
            "ANTHROPIC_IDENTITY_TOKEN_FILE": str(token_file),
            "HINDSIGHT_API_WORKER_ENABLED": "false",
        },
    )
    yield server
    server.stop()


@pytest.fixture(scope="session")
def scoped_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A server whose operation validator confines each API key to ``user:<key>`` + ``kind:rule``.

    The validator (``server_extensions/tag_scope_validator.py``) is server-level, so it costs
    a process of its own. Session-scoped and started lazily, like ``typesafe_server``.
    """
    log_path: Path = tmp_path_factory.mktemp("scoped-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={
            "HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION": "tag_scope_validator:UserTagScope",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "server_extensions"),
        },
    )
    yield server
    server.stop()


@pytest.fixture(scope="session")
def retrying_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A server that will actually re-ask the extraction model once.

    ``stub_environment`` pins ``HINDSIGHT_API_LLM_MAX_RETRIES`` to 0 for every other
    story, because a retry would double the stubbed calls and prove nothing. The
    schema-drift recovery in #5280 lives *entirely* in the retry — the correction is
    what the second attempt carries — so with no budget there is no second attempt and
    the mechanism cannot be observed at all. The budget is server-level (it is not in
    ``_CONFIGURABLE_FIELDS``), hence a process rather than a bank setting. Scoped to
    retain, so recall, reflect and consolidation keep the suite's zero-retry rule.

    The worker stays enabled, unlike ``typesafe_server``: the only difference here is
    how many times a drifted extraction may be re-asked, and a story that never drifts
    is unaffected by this worker claiming its queued work.
    """
    log_path: Path = tmp_path_factory.mktemp("retrying-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={"HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES": "1"},
    )
    yield server
    server.stop()


#: The tenant schema of the LLM-keys deployment. Its own, so its worker claims only its
#: own banks' work and the default server's worker never claims theirs.
KEYED_SCHEMA = "systest_llmkeys"

#: Bank-id prefix → the key the bank-llm-routing extension bills that bank's calls to.
KEYED_BANKS = {f"{BANK_PREFIX}alice-": "alice-key", f"{BANK_PREFIX}bob-": "bob-key"}

#: A bank prefix routed to a whole different LLM rather than a different key.
CARL_PREFIX = f"{BANK_PREFIX}carl-"
CARL_MODEL = "carl-model"
CARL_RETAIN_MODEL = "carl-retain-model"

BANK_LLM_ROUTING_EXTENSION_DIR = Path(__file__).resolve().parents[2] / "hindsight-extensions" / "bank-llm-routing"


@pytest.fixture(scope="session")
def keyed_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A split deployment — HTTP server plus a standalone worker — that sends each bank's
    LLM calls with its own key, or to its own LLM, through the ``bank-llm-routing``
    community extension.

    Split on purpose: the extension must hold in the API process (sync retain, reflect)
    and in the worker process (async retain, consolidation, mental-model refresh), and
    with the in-process worker on, a story could not tell which one ran a step.
    """
    routes = [
        {
            "bank": f"{prefix}*",
            "schema": KEYED_SCHEMA,
            "api_key": key,
            "headers": {"x-gateway-customer": key.removesuffix("-key")},
        }
        for prefix, key in KEYED_BANKS.items()
    ]
    # Carl's banks run on an LLM of their own — model, endpoint and key — and retain on a
    # different one again, so a story can tell the operations apart by the model they reached.
    carl_llm = {"provider": "openai", "base_url": f"{stub_server.url}/v1", "api_key": "carl-key"}
    routes += [
        {
            "bank": f"{CARL_PREFIX}*",
            "schema": KEYED_SCHEMA,
            "operation": "retain",
            "llm": {**carl_llm, "model": CARL_RETAIN_MODEL},
        },
        {"bank": f"{CARL_PREFIX}*", "schema": KEYED_SCHEMA, "llm": {**carl_llm, "model": CARL_MODEL}},
    ]
    env = {
        "HINDSIGHT_API_DATABASE_SCHEMA": KEYED_SCHEMA,
        "HINDSIGHT_API_LLM_REQUEST_EXTENSION": "hindsight_ext_bank_llm_routing:BankLLMRoutingExtension",
        "HINDSIGHT_API_LLM_REQUEST_ROUTES": json.dumps(routes),
        "HINDSIGHT_API_LLM_REQUEST_UNMATCHED": "deny",
        "PYTHONPATH": str(BANK_LLM_ROUTING_EXTENSION_DIR),
    }
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path_factory.mktemp("keyed-server") / "server.log",
        extra_env={**env, "HINDSIGHT_API_WORKER_ENABLED": "false"},
    )
    worker = start_hindsight_worker(
        stub_url=stub_server.url,
        log_path=tmp_path_factory.mktemp("keyed-worker") / "worker.log",
        extra_env=env,
    )
    yield server
    worker.stop()
    server.stop()


@pytest.fixture
async def keyed_client(keyed_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=keyed_server.url)
    yield client
    await client.aclose()


@pytest.fixture
async def retrying_client(retrying_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=retrying_server.url)
    yield client
    await client.aclose()


@pytest.fixture
async def wif_client(wif_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=wif_server.url)
    yield client
    await client.aclose()


@pytest.fixture
async def typesafe_client(typesafe_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=typesafe_server.url)
    yield client
    await client.aclose()


@pytest.fixture
async def decision_model_client(decision_model_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=decision_model_server.url)
    yield client
    await client.aclose()


@pytest.fixture
def llm(stubs: Stubs) -> LLMStub:
    """The chat-completions rulebook. Declare what the model says before acting."""
    return stubs.llm


@pytest.fixture(autouse=True)
def _reset_stubs(stubs: Stubs) -> Iterator[None]:
    stubs.reset()
    yield
    # A net, not the mechanism: ``async with llm.hold(...)`` releases inside the story,
    # which is what keeps a parked call from outliving the bank it belongs to. This
    # catches the story that never entered the block, or died before it — the stub
    # server and its rulebook are session-scoped, so one wedged request would
    # otherwise reach the next test.
    stubs.llm.release_all()


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
