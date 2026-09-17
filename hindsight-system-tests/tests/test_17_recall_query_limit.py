"""Disabling the query cap must let a long query reach retrieval through the public client."""

from collections.abc import Awaitable, Callable, Iterator

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import HindsightServer, LLMStub, StubServer, start_hindsight_server
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.server import free_port


@pytest.fixture(scope="module")
def hindsight_server(stub_server: StubServer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[HindsightServer]:
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path_factory.mktemp("uncapped-recall") / "server.log",
        extra_env={
            "HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS": "0",
            "HINDSIGHT_API_DATABASE_URL": f"pg0://hindsight-systest-query-cap:{free_port()}",
        },
    )
    try:
        yield server
    finally:
        server.stop()


async def test_zero_query_limit_still_recalls_retained_memories(
    client: Hindsight, llm: LLMStub, bank_id: str, settled: Callable[[str], Awaitable[None]]
) -> None:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin.")
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query="Where does Alice live? " * 150)

    assert [result.text for result in response.results] == ["Alice lives in Berlin | Involving: Alice"]
    assert [result.type for result in response.results] == ["world"]
