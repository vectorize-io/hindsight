"""Every model call reaches a model call extension, from the API and from the worker.

A deployment that keeps its own per-call record of model spend and content (object
storage, a warehouse) loads a model call extension and expects it to be complete: a
call it never sees is spend nobody can account for. The calls are split across two
processes — sync work (recall, reflect) in the HTTP server, async retain and
consolidation in the worker — and each process hands its own calls to its own copy of
the extension, which buffers them and writes them out only when the process stops.

So the story runs one operation of each kind on a split deployment, stops it, and
reads both files: each call attributed to the bank and operation that made it, with
what went in and what came back.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.server import HindsightServer, start_hindsight_server, start_hindsight_worker

from .conftest import BANK_PREFIX

pytestmark = pytest.mark.asyncio

#: Its own schema, so this worker claims only this story's work.
MODEL_CALLS_SCHEMA = "systest_model_calls"
EXTENSIONS_DIR = Path(__file__).resolve().parent.parent / "server_extensions"
ANSWER = "Theodora lives in Reykjavík."
QUERY = "Where does Theodora live?"


@dataclass
class Deployment:
    server: HindsightServer
    worker: HindsightServer
    api_calls: Path
    worker_calls: Path

    def stop(self) -> None:
        self.worker.stop()
        self.server.stop()


@pytest.fixture
def deployment(stub_server, tmp_path: Path) -> Iterator[Deployment]:
    """A split deployment whose two processes each write their model calls to a file.

    Function-scoped: the files are written on shutdown, so the story stops it itself.
    """
    env = {
        "HINDSIGHT_API_DATABASE_SCHEMA": MODEL_CALLS_SCHEMA,
        "HINDSIGHT_API_MODEL_CALL_EXTENSION": "jsonl_model_calls:JsonlModelCalls",
        "PYTHONPATH": str(EXTENSIONS_DIR),
    }
    api_calls, worker_calls = tmp_path / "api.jsonl", tmp_path / "worker.jsonl"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "server.log",
        extra_env={**env, "HINDSIGHT_API_WORKER_ENABLED": "false", "HINDSIGHT_API_MODEL_CALL_PATH": str(api_calls)},
    )
    worker = start_hindsight_worker(
        stub_url=stub_server.url,
        log_path=tmp_path / "worker.log",
        extra_env={**env, "HINDSIGHT_API_MODEL_CALL_PATH": str(worker_calls)},
    )
    deployed = Deployment(server=server, worker=worker, api_calls=api_calls, worker_calls=worker_calls)
    yield deployed
    deployed.stop()


def _read(path: Path, bank: str) -> list[dict[str, Any]]:
    """The bank's calls from one process's file, as the JSON the extension wrote. Plain JSON
    on purpose: the suite drives the server from outside and does not import its models."""
    return [call for line in path.read_text().splitlines() if (call := json.loads(line))["bank_id"] == bank]


async def test_every_model_call_reaches_the_extension(deployment: Deployment, llm):
    bank = f"{BANK_PREFIX}{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Theodora").returns(
        extracted(fact("Theodora lives in Reykjavík", who="Theodora"))
    )
    llm.on_step("consolidate", contains="Theodora").answers_with(observes("Theodora is settled in Reykjavík"))
    reflect_loop(llm, answer=ANSWER, query=QUERY)

    client = Hindsight(base_url=deployment.server.url)
    try:
        # Worker: async retain extracts and embeds there; consolidation always runs there.
        await client.aretain(bank_id=bank, content="Theodora lives in Reykjavík.", retain_async=True)
        await wait_until_settled(client, bank)
        # HTTP process: recall embeds the query (and reranks); reflect walks its tool loop.
        await client.arecall(bank_id=bank, query=QUERY)
        reflected = await client.areflect(bank_id=bank, query=QUERY)
        assert reflected.text.strip() == ANSWER
        await client.banks.delete_bank(bank)
    finally:
        await client.aclose()
    deployment.stop()

    worker_calls, api_calls = _read(deployment.worker_calls, bank), _read(deployment.api_calls, bank)
    every = worker_calls + api_calls
    assert every and all(c["tenant"] == MODEL_CALLS_SCHEMA and c["success"] for c in every)
    assert all(c["operation"] and c["trace_id"] for c in every), "every call belongs to an operation run"
    assert len({c["id"] for c in every}) == len(every)

    by_step = {(c["operation"], c["scope"]) for c in worker_calls}
    assert {("retain", "embedding_document"), ("consolidation", "consolidation")} <= by_step
    extraction = next(c for c in worker_calls if c["operation"] == "retain" and c["call_type"] == "chat")
    assert any("Theodora lives in Reykjavík." in m["content"] for m in extraction["input"])
    # The model's reply, verbatim: what the stub was told to answer.
    assert [f["what"] for f in json.loads(extraction["output"])["facts"]] == ["Theodora lives in Reykjavík"]

    query = next(c for c in api_calls if c["scope"] == "embedding_query" and c["operation"] == "recall")
    assert query["input"] == {"texts": [QUERY]} and query["input_count"] == 1
    # At most one rerank per recall. None when the memories store ranks the results itself and
    # the engine's reranker never runs; when it does run, it is recorded in full.
    reranks = [c for c in api_calls if c["scope"] == "rerank" and c["trace_id"] == query["trace_id"]]
    assert len(reranks) <= 1
    for rerank in reranks:
        assert rerank["input"]["queries"] == [QUERY]
        assert len(rerank["output"]["scores"]) == len(rerank["input"]["documents"])

    reflect = [c for c in api_calls if c["operation"] == "reflect" and c["call_type"] == "chat"]
    assert len({c["trace_id"] for c in reflect}) == 1, "one reflect, one trace"
    done = [tc for c in reflect for tc in c["tool_calls"] or [] if tc["name"] == "done"]
    # Each tool-calling turn's tool calls are its output; how many turns end in ``done``
    # is the reflect loop's business, not this story's.
    assert done and {tc["arguments"]["answer"] for tc in done} == {ANSWER}
