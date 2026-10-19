"""Recall reranked by a real cross-encoder running in-process on ONNX Runtime (#5567).

Opt in with a local cross-encoder snapshot laid out like the Hugging Face repos
(``onnx/model.onnx`` plus the tokenizer files at the root), e.g. the
``cross-encoder/ms-marco-MiniLM-L-6-v2`` snapshot, via
HINDSIGHT_TEST_ONNX_RERANKER_MODEL_DIR. Only the LLM and embedder are scripted
through the HTTP stub; the tokenizer, ORT session, DB and API are real, so the
ranking below is the model's, not the stub's.
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import start_hindsight_server, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.server import HindsightServer

MODEL_DIR = os.environ.get("HINDSIGHT_TEST_ONNX_RERANKER_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="requires a local ONNX cross-encoder snapshot")


@pytest.fixture
def onnx_reranker_server(stub_server, tmp_path) -> Iterator[HindsightServer]:
    root = Path(MODEL_DIR).resolve()
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "onnx-reranker-server.log",
        extra_env={
            "UV_NO_SYNC": "1",
            "HINDSIGHT_API_RERANKER_PROVIDER": "onnx",
            "HINDSIGHT_API_RERANKER_ONNX_MODEL_PATH": str(root / "onnx/model.onnx"),
            "HINDSIGHT_API_RERANKER_ONNX_TOKENIZER_NAME_OR_PATH": str(root),
            # Two pairs per pass, so the three candidates span more than one batch.
            "HINDSIGHT_API_RERANKER_ONNX_BATCH_SIZE": "2",
            "HINDSIGHT_API_RERANKER_ONNX_THREADS": "2",
        },
    )
    try:
        yield server
    finally:
        server.stop()


async def test_recall_is_ranked_by_the_onnx_cross_encoder(onnx_reranker_server, llm):
    client = Hindsight(base_url=onnx_reranker_server.url)
    bank = f"systest-onnx-rerank-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice plays the cello", who="Alice", entities=["Alice", "cello"]),
            fact("Alice rides a blue bicycle", who="Alice", entities=["Alice", "bicycle"]),
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    try:
        await client.aretain(bank_id=bank, content="Alice lives in Berlin, plays cello and cycles.")
        await wait_until_settled(client, bank)

        result = await client.arecall(bank_id=bank, query="Which city does Alice live in?", types=["world"])

        assert [r.text for r in result.results] == [
            "Alice lives in Berlin | Involving: Alice",
            "Alice rides a blue bicycle | Involving: Alice",
            "Alice plays the cello | Involving: Alice",
        ]
        rerank = [r.scores.reranker for r in result.results]
        assert all(score is not None and 0.0 < score < 1.0 for score in rerank)
        # The answer is the only relevant candidate: the model says so, the others it rejects.
        assert rerank[0] > 0.5 > max(rerank[1:])
    finally:
        with suppress(Exception):
            await client.banks.delete_bank(bank)
        await client.aclose()
