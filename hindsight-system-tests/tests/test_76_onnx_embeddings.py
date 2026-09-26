"""Real ONNX embeddings across retain, background work, recall and replacement.

Opt in with a local multilingual-e5-small snapshot via HINDSIGHT_TEST_ONNX_MODEL_DIR.
Set HINDSIGHT_TEST_ONNX_CUDA=1 to also exercise CUDA. Explicit GPU runs fail when
unusable; they never silently skip a missing provider. Only the LLM/reranker are
scripted through the existing HTTP stub; the tokenizer, ORT, DB and API are real.
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

MODEL_DIR = os.environ.get("HINDSIGHT_TEST_ONNX_MODEL_DIR")
DEVICES = ["cpu", "cuda"] if os.environ.get("HINDSIGHT_TEST_ONNX_CUDA") == "1" else ["cpu"]
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="requires local E5 ONNX model; see CUDA ONNX recipe README")


@pytest.fixture(params=DEVICES)
def onnx_server(request, stub_server, tmp_path) -> Iterator[HindsightServer]:
    root = Path(MODEL_DIR).resolve()
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "onnx-server.log",
        extra_env={
            "UV_NO_SYNC": "1",
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "onnx",
            "HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_PATH": str(root / "onnx/model.onnx"),
            "HINDSIGHT_API_EMBEDDINGS_ONNX_TOKENIZER_NAME_OR_PATH": str(root),
            "HINDSIGHT_API_EMBEDDINGS_ONNX_DEVICE": request.param,
            "HINDSIGHT_API_EMBEDDINGS_ONNX_CUDA_DEVICE_ID": "0",
            "HINDSIGHT_API_EMBEDDINGS_ONNX_BATCH_SIZE": "2",
            "HINDSIGHT_API_EMBEDDINGS_ONNX_DIMENSIONS": "384",
        },
    )
    try:
        yield server
    finally:
        server.stop()


async def test_real_onnx_memory_lifecycle(onnx_server, llm):
    client = Hindsight(base_url=onnx_server.url)
    bank = f"systest-onnx-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello", who="Alice", entities=["Alice", "cello"]),
            fact("Alice rides a blue bicycle", who="Alice", entities=["Alice", "bicycle"]),
        )
    )
    llm.on_step("extract_facts", contains="Paris").returns(
        extracted(
            fact("Alice now lives in Paris", who="Alice", entities=["Alice", "Paris"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    try:
        await client.aretain(
            bank_id=bank, document_id="residence", content="Alice lives in Berlin, plays cello and cycles."
        )
        await wait_until_settled(client, bank)
        result = await client.arecall(bank_id=bank, query="Where does Alice live?", types=["world"])
        assert result.results[0].text == "Alice lives in Berlin | Involving: Alice"
        assert {r.text for r in result.results} == {
            "Alice lives in Berlin | Involving: Alice",
            "Alice plays the cello | Involving: Alice",
            "Alice rides a blue bicycle | Involving: Alice",
        }
        for row in result.results:
            assert row.document_id == "residence"
            assert row.type == "world"
            assert row.scores.semantic is not None and row.scores.semantic > 0
        await client.aretain(bank_id=bank, document_id="residence", content="Alice now lives in Paris.")
        await wait_until_settled(client, bank)
        replaced = await client.arecall(bank_id=bank, query="Where does Alice live?", types=["world"])
        assert [r.text for r in replaced.results] == ["Alice now lives in Paris | Involving: Alice"]
        assert replaced.results[0].scores.semantic > 0
        document = await client.documents.get_document(bank, "residence")
        assert document.original_text == "Alice now lives in Paris."
    finally:
        with suppress(Exception):
            await client.banks.delete_bank(bank)
        await client.aclose()
