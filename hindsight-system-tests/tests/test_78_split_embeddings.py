"""Recall queries can embed with their own model, while everything stored keeps the main one.

Two deployments want this. One runs the same model on two endpoints, so a long retain
or import no longer queues recall queries behind it (#6078). The other pairs a document
model with a smaller query model trained for its vector space (Qdrant's Constella). Both
come down to the same promise, and it only holds end to end: every stored vector comes
from the passage side, every search vector from the query side, and a memory stored
through one is still found through the other. A query that slipped through to the
document model would rank fine in this suite and quietly defeat the setup in production.

The server here specializes only the query side's model (``QUERY_EMBEDDINGS_MODEL``);
the stub records which model each text was sent to.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests.payloads import extracted, fact

from .conftest import BANK_PREFIX, QUERY_EMBEDDINGS_MODEL

pytestmark = pytest.mark.asyncio

# Unusual names: the stub is shared with every server in the session.
FACT = "Quillon repairs harpsichords in Ljubljana"
QUERY = "What does Quillon repair?"


def _models_for(stubs, text: str) -> set[str]:
    return {call.model for call in stubs.embeddings.calls if any(text in i for i in call.inputs)}


async def test_queries_embed_with_the_query_model_and_still_find_stored_facts(
    split_embeddings_client: Hindsight, llm, stubs
):
    bank = f"{BANK_PREFIX}{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Quillon").returns(
        extracted(fact(FACT, who="Quillon", where="Ljubljana", entities=["Quillon", "Ljubljana"]))
    )
    try:
        await split_embeddings_client.aretain(bank_id=bank, content="Quillon repairs harpsichords in Ljubljana.")
        response = await split_embeddings_client.arecall(bank_id=bank, query=QUERY)
    finally:
        with contextlib.suppress(Exception):
            await split_embeddings_client.banks.delete_bank(bank)

    # Stored through the passage side, searched through the query side, and found.
    assert [r.text.split(" | ")[0] for r in response.results] == [FACT]

    query_models = _models_for(stubs, QUERY)
    stored_models = _models_for(stubs, FACT)
    assert query_models == {QUERY_EMBEDDINGS_MODEL}, query_models
    assert stored_models == {"stub-embeddings"}, stored_models
