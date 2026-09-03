"""A duplicate mental-model id is a conflict, not an unhandled 500.

Clients that sweep a list of external ids and create a model for each one re-POST
ids they already created. Before this was handled, every such call propagated the
raw ``mental_models_pkey`` violation out of the engine and surfaced as a 500.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.extensions import OperationValidationError


async def test_duplicate_mental_model_id_raises_conflict(memory: MemoryEngine, request_context):
    bank_id = f"test-mm-dup-{uuid.uuid4().hex[:8]}"
    mental_model_id = "account-0013a00001aATOhAAO"
    create = dict(
        bank_id=bank_id,
        name="Account standing",
        source_query="what is the standing of this account",
        content="Generating content...",
        mental_model_id=mental_model_id,
        request_context=request_context,
    )
    # Outside the try: the bank only exists once this has returned, and cleaning up
    # a bank that was never created would mask whatever made the first create fail.
    first = await memory.create_mental_model(**create)
    assert first["id"] == mental_model_id
    try:
        with pytest.raises(OperationValidationError) as excinfo:
            await memory.create_mental_model(**create)
        assert excinfo.value.status_code == 409
        assert "already exists" in excinfo.value.reason
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
