"""Operation-validator hooks for memory curation (edit / invalidate / revert).

``validate_memory_update`` gates a curation before any work runs and
``on_memory_update_complete`` reports what committed, including the text that was
re-embedded, so an extension can quota or meter curation the way it does retain.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine, count_tokens
from hindsight_api.engine.retain import embedding_processing
from hindsight_api.extensions import (
    MemoryUpdateContext,
    MemoryUpdateResult,
    OperationValidationError,
    OperationValidatorExtension,
    ValidationResult,
)

# Seeds with a direct INSERT into `memory_units`, which an alternative MEMORIES
# store would not see (same reason as test_curation_http.py).
pytestmark = pytest.mark.memory_backend_incompatible


class RecordingValidator(OperationValidatorExtension):
    def __init__(self, reject: ValidationResult | None = None, fail_complete: bool = False):
        super().__init__({})
        self.reject = reject
        self.fail_complete = fail_complete
        self.validated: list[MemoryUpdateContext] = []
        self.completed: list[MemoryUpdateResult] = []

    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def validate_memory_update(self, ctx):
        self.validated.append(ctx)
        return self.reject or ValidationResult.accept()

    async def on_memory_update_complete(self, result):
        self.completed.append(result)
        if self.fail_complete:
            raise RuntimeError("metering backend down")


async def _insert_fact(memory: MemoryEngine, bank_id: str, text: str) -> str:
    """Insert one world fact with a real embedding; returns its id."""
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    mem_id = uuid.uuid4()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, created_at, updated_at, consolidated_at)
            VALUES ($1, $2, $3, 'world', $4::vector, NOW(), NOW(), NOW(), NOW())
            """,
            mem_id,
            bank_id,
            text,
            str(emb[0]),
        )
    return str(mem_id)


@pytest_asyncio.fixture
async def validated_memory(memory):
    original = memory._operation_validator
    yield memory
    memory._operation_validator = original


@pytest.mark.asyncio
async def test_edit_reports_the_reembedded_text(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "srv-04 runs PostgreSQL 14.")
    validator = RecordingValidator()
    memory._operation_validator = validator
    ctx = RequestContext()

    new_text = "srv-04 runs PostgreSQL 16 since the September upgrade."
    await memory.update_memory_unit(bank_id, mem_id, text=new_text, request_context=ctx)

    assert len(validator.validated) == 1
    pre = validator.validated[0]
    assert (pre.bank_id, pre.memory_id, pre.text, pre.state, pre.edits_fields) == (
        bank_id,
        mem_id,
        new_text,
        None,
        True,
    )
    assert len(validator.completed) == 1
    post = validator.completed[0]
    assert post.action == "edit"
    assert post.reembedded_text == new_text
    assert post.reembedded_tokens == count_tokens(new_text) > 0

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_invalidate_reembeds_nothing_and_revert_reembeds_the_restored_text(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    original_text = "The billing service is owned by the payments team."
    mem_id = await _insert_fact(memory, bank_id, original_text)
    validator = RecordingValidator()
    memory._operation_validator = validator
    ctx = RequestContext()

    await memory.update_memory_unit(bank_id, mem_id, state="invalidated", reason="wrong team", request_context=ctx)
    await memory.update_memory_unit(bank_id, mem_id, state="valid", request_context=ctx)

    assert [v.edits_fields for v in validator.validated] == [False, False]
    assert [v.state for v in validator.validated] == ["invalidated", "valid"]
    invalidate, revert = validator.completed
    assert (invalidate.action, invalidate.reembedded_text, invalidate.reembedded_tokens) == ("invalidate", None, 0)
    assert revert.action == "revert"
    assert revert.reembedded_text == original_text
    assert revert.reembedded_tokens == count_tokens(original_text) > 0

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_edit_and_invalidate_reports_the_state_change_and_the_edited_text(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "The cache TTL is 5 minutes.")
    validator = RecordingValidator()
    memory._operation_validator = validator
    ctx = RequestContext()

    new_text = "The cache TTL is 10 minutes."
    await memory.update_memory_unit(
        bank_id, mem_id, text=new_text, state="invalidated", reason="obsolete", request_context=ctx
    )

    (post,) = validator.completed
    # The edit was embedded before the row was archived, so its text is still metered.
    assert (post.action, post.reembedded_text, post.reembedded_tokens) == (
        "invalidate",
        new_text,
        count_tokens(new_text),
    )

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_reason_only_update_reembeds_nothing(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "The office moved to Ottawa.")
    ctx = RequestContext()
    await memory.update_memory_unit(bank_id, mem_id, state="invalidated", reason="stale", request_context=ctx)

    validator = RecordingValidator()
    memory._operation_validator = validator
    # Re-invalidating an already invalidated memory with a new reason only updates the reason.
    await memory.update_memory_unit(
        bank_id, mem_id, state="invalidated", reason="superseded by the Toronto move", request_context=ctx
    )

    (post,) = validator.completed
    assert (post.action, post.reembedded_text, post.reembedded_tokens) == ("reason", None, 0)

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_rejected_curation_changes_nothing_and_skips_the_completion_hook(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    original_text = "The API rate limit is 100 requests per minute."
    mem_id = await _insert_fact(memory, bank_id, original_text)
    validator = RecordingValidator(reject=ValidationResult.reject("insufficient credits", status_code=402))
    memory._operation_validator = validator
    ctx = RequestContext()

    with pytest.raises(OperationValidationError) as exc_info:
        await memory.update_memory_unit(bank_id, mem_id, text="The API rate limit is 500.", request_context=ctx)

    assert exc_info.value.status_code == 402
    assert validator.completed == []
    memory._operation_validator = None
    unit = await memory.get_memory_unit(bank_id=bank_id, memory_id=mem_id, request_context=ctx)
    assert unit["text"] == original_text

    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_completion_hook_failure_does_not_fail_the_curation(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "Deploys happen on Tuesdays.")
    validator = RecordingValidator(fail_complete=True)
    memory._operation_validator = validator
    ctx = RequestContext()

    result = await memory.update_memory_unit(bank_id, mem_id, text="Deploys happen on Thursdays.", request_context=ctx)

    assert result is not None and result["text"] == "Deploys happen on Thursdays."
    assert len(validator.completed) == 1

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_unknown_memory_skips_the_completion_hook(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext()
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=ctx)
    validator = RecordingValidator()
    memory._operation_validator = validator

    result = await memory.update_memory_unit(bank_id, str(uuid.uuid4()), text="anything", request_context=ctx)

    assert result is None
    assert validator.completed == []

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=ctx)


@pytest.mark.asyncio
async def test_http_patch_returns_the_validator_status(validated_memory):
    memory = validated_memory
    bank_id = f"curation-hooks-{uuid.uuid4().hex[:8]}"
    mem_id = await _insert_fact(memory, bank_id, "The staging cluster has 3 nodes.")
    memory._operation_validator = RecordingValidator(
        reject=ValidationResult.reject("insufficient credits", status_code=402)
    )
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/v1/default/banks/{bank_id}/memories/{mem_id}", json={"text": "The staging cluster has 5 nodes."}
        )

    assert resp.status_code == 402, resp.text
    assert "insufficient credits" in resp.json()["detail"]

    memory._operation_validator = None
    await memory.delete_bank(bank_id, request_context=RequestContext())
