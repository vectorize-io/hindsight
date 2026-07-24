"""Regression tests for async Retain submission idempotency."""

import asyncio
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import (
    MemoryEngine,
    RetainIdempotencyConflictError,
    _retain_submission_fingerprint,
)
from hindsight_api.engine.task_backend import WorkerTaskBackend
from hindsight_api.extensions import OperationValidationError, OperationValidatorExtension, ValidationResult


class _NoopEmbeddings:
    provider_name = "test"
    dimension = 384

    async def initialize(self) -> None:
        return None


class _NoopCrossEncoder:
    provider_name = "test"

    async def initialize(self) -> None:
        return None


class _NoopQueryAnalyzer:
    def load(self) -> None:
        return None


@pytest_asyncio.fixture
async def idempotency_memory(pg0_db_url):
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_NoopEmbeddings(),
        cross_encoder=_NoopCrossEncoder(),
        query_analyzer=_NoopQueryAnalyzer(),
        run_migrations=False,
        task_backend=WorkerTaskBackend(),
    )
    await memory.initialize()
    yield memory
    await memory.close()


@pytest_asyncio.fixture
async def idempotency_api_client(idempotency_memory):
    app = create_app(idempotency_memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@dataclass(frozen=True)
class _OperationCounts:
    parents: int
    children: int


async def _operation_counts(memory, bank_id: str) -> _OperationCounts:
    pool = await memory._get_pool()
    parent_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'batch_retain'
        """,
        bank_id,
    )
    child_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'retain'
        """,
        bank_id,
    )
    return _OperationCounts(parents=parent_count, children=child_count)


async def _child_for_parent(memory, parent_operation_id: str):
    pool = await memory._get_pool()
    return await pool.fetchrow(
        """
        SELECT operation_id, blocked_by_operation_id, next_retry_at
        FROM async_operations
        WHERE operation_type = 'retain'
          AND result_metadata->>'parent_operation_id' = $1
        """,
        parent_operation_id,
    )


async def _parent(memory, parent_operation_id: str):
    pool = await memory._get_pool()
    return await pool.fetchrow(
        """
        SELECT operation_id, blocked_by_operation_id, serialization_key, status
        FROM async_operations
        WHERE operation_id = $1
        """,
        parent_operation_id,
    )


@pytest.mark.asyncio
async def test_async_retain_without_key_preserves_create_each_time(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-unkeyed"
    contents = [{"content": "One stable fact.", "document_id": "unkeyed-doc"}]

    first = await memory.submit_async_retain(bank_id, contents, request_context=request_context)
    second = await memory.submit_async_retain(bank_id, contents, request_context=request_context)

    assert first["operation_id"] != second["operation_id"]
    assert await _operation_counts(memory, bank_id) == _OperationCounts(parents=2, children=2)


@pytest.mark.asyncio
async def test_identical_key_returns_original_operation_without_new_work(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-repeat"
    contents = [{"content": "One stable fact.", "document_id": "repeat-doc", "tags": ["b", "a"]}]

    first = await memory.submit_async_retain(
        bank_id,
        contents,
        request_context=request_context,
        idempotency_key="repeat-key",
    )
    second = await memory.submit_async_retain(
        bank_id,
        [{"content": "One stable fact.", "document_id": "repeat-doc", "tags": ["a", "b"]}],
        request_context=request_context,
        idempotency_key="repeat-key",
    )

    assert second["operation_id"] == first["operation_id"]
    assert await _operation_counts(memory, bank_id) == _OperationCounts(parents=1, children=1)


@pytest.mark.asyncio
async def test_same_key_different_payload_fails_without_mutation(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-conflict"
    first = await memory.submit_async_retain(
        bank_id,
        [{"content": "Original fact.", "document_id": "conflict-doc"}],
        request_context=request_context,
        idempotency_key="conflict-key",
    )
    counts_before = await _operation_counts(memory, bank_id)

    with pytest.raises(RetainIdempotencyConflictError, match="different async retain payload"):
        await memory.submit_async_retain(
            bank_id,
            [{"content": "Different fact.", "document_id": "conflict-doc"}],
            request_context=request_context,
            idempotency_key="conflict-key",
        )

    assert first["operation_id"]
    assert await _operation_counts(memory, bank_id) == counts_before


@pytest.mark.asyncio
async def test_retry_still_requires_current_retain_authorization(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-revoked"
    contents = [{"content": "Authorized once.", "document_id": "revoked-doc"}]
    first = await memory.submit_async_retain(
        bank_id,
        contents,
        request_context=request_context,
        idempotency_key="revoked-key",
    )
    counts_before = await _operation_counts(memory, bank_id)

    class _RejectingValidator:
        async def validate_retain(self, ctx):
            return ValidationResult.reject("bank access revoked", status_code=403)

    memory._operation_validator = _RejectingValidator()

    with pytest.raises(OperationValidationError, match="bank access revoked"):
        await memory.submit_async_retain(
            bank_id,
            contents,
            request_context=request_context,
            idempotency_key="revoked-key",
        )

    assert first["operation_id"]
    assert await _operation_counts(memory, bank_id) == counts_before


@pytest.mark.asyncio
async def test_validator_rewrites_do_not_change_caller_payload_identity(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-validator-rewrite"
    contents = [{"content": "Stable caller payload.", "document_id": "validator-doc"}]

    class _StatefulValidator(OperationValidatorExtension):
        calls = 0

        async def validate_retain(self, ctx):
            self.calls += 1
            rewritten = [dict(item) for item in ctx.contents]
            rewritten[0]["content"] = f"validated-version-{self.calls}"
            if self.calls > 1:
                rewritten.append({"content": "extra validator item", "document_id": "validator-extra"})
            return ValidationResult.accept_with(contents=rewritten)

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

    validator = _StatefulValidator({})
    memory._operation_validator = validator

    first = await memory.submit_async_retain(
        bank_id,
        contents,
        request_context=request_context,
        idempotency_key="validator-rewrite-key",
    )
    second = await memory.submit_async_retain(
        bank_id,
        contents,
        request_context=request_context,
        idempotency_key="validator-rewrite-key",
    )

    assert validator.calls == 2
    assert second["operation_id"] == first["operation_id"]
    assert first["items_count"] == second["items_count"] == 1
    assert await _operation_counts(memory, bank_id) == _OperationCounts(parents=1, children=1)


@pytest.mark.asyncio
async def test_validator_enrichment_preserves_same_document_serialization(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-validator-enrichment"

    class _AddingValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept_with(
                contents=[
                    *ctx.contents,
                    {"content": "validator-added", "document_id": f"extra-{ctx.contents[0]['content']}"},
                ]
            )

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

    memory._operation_validator = _AddingValidator({})
    first = await memory.submit_async_retain(
        bank_id,
        [{"content": "first", "document_id": "validator-shared-doc"}],
        request_context=request_context,
        idempotency_key="validator-serialization-1",
    )
    second = await memory.submit_async_retain(
        bank_id,
        [{"content": "second", "document_id": "validator-shared-doc"}],
        request_context=request_context,
        idempotency_key="validator-serialization-2",
    )

    assert str((await _parent(memory, second["operation_id"]))["blocked_by_operation_id"]) == first["operation_id"]


def test_nullable_entity_type_has_a_stable_fingerprint():
    fingerprint = _retain_submission_fingerprint(
        [
            {
                "content": "Entities with optional types.",
                "entities": [
                    {"text": "same", "type": None},
                    {"text": "same", "type": "PERSON"},
                ],
            }
        ],
        None,
        None,
    )

    assert len(fingerprint) == 64


def test_empty_document_tags_match_omitted_document_tags():
    contents = [{"content": "same", "document_id": "doc"}]

    assert _retain_submission_fingerprint(
        contents,
        [],
        None,
    ) == _retain_submission_fingerprint(
        contents,
        None,
        None,
    )


def test_set_like_fields_ignore_duplicate_values():
    one_each = [
        {
            "content": "same",
            "document_id": "doc",
            "tags": ["a"],
            "entities": [{"text": "Vitor", "type": "PERSON"}],
            "observation_scopes": [["work", "project"]],
        }
    ]
    duplicated = [
        {
            "content": "same",
            "document_id": "doc",
            "tags": ["a", "a"],
            "entities": [
                {"text": "Vitor", "type": "PERSON"},
                {"type": "PERSON", "text": "Vitor"},
            ],
            "observation_scopes": [
                ["project", "work"],
                ["work", "project"],
            ],
        }
    ]

    assert _retain_submission_fingerprint(
        one_each,
        ["session"],
        None,
    ) == _retain_submission_fingerprint(
        duplicated,
        ["session", "session"],
        None,
    )


@pytest.mark.asyncio
async def test_unkeyed_response_preserves_processed_item_count(idempotency_memory, request_context):
    memory = idempotency_memory

    class _AddingValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept_with(
                contents=[
                    *ctx.contents,
                    {"content": "validator-added", "document_id": "validator-added-doc"},
                ]
            )

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

    memory._operation_validator = _AddingValidator({})
    result = await memory.submit_async_retain(
        "retain-unkeyed-validator-count",
        [{"content": "caller item", "document_id": "caller-doc"}],
        request_context=request_context,
    )

    assert result["items_count"] == 2


@pytest.mark.asyncio
async def test_concurrent_identical_submissions_converge(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-idempotency-concurrent"
    contents = [{"content": "Concurrent fact.", "document_id": "concurrent-doc"}]

    first, second = await asyncio.gather(
        memory.submit_async_retain(
            bank_id,
            contents,
            request_context=request_context,
            idempotency_key="concurrent-key",
        ),
        memory.submit_async_retain(
            bank_id,
            contents,
            request_context=request_context,
            idempotency_key="concurrent-key",
        ),
    )

    assert first["operation_id"] == second["operation_id"]
    assert await _operation_counts(memory, bank_id) == _OperationCounts(parents=1, children=1)


@pytest.mark.asyncio
async def test_same_document_distinct_requests_are_serialized(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-same-document"
    first = await memory.submit_async_retain(
        bank_id,
        [{"content": "First snapshot.", "document_id": "shared-doc"}],
        request_context=request_context,
        idempotency_key="snapshot-1",
    )
    second = await memory.submit_async_retain(
        bank_id,
        [{"content": "Second snapshot.", "document_id": "shared-doc"}],
        request_context=request_context,
        idempotency_key="snapshot-2",
    )

    first_child = await _child_for_parent(memory, first["operation_id"])
    second_child = await _child_for_parent(memory, second["operation_id"])

    assert first_child["blocked_by_operation_id"] is None
    assert str(second_child["blocked_by_operation_id"]) == first["operation_id"]
    assert second_child["next_retry_at"].year == 9999


@pytest.mark.asyncio
async def test_concurrent_same_document_requests_form_one_serialized_pair(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-concurrent-same-document"
    first, second = await asyncio.gather(
        memory.submit_async_retain(
            bank_id,
            [{"content": "Concurrent snapshot A.", "document_id": "shared-doc"}],
            request_context=request_context,
            idempotency_key="concurrent-snapshot-a",
        ),
        memory.submit_async_retain(
            bank_id,
            [{"content": "Concurrent snapshot B.", "document_id": "shared-doc"}],
            request_context=request_context,
            idempotency_key="concurrent-snapshot-b",
        ),
    )

    first_child = await _child_for_parent(memory, first["operation_id"])
    second_child = await _child_for_parent(memory, second["operation_id"])
    dependencies = {
        first["operation_id"]: first_child["blocked_by_operation_id"],
        second["operation_id"]: second_child["blocked_by_operation_id"],
    }
    unblocked = [operation_id for operation_id, dependency in dependencies.items() if dependency is None]
    blocked = [
        (operation_id, str(dependency)) for operation_id, dependency in dependencies.items() if dependency is not None
    ]

    assert len(unblocked) == 1
    assert len(blocked) == 1
    assert blocked[0][0] != unblocked[0]
    assert blocked[0][1] == unblocked[0]


@pytest.mark.asyncio
async def test_serialization_forms_a_chain_not_a_fan_in(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-chain"
    operations = []
    for number in range(3):
        operations.append(
            await memory.submit_async_retain(
                bank_id,
                [{"content": f"Snapshot {number}.", "document_id": "chain-doc"}],
                request_context=request_context,
                idempotency_key=f"chain-{number}",
            )
        )

    second_child = await _child_for_parent(memory, operations[1]["operation_id"])
    third_child = await _child_for_parent(memory, operations[2]["operation_id"])

    assert str(second_child["blocked_by_operation_id"]) == operations[0]["operation_id"]
    assert str(third_child["blocked_by_operation_id"]) == operations[1]["operation_id"]


@pytest.mark.asyncio
async def test_serialization_tail_does_not_depend_on_created_at(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-created-at-inversion"
    first = await memory.submit_async_retain(
        bank_id,
        [{"content": "first", "document_id": "inverted-doc"}],
        request_context=request_context,
        idempotency_key="inverted-1",
    )
    second = await memory.submit_async_retain(
        bank_id,
        [{"content": "second", "document_id": "inverted-doc"}],
        request_context=request_context,
        idempotency_key="inverted-2",
    )
    pool = await memory._get_pool()
    await pool.execute(
        "UPDATE async_operations SET created_at = created_at - INTERVAL '1 day' WHERE operation_id = $1",
        second["operation_id"],
    )

    third = await memory.submit_async_retain(
        bank_id,
        [{"content": "third", "document_id": "inverted-doc"}],
        request_context=request_context,
        idempotency_key="inverted-3",
    )

    assert str((await _parent(memory, second["operation_id"]))["blocked_by_operation_id"]) == first["operation_id"]
    assert str((await _parent(memory, third["operation_id"]))["blocked_by_operation_id"]) == second["operation_id"]


@pytest.mark.asyncio
async def test_serialized_operations_reject_manual_cancel_and_retry(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-lifecycle"
    operation = await memory.submit_async_retain(
        bank_id,
        [{"content": "snapshot", "document_id": "lifecycle-doc"}],
        request_context=request_context,
        idempotency_key="lifecycle-1",
    )
    child = await _child_for_parent(memory, operation["operation_id"])

    with pytest.raises(OperationValidationError, match="cannot be cancelled"):
        await memory.cancel_operation(
            bank_id,
            str(child["operation_id"]),
            request_context=request_context,
        )

    pool = await memory._get_pool()
    await pool.execute(
        "UPDATE async_operations SET status = 'failed' WHERE operation_id = $1",
        child["operation_id"],
    )
    with pytest.raises(OperationValidationError, match="cannot be retried"):
        await memory.retry_operation(
            bank_id,
            str(child["operation_id"]),
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_different_documents_remain_parallel(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-different-documents"
    first, second = await asyncio.gather(
        memory.submit_async_retain(
            bank_id,
            [{"content": "First document.", "document_id": "doc-a"}],
            request_context=request_context,
            idempotency_key="doc-a-key",
        ),
        memory.submit_async_retain(
            bank_id,
            [{"content": "Second document.", "document_id": "doc-b"}],
            request_context=request_context,
            idempotency_key="doc-b-key",
        ),
    )

    assert (await _child_for_parent(memory, first["operation_id"]))["blocked_by_operation_id"] is None
    assert (await _child_for_parent(memory, second["operation_id"]))["blocked_by_operation_id"] is None


@pytest.mark.asyncio
async def test_terminal_predecessor_releases_child_for_claim(idempotency_memory, request_context):
    memory = idempotency_memory
    bank_id = "retain-serialization-release"
    first = await memory.submit_async_retain(
        bank_id,
        [{"content": "First snapshot.", "document_id": "release-doc"}],
        request_context=request_context,
        idempotency_key="release-1",
    )
    second = await memory.submit_async_retain(
        bank_id,
        [{"content": "Second snapshot.", "document_id": "release-doc"}],
        request_context=request_context,
        idempotency_key="release-2",
    )

    pool = await memory._get_pool()
    await pool.execute(
        """
        UPDATE async_operations
        SET status = 'completed', completed_at = NOW(), updated_at = NOW()
        WHERE operation_id = $1
        """,
        first["operation_id"],
    )
    async with memory._backend.acquire() as conn:
        async with conn.transaction():
            await memory._backend.ops._release_serialized_retain_tasks(conn, "async_operations")

    released = await _child_for_parent(memory, second["operation_id"])
    assert released["blocked_by_operation_id"] is None
    assert released["next_retry_at"] is None


@pytest.mark.asyncio
async def test_http_conflict_is_explicit(idempotency_api_client):
    api_client = idempotency_api_client
    bank_id = "retain-idempotency-http"
    first = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": "Original HTTP fact.", "document_id": "http-doc"}],
            "async": True,
            "idempotency_key": "http-key",
        },
    )
    conflict = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": "Changed HTTP fact.", "document_id": "http-doc"}],
            "async": True,
            "idempotency_key": "http-key",
        },
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert "different async retain payload" in conflict.json()["detail"]


@pytest.mark.asyncio
async def test_http_rejects_nul_in_idempotency_key(idempotency_api_client):
    response = await idempotency_api_client.post(
        "/v1/default/banks/retain-idempotency-nul/memories",
        json={
            "items": [{"content": "Never queued.", "document_id": "nul-doc"}],
            "async": True,
            "idempotency_key": "invalid\u0000key",
        },
    )

    assert response.status_code == 422
    assert "cannot contain NUL bytes" in response.text
