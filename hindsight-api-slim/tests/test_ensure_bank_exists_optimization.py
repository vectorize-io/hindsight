"""Tests for _ensure_bank_exists optimization and bank_utils primitives.

Verifies that:
1. Both validator-enabled and default non-validator paths execute a SELECT-first check
   and do not issue write statements for already existing banks.
2. Existing banks do not trigger validate_create_bank or duplicate queries.
3. Missing banks execute exactly one pre-check query and one atomic insert.
4. Concurrent creation is idempotent and applies default template exactly once.
5. Validator rejection prevents bank row insertion.
6. Primitives in bank_utils are properly typed and behave as expected.
"""

import asyncio
import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import bank_utils
from hindsight_api.extensions import CreateBankContext, OperationValidationError, ValidationResult


class TrackingValidator:
    """Validator that tracks validate_create_bank invocations and can reject."""

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.create_bank_calls: list[CreateBankContext] = []

    async def validate_create_bank(self, ctx: CreateBankContext) -> ValidationResult:
        self.create_bank_calls.append(ctx)
        if self.reject:
            return ValidationResult.reject("bank creation not allowed", status_code=403)
        return ValidationResult.accept()


@pytest.mark.asyncio
async def test_ensure_bank_exists_skips_validation_and_duplicate_query_for_existing_bank(
    memory: MemoryEngine, request_context: RequestContext
) -> None:
    """When bank exists, _ensure_bank_exists runs only 1 query and skips validate_create_bank."""
    bank_id = f"test-opt-exist-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    # Pre-create the bank
    await bank_utils.create_bank_if_missing(backend, bank_id)

    validator = TrackingValidator()
    memory._operation_validator = validator

    # Call _ensure_bank_exists without conn
    created = await memory._ensure_bank_exists(bank_id, request_context)
    assert created is False
    assert len(validator.create_bank_calls) == 0

    # Call _ensure_bank_exists with conn
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            created_on_conn = await memory._ensure_bank_exists(bank_id, request_context, conn=conn)
            assert created_on_conn is False
    assert len(validator.create_bank_calls) == 0


@pytest.mark.asyncio
async def test_ensure_bank_exists_non_validator_path_is_read_only_for_existing_bank(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under default (no-validator) mode, existing banks perform only a SELECT and no INSERT."""
    bank_id = f"test-opt-novalidator-exist-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    # Pre-create the bank
    await bank_utils.create_bank_if_missing(backend, bank_id)

    memory._operation_validator = None

    create_row_calls = 0
    real_create_row = bank_utils.create_bank_row_on_conn

    async def spy_create_row(conn, b_id, *, ops):
        nonlocal create_row_calls
        if b_id == bank_id:
            create_row_calls += 1
        return await real_create_row(conn, b_id, ops=ops)

    monkeypatch.setattr(bank_utils, "create_bank_row_on_conn", spy_create_row)

    # Calling on existing bank without conn must NOT call create_bank_row_on_conn
    created = await memory._ensure_bank_exists(bank_id, request_context)
    assert created is False
    assert create_row_calls == 0, "Existing bank in non-validator path must not execute insert"

    # Calling on existing bank with conn must NOT call create_bank_row_on_conn
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            created_on_conn = await memory._ensure_bank_exists(bank_id, request_context, conn=conn)
            assert created_on_conn is False
    assert create_row_calls == 0, "Existing bank in non-validator on-conn path must not execute insert"


@pytest.mark.asyncio
async def test_ensure_bank_exists_missing_bank_query_count(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When bank is missing, _ensure_bank_exists validates and inserts without duplicate SELECT."""
    bank_id = f"test-opt-missing-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    validator = TrackingValidator()
    memory._operation_validator = validator

    # Spy on get_bank_profile_if_exists to ensure it is called only once per _ensure_bank_exists invocation
    real_get_if_exists = bank_utils.get_bank_profile_if_exists
    get_if_exists_calls = 0

    async def spy_get_if_exists(pool, b_id):
        nonlocal get_if_exists_calls
        if b_id == bank_id:
            get_if_exists_calls += 1
        return await real_get_if_exists(pool, b_id)

    # Spy on create_bank_row_on_conn to ensure it is called directly
    real_create_row = bank_utils.create_bank_row_on_conn
    create_row_calls = 0

    async def spy_create_row(conn, b_id, *, ops):
        nonlocal create_row_calls
        if b_id == bank_id:
            create_row_calls += 1
        return await real_create_row(conn, b_id, ops=ops)

    monkeypatch.setattr(bank_utils, "get_bank_profile_if_exists", spy_get_if_exists)
    monkeypatch.setattr(bank_utils, "create_bank_row_on_conn", spy_create_row)

    # Execute _ensure_bank_exists (without conn)
    created = await memory._ensure_bank_exists(bank_id, request_context)

    assert created is True
    assert len(validator.create_bank_calls) == 1
    assert validator.create_bank_calls[0].bank_id == bank_id
    assert get_if_exists_calls == 1, "Expected exactly 1 existence query before validation"
    assert create_row_calls == 1, "Expected exactly 1 atomic insert call"

    # Subsequent call on now-existing bank: no validate, no insert
    created_again = await memory._ensure_bank_exists(bank_id, request_context)
    assert created_again is False
    assert len(validator.create_bank_calls) == 1, "Existing bank must not trigger validation"
    assert get_if_exists_calls == 2, "Expected 2 total existence checks across the two invocations"
    assert create_row_calls == 1, "Insert must not be called again"


@pytest.mark.asyncio
async def test_ensure_bank_exists_on_conn_missing_bank_query_count(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When bank is missing and conn is provided, _ensure_bank_exists executes 1 SELECT and 1 INSERT."""
    bank_id = f"test-opt-conn-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    validator = TrackingValidator()
    memory._operation_validator = validator

    create_row_calls = 0
    real_create_row = bank_utils.create_bank_row_on_conn

    async def spy_create_row(conn, b_id, *, ops):
        nonlocal create_row_calls
        if b_id == bank_id:
            create_row_calls += 1
        return await real_create_row(conn, b_id, ops=ops)

    monkeypatch.setattr(bank_utils, "create_bank_row_on_conn", spy_create_row)

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            created = await memory._ensure_bank_exists(bank_id, request_context, conn=conn)

    assert created is True
    assert len(validator.create_bank_calls) == 1
    assert create_row_calls == 1

    # Verify on-conn check on existing bank returns False and does not call create_row
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            created_again = await memory._ensure_bank_exists(bank_id, request_context, conn=conn)

    assert created_again is False
    assert len(validator.create_bank_calls) == 1
    assert create_row_calls == 1


@pytest.mark.asyncio
async def test_ensure_bank_exists_validator_rejection_prevents_insert(
    memory: MemoryEngine, request_context: RequestContext
) -> None:
    """When validate_create_bank rejects, no bank row is created in the database."""
    bank_id = f"test-opt-reject-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    validator = TrackingValidator(reject=True)
    memory._operation_validator = validator

    with pytest.raises(OperationValidationError) as exc_info:
        await memory._ensure_bank_exists(bank_id, request_context)

    assert "bank creation not allowed" in str(exc_info.value)
    assert len(validator.create_bank_calls) == 1

    # Ensure no row was created
    profile = await bank_utils.get_bank_profile_if_exists(backend, bank_id)
    assert profile is None


@pytest.mark.asyncio
async def test_ensure_bank_exists_concurrent_creation_applies_template_once(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent calls to _ensure_bank_exists result in exactly one created=True and one template application."""
    bank_id = f"test-opt-concurrent-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    validator = TrackingValidator()
    memory._operation_validator = validator

    template_applied_count = 0
    real_apply_template = memory._apply_default_bank_template

    async def spy_apply_template(b_id, ctx):
        nonlocal template_applied_count
        if b_id == bank_id:
            template_applied_count += 1
        return await real_apply_template(b_id, ctx)

    monkeypatch.setattr(memory, "_apply_default_bank_template", spy_apply_template)

    # Launch two concurrent _ensure_bank_exists calls
    results = await asyncio.gather(
        memory._ensure_bank_exists(bank_id, request_context),
        memory._ensure_bank_exists(bank_id, request_context),
    )

    # Exactly one must return True (created) and one False (already created by the other)
    assert sorted(results) == [False, True]
    assert template_applied_count == 1, "Default bank template must be applied exactly once"

    # Verify bank profile exists
    profile = await bank_utils.get_bank_profile_if_exists(backend, bank_id)
    assert profile is not None
    assert profile["name"] == bank_id


@pytest.mark.asyncio
async def test_ensure_bank_exists_creation_race_barrier_delayed_validation(
    memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Barrier test: Caller 2 observes missing bank, Caller 1 commits insert, then Caller 2 validates & inserts."""
    bank_id = f"test-opt-race-barrier-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    rc1 = RequestContext(tenant_id="tenant-caller-1")
    rc2 = RequestContext(tenant_id="tenant-caller-2")

    validator = TrackingValidator()
    memory._operation_validator = validator

    template_applied_count = 0
    real_apply_template = memory._apply_default_bank_template

    async def spy_apply_template(b_id, ctx):
        nonlocal template_applied_count
        if b_id == bank_id:
            template_applied_count += 1
        return await real_apply_template(b_id, ctx)

    monkeypatch.setattr(memory, "_apply_default_bank_template", spy_apply_template)

    caller1_committed_event = asyncio.Event()
    caller2_probed_missing_event = asyncio.Event()

    real_get_if_exists = bank_utils.get_bank_profile_if_exists

    async def spy_get_if_exists(pool, b_id):
        res = await real_get_if_exists(pool, b_id)
        if b_id == bank_id and not caller2_probed_missing_event.is_set():
            caller2_probed_missing_event.set()
        return res

    monkeypatch.setattr(bank_utils, "get_bank_profile_if_exists", spy_get_if_exists)

    real_validate = validator.validate_create_bank

    async def synchronized_validate(ctx: CreateBankContext) -> ValidationResult:
        if ctx.request_context and ctx.request_context.tenant_id == "tenant-caller-2":
            # Caller 2: pause inside validation until Caller 1 has committed its insert
            await caller1_committed_event.wait()
        return await real_validate(ctx)

    validator.validate_create_bank = synchronized_validate

    async def run_caller1():
        # Wait until Caller 2 has probed missing bank state
        await caller2_probed_missing_event.wait()
        res = await memory._ensure_bank_exists(bank_id, rc1)
        caller1_committed_event.set()
        return res

    async def run_caller2():
        return await memory._ensure_bank_exists(bank_id, rc2)

    # Start Caller 2 first so its probe runs before Caller 1 starts
    task2 = asyncio.create_task(run_caller2())
    task1 = asyncio.create_task(run_caller1())

    res2, res1 = await asyncio.gather(task2, task1)

    assert res1 is True, "Caller 1 was the winning insert and should report created=True"
    assert res2 is False, "Caller 2 was the losing insert and should report created=False"
    assert len(validator.create_bank_calls) == 2, "Both concurrent callers executed validation"
    assert template_applied_count == 1, "Default bank template must be applied exactly once"

    # Verify exactly 1 bank in database
    profile = await bank_utils.get_bank_profile_if_exists(backend, bank_id)
    assert profile is not None
    assert profile["name"] == bank_id


@pytest.mark.asyncio
async def test_ensure_bank_exists_creation_race_barrier_delayed_validation_rejected(
    memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Barrier test: Caller 2 observes missing bank, Caller 1 commits, Caller 2 delayed validation is rejected."""
    bank_id = f"test-opt-race-reject-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    rc1 = RequestContext(tenant_id="tenant-caller-1")
    rc2 = RequestContext(tenant_id="tenant-caller-2")

    validator = TrackingValidator()
    memory._operation_validator = validator

    template_applied_count = 0
    real_apply_template = memory._apply_default_bank_template

    async def spy_apply_template(b_id, ctx):
        nonlocal template_applied_count
        if b_id == bank_id:
            template_applied_count += 1
        return await real_apply_template(b_id, ctx)

    monkeypatch.setattr(memory, "_apply_default_bank_template", spy_apply_template)

    caller1_committed_event = asyncio.Event()
    caller2_probed_missing_event = asyncio.Event()

    real_get_if_exists = bank_utils.get_bank_profile_if_exists

    async def spy_get_if_exists(pool, b_id):
        res = await real_get_if_exists(pool, b_id)
        if b_id == bank_id and not caller2_probed_missing_event.is_set():
            caller2_probed_missing_event.set()
        return res

    monkeypatch.setattr(bank_utils, "get_bank_profile_if_exists", spy_get_if_exists)

    async def synchronized_validate(ctx: CreateBankContext) -> ValidationResult:
        validator.create_bank_calls.append(ctx)
        if ctx.request_context and ctx.request_context.tenant_id == "tenant-caller-2":
            # Caller 2: wait until Caller 1 has committed, then reject
            await caller1_committed_event.wait()
            return ValidationResult.reject("caller 2 quota exhausted", status_code=403)
        return ValidationResult.accept()

    validator.validate_create_bank = synchronized_validate

    async def run_caller1():
        await caller2_probed_missing_event.wait()
        res = await memory._ensure_bank_exists(bank_id, rc1)
        caller1_committed_event.set()
        return res

    async def run_caller2():
        return await memory._ensure_bank_exists(bank_id, rc2)

    task2 = asyncio.create_task(run_caller2())
    task1 = asyncio.create_task(run_caller1())

    res1 = await task1
    assert res1 is True, "Caller 1 created bank successfully"

    with pytest.raises(OperationValidationError) as exc_info:
        await task2

    assert "caller 2 quota exhausted" in str(exc_info.value)
    assert len(validator.create_bank_calls) == 2
    assert template_applied_count == 1, "Default bank template applied by Caller 1"

    # Bank created by Caller 1 remains intact in database
    profile = await bank_utils.get_bank_profile_if_exists(backend, bank_id)
    assert profile is not None
    assert profile["name"] == bank_id


@pytest.mark.asyncio
async def test_bank_utils_primitives_direct(memory: MemoryEngine) -> None:
    """Test get_bank_profile_if_exists_on_conn and create_bank_row_on_conn directly."""
    bank_id = f"test-opt-primitives-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            # Initially missing
            profile = await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id)
            assert profile is None

            # First create: created=True
            created = await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops)
            assert created is True

            # Second create (same tx or after): created=False (ON CONFLICT DO NOTHING)
            created_again = await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops)
            assert created_again is False

            # Now exists
            profile_after = await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id)
            assert profile_after is not None
            assert profile_after["name"] == bank_id
