"""Coverage for the lazy bank-create fast path in ``MemoryEngine._ensure_bank_exists``.

The contract under test:

* An already-existing bank costs one bare SELECT — no write statement, no
  ``validate_create_bank`` call, no surrounding transaction.
* A missing bank probes once, validates once, then issues one idempotent insert.
* A rejected ``validate_create_bank`` leaves no bank row behind.
* Concurrent callers may all validate, but exactly one wins the insert and
  exactly one applies the default bank template.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import bank_utils
from hindsight_api.extensions import CreateBankContext, OperationValidationError, ValidationResult


class TrackingValidator:
    """Records ``validate_create_bank`` calls, optionally rejecting them all."""

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.create_bank_calls: list[CreateBankContext] = []

    async def validate_create_bank(self, ctx: CreateBankContext) -> ValidationResult:
        self.create_bank_calls.append(ctx)
        if self.reject:
            return ValidationResult.reject("bank creation not allowed", status_code=403)
        return ValidationResult.accept()


@pytest_asyncio.fixture
async def bank_name(memory: MemoryEngine, request_context: RequestContext) -> AsyncIterator[Callable[[str], str]]:
    """Mint unique bank ids and drop whatever they named at teardown."""
    created: list[str] = []

    def _mint(prefix: str) -> str:
        bank_id = f"test-{prefix}-{uuid.uuid4().hex[:8]}"
        created.append(bank_id)
        return bank_id

    yield _mint

    # Cleanup runs unvalidated: the tracking validators here implement only
    # validate_create_bank.
    memory._operation_validator = None
    for bank_id in created:
        await memory.delete_bank(bank_id, request_context=request_context)


def spy_on(module_or_obj, name: str, bank_id: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``name`` with a pass-through that appends to the returned list for ``bank_id``.

    The wrapped callable takes the bank id as its second positional argument,
    which is true of every ``bank_utils`` helper spied on here.
    """
    calls: list[str] = []
    real = getattr(module_or_obj, name)

    async def _spy(first, b_id, *args, **kwargs):
        if b_id == bank_id:
            calls.append(b_id)
        return await real(first, b_id, *args, **kwargs)

    monkeypatch.setattr(module_or_obj, name, _spy)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("with_validator", [True, False], ids=["validator", "no-validator"])
async def test_existing_bank_is_read_only_and_skips_validation(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
    with_validator: bool,
) -> None:
    """An existing bank issues no insert and never reaches ``validate_create_bank``."""
    bank_id = bank_name("exists")
    backend = await memory._get_backend()
    await bank_utils.create_bank_if_missing(backend, bank_id)

    validator = TrackingValidator() if with_validator else None
    memory._operation_validator = validator

    inserts = spy_on(bank_utils, "create_bank_row_on_conn", bank_id, monkeypatch)

    assert await memory._ensure_bank_exists(bank_id, request_context) is False
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            assert await memory._ensure_bank_exists(bank_id, request_context, conn=conn) is False

    assert inserts == [], "An existing bank must not issue an insert on either path"
    if validator is not None:
        assert validator.create_bank_calls == [], "An existing bank must not be re-validated"


@pytest.mark.asyncio
@pytest.mark.parametrize("on_conn", [True, False], ids=["on-conn", "own-conn"])
async def test_missing_bank_probes_once_then_inserts_once(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
    on_conn: bool,
) -> None:
    """A missing bank costs one existence probe, one validation and one insert."""
    bank_id = bank_name("missing")
    backend = await memory._get_backend()

    validator = TrackingValidator()
    memory._operation_validator = validator

    probes = spy_on(bank_utils, "bank_exists_on_conn" if on_conn else "bank_exists", bank_id, monkeypatch)
    inserts = spy_on(bank_utils, "create_bank_row_on_conn", bank_id, monkeypatch)

    async def ensure() -> bool:
        if not on_conn:
            return await memory._ensure_bank_exists(bank_id, request_context)
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                return await memory._ensure_bank_exists(bank_id, request_context, conn=conn)

    assert await ensure() is True
    assert len(validator.create_bank_calls) == 1
    assert validator.create_bank_calls[0].bank_id == bank_id
    assert len(probes) == 1, "Expected exactly one existence probe before validation"
    assert len(inserts) == 1, "Expected exactly one insert"

    # The bank now exists: the second call returns on the probe.
    assert await ensure() is False
    assert len(validator.create_bank_calls) == 1, "Existing bank must not trigger validation"
    assert len(probes) == 2
    assert len(inserts) == 1, "Existing bank must not re-insert"


@pytest.mark.asyncio
async def test_validator_rejection_creates_no_row(
    memory: MemoryEngine, request_context: RequestContext, bank_name: Callable[[str], str]
) -> None:
    """A rejected ``validate_create_bank`` leaves the bank uncreated."""
    bank_id = bank_name("reject")
    backend = await memory._get_backend()

    validator = TrackingValidator(reject=True)
    memory._operation_validator = validator

    with pytest.raises(OperationValidationError, match="bank creation not allowed"):
        await memory._ensure_bank_exists(bank_id, request_context)

    assert len(validator.create_bank_calls) == 1
    assert await bank_utils.bank_exists(backend, bank_id) is False


@pytest.mark.asyncio
async def test_concurrent_creation_applies_the_template_once(
    memory: MemoryEngine,
    request_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
) -> None:
    """Two racing callers produce one created=True and one template application."""
    bank_id = bank_name("concurrent")
    backend = await memory._get_backend()
    memory._operation_validator = TrackingValidator()

    templates: list[str] = []
    real_apply = memory._apply_default_bank_template

    async def spy_apply(b_id, ctx):
        if b_id == bank_id:
            templates.append(b_id)
        return await real_apply(b_id, ctx)

    monkeypatch.setattr(memory, "_apply_default_bank_template", spy_apply)

    results = await asyncio.gather(
        memory._ensure_bank_exists(bank_id, request_context),
        memory._ensure_bank_exists(bank_id, request_context),
    )

    assert sorted(results) == [False, True], "Exactly one caller must win the insert"
    assert templates == [bank_id], "Default bank template must be applied exactly once"
    assert await bank_utils.bank_exists(backend, bank_id) is True


class _CreationRace:
    """Drives the interleaving the fast path cannot serialise.

    Caller 2 probes the bank as missing and then stalls inside its validator
    until caller 1 has committed the insert. Caller 2 therefore reaches its own
    insert against a bank that already exists — the case that decides whether
    losing a creation race is reported as ``created=False`` or corrupts the row.
    """

    def __init__(self, memory: MemoryEngine, bank_id: str, monkeypatch: pytest.MonkeyPatch):
        self.memory = memory
        self.bank_id = bank_id
        self.validate_calls: list[str] = []
        self.templates: list[str] = []
        self.caller1_committed = asyncio.Event()
        self.caller2_probed = asyncio.Event()
        self.rc1 = RequestContext(tenant_id="tenant-caller-1")
        self.rc2 = RequestContext(tenant_id="tenant-caller-2")
        self._monkeypatch = monkeypatch

    def arm(self, *, reject_caller2: bool) -> None:
        real_apply = self.memory._apply_default_bank_template

        async def spy_apply(b_id, ctx):
            if b_id == self.bank_id:
                self.templates.append(b_id)
            return await real_apply(b_id, ctx)

        self._monkeypatch.setattr(self.memory, "_apply_default_bank_template", spy_apply)

        real_probe = bank_utils.bank_exists

        async def spy_probe(pool, b_id):
            result = await real_probe(pool, b_id)
            if b_id == self.bank_id:
                self.caller2_probed.set()
            return result

        self._monkeypatch.setattr(bank_utils, "bank_exists", spy_probe)

        race = self

        class _Validator:
            async def validate_create_bank(self, ctx: CreateBankContext) -> ValidationResult:
                race.validate_calls.append(ctx.bank_id)
                if ctx.request_context and ctx.request_context.tenant_id == "tenant-caller-2":
                    await race.caller1_committed.wait()
                    if reject_caller2:
                        return ValidationResult.reject("caller 2 quota exhausted", status_code=403)
                return ValidationResult.accept()

        self.memory._operation_validator = _Validator()

    def start(self) -> tuple[asyncio.Task, asyncio.Task]:
        async def caller1() -> bool:
            await self.caller2_probed.wait()
            try:
                return await self.memory._ensure_bank_exists(self.bank_id, self.rc1)
            finally:
                self.caller1_committed.set()

        async def caller2() -> bool:
            return await self.memory._ensure_bank_exists(self.bank_id, self.rc2)

        # Caller 2 starts first so its probe is what unblocks caller 1.
        task2 = asyncio.create_task(caller2())
        task1 = asyncio.create_task(caller1())
        return task1, task2


@pytest.mark.asyncio
async def test_losing_a_creation_race_reports_not_created(
    memory: MemoryEngine,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
) -> None:
    """The caller whose insert loses to a committed row reports created=False."""
    bank_id = bank_name("race")
    backend = await memory._get_backend()

    race = _CreationRace(memory, bank_id, monkeypatch)
    race.arm(reject_caller2=False)
    task1, task2 = race.start()
    created1, created2 = await asyncio.gather(task1, task2)

    assert created1 is True, "Caller 1 won the insert"
    assert created2 is False, "Caller 2 lost the insert and must not claim creation"
    assert race.validate_calls == [bank_id, bank_id], "Both callers validate — the documented contract"
    assert race.templates == [bank_id], "Default bank template must be applied exactly once"
    assert await bank_utils.bank_exists(backend, bank_id) is True


@pytest.mark.asyncio
async def test_late_rejection_leaves_the_winners_bank_intact(
    memory: MemoryEngine,
    monkeypatch: pytest.MonkeyPatch,
    bank_name: Callable[[str], str],
) -> None:
    """Rejecting the losing caller must not undo the bank the winner created."""
    bank_id = bank_name("race-reject")
    backend = await memory._get_backend()

    race = _CreationRace(memory, bank_id, monkeypatch)
    race.arm(reject_caller2=True)
    task1, task2 = race.start()

    assert await task1 is True
    with pytest.raises(OperationValidationError, match="caller 2 quota exhausted"):
        await task2

    assert race.validate_calls == [bank_id, bank_id]
    assert race.templates == [bank_id]
    assert await bank_utils.bank_exists(backend, bank_id) is True


@pytest.mark.asyncio
async def test_bank_utils_primitives(memory: MemoryEngine, bank_name: Callable[[str], str]) -> None:
    """The extracted primitives behave standalone, including the ON CONFLICT arm."""
    bank_id = bank_name("primitives")
    backend = await memory._get_backend()

    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            assert await bank_utils.bank_exists_on_conn(conn, bank_id) is False
            assert await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id) is None

            assert await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops) is True
            # ON CONFLICT DO NOTHING: the second insert reports no creation.
            assert await bank_utils.create_bank_row_on_conn(conn, bank_id, ops=backend.ops) is False

            assert await bank_utils.bank_exists_on_conn(conn, bank_id) is True
            profile = await bank_utils.get_bank_profile_if_exists_on_conn(conn, bank_id)
            assert profile is not None
            assert profile["name"] == bank_id
