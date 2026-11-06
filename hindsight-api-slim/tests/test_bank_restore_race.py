"""Two whole-bank restores into the same fresh bank id: one creates the bank, the other refuses.

Both pass the "target exists" check while neither has committed. Before the fix, the
second one's bank-row insert waited for the first, inserted nothing, and the restore
went on to write a second copy of every fact into the first one's bank (#6335).
"""

import asyncio
import io
import uuid
import zipfile

import pytest

from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.transfer import TransferScope, export_bank
from hindsight_api.engine.transfer.importer import import_bank


async def _facts(memory, bank_id: str, request_context) -> list[str]:
    units = await memory.list_memory_units(bank_id, request_context=request_context)
    return sorted(u["text"] for u in units["items"] if u["fact_type"] != "observation")


@pytest.mark.asyncio
@pytest.mark.parametrize("bank_config", [True, False], ids=["with-config", "without-config"])
async def test_a_second_restore_into_the_same_fresh_bank_is_refused(memory, request_context, bank_config):
    src = f"race-src-{uuid.uuid4().hex[:8]}"
    dst = f"race-dst-{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()
    try:
        await memory.retain_async(
            bank_id=src, content="Alice joined Acme in 2024.", document_id="doc", request_context=request_context
        )
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, src)
        source_facts = await _facts(memory, src, request_context)
        assert source_facts
        scope = TransferScope(bank_config=bank_config)
        second: asyncio.Task | None = None

        async def start_second_restore_while_first_is_open(conn) -> None:
            # The first restore's bank row is written but not committed: the second one
            # finds no bank, then blocks on the first one's row. Wait until it does.
            nonlocal second
            second = asyncio.create_task(
                memory.import_bank_async(archive, request_context, target_bank_id=dst, scope=scope)
            )
            first_pid = await conn.fetchval("SELECT pg_backend_pid()")
            async with acquire_with_retry(backend) as probe:
                for _ in range(200):
                    if await probe.fetchval(
                        "SELECT 1 FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))", first_pid
                    ):
                        return
                    await asyncio.sleep(0.05)
            raise AssertionError("the second restore never blocked on the first one's bank row")

        await import_bank(
            backend=backend,
            embeddings_model=memory.embeddings,
            entity_resolver=memory.entity_resolver,
            resolve_config=lambda: memory._config_resolver.resolve_full_config(dst, request_context),
            format_date_fn=memory._format_readable_date,
            archive=archive,
            target_bank_id=dst,
            scope=scope,
            file_storage=memory._file_storage,
            on_bank_created=start_second_restore_while_first_is_open,
        )

        assert second is not None
        with pytest.raises(ValueError, match="created by another restore"):
            await second
        assert await _facts(memory, dst, request_context) == source_facts
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_an_archive_whose_config_has_no_bank_row_is_refused(memory, request_context):
    """With the config component asked for but no bank row in the archive, the restore says so
    and writes nothing, instead of failing on a foreign key further down."""
    src = f"race-src-{uuid.uuid4().hex[:8]}"
    dst = f"race-dst-{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=src, content="Alice joined Acme in 2024.", document_id="doc", request_context=request_context
        )
        async with acquire_with_retry(await memory._get_backend()) as conn:
            archive = await export_bank(conn, src)
        stripped = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(archive)) as zin, zipfile.ZipFile(stripped, "w") as zout:
            for name in zin.namelist():
                zout.writestr(name, b"[]" if name == "banks.json" else zin.read(name))

        with pytest.raises(ValueError, match="no bank row"):
            await memory.import_bank_async(stripped.getvalue(), request_context, target_bank_id=dst)
        assert await memory.get_bank_profile(dst, request_context=request_context) is None
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)
