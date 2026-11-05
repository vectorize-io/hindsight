"""Keep a background job from writing into a bank deleted after the job started (#6344).

A job binds the bank's ``internal_id`` when it starts. ``internal_id`` is minted with the bank
row, so a bank deleted and created again under the same name has a new one: a mismatch means
"the bank this job was working for is gone", whether or not a new bank now holds the name.

Only Hindsight can tell this apart from a plain write: a store that owns its storage recreates a
missing bank's storage on any write, so the check has to run before the write reaches it.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .db.base import DatabaseBackend, DatabaseConnection
from .db_utils import acquire_with_retry
from .schema import _is_oracle, fq_table


class BankDeletedError(Exception):
    """The bank a background job started for was deleted (or replaced) while it ran."""


@dataclass(frozen=True)
class _JobBank:
    bank_id: str
    internal_id: str


_job_bank: ContextVar[_JobBank | None] = ContextVar("hindsight_job_bank", default=None)


async def _internal_id_on_conn(conn: DatabaseConnection, bank_id: str, *, lock: bool) -> str | None:
    # FOR KEY SHARE holds the row until the caller's transaction ends, so a delete waits for the
    # write it is checking instead of committing between the check and the write. Of what runs
    # against a bank row, only deleting it (or FOR UPDATE) waits on it. Oracle has no shared row
    # lock, so there the check is a plain read.
    suffix = " FOR KEY SHARE" if lock and not _is_oracle() else ""
    value = await conn.fetchval(f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1{suffix}", bank_id)
    return None if value is None else str(value)


@contextmanager
def _bind(bank_id: str, internal_id: str) -> Iterator[None]:
    token = _job_bank.set(_JobBank(bank_id=bank_id, internal_id=internal_id))
    try:
        yield
    finally:
        _job_bank.reset(token)


async def bind_job_bank(pool: DatabaseBackend, bank_id: str) -> AbstractContextManager[None]:
    """Bind ``bank_id``'s current life to the running job. Raises if the bank is already gone.

    Use as ``with await bind_job_bank(pool, bank_id): ...`` around the job.
    """
    async with acquire_with_retry(pool) as conn:
        internal_id = await _internal_id_on_conn(conn, bank_id, lock=False)
    if internal_id is None:
        raise BankDeletedError(f"Bank '{bank_id}' no longer exists")
    return _bind(bank_id, internal_id)


def job_bank_bound(bank_id: str) -> bool:
    """True when the running job is bound to ``bank_id`` and must not create it."""
    bound = _job_bank.get()
    return bound is not None and bound.bank_id == bank_id


async def assert_job_bank_alive(conn: DatabaseConnection, bank_id: str) -> None:
    """Raise :class:`BankDeletedError` if the job's bank was deleted since the job started.

    A no-op outside a job bound to ``bank_id``. Inside a transaction it also holds the bank row
    until commit, so the write that follows cannot land after a delete.
    """
    bound = _job_bank.get()
    if bound is None or bound.bank_id != bank_id:
        return
    if await _internal_id_on_conn(conn, bank_id, lock=True) != bound.internal_id:
        raise BankDeletedError(f"Bank '{bank_id}' was deleted while a background job was running for it")


async def assert_job_bank_alive_on_pool(pool: DatabaseBackend, bank_id: str) -> None:
    """:func:`assert_job_bank_alive` on a connection of its own, for a caller with no transaction."""
    if not job_bank_bound(bank_id):
        return
    async with acquire_with_retry(pool) as conn:
        await assert_job_bank_alive(conn, bank_id)
