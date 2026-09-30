"""Extra ids a bank answers to, and the one place they become the real id.

Why this exists
---------------
``bank_id`` is a TEXT primary key and the foreign key in every bank-scoped table,
so changing it means rewriting every row and stopping the bank's clients first
(``hindsight-admin rename-bank``). An alias gives the same outcome without the
stop: the bank keeps its id and its rows, and additionally answers to a new one,
so callers move over in phases.

Why resolution happens at the HTTP edge
---------------------------------------
Roughly 337 ``WHERE bank_id = $1`` queries span 36 modules. Resolving deeper --
say in ``_ensure_bank_exists`` -- would leave all of them keyed on the alias, so
the id has to become canonical *before* anything reads it. See
``api/unknown_params.py``, which rewrites the path param for every bank endpoint
at once. Everything below that point only ever sees a real ``bank_id``, and no
row anywhere is stored under an alias.

Resolution order: a real bank always wins
-----------------------------------------
:func:`resolve` consults ``banks`` first and ``bank_aliases`` only on a miss, so
an alias can never shadow an existing bank. That ordering is also what makes the
one unclosed race safe -- see :func:`create_alias`.

An unknown id resolves to itself, so behaviour for every id that is not an alias
(including retain's lazy auto-create) is exactly what it was before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import get_config
from .bank_stats_cache import BankStatsCache
from .db_utils import acquire_with_retry
from .memory_engine import fq_table, get_current_schema

if TYPE_CHECKING:
    from .db.base import DatabaseConnection

logger = logging.getLogger(__name__)

_cache: BankStatsCache | None = None


def _get_cache() -> BankStatsCache:
    global _cache
    if _cache is None:
        cfg = get_config()
        _cache = BankStatsCache(
            ttl_seconds=cfg.bank_alias_cache_ttl_seconds,
            max_entries=cfg.bank_alias_cache_max_entries,
        )
    return _cache


def _key(alias: str) -> tuple[str, str]:
    """Keyed on ``(schema, alias)``, never the alias alone: schema is the tenant
    boundary, and aliases are caller-chosen, so two tenants routinely pick the same
    string. A key without the schema would route one tenant's traffic into another's
    bank."""
    return (get_current_schema() or "", f"alias:{alias}")


async def _lookup(pool, alias: str) -> str | None:
    """Read the canonical bank id for ``alias``, uncached. None when it is not an alias."""
    async with acquire_with_retry(pool) as conn:
        return await conn.fetchval(f"SELECT bank_id FROM {fq_table('bank_aliases')} WHERE alias = $1", alias)


async def resolve(pool, bank_id: str) -> str:
    """Return the canonical bank id for ``bank_id``, which may be an alias.

    An id that names a real bank, or names nothing at all, is returned unchanged.

    Cached per process, misses included: nearly every request names a real bank,
    and without a negative entry each one would pay a pooled read that can only
    ever answer "not an alias". A deleted alias therefore keeps routing, and a new
    one stays invisible, for up to ``bank_alias_cache_ttl_seconds`` on pods other
    than the one that made the change.
    """
    from .retain import bank_utils

    if await bank_utils.get_bank_profile_if_exists(pool, bank_id) is not None:
        # The overwhelmingly common case. Read through bank_info_cache rather than
        # `bank_exists`, which is an uncached bare SELECT: the profile answers the
        # same question and every request is about to read it anyway, so on a hit
        # this costs no pooled connection at all. A real bank is never shadowed by
        # an alias, so there is nothing else to look up.
        return bank_id

    cache = _get_cache()
    schema, key = _key(bank_id)

    async def _load() -> dict[str, Any]:
        target = await _lookup(pool, bank_id)
        # `{}` is the cached "not an alias", which is the answer worth caching here.
        return {"bank_id": target} if target else {}

    row = await cache.get_or_load(schema, key, _load)
    return row.get("bank_id") or bank_id


async def invalidate(alias: str) -> None:
    """Drop a cached alias entry. Called by every path that writes one.

    In-process only, like the caches it sits beside: other pods wait out the TTL.
    """
    schema, key = _key(alias)
    await _get_cache().invalidate(schema, key)


async def alias_exists_on_conn(conn: "DatabaseConnection", alias: str) -> bool:
    """Is this string already an alias? Uncached, and on the caller's connection.

    Used by ``bank_utils.create_bank_row_on_conn`` to refuse creating a bank whose
    id is already routing elsewhere. Deliberately not cached: it guards a write, and
    a stale "no" here would create exactly the collision it exists to prevent.
    """
    return await conn.fetchval(f"SELECT 1 FROM {fq_table('bank_aliases')} WHERE alias = $1", alias) is not None


@dataclass(frozen=True)
class BankAlias:
    """One id a bank answers to, and whether it is the one shown in its place."""

    alias: str
    primary: bool


async def list_aliases(pool, bank_id: str) -> list[BankAlias]:
    """Every alias pointing at ``bank_id``, the primary one first then oldest first.

    Primary first because that is the one a reader is looking for: it is the id the
    bank is presented as, and the rest are the ones still being retired.
    """
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"SELECT alias, is_primary FROM {fq_table('bank_aliases')} WHERE bank_id = $1 "
            "ORDER BY is_primary DESC, created_at, alias",
            bank_id,
        )
    return [BankAlias(alias=row["alias"], primary=bool(row["is_primary"])) for row in rows]


async def primary_aliases(pool, bank_ids: list[str]) -> dict[str, str]:
    """Map each of ``bank_ids`` that has a primary alias to it.

    One query for a whole page of banks rather than one per bank, and every
    primary in the schema is read rather than building an ``IN`` list: at most one
    row per aliased bank exists, so the table is far smaller than the bank list it
    decorates, and a fixed statement stays portable across both dialects.
    """
    if not bank_ids:
        return {}
    wanted = set(bank_ids)
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(f"SELECT bank_id, alias FROM {fq_table('bank_aliases')} WHERE is_primary = TRUE")
    return {row["bank_id"]: row["alias"] for row in rows if row["bank_id"] in wanted}


async def create_alias(pool, bank_id: str, alias: str, *, primary: bool = False) -> None:
    """Point ``alias`` at ``bank_id``. Raises 409 if the name is already taken.

    Two collisions have to be refused, and only one of them is a constraint:

    * **alias vs alias** -- the PRIMARY KEY on ``alias``. A second bank claiming the
      same name gets a unique violation, so there is no read-then-write to race.
    * **alias vs an existing bank id** -- two different tables, which PostgreSQL
      cannot police, so the ``WHERE NOT EXISTS`` does it in the same statement.

    That leaves one race: a bank auto-created under this exact string between the
    NOT EXISTS and the commit. It is left open on purpose rather than paid for with
    an advisory lock on the bank-create hot path, because :func:`resolve` checks
    ``banks`` first -- so the real bank wins, and the loser is a dead alias rather
    than misrouted data. Deleting the alias fixes it.
    """
    from hindsight_api.extensions import OperationValidationError

    from .retain.bank_utils import validate_new_bank_id

    # An alias is used exactly like a bank id, so it lives under the same rules.
    validate_new_bank_id(alias)
    if alias == bank_id:
        raise OperationValidationError("A bank cannot be an alias of itself", status_code=400)

    async with acquire_with_retry(pool) as conn:
        inserted = await conn.fetchval(
            f"""
            INSERT INTO {fq_table("bank_aliases")} (alias, bank_id)
            SELECT $1, $2
            WHERE NOT EXISTS (SELECT 1 FROM {fq_table("banks")} WHERE bank_id = $1)
            ON CONFLICT (alias) DO NOTHING
            RETURNING alias
            """,
            alias,
            bank_id,
        )
    if inserted is None:
        raise OperationValidationError(f"'{alias}' is already taken by another bank or alias", status_code=409)
    # The negative cached by an earlier resolve() of this same string would otherwise
    # keep the new alias invisible on this pod for the whole TTL.
    await invalidate(alias)
    if primary:
        # Separate statement rather than part of the INSERT: promotion has to demote
        # whatever holds the flag today, which is the same work `set_primary` does.
        await set_primary(pool, bank_id, alias, True)


async def set_primary(pool, bank_id: str, alias: str, primary: bool) -> bool:
    """Make ``alias`` the id shown for ``bank_id``, or stop it being that.

    Promotion demotes the incumbent in the SAME transaction. The unique index makes
    "two primaries" unrepresentable, so doing it in two statements outside a
    transaction would not corrupt anything — it would fail the second write and
    leave the bank with none, which is worse than either outcome the caller asked
    for. Inside one transaction the bank goes straight from its old primary to its
    new one, and a concurrent promotion of a different alias loses the index race
    and rolls back whole.

    Returns False when the bank has no such alias, which the caller turns into a 404.
    """
    aliases = fq_table("bank_aliases")
    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            owned = await conn.fetchval(f"SELECT 1 FROM {aliases} WHERE alias = $1 AND bank_id = $2", alias, bank_id)
            if owned is None:
                return False
            if primary:
                # Demote first: the index forbids the overlap that setting the new
                # one first would create.
                await conn.execute(
                    f"UPDATE {aliases} SET is_primary = FALSE WHERE bank_id = $1 AND is_primary = TRUE AND alias <> $2",
                    bank_id,
                    alias,
                )
            await conn.execute(
                f"UPDATE {aliases} SET is_primary = $1 WHERE alias = $2 AND bank_id = $3",
                primary,
                alias,
                bank_id,
            )
    return True


async def delete_alias(pool, bank_id: str, alias: str) -> bool:
    """Stop ``alias`` routing to ``bank_id``. False when it was not one of its aliases.

    Scoped to the bank on purpose: deleting through an alias must not let a caller
    detach a name from a bank it never reached.
    """
    async with acquire_with_retry(pool) as conn:
        deleted = await conn.fetchval(
            f"DELETE FROM {fq_table('bank_aliases')} WHERE alias = $1 AND bank_id = $2 RETURNING alias",
            alias,
            bank_id,
        )
    await invalidate(alias)
    return deleted is not None
