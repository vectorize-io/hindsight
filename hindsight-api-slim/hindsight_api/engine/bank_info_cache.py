"""Per-process cache for the two bank rows a retain reads on every call.

A retain into a store-owned bank writes nothing to Postgres. What kept it holding a pooled
connection anyway was two reads that are the same on every call for the life of a bank:

    SELECT name, disposition, mission FROM {fq_table("banks")} WHERE bank_id = $1  -- the profile
    SELECT config FROM {fq_table("banks")} WHERE bank_id = $1                      -- its config

Neither is free. Each is a pool acquire, and an acquire costs more than the query it carries:
the pool runs five ``set_config`` calls on checkout and a full ``RESET ALL`` on release, so one
cached read removes roughly three statements, not one. Removing both is what lets such a retain
run without touching the database at all.

**This is a real consistency trade, not a free win.** A bank edited in ANOTHER process is not
visible to a reader holding a cached entry until that entry expires, so across pods the guarantee
is "eventually consistent within the TTL".

Within the writing process it is not a trade, because :func:`invalidate` is called on every path
that writes the bank row. That is not an optimisation for one pod -- it is what keeps the API's
own contract. `update_bank` returns the profile it reads back after writing, so without
invalidation the endpoint answers a successful update with the values it just replaced; and a
caller that PUTs a bank and immediately GETs it is the ordinary case, not a race.

The cost is that every bank-write path has to invalidate, and a forgotten one produces staleness
the TTL no longer bounds. `test_bank_info_cache_invalidation` is what makes a forgotten site fail
loudly instead of silently.

What that permits is bounded by what these two rows carry: a bank's display name, its
disposition, its mission and its config. None is a correctness gate on a write, and this cache is
deliberately NOT used for anything that authorises or routes a request. Set the TTL to 0 to
disable it and read on every call.

Keyed on ``(schema, bank_id)``, never ``bank_id`` alone: schema is the tenant boundary, and a
cache keyed on the bank id alone would serve one tenant's bank row to another whenever two
tenants happen to use the same bank id -- which they routinely do, because bank ids are chosen
by callers.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from ..config import get_config
from .bank_stats_cache import BankStatsCache

logger = logging.getLogger(__name__)

# `BankStatsCache` is named for its first caller but is a generic
# `(schema, bank_id) -> dict` TTL cache with miss coalescing and an LRU bound. Reused here rather
# than copied: a second implementation of the same eviction and coalescing logic is a second place
# for it to be wrong. (Its name and docstring deserve generalising -- a follow-up, not worth
# churning its 20-odd test references for.)
_cache: BankStatsCache | None = None


def _get_cache() -> BankStatsCache:
    global _cache
    if _cache is None:
        cfg = get_config()
        _cache = BankStatsCache(
            ttl_seconds=cfg.bank_info_cache_ttl_seconds,
            max_entries=cfg.bank_info_cache_max_entries,
        )
    return _cache


def _key(bank_id: str, kind: str) -> tuple[str, str]:
    """The two rows are cached under separate keys so invalidating one does not drop the other,
    and so a caller that needs only the profile does not pull the config into memory."""
    return (_schema(), f"{kind}:{bank_id}")


def _schema() -> str:
    try:
        from .memory_engine import get_current_schema

        return get_current_schema() or ""
    except Exception:
        # No schema context (a background loop, say). Cache under the empty key rather than
        # failing -- it is still a per-tenant-process boundary, and a wrong-but-consistent key
        # would be worse than a shared one.
        return ""


async def get_or_load(bank_id: str, kind: str, loader: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    """Cached read of one bank row. `kind` is "profile" or "config"."""
    cache = _get_cache()
    schema, key = _key(bank_id, kind)
    value = await cache.get_or_load(schema, key, loader)
    # NEVER retain a negative. "This bank does not exist" is the one answer guaranteed to change,
    # and usually within milliseconds -- the caller's next move on a miss is to create the bank.
    # Caching it sends every subsequent call down the create path for a bank that now exists,
    # which is slower than not caching at all: the create path re-reads the row inside a
    # transaction, so a cached miss costs a BEGIN, a SELECT and a COMMIT to reach the same answer.
    if not value:
        await cache.invalidate(schema, key)
    return value


async def invalidate(bank_id: str, kind: str | None = None) -> None:
    """Drop this bank's cached rows. `kind` narrows it to "profile" or "config"; None drops both.

    Called by every path that writes the bank row. The two kinds are dropped together by default
    because a caller that has just written one rarely knows which reads the request will make next,
    and dropping the other costs a single re-read.
    """
    cache = _get_cache()
    for k in (kind,) if kind else ("profile", "config"):
        schema, key = _key(bank_id, k)
        await cache.invalidate(schema, key)
