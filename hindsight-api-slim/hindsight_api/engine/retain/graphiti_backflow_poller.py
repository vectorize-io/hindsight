"""Channel C polling worker — drains ``graphiti_backflow_poller_state`` for each bank.

Channel C of C4 (deep-dive 4 §1.2, deep-dive 5 §3) is the **fallback** for
deployments that mix in native Graphiti ``add_episode`` writes that bypass
Hindsight's ``/triplet`` C1 path. The stock ``/search`` endpoint is
query-only and cannot list invalidated edges — this worker periodically
calls the new ``/invalidated-edges`` endpoint (deep-dive 5 §2.1) to
discover edges that were invalidated upstream of Hindsight, then replays
each one through the same ``MemoryEngine.handle_graphiti_edge_invalidated``
primitive channel A and channel B use. Channels A and B together already
give 100% coverage for the C1-mediated path; channel C closes the gap for
legacy / out-of-band writes.

Design (deep-dive 5 §3):

* Lifecycle — runs as an independent ``asyncio.Task`` started by
  ``main.py``'s lifespan when ``graphiti_backflow_polling_enabled`` is
  true at global config. Cancelled cleanly on shutdown. Not submitted
  through the ``_submit_async_operation`` system because that system is
  event-driven (one retain → one drain); channel C is time-driven.
* Per-bank state — the ``graphiti_backflow_poller_state`` table tracks
  ``last_seen_invalid_at`` (the cursor) plus diagnostic columns for
  observability. Missing row → ``last_seen_invalid_at = epoch``.
* Replay — filters by ``source_uri`` prefix
  ``hindsight://bank/{bank_id}/memory/`` and forwards each matching edge
  to ``MemoryEngine.handle_graphiti_edge_invalidated`` (the same
  primitive channels A and B use, so all three channels produce
  identical DB writes + audit entries).
* Truncation handling — if the response sets ``truncated=True``, the
  worker does NOT advance the cursor (it must keep its place so
  Graphiti can resume the next page). Diagnostic columns are still
  updated.
* Idempotency — guaranteed by the engine method itself (deep-dive 4
  §1.3): step 2's ``consolidated_at = NULL`` is a no-op when already
  NULL; step 4's ``valid_until IS NULL`` guard prevents re-stomping
  B1 supersession. Re-running on a seen edge is safe.
* Backpressure — a per-iteration ``asyncio.sleep(poll_interval)`` at
  the end, plus a ``try/except`` around the whole body so a transient
  failure (network blip, DB unavailable) does not kill the worker.
  ``last_seen_invalid_at`` is **not** advanced on error — the next
  iteration re-queries the same range and replays any new edges.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from ..db_utils import acquire_with_retry
from ..federation.graphiti_client import (
    EdgeResult,
    GraphitiClient,
    GraphitiClientError,
    InvalidatedEdgesResponse,
)
from ..memory_engine import fq_table
from .graphiti_forward import _parse_iso_to_utc

if TYPE_CHECKING:
    from ..memory_engine import MemoryEngine


logger = logging.getLogger(__name__)

# Default poll interval (seconds) — matches Graphiti's bi-temporal
# invalidation latency budget for add_episode writes. The operator
# can lower this via ``graphiti_backflow_polling_interval_seconds``;
# the worker's per-iteration sleep is the upper bound, not a precise
# timer (it races with the per-bank network/DB latency).
_DEFAULT_POLL_INTERVAL_S = 60.0

# The list_invalidated_edges response ceiling (deep-dive 5 §2.2). The
# worker passes this straight through; Graphiti enforces it server-side.
_DEFAULT_MAX_EDGES = 100

# Epoch cursor for the first poll. UTC + tz-aware so it round-trips
# through the engine's timezone-aware columns without surprise.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass
class PollTickResult:
    """Per-bank summary logged at the end of each poll iteration.

    Pure data — no DB connection held. Returned to the worker loop and
    folded into the lifespan-level periodic summary log.
    """

    bank_id: str
    polled: bool
    edges_seen: int = 0
    edges_replayed: int = 0
    not_found: int = 0
    errors: int = 0
    truncated: bool = False
    new_since: datetime | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "polled": self.polled,
            "edges_seen": self.edges_seen,
            "edges_replayed": self.edges_replayed,
            "not_found": self.not_found,
            "errors": self.errors,
            "truncated": self.truncated,
            "new_since": self.new_since.isoformat() if self.new_since else None,
            "error_message": self.error_message,
        }


@dataclass
class PollerRunResult:
    """End-of-tick summary across all banks in one poll cycle."""

    banks_polled: int = 0
    banks_skipped: int = 0
    edges_replayed: int = 0
    errors: int = 0
    tick_results: list[PollTickResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "banks_polled": self.banks_polled,
            "banks_skipped": self.banks_skipped,
            "edges_replayed": self.edges_replayed,
            "errors": self.errors,
        }


@dataclass
class ReplayResult:
    """Per-batch outcome from ``_replay_local_edges``.

    Sum of ``replayed`` + ``not_found`` + ``errors`` equals the number of
    edges the worker attempted to replay. Splitting them keeps the tick
    summary able to distinguish "edge was a no-op because the memory
    unit was already consolidated" (not_found) from "edge replay
    actually executed" (replayed) from "edge replay raised" (errors).
    """

    replayed: int = 0
    not_found: int = 0
    errors: int = 0


def _build_client(bank_config) -> GraphitiClient:
    """Construct a client from the bank's resolved config + env defaults.

    Mirrors ``graphiti_forward._build_client`` — same env-var fallback
    chain, same auth header, same timeout/breaker defaults. Kept as a
    separate function rather than imported so the poller doesn't grow
    an undeclared dependency on the forwarder module.
    """
    import os

    base_url = (
        getattr(bank_config, "graphiti_base_url", None)
        or os.getenv("HINDSIGHT_API_GRAPHITI_BASE_URL")
        or os.getenv("GRAPHITI_BASE_URL", "")
    )
    if not base_url:
        raise GraphitiClientError("HINDSIGHT_API_GRAPHITI_BASE_URL is not set; cannot poll")
    api_key = (
        getattr(bank_config, "graphiti_api_key", None)
        or os.getenv("HINDSIGHT_API_GRAPHITI_API_KEY")
        or os.getenv("GRAPHITI_API_KEY")
    )
    return GraphitiClient(base_url=base_url, api_key=api_key)


async def _read_poller_state(
    backend: Any,
    bank_id: str,
) -> datetime | None:
    """Read the cursor for a bank, returning ``None`` when no row exists.

    Caller treats ``None`` as "use the epoch" — i.e. start of time. We
    keep the "None means epoch" decision at the call site so the
    semantics are obvious to readers.
    """
    async with acquire_with_retry(backend) as conn:
        row = await conn.fetchrow(
            f"SELECT last_seen_invalid_at FROM {fq_table('graphiti_backflow_poller_state')} WHERE bank_id = $1",
            bank_id,
        )
    if row is None:
        return None
    return row["last_seen_invalid_at"]


async def _write_poller_state(
    backend: Any,
    bank_id: str,
    *,
    new_since: datetime,
    last_poll_at: datetime,
    last_poll_edges: int,
    truncated: bool,
    error: str | None,
) -> None:
    """UPSERT the poller state row.

    Note: when ``error`` is non-None we still write the diagnostic
    columns (``last_poll_at``, ``last_poll_error``) but **do not
    advance** the cursor — the caller is responsible for that
    decision. This function just persists whatever fields the caller
    decided to commit.
    """
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            f"""
            INSERT INTO {fq_table("graphiti_backflow_poller_state")}
                (bank_id, last_seen_invalid_at, last_poll_at, last_poll_edges, last_poll_truncated, last_poll_error, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, now())
            ON CONFLICT (bank_id) DO UPDATE SET
                last_seen_invalid_at = EXCLUDED.last_seen_invalid_at,
                last_poll_at = EXCLUDED.last_poll_at,
                last_poll_edges = EXCLUDED.last_poll_edges,
                last_poll_truncated = EXCLUDED.last_poll_truncated,
                last_poll_error = EXCLUDED.last_poll_error,
                updated_at = EXCLUDED.updated_at
            """,
            bank_id,
            new_since,
            last_poll_at,
            last_poll_edges,
            truncated,
            error,
        )


def _is_local_edge(bank_id: str, edge: EdgeResult) -> bool:
    """Channel C source_uri filter — same prefix channel A uses.

    Centralized so the deep-dive 5 §3 "local-only filtering" invariant
    is grep-able in one place; if the URI format ever changes (e.g.
    adding a query string) only this function updates.
    """
    if not edge.source_uri:
        return False
    prefix = f"hindsight://bank/{bank_id}/memory/"
    return edge.source_uri.startswith(prefix)


async def _replay_local_edges(
    engine: "MemoryEngine",
    bank_id: str,
    edges: list[EdgeResult],
    internal: Any,
) -> ReplayResult:
    """Replay edges that target this bank through the shared engine method.

    Mirrors ``graphiti_forward._trigger_backflow_actions`` (channel A):
    build a UTC datetime from each edge's ``invalid_at`` (the wire
    format may include ``Z`` or ``+00:00``), then call
    ``engine.handle_graphiti_edge_invalidated``. Failures are caught
    per-edge so a single bad edge cannot poison the rest of the batch.

    Returns a ``ReplayResult`` with ``replayed``, ``not_found``, and
    ``errors`` counts. (Sum equals the number of edges attempted.)
    """
    result = ReplayResult()
    for edge in edges:
        if edge.invalid_at is None:
            # Defensive: the parser already filters for ``invalid_at is
            # not None`` upstream, but if a future schema change drops
            # that filter we want a clear log rather than a NoneType
            # crash.
            logger.warning(
                "graphiti_backflow_poller: edge %s for bank %s has no invalid_at; skipping",
                edge.uuid,
                bank_id,
            )
            result.errors += 1
            continue
        try:
            invalid_at_dt = _parse_iso_to_utc(edge.invalid_at)
        except ValueError:
            logger.warning(
                "graphiti_backflow_poller: edge %s has unparseable invalid_at %r; skipping",
                edge.uuid,
                edge.invalid_at,
            )
            result.errors += 1
            continue
        try:
            engine_result = await engine.handle_graphiti_edge_invalidated(
                bank_id=bank_id,
                edge_uuid=str(edge.uuid),
                source_uri=edge.source_uri or "",
                invalid_at=invalid_at_dt,
                request_context=internal,
            )
            result.replayed += 1
            if engine_result.not_found:
                result.not_found += 1
        except Exception:
            logger.exception(
                "graphiti_backflow_poller: failed to replay edge %s for bank %s",
                edge.uuid,
                bank_id,
            )
            result.errors += 1
    return result


def _compute_new_since(
    edges: list[EdgeResult],
    current_since: datetime,
) -> datetime | None:
    """Pick the new cursor: max parseable ``invalid_at`` across the page.

    Returns ``None`` if no edges yielded a parseable timestamp — in
    which case the caller should keep the cursor unchanged. The
    ``current_since`` argument is the safety floor: if a malformed
    edge has a parseable ``invalid_at`` *earlier* than the current
    cursor (shouldn't happen — Graphiti filters ``>= since`` — but the
    contract allows ``>=`` not ``>``), the result is clamped to
    ``current_since`` so the cursor never moves backward.
    """
    best: datetime | None = None
    for e in edges:
        if e.invalid_at is None:
            continue
        try:
            dt = _parse_iso_to_utc(e.invalid_at)
        except ValueError:
            continue
        if best is None or dt > best:
            best = dt
    if best is None:
        return None
    if best < current_since:
        return current_since
    return best


async def _poll_one_bank(
    engine: "MemoryEngine",
    client: GraphitiClient,
    bank_id: str,
    internal: Any,
    max_edges: int = _DEFAULT_MAX_EDGES,
    *,
    clock: Callable[[], datetime] | None = None,
) -> PollTickResult:
    """One bank's worth of channel-C work.

    Reads the cursor, queries ``/invalidated-edges`` with it, filters
    down to this bank's edges, replays them, and writes the new state
    (or, on error, writes diagnostic columns without advancing the
    cursor).

    ``clock`` is a test seam — production uses ``datetime.now(timezone.utc)``
    so the diagnostic ``last_poll_at`` reflects real wall time; tests
    inject a fixed clock for deterministic assertions. The
    ``last_seen_invalid_at`` cursor is **not** driven by ``clock`` —
    it comes from the edges Graphiti returns, so the replay logic is
    clock-independent.
    """
    if clock is None:

        def clock() -> datetime:
            return datetime.now(timezone.utc)

    backend = await engine._get_backend()
    result = PollTickResult(bank_id=bank_id, polled=True)

    current_since = await _read_poller_state(backend, bank_id)
    if current_since is None:
        current_since = _EPOCH

    try:
        response: InvalidatedEdgesResponse = await client.list_invalidated_edges(
            group_ids=[bank_id],  # graphiti_group_id == bank_id by convention
            since=current_since,
            max_edges=max_edges,
        )
    except (GraphitiClientError, Exception) as e:
        logger.warning(
            "graphiti_backflow_poller: bank %s poll failed: %s; cursor unchanged",
            bank_id,
            e,
        )
        result.errors += 1
        result.error_message = str(e)
        try:
            await _write_poller_state(
                backend,
                bank_id,
                new_since=current_since,
                last_poll_at=clock(),
                last_poll_edges=0,
                truncated=False,
                error=str(e),
            )
        except Exception:
            logger.exception("graphiti_backflow_poller: failed to write error state for bank %s", bank_id)
        return result

    result.edges_seen = len(response.edges)
    result.truncated = response.truncated

    # Filter to local edges (this bank's source_uri prefix).
    local_edges = [e for e in response.edges if _is_local_edge(bank_id, e)]

    if local_edges:
        replay_result = await _replay_local_edges(engine, bank_id, local_edges, internal)
        result.edges_replayed = replay_result.replayed
        result.not_found = replay_result.not_found
        result.errors += replay_result.errors

    # Cursor management: advance only if the response wasn't truncated
    # (deep-dive 5 §3.3 — truncated page means Graphiti is paging
    # forward; keeping ``since`` fixed is the only way to resume).
    new_since_for_write = current_since
    if not response.truncated:
        computed = _compute_new_since(response.edges, current_since)
        if computed is not None:
            new_since_for_write = computed
    result.new_since = new_since_for_write

    try:
        await _write_poller_state(
            backend,
            bank_id,
            new_since=new_since_for_write,
            last_poll_at=clock(),
            last_poll_edges=result.edges_seen,
            truncated=response.truncated,
            error=None,
        )
    except Exception:
        # Cursor write failure is a soft error: the in-memory replay
        # already happened, so the data path is correct. A repeated
        # failure would just cause the same edges to be replayed on
        # the next tick — which is fine because the engine method is
        # idempotent. Log and move on.
        logger.exception("graphiti_backflow_poller: failed to persist state for bank %s", bank_id)
        result.errors += 1

    return result


async def run_graphiti_backflow_poller(
    memory_engine: "MemoryEngine",
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_S,
    *,
    client_factory: Callable[[Any], GraphitiClient | Awaitable[GraphitiClient]] | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Main entry point: poll every enabled bank on a fixed cadence.

    Runs forever (until cancelled) or until ``stop_event`` is set. Each
    tick:

    1. Resolves ``RequestContext(internal=True)`` for backend auth.
    2. Lists all banks; for each one that has
       ``graphiti_backflow_polling_enabled`` and a non-empty
       ``graphiti_group_id`` (set by graphiti_forward §3.4 of deep-dive
       5), calls ``_poll_one_bank``.
    3. Sleeps ``poll_interval_seconds`` (cancellable).

    The single ``try/except`` around the whole tick is the backstop —
    a tick-level exception logs and continues, never kills the loop.
    """
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be > 0")
    if stop_event is None:
        stop_event = asyncio.Event()

    from hindsight_api.models import RequestContext

    internal = RequestContext(internal=True)

    logger.info("graphiti_backflow_poller: starting (poll_interval=%.1fs)", poll_interval_seconds)
    try:
        while not stop_event.is_set():
            tick = PollerRunResult()
            try:
                banks = await memory_engine.list_banks(request_context=internal)
                for bank in banks:
                    bank_id = bank.get("bank_id")
                    if not bank_id:
                        continue
                    try:
                        cfg = await memory_engine._config_resolver.resolve_full_config(bank_id, internal)
                    except Exception:
                        logger.exception(
                            "graphiti_backflow_poller: failed to resolve config for bank %s; skipping",
                            bank_id,
                        )
                        tick.banks_skipped += 1
                        continue
                    if not getattr(cfg, "graphiti_backflow_polling_enabled", False):
                        tick.banks_skipped += 1
                        continue
                    if not getattr(cfg, "graphiti_group_id", ""):
                        # Same precondition as graphiti_forward: the
                        # bank needs a federation identity before we
                        # can know which group_ids to query.
                        tick.banks_skipped += 1
                        continue
                    if client_factory is not None:
                        # ``client_factory`` may be sync (return a client
                        # directly — matches ``graphiti_forward``'s API)
                        # or async (return an awaitable that resolves to
                        # a client). Branch on type rather than rely on
                        # the static type checker to figure out the union
                        # — ``Awaitable[X]`` is not guaranteed to be
                        # awaitable in a way ``ty`` can prove.
                        result_obj = client_factory(cfg)
                        if inspect.isawaitable(result_obj):
                            client = await result_obj
                        else:
                            client = result_obj
                    else:
                        client = _build_client(cfg)
                    tick.banks_polled += 1
                    tick_result = await _poll_one_bank(memory_engine, client, bank_id, internal)
                    tick.tick_results.append(tick_result)
                    tick.edges_replayed += tick_result.edges_replayed
                    tick.errors += tick_result.errors
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("graphiti_backflow_poller: tick failed; continuing")

            if tick.banks_polled > 0 or tick.banks_skipped > 0:
                logger.info("graphiti_backflow_poller: %s", json.dumps(tick.as_dict()))

            # Cancellable sleep — when stop_event is set (or the task
            # is cancelled), the await returns immediately so the loop
            # can exit on the next while-iteration check.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_seconds)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        logger.info("graphiti_backflow_poller: cancelled, shutting down")
        raise
    finally:
        logger.info("graphiti_backflow_poller: stopped")
