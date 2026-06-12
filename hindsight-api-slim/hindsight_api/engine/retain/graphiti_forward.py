"""``graphiti_forward`` worker — drains ``graphiti_outbox`` into Graphiti (C1 + C4-A).

Per drained row:

1. For each relation in the row's payload, build a deterministic edge UUID
   and call ``GraphitiClient.add_triplet`` once (deep-dive 2 §2.4). Edge
   UUID = ``uuid5(NAMESPACE, memory_id:src_idx:tgt_idx:predicate)`` so
   at-least-once retries land on the same edge — the overlay's
   ``resolve_extracted_edge`` re-resolves by endpoint pair, so even if
   endpoints merged in Graphiti the second attempt still re-uses the
   original UUID and updates rather than duplicates.

2. Write back the resolved node UUIDs to ``entities.graphiti_uuid`` so the
   next forwarder run for the same bank/entity can skip the add_triplet
   resolution step (C5 cold→hot mapping payoff, deep-dive 2 §2.5).

3. **Channel A of C4** (deep-dive 4 §1.2-1.3): for each ``invalidated_edge``
   in the response whose ``source_uri`` matches the current bank, replay
   the edge through ``MemoryEngine.handle_graphiti_edge_invalidated`` —
   the *same* primitive the channel-B HTTP endpoint uses. The engine
   method handles per-edge work: parse source_uri, clear observations
   derived from the memory, optionally stamp the B1 supersession ledger
   (gated on ``graphiti_backflow_supersession=true``), and submit a
   gated consolidation job. Replay is best-effort — per-edge failures
   log and continue; the affected memory's cleared state still picks up
   on the next natural consolidation cycle. The default action is
   re-consolidation only — never rewrites the private memory text
   (private memory owner is always the bank, per main plan §6-5). The
   B1 schema enforces the CHECK constraint that prevents landing rows
   without ``occurred_start``.

Errors:

* Transient (timeout, 5xx, circuit open) → the entire row's id list is
  re-enqueued via ``reschedule_graphiti_outbox_rows`` with exponential
  backoff (1s, 4s, 16s, 64s, 256s, 5min cap).
* Permanent (4xx other than 408/429) → log + drop the row, never wedge the
  bank queue. (4xx other than 429 are configuration errors — the row is
  poisoned for this Graphiti config and retrying will not help.)
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid5

from ..db_utils import acquire_with_retry
from ..federation.circuit_breaker import CircuitOpenError
from ..federation.graphiti_client import (
    AddTripletResults,
    EdgePayload,
    EdgeResult,
    GraphitiClient,
    GraphitiClientError,
    NodePayload,
    TripletRequest,
)
from ..memory_engine import fq_table

if TYPE_CHECKING:
    from ..memory_engine import MemoryEngine


logger = logging.getLogger(__name__)

# Deterministic node UUID namespace — used to derive a Graphiti node UUID
# from (group_id, normalized_name) when no prior mapping exists. Stable
# across retries (the alternative — random UUID per attempt — would force
# Graphiti to merge nodes on every retry).
_NODE_UUID_NS = UUID("4f5a2b1e-9a8c-4d2b-8e3a-1c5b7d9e2f4a")

# Deterministic edge UUID namespace.
_EDGE_UUID_NS = UUID("7c3d8f2a-1b5e-4a9c-8d2e-6f3a4b8c1d5e")

# Bank-claim batch size — matches the supersession queue's _DRAIN_BATCH_SIZE
# shape. 50 keeps the worker's transaction time bounded and lets the
# circuit breaker recover between batches.
_DRAIN_BATCH_SIZE = 50

# Per-bank invalidated-edge memo. Owned by ``run_graphiti_forward_job`` as a
# local dict and threaded through the helpers (see _handle_invalidated_edges
# and _trigger_backflow_actions). Module-scope state would be safe today
# (per-bank filtering on ``source_uri`` keeps concurrent drains isolated) but
# is one careless refactor away from leaking edges across banks — keeping it
# local makes the isolation invariant structural rather than a comment.
#
# Stores the full ``EdgeResult`` (not just memory_id) so end-of-drain can
# call the shared engine primitive ``handle_graphiti_edge_invalidated`` with
# the original edge_uuid / source_uri / invalid_at — same call path as the
# channel-B HTTP endpoint, so a forwarder-driven invalidation and an
# overlay-driven invalidation produce identical writes + audit entries.


@dataclass
class ForwardJobResult:
    drained: int = 0
    forwarded: int = 0
    relations: int = 0
    invalidated_edges: int = 0
    rescheduled: int = 0
    dropped: int = 0
    write_backs: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "drained": self.drained,
            "forwarded": self.forwarded,
            "relations": self.relations,
            "invalidated_edges": self.invalidated_edges,
            "rescheduled": self.rescheduled,
            "dropped": self.dropped,
            "write_backs": self.write_backs,
            "errors": len(self.errors),
        }


def _deterministic_node_uuid(group_id: str, name: str) -> UUID:
    return uuid5(_NODE_UUID_NS, f"{group_id}:{name.lower()}")


def _deterministic_edge_uuid(memory_id: UUID, rel: dict) -> UUID:
    return uuid5(
        _EDGE_UUID_NS,
        f"{memory_id}:{rel['source_entity_index']}:{rel['target_entity_index']}:{rel['predicate']}",
    )


def _is_transient_error(exc: Exception) -> bool:
    """4xx other than 408/429 are permanent; everything else transient.

    The Graphiti client wraps httpx errors with the status code in the
    message ("HTTP error: 503 ..."), so we extract it and apply the
    standard retry-after-408/429 rule. 4xx responses with no status
    embedded are treated as permanent (a misconfigured bank should not
    wedge the queue).
    """
    if isinstance(exc, CircuitOpenError):
        return True
    if isinstance(exc, GraphitiClientError):
        msg = str(exc).lower()
        # Try to extract a status code from the wrapped httpx message.
        m = re.search(r"\b([1-5]\d\d)\b", msg)
        if m:
            status = int(m.group(1))
            if 500 <= status < 600:
                return True
            if status in (408, 429):
                return True
            if 400 <= status < 500:
                return False
        # No status code present — default transient (better to retry
        # than to silently drop a row whose error we can't classify).
        return True
    return True  # default: treat unknown errors as transient (better to retry than drop)


async def _write_back_entity_uuids(
    engine: "MemoryEngine",
    bank_id: str,
    nodes: list,
) -> int:
    """Map resolved node UUIDs back to ``entities.graphiti_uuid`` by name.

    Returns the number of rows updated. Done per-add_triplet (so cold→hot
    mapping happens in-flight) rather than in a single end-of-drain batch —
    keeps the working set small and lets an interrupted worker leave the
    table in a self-consistent state.
    """
    if not nodes:
        return 0
    backend = await engine._get_backend()
    # Group by name; Graphiti may return multiple node rows for the same
    # name across different groups (we only care about the one matching
    # our bank).
    name_to_uuid: dict[str, UUID] = {}
    for n in nodes:
        if n.group_id and n.name:
            name_to_uuid.setdefault(n.name, n.uuid)

    if not name_to_uuid:
        return 0

    # entities table is bank-scoped — match by (bank_id, LOWER(canonical_name)).
    # The LOWER() prevents the case-folding bug noted in the entity_resolver
    # (#626) from giving a different answer than the add_triplet path.
    updates = 0
    async with acquire_with_retry(backend) as conn:
        for name, uuid_val in name_to_uuid.items():
            result = await conn.execute(
                f"""
                UPDATE {fq_table("entities")}
                SET graphiti_uuid = $1
                WHERE bank_id = $2
                  AND LOWER(canonical_name) = LOWER($3)
                  AND graphiti_uuid IS NULL
                """,
                uuid_val,
                bank_id,
                name,
            )
            # asyncpg returns "UPDATE N" — extract the count.
            try:
                tag = result.split(" ")[-1]
                updates += int(tag)
            except (ValueError, IndexError):
                pass
    return updates


async def _handle_invalidated_edges(
    engine: "MemoryEngine",
    bank_id: str,
    results: AddTripletResults,
    memo: dict[UUID, EdgeResult],
) -> int:
    """Channel A: collect invalidated edges whose ``source_uri`` belongs to
    *this* bank. End-of-drain hands each one to
    ``MemoryEngine.handle_graphiti_edge_invalidated`` (the same primitive
    the channel-B HTTP endpoint uses) so both write paths produce
    identical DB writes + audit entries.

    Per deep-dive 4 §1.3, the per-memory work (parse URI, clear obs,
    optional B1 supersession, audit) lives in the engine method — the
    forwarder just *accumulates* during the drain.

    ``memo`` is owned by the calling ``run_graphiti_forward_job`` and
    lives for the duration of one bank drain; passing it in (instead of
    relying on module state) keeps the per-bank isolation invariant
    structural.
    """
    invalidated = results.invalidated_edges
    if not invalidated:
        return 0

    prefix = f"hindsight://bank/{bank_id}/memory/"
    for edge in invalidated:
        if not edge.source_uri or not edge.source_uri.startswith(prefix):
            continue
        # Validate the URI parses to a UUID before we accumulate. The
        # engine method will re-validate, but failing fast here keeps
        # the memo clean — an unparseable URI cannot produce a valid
        # backflow call, so we drop it (logged) instead of letting the
        # engine method log a not_found warning at end-of-drain.
        try:
            UUID(edge.source_uri[len(prefix) :])
        except ValueError:
            logger.warning("graphiti_forward: unparseable source_uri %r", edge.source_uri)
            continue
        # ``edge.invalid_at`` is guaranteed non-None here — the
        # ``AddTripletResults.invalidated_edges`` property already
        # filters by ``e.invalid_at is not None`` (per the
        # ``EdgeResult`` docstring). The ``assert`` in
        # ``_trigger_backflow_actions`` is the type-guard the static
        # checker is happy with.
        memo[edge.uuid] = edge
    return len(invalidated)


async def _process_one_row(
    engine: "MemoryEngine",
    client: GraphitiClient,
    bank_id: str,
    row: dict,
    memo: dict[UUID, EdgeResult],
) -> tuple[int, int, int, int]:
    """Forward a single outbox row. Returns (forwarded, relations, invalidated, write_backs).

    ``memo`` is the per-bank invalidated-edge scratch dict owned by
    ``run_graphiti_forward_job`` — passed in so the module doesn't rely
    on hidden global state.
    """
    memory_id = row["memory_id"] if isinstance(row["memory_id"], UUID) else UUID(row["memory_id"])
    group_id = row["group_id"]
    fact_text = row["fact_text"]
    entities = row["entities"] if isinstance(row["entities"], list) else json.loads(row["entities"])
    relations = row["relations"] if isinstance(row["relations"], list) else json.loads(row["relations"])
    tags = row["tags"]
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []

    forwarded = 0
    invalidated = 0
    write_backs = 0
    all_nodes: list = []

    for rel in relations:
        src = entities[rel["source_entity_index"]]
        tgt = entities[rel["target_entity_index"]]

        src_uuid = src.get("graphiti_uuid")
        if not src_uuid:
            src_uuid = _deterministic_node_uuid(group_id, src["text"])
        tgt_uuid = tgt.get("graphiti_uuid")
        if not tgt_uuid:
            tgt_uuid = _deterministic_node_uuid(group_id, tgt["text"])

        edge_uuid = _deterministic_edge_uuid(memory_id, rel)
        payload = TripletRequest(
            source_node=NodePayload(uuid=UUID(str(src_uuid)), name=src["text"], group_id=group_id),
            target_node=NodePayload(uuid=UUID(str(tgt_uuid)), name=tgt["text"], group_id=group_id),
            edge=EdgePayload(
                uuid=edge_uuid,
                name=rel["predicate"],
                fact=fact_text,
                valid_at=rel.get("rel_valid_at"),
                invalid_at=rel.get("rel_invalid_at"),
                group_id=group_id,
                attributes={
                    "source_uri": f"hindsight://bank/{bank_id}/memory/{memory_id}",
                    "hindsight_tags": tags or [],
                },
            ),
        )
        results = await client.add_triplet(payload)
        forwarded += 1
        all_nodes.extend(results.nodes)
        invalidated += await _handle_invalidated_edges(engine, bank_id, results, memo)

    write_backs = await _write_back_entity_uuids(engine, bank_id, all_nodes)
    return forwarded, len(relations), invalidated, write_backs


async def _trigger_backflow_actions(
    engine: "MemoryEngine",
    bank_id: str,
    memo: dict[UUID, EdgeResult],
) -> None:
    """End-of-drain housekeeping: replay each invalidated edge through the
    shared ``MemoryEngine.handle_graphiti_edge_invalidated`` primitive.

    Channel A and channel B (the HTTP endpoint) both call this same
    method, so a forwarder-driven invalidation and an overlay-driven
    invalidation produce identical DB writes + audit entries. The
    primitive handles per-edge work (parse source_uri, clear
    observations, optional B1 supersession, gated consolidation submit)
    and writes its own fire-and-forget audit log — we just iterate.

    Best-effort: a per-edge failure logs and continues. Channel A's
    invariant is "the affected memory's observations get refreshed on
    the next consolidation cycle" — missing one trigger simply means
    that memory's cleared state picks up on the next natural
    consolidation, so a dropped call is recoverable.

    Idempotency: the engine method is idempotent (step 2's
    ``consolidated_at = NULL`` no-ops when already NULL; step 4's
    ``valid_until IS NULL`` guard prevents re-stomp). It is safe to
    re-run the same edge in a later drain.

    ``memo`` is the per-bank invalidated-edge scratch dict owned by the
    calling ``run_graphiti_forward_job`` (passed in to keep the module's
    state local — see the docstring at the top of this file).
    """
    if not memo:
        return
    # Snapshot + clear immediately so a per-edge failure (or an
    # exception during iteration) cannot cause us to drop the whole
    # batch on the next drain — they'll just accumulate again.
    edges = list(memo.values())
    memo.clear()

    from hindsight_api.models import RequestContext

    internal = RequestContext(internal=True)

    replayed = 0
    not_found = 0
    errors = 0
    for edge in edges:
        assert edge.invalid_at is not None  # gated in _handle_invalidated_edges
        try:
            invalid_at_dt = _parse_iso_to_utc(edge.invalid_at)
        except ValueError:
            logger.warning(
                "graphiti_forward: edge %s has unparseable invalid_at %r; skipping",
                edge.uuid,
                edge.invalid_at,
            )
            errors += 1
            continue
        try:
            result = await engine.handle_graphiti_edge_invalidated(
                bank_id=bank_id,
                edge_uuid=str(edge.uuid),
                source_uri=edge.source_uri or "",
                invalid_at=invalid_at_dt,
                request_context=internal,
            )
            replayed += 1
            if result.not_found:
                # The edge pointed at a memory the bank has since
                # deleted. Per deep-dive 4 §1.3, that's a normal
                # outcome (not an error), but we still want to count
                # it so the worker log can show "N replayed, M gone".
                not_found += 1
        except Exception:
            # Per-edge failures must not abort the rest of the batch —
            # see deep-dive 4 §1.3 invariant: "at-least-once replay,
            # best-effort delivery".
            logger.exception(
                "graphiti_forward: failed to replay invalidated edge %s for bank %s",
                edge.uuid,
                bank_id,
            )
            errors += 1

    logger.info(
        "graphiti_forward: replayed %d invalidated edges for bank %s (not_found=%d, errors=%d)",
        replayed,
        bank_id,
        not_found,
        errors,
    )


def _parse_iso_to_utc(value: str) -> datetime:
    """Parse an ISO-8601 string (with or without trailing 'Z') to a UTC
    ``datetime``. Mirrors the field_validator in
    ``GraphitiEdgeInvalidatedRequest`` so the engine method sees a
    timezone-aware ``datetime`` regardless of the upstream wire format.
    """
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # The engine writes ``invalid_at`` straight into the DB's
        # ``valid_until`` column via asyncpg, which expects a
        # timezone-aware datetime. Naive timestamps from upstream are
        # assumed UTC (Graphiti serializes everything in UTC; a
        # non-UTC value here would be an upstream bug, not our problem).
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_client(bank_config) -> GraphitiClient:
    """Construct a client from the bank's resolved config + env defaults.

    ``HINDSIGHT_API_GRAPHITI_BASE_URL`` is the documented env name (matches
    the entry in ``.env.example`` and the C3 reflect tool's lookup); the
    bare ``GRAPHITI_BASE_URL`` is honored as a backward-compat alias for
    operators who already had the unprefixed form set. The API key follows
    the same pattern. Timeouts/breaker thresholds come from ``bank_config``
    if present, else from env, else from the client defaults (which match
    the C-track config table in deep-dive 2 §2.6).
    """
    base_url = (
        getattr(bank_config, "graphiti_base_url", None)
        or os.getenv("HINDSIGHT_API_GRAPHITI_BASE_URL")
        or os.getenv("GRAPHITI_BASE_URL", "")
    )
    if not base_url:
        raise GraphitiClientError("HINDSIGHT_API_GRAPHITI_BASE_URL is not set; cannot forward")
    api_key = (
        getattr(bank_config, "graphiti_api_key", None)
        or os.getenv("HINDSIGHT_API_GRAPHITI_API_KEY")
        or os.getenv("GRAPHITI_API_KEY")
    )
    return GraphitiClient(base_url=base_url, api_key=api_key)


async def run_graphiti_forward_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context,  # RequestContext
    operation_id: str | None = None,
    batch_size: int = _DRAIN_BATCH_SIZE,
    client_factory: Callable[[Any], GraphitiClient] | None = None,
) -> dict:
    """Drain ``graphiti_outbox`` for one bank.

    Loop structure mirrors ``run_fact_supersession_job``: claim a batch,
    process, reschedule failures, repeat until the queue is empty.
    """
    backend = await memory_engine._get_backend()
    ops = backend.ops
    bank_config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # If the bank has lost its federation identity, no point draining — the
    # rows were enqueued under the assumption that the bank was federated.
    # We drop them with a one-time warning rather than re-enqueueing.
    if not getattr(bank_config, "graphiti_group_id", ""):
        logger.warning(
            "graphiti_forward: bank %s no longer has graphiti_group_id; "
            "drain is a no-op (rows will be picked up on next submission)",
            bank_id,
        )
        return ForwardJobResult().as_dict()

    client = client_factory(bank_config) if client_factory is not None else _build_client(bank_config)
    result = ForwardJobResult()
    # Per-bank invalidated-edge scratch. Owned by this call so the module
    # has no hidden global state — see the docstring at the top of the
    # file for the isolation invariant this preserves.
    invalidated_memo: dict[UUID, EdgeResult] = {}

    try:
        while True:
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    rows = await ops.claim_graphiti_outbox_batch(
                        conn,
                        fq_table("graphiti_outbox"),
                        bank_id,
                        batch_size,
                    )
            if not rows:
                break
            result.drained += len(rows)

            failed_ids: list[int] = []
            last_error: str | None = None
            for row in rows:
                try:
                    forwarded, n_relations, invalidated, write_backs = await _process_one_row(
                        memory_engine, client, bank_id, row, invalidated_memo
                    )
                    result.forwarded += forwarded
                    result.relations += n_relations
                    result.invalidated_edges += invalidated
                    result.write_backs += write_backs
                except Exception as e:
                    if _is_transient_error(e):
                        failed_ids.append(row["id"])
                        last_error = f"{type(e).__name__}: {e}"
                    else:
                        logger.exception(
                            "graphiti_forward: permanent error on row %s for bank %s; dropping",
                            row["id"],
                            bank_id,
                        )
                        result.dropped += 1

            if failed_ids:
                async with acquire_with_retry(backend) as conn:
                    await ops.reschedule_graphiti_outbox_rows(
                        conn,
                        fq_table("graphiti_outbox"),
                        failed_ids,
                        last_error or "unknown transient error",
                    )
                result.rescheduled += len(failed_ids)

        await _trigger_backflow_actions(memory_engine, bank_id, invalidated_memo)
    finally:
        await client.aclose()

    logger.info(f"[GRAPHITI_FORWARD] bank={bank_id} {result.as_dict()}")
    return result.as_dict()
