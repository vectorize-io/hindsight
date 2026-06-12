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
   in the response whose ``source_uri`` matches the current bank, mark the
   local memory as needing re-consolidation. The default action is
   re-consolidation only — never rewrites the private memory text (private
   memory owner is always the bank, per main plan §6-5). B1 ledger stamping
   is gated on the explicit ``graphiti_backflow_supersession=true`` flag and
   is a follow-up; the B1 schema already enforces the CHECK constraint that
   prevents landing rows without ``occurred_start``.

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
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid5

from ..db_utils import acquire_with_retry
from ..federation.circuit_breaker import CircuitOpenError
from ..federation.graphiti_client import (
    AddTripletResults,
    EdgePayload,
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

# ``memory_id -> re-consolidation flag`` set used by channel A within a
# single bank drain. The actual re-consolidation request fires once at the
# end of the drain to avoid submitting one job per invalidated edge.
_INVALIDATED_MEMO: dict[UUID, None] = {}


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
) -> int:
    """Channel A: parse ``source_uri`` from invalidated edges, mark local
    memories for re-consolidation.

    Per deep-dive 4 §1.3 the only action taken is *marking* — the actual
    re-consolidation is submitted as a single job at the end of the drain.
    """
    invalidated = results.invalidated_edges
    if not invalidated:
        return 0

    prefix = f"hindsight://bank/{bank_id}/memory/"
    for edge in invalidated:
        if not edge.source_uri or not edge.source_uri.startswith(prefix):
            continue
        try:
            memory_id = UUID(edge.source_uri[len(prefix) :])
        except ValueError:
            logger.warning("graphiti_forward: unparseable source_uri %r", edge.source_uri)
            continue
        _INVALIDATED_MEMO[memory_id] = None
    return len(invalidated)


async def _process_one_row(
    engine: "MemoryEngine",
    client: GraphitiClient,
    bank_id: str,
    row: dict,
) -> tuple[int, int, int, int]:
    """Forward a single outbox row. Returns (forwarded, relations, invalidated, write_backs)."""
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
        invalidated += await _handle_invalidated_edges(engine, bank_id, results)

    write_backs = await _write_back_entity_uuids(engine, bank_id, all_nodes)
    return forwarded, len(relations), invalidated, write_backs


async def _trigger_backflow_actions(engine: "MemoryEngine", bank_id: str) -> None:
    """End-of-drain housekeeping: submit re-consolidation for memories that
    were invalidated upstream.

    The submit is best-effort: a failure here does not wedge the queue.
    Channel A's only invariant is "the local memory's observations get
    refreshed on the next consolidation cycle" — missing the trigger
    simply means consolidation re-runs naturally the next time the bank's
    source facts change, so a dropped job is recoverable.
    """
    if not _INVALIDATED_MEMO:
        return
    memory_ids = list(_INVALIDATED_MEMO.keys())
    _INVALIDATED_MEMO.clear()
    try:
        from hindsight_api.models import RequestContext

        internal = RequestContext(internal=True)
        # Re-consolidation is a no-op for the data itself (the local memory
        # text is never rewritten); it just re-runs the observation refresh
        # over the bank's world facts. We piggyback on the existing
        # consolidation task rather than introducing a new task type —
        # consolidation already gates on source-fact count and is idempotent.
        await engine.submit_async_consolidation(bank_id=bank_id, request_context=internal)
        logger.info(
            "graphiti_forward: triggered re-consolidation for bank %s (invalidated upstream edges → %d local memories)",
            bank_id,
            len(memory_ids),
        )
    except Exception:
        logger.exception("graphiti_forward: failed to submit backflow consolidation for bank %s", bank_id)


def _build_client(bank_config) -> GraphitiClient:
    """Construct a client from the bank's resolved config + env defaults.

    ``GRAPHITI_BASE_URL`` is the single source for the overlay URL; the
    API key is per-bank where set, otherwise falls back to the env
    ``GRAPHITI_API_KEY``. Timeouts/breaker thresholds come from
    ``bank_config`` if present, else from env, else from the client
    defaults (which match the C-track config table in deep-dive 2 §2.6).
    """
    base_url = getattr(bank_config, "graphiti_base_url", None) or os.getenv("GRAPHITI_BASE_URL", "")
    if not base_url:
        raise GraphitiClientError("GRAPHITI_BASE_URL is not set; cannot forward")
    api_key = getattr(bank_config, "graphiti_api_key", None) or os.getenv("GRAPHITI_API_KEY")
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
                        memory_engine, client, bank_id, row
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

        await _trigger_backflow_actions(memory_engine, bank_id)
    finally:
        await client.aclose()

    logger.info(f"[GRAPHITI_FORWARD] bank={bank_id} {result.as_dict()}")
    return result.as_dict()
