"""Automatic fact supersession: contradiction detection over the fact ledger.

When a new world fact contradicts an older one whose validity interval
overlaps, the older fact is *superseded* — ``valid_until``/``superseded_at``/
``superseded_by`` are set (PR: migration ``e7f8a9b0c1d2``), default retrieval
hides it, and ``as_of`` recalls can still see it. Rows are never deleted and
``memory_links`` are untouched, so graph maintenance is a non-event.

Naming: "supersession" (temporal ledger, rows stay) is deliberately distinct
from "invalidation" (reversible curation, rows MOVE to the cold archive).

Pipeline shape clones ``graph_maintenance``: retain Phase 2 enqueues one
``supersession_queue`` row per eligible fact inside the insert transaction;
``_submit_post_insert_maintenance`` submits a bank-level drain task (empty
queue short-circuits, pending jobs dedupe by bank); the worker claims rows,
recalls contradiction candidates, asks a small LLM for a verdict, and applies
pure-function interval algebra.

Only ``world`` facts with a real ``occurred_start`` participate: experience
facts carry no "was replaced" semantics, and interval algebra over ingestion
fallback timestamps would supersede true facts on garbage data (the DB CHECK
``chk_mu_supersession_needs_occurred`` backs the same rule).
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..audit import AuditEntry
from ..db_utils import acquire_with_retry
from ..response_models import TokenUsage
from ..schema import fq_table

if TYPE_CHECKING:
    from hindsight_api.models import RequestContext

    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

_DRAIN_BATCH_SIZE = 25
# LLM/transient errors park the remaining batch back in the queue and retry the
# whole task after this delay (the poller resets it to pending via RetryTaskAt).
_RETRY_DELAY_SECONDS = 120


# ---------------------------------------------------------------------------
# Interval algebra (pure functions — every branch is a unit-test row)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactTimeline:
    """The temporal coordinates of one fact, as needed by interval algebra."""

    id: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None


@dataclass(frozen=True)
class SupersessionAction:
    """Verdict of interval algebra: ``loser`` stops being true at ``valid_until``."""

    loser_id: str
    winner_id: str
    valid_until: datetime


def resolve_supersession(new_fact: FactTimeline, candidate: FactTimeline) -> SupersessionAction | None:
    """Decide which of two *contradicting* facts supersedes the other.

    Caller guarantees the facts contradict (LLM verdict); this function only
    runs the temporal logic:

    * either side missing ``occurred_start`` → None (defensive; enqueue and
      candidate filters should have excluded it),
    * disjoint intervals → None — both stay ("worked at X until March" and
      "works at Y since April" coexist),
    * the earlier-starting fact loses, ``valid_until`` = the later start. This
      includes the reversal case: when the *candidate* starts later than the
      new fact, we are ingesting a late-arriving old message and the NEW fact
      is the one superseded (Graphiti edge_operations late-arrival semantics),
    * equal starts → later ``mentioned_at`` wins (the system's newer knowledge);
      still tied → None (undecidable, left to observation-layer reconciliation).

    ``valid_until`` must end up strictly after the loser's ``occurred_start``
    (DB CHECK ``chk_mu_valid_until_after_start``); equal-start ties would
    violate that, which is the second reason they resolve via ``mentioned_at``
    or not at all.
    """
    if new_fact.occurred_start is None or candidate.occurred_start is None:
        logger.warning(
            "resolve_supersession called with missing occurred_start (new=%s, candidate=%s)",
            new_fact.id,
            candidate.id,
        )
        return None

    if _intervals_disjoint(new_fact, candidate):
        return None

    if new_fact.occurred_start != candidate.occurred_start:
        if new_fact.occurred_start < candidate.occurred_start:
            earlier, later = new_fact, candidate
        else:
            earlier, later = candidate, new_fact
        return SupersessionAction(loser_id=earlier.id, winner_id=later.id, valid_until=later.occurred_start)

    # Equal occurred_start: prefer what the system learned later.
    new_seen = new_fact.mentioned_at
    cand_seen = candidate.mentioned_at
    if new_seen is None or cand_seen is None or new_seen == cand_seen:
        return None
    loser, winner, boundary = (
        (candidate, new_fact, new_seen) if new_seen > cand_seen else (new_fact, candidate, cand_seen)
    )
    # valid_until must be strictly > occurred_start (CHECK constraint); with
    # equal starts the only defensible boundary is "when the superseding
    # knowledge arrived", floored just past the shared start.
    boundary = max(boundary, new_fact.occurred_start + timedelta(seconds=1))
    return SupersessionAction(loser_id=loser.id, winner_id=winner.id, valid_until=boundary)


def _intervals_disjoint(a: FactTimeline, b: FactTimeline) -> bool:
    """True when both validity intervals are closed and don't overlap."""
    if a.occurred_end is not None and b.occurred_start is not None and a.occurred_end < b.occurred_start:
        return True
    if b.occurred_end is not None and a.occurred_start is not None and b.occurred_end < a.occurred_start:
        return True
    return False


# ---------------------------------------------------------------------------
# LLM arbitration
# ---------------------------------------------------------------------------


class SupersessionVerdict(BaseModel):
    """One arbitration call classifies all candidates at once (contiguous indices)."""

    duplicate_indices: list[int] = Field(
        default_factory=list,
        description="Candidates stating the SAME fact as the new fact (re-statements, paraphrases).",
    )
    contradicted_indices: list[int] = Field(
        default_factory=list,
        description="Candidates the new fact CONTRADICTS or makes obsolete "
        "(same subject, incompatible state). NOT mere wording differences.",
    )


_ARBITRATION_PROMPT = """You compare a NEW fact against numbered CANDIDATE facts from the same memory bank.

Classify each candidate index into at most one list:
- duplicate_indices: the candidate states the SAME thing as the new fact (re-statement or paraphrase).
- contradicted_indices: the candidate and the new fact CANNOT both be true of the same subject at the
  same time (e.g. a changed employer, location, status, ownership).

Rules:
- Wording differences, partial overlaps, or related-but-compatible facts belong in NEITHER list.
- Facts about different subjects are never duplicates or contradictions.
- When unsure, leave the candidate out. Missing a contradiction is recoverable; a false one hides a true fact.
"""


def validate_verdict_indices(verdict: SupersessionVerdict, candidate_count: int) -> SupersessionVerdict:
    """Drop out-of-range indices (defensive parse, mirrors relation_extraction)."""

    def _in_range(indices: list[int], label: str) -> list[int]:
        kept = [i for i in indices if 0 <= i < candidate_count]
        if len(kept) != len(indices):
            logger.warning(
                "Dropping out-of-range %s indices from supersession verdict: %s (candidate_count=%d)",
                label,
                [i for i in indices if not 0 <= i < candidate_count],
                candidate_count,
            )
        return kept

    return SupersessionVerdict(
        duplicate_indices=_in_range(verdict.duplicate_indices, "duplicate"),
        contradicted_indices=_in_range(verdict.contradicted_indices, "contradicted"),
    )


@dataclass(frozen=True)
class ArbitrationResult:
    verdict: SupersessionVerdict
    usage: TokenUsage


async def _arbitrate(
    llm_config: Any,
    new_fact_text: str,
    candidate_texts: list[str],
) -> ArbitrationResult:
    """One low-cost LLM call returning duplicates + contradictions together."""
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    user_message = f"NEW FACT:\n{new_fact_text}\n\nCANDIDATES:\n{numbered}"
    raw, usage = await llm_config.call(
        messages=[
            {"role": "system", "content": _ARBITRATION_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=SupersessionVerdict,
        scope="fact_supersession_arbitrate",
        temperature=0.0,
        skip_validation=True,
        return_usage=True,
    )
    verdict = SupersessionVerdict.model_validate_json(raw) if isinstance(raw, str) else SupersessionVerdict()
    return ArbitrationResult(verdict=validate_verdict_indices(verdict, len(candidate_texts)), usage=usage)


# ---------------------------------------------------------------------------
# Enqueue (inside the retain Phase 2 transaction)
# ---------------------------------------------------------------------------


async def enqueue_supersession_checks(
    conn: Any,
    bank_id: str,
    unit_ids: list[str],
    processed_facts: list[Any],
) -> int:
    """Queue eligible freshly-inserted facts for the supersession worker.

    Filters at write time (queue rows cost worker recall+LLM calls, so garbage
    must not enter): world facts with a real ``occurred_start`` only. The
    caller gates on the bank's ``enable_fact_supersession`` config.
    """
    eligible = [
        uuid.UUID(uid)
        for uid, fact in zip(unit_ids, processed_facts)
        if fact.fact_type == "world" and fact.occurred_start is not None
    ]
    if not eligible:
        return 0
    await conn.executemany(
        f"INSERT INTO {fq_table('supersession_queue')} (bank_id, memory_id) VALUES ($1, $2)",
        [(bank_id, mid) for mid in eligible],
    )
    return len(eligible)


# ---------------------------------------------------------------------------
# Worker job
# ---------------------------------------------------------------------------


@dataclass
class SupersessionJobResult:
    """Counters reported by one drain run."""

    checked: int = 0
    superseded: int = 0
    duplicates_noted: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "checked": self.checked,
            "superseded": self.superseded,
            "duplicates_noted": self.duplicates_noted,
            "skipped": self.skipped,
        }


async def run_fact_supersession_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
) -> dict[str, int]:
    """Drain ``supersession_queue`` for one bank.

    Claim-then-process (claim commits before any LLM call — long transactions
    around LLM latency are not acceptable). On a transient failure mid-batch
    the unprocessed ids are re-enqueued and the task retries via RetryTaskAt;
    per-fact judgment errors drop that row with an audit entry so one poison
    fact can't wedge the bank's queue.
    """
    from ..memory_engine import Budget

    backend = await memory_engine._get_backend()
    ops = backend.ops
    bank_config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)
    candidate_limit = bank_config.fact_supersession_candidate_limit
    recall_budget = Budget(bank_config.fact_supersession_recall_budget)

    result = SupersessionJobResult()
    any_superseded = False

    while True:
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                memory_ids = await ops.claim_supersession_batch(
                    conn,
                    fq_table("supersession_queue"),
                    bank_id,
                    _DRAIN_BATCH_SIZE,
                )
        if not memory_ids:
            break

        for position, memory_id in enumerate(memory_ids):
            try:
                outcome = await _check_one_fact(
                    memory_engine,
                    backend,
                    bank_id,
                    memory_id,
                    candidate_limit,
                    recall_budget,
                    request_context,
                    operation_id,
                )
                if outcome is None:
                    # Fact vanished (curation/delta delete) or was already
                    # superseded by an earlier batch — a non-event, not a check.
                    continue
                result.checked += 1
                result.superseded += outcome.superseded
                result.duplicates_noted += outcome.duplicates
                any_superseded = any_superseded or outcome.superseded > 0
            except _TransientSupersessionError as e:
                # Park this id and everything after it back in the queue, then
                # let the poller reschedule the whole task.
                remaining = memory_ids[position:]
                async with acquire_with_retry(backend) as conn:
                    await conn.executemany(
                        f"INSERT INTO {fq_table('supersession_queue')} (bank_id, memory_id) VALUES ($1, $2)",
                        [(bank_id, uuid.UUID(mid)) for mid in remaining],
                    )
                from ...worker.exceptions import RetryTaskAt

                raise RetryTaskAt(
                    datetime.now(UTC) + timedelta(seconds=_RETRY_DELAY_SECONDS),
                    f"transient supersession failure ({e}); re-enqueued {len(remaining)} facts",
                ) from e
            except Exception:
                # Poison row: drop it (the queue row is already claimed) but
                # never wedge the batch.
                logger.exception(
                    "fact_supersession: skipping fact %s in bank %s after unexpected error", memory_id, bank_id
                )
                result.skipped += 1
                memory_engine.audit_logger.log_fire_and_forget(
                    AuditEntry(
                        action="fact_supersession_skipped",
                        transport="system",
                        bank_id=bank_id,
                        metadata={"memory_id": memory_id, "operation_id": operation_id},
                    )
                )

    # Superseded facts feed belief evolution: consolidation re-reads them
    # (its source pull is deliberately exempt from the validity filter).
    if any_superseded and bank_config.enable_observations and bank_config.enable_auto_consolidation:
        try:
            await memory_engine.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        except Exception as e:
            logger.warning(f"fact_supersession: failed to submit consolidation for bank {bank_id}: {e}")

    logger.info(f"[FACT_SUPERSESSION] bank={bank_id} {result.as_dict()}")
    return result.as_dict()


class _TransientSupersessionError(Exception):
    """LLM/infra failure that warrants re-enqueue + task retry."""


@dataclass(frozen=True)
class _CheckOutcome:
    """Counters from checking one queued fact."""

    superseded: int
    duplicates: int


async def _check_one_fact(
    memory_engine: "MemoryEngine",
    backend: Any,
    bank_id: str,
    memory_id: str,
    candidate_limit: int,
    recall_budget: Any,
    request_context: "RequestContext",
    operation_id: str | None,
) -> "_CheckOutcome | None":
    """Run steps 0-5 for one queued fact. None = step-0 self-check miss (non-event)."""
    mu = fq_table("memory_units")

    # Step 0: self-check — the fact may have been curated away, deleted by
    # delta retain, or already superseded by an earlier batch.
    async with acquire_with_retry(backend) as conn:
        row = await conn.fetchrow(
            f"SELECT text, document_id, occurred_start, occurred_end, mentioned_at "
            f"FROM {mu} WHERE id = $1 AND bank_id = $2 AND valid_until IS NULL",
            uuid.UUID(memory_id),
            bank_id,
        )
    if row is None or row["occurred_start"] is None:
        return None

    new_fact = FactTimeline(
        id=memory_id,
        occurred_start=row["occurred_start"],
        occurred_end=row["occurred_end"],
        mentioned_at=row["mentioned_at"],
    )

    # Step 1: candidate recall — same internal-recall shape as consolidation's
    # dedup recall (interleave guarantees each arm's top hit a slot; RRF and the
    # cross-encoder were both measured to bury the near-identical twin).
    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=row["text"],
            budget=recall_budget,
            max_tokens=2048,
            fact_type=["world"],
            request_context=request_context,
            reranking="interleave",
            _quiet=True,
        )
    except Exception as e:
        raise _TransientSupersessionError(f"candidate recall failed: {e}") from e

    candidates = [
        r
        for r in recall_result.results
        if r.id != memory_id
        and r.occurred_start is not None
        # Same-document "contradictions" are extraction-granularity noise, not
        # temporal succession; the observation layer reconciles those.
        and (r.document_id is None or r.document_id != row["document_id"])
    ][:candidate_limit]
    if not candidates:
        return _CheckOutcome(superseded=0, duplicates=0)

    # Step 2: LLM arbitration (one call for the whole candidate list).
    try:
        arbitration = await _arbitrate(
            memory_engine._consolidation_llm_config,
            row["text"],
            [c.text for c in candidates],
        )
    except Exception as e:
        raise _TransientSupersessionError(f"arbitration failed: {e}") from e
    verdict = arbitration.verdict

    duplicates = len(verdict.duplicate_indices)
    if duplicates:
        # v1 records duplicates without acting — chunk-level dedup and
        # consolidation merging already exist; this is decision data for a
        # future fact-level dedup, collected at zero marginal LLM cost.
        memory_engine.audit_logger.log_fire_and_forget(
            AuditEntry(
                action="fact_supersession_duplicates",
                transport="system",
                bank_id=bank_id,
                metadata={
                    "memory_id": memory_id,
                    "duplicate_ids": [candidates[i].id for i in verdict.duplicate_indices],
                },
            )
        )

    if not verdict.contradicted_indices:
        return _CheckOutcome(superseded=0, duplicates=duplicates)

    # Step 3 needs exact datetimes (recall results carry ISO strings) and must
    # re-verify the candidates are still live — refetch in one round-trip.
    contradicted_ids = [uuid.UUID(candidates[i].id) for i in verdict.contradicted_indices]
    async with acquire_with_retry(backend) as conn:
        cand_rows = await conn.fetch(
            f"SELECT id, occurred_start, occurred_end, mentioned_at FROM {mu} "
            f"WHERE id = ANY($1::uuid[]) AND bank_id = $2 AND valid_until IS NULL",
            contradicted_ids,
            bank_id,
        )

        superseded = 0
        for cand in cand_rows:
            action = resolve_supersession(
                new_fact,
                FactTimeline(
                    id=str(cand["id"]),
                    occurred_start=cand["occurred_start"],
                    occurred_end=cand["occurred_end"],
                    mentioned_at=cand["mentioned_at"],
                ),
            )
            if action is None:
                continue
            # Step 4: idempotent write — "first verdict wins". A concurrent or
            # retried run hits valid_until IS NULL = false and silently yields.
            # consolidated_at resets so the next consolidation re-reads the
            # superseded fact and narrates the belief change (step 5).
            updated = await conn.execute(
                f"""
                UPDATE {mu}
                SET valid_until = $1, superseded_at = now(), superseded_by = $2, consolidated_at = NULL
                WHERE id = $3 AND bank_id = $4 AND valid_until IS NULL
                """,
                action.valid_until,
                uuid.UUID(action.winner_id),
                uuid.UUID(action.loser_id),
                bank_id,
            )
            if updated.endswith("1"):
                superseded += 1
                memory_engine.audit_logger.log_fire_and_forget(
                    AuditEntry(
                        action="fact_superseded",
                        transport="system",
                        bank_id=bank_id,
                        metadata={
                            "loser_id": action.loser_id,
                            "winner_id": action.winner_id,
                            "valid_until": action.valid_until.isoformat(),
                            "detector": "fact_supersession_worker",
                            "operation_id": operation_id,
                        },
                    )
                )

    return _CheckOutcome(superseded=superseded, duplicates=duplicates)
