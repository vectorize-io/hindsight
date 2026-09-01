"""PostgreSQL implementation of DataAccessOps.

Uses unnest(), LATERAL, DISTINCT ON, and native array operations for
efficient batch operations.
"""

import asyncio
from datetime import datetime

from .base import DatabaseConnection
from .ops import (
    ClaimedOperations,
    DataAccessOps,
    LinkExpansionRows,
    TagListingParts,
    UpdatedWindow,
    bank_serialization_sql,
    document_serialization_sql,
)
from .result import ResultRow


def pg_search_vector_expr(
    config,
    *,
    text_col: str = "text",
    context_col: str = "context",
    signals_col: str | None = "text_signals",
    native_inline: bool = True,
) -> str | None:
    """SQL expression that builds ``search_vector`` for the configured PG text-search backend.

    Single source of truth shared by ``memory_units`` (the batch insert over the
    ``input_data`` CTE columns and the curation-revert recompute) and
    ``mental_models`` (the knowledge-page writes), so the per-backend tokenization
    can never drift between the two tables. Returns ``None`` for backends that
    leave ``search_vector`` unpopulated — pgroonga / pg_textsearch / pg_search
    index the base text columns directly and keep only a dummy column, so there is
    nothing to build.

    The ``*_col`` arguments are the SQL for each text source (a column name or a
    bind placeholder); pass ``signals_col=None`` for a two-column table like
    ``mental_models`` (name + content). Pass ``native_inline=False`` when the
    table's native ``search_vector`` is a GENERATED column that populates itself
    (``mental_models``) — writing it inline would fail; only vchord's plain
    bm25vector column then needs an explicit value.

    ``text_search_extension_native_language`` is validated as a PG identifier in
    ``HindsightConfig.validate()``, so embedding it as a SQL literal is safe.
    """
    cols = [text_col, context_col] + ([signals_col] if signals_col is not None else [])
    combined = " || ' ' || ".join(f"COALESCE({c}, '')" for c in cols)
    if config.text_search_extension == "vchord":
        return f"tokenize({combined}, 'llmlingua2')::bm25_catalog.bm25vector"
    if config.text_search_extension == "native" and native_inline:
        return f"to_tsvector('{config.text_search_extension_native_language}'::regconfig, {combined})"
    return None


class PostgreSQLOps(DataAccessOps):
    """PostgreSQL-specific data access operations using unnest and LATERAL."""

    def __init__(self) -> None:
        # Per-table serialization of per-bank vector-index DDL within this
        # process. Concurrent index DDL on one relation deadlocks by design:
        # DROP INDEX CONCURRENTLY holds ShareUpdateExclusive while it waits out
        # every transaction whose snapshot could still see the index — including
        # other sessions' index DDL queued on that same lock — so many banks
        # deleted at once form a wait cycle Postgres resolves by killing one.
        # A session advisory lock would serialize this across processes too, but
        # advisory locks are banned here (poolers hand sessions around; see the
        # Database Locking standard). In-process the asyncio lock removes the
        # cycle outright; across processes the callers' retry-with-backoff
        # absorbs the (now much rarer) collisions.
        self._index_ddl_locks: dict[str, asyncio.Lock] = {}

    def _index_ddl_lock(self, table: str) -> asyncio.Lock:
        return self._index_ddl_locks.setdefault(table, asyncio.Lock())

    @property
    def uses_observation_sources_table(self) -> bool:
        return False  # PG uses native array ops on source_memory_ids

    async def bulk_upsert_chunks(
        self,
        conn: DatabaseConnection,
        table: str,
        chunk_ids: list[str],
        document_ids: list[str],
        bank_ids: list[str],
        chunk_texts: list[str],
        chunk_indices: list[int],
        content_hashes: list[str],
    ) -> None:
        await conn.execute(
            f"""
            INSERT INTO {table} (chunk_id, document_id, bank_id, chunk_text, chunk_index, content_hash)
            SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::integer[], $6::text[])
            ON CONFLICT (chunk_id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                chunk_index = EXCLUDED.chunk_index,
                content_hash = EXCLUDED.content_hash
            """,
            chunk_ids,
            document_ids,
            bank_ids,
            chunk_texts,
            chunk_indices,
            content_hashes,
        )

    async def lock_document_for_write(
        self,
        conn: DatabaseConnection,
        table: str,
        doc_id: str,
        bank_id: str,
    ) -> str | None:
        # Single upsert that both creates the row (if absent) and locks it (if
        # present) atomically. ON CONFLICT DO UPDATE always takes the row lock as
        # part of the statement, so all concurrent same-document writers serialize
        # on the document row in one consistent step (the earlier two-step form —
        # DO NOTHING + a separate SELECT FOR UPDATE — could deadlock because
        # DO NOTHING takes no lock on an existing row). The SET is a no-op
        # self-assignment used only to acquire the lock; RETURNING yields the
        # pre-existing hash (or '__pending__' for a freshly inserted row).
        return await conn.fetchval(
            f"INSERT INTO {table} (id, bank_id, original_text, content_hash) "
            f"VALUES ($1, $2, '', '__pending__') "
            f"ON CONFLICT (id, bank_id) DO UPDATE SET content_hash = {table}.content_hash "
            f"RETURNING content_hash",
            doc_id,
            bank_id,
        )

    async def insert_facts_batch(
        self,
        conn: DatabaseConnection,
        bank_id: str,
        fact_texts: list[str],
        embeddings: list[str],
        event_dates: list,
        occurred_starts: list,
        occurred_ends: list,
        mentioned_ats: list,
        contexts: list[str],
        fact_types: list[str],
        metadata_jsons: list[str],
        chunk_ids: list,
        document_ids: list,
        tags_list: list[str],
        observation_scopes_list: list,
        text_signals_list: list,
        text_search_extension: str = "native",
    ) -> list[str]:
        from ...config import get_config

        config = get_config()
        table = self._get_mu_table()

        # search_vector is populated inline for backends that store a real vector
        # (native tsvector, vchord bm25vector). pgroonga / pg_textsearch / pg_search
        # index the base text columns directly and keep only a dummy column, so the
        # expression is None and the column is left out of the insert entirely.
        # Same expression is reused by curation revert (see pg_search_vector_expr).
        sv_expr = pg_search_vector_expr(config)
        sv_insert_col = ", search_vector" if sv_expr else ""
        sv_select_val = f",\n                    {sv_expr}" if sv_expr else ""
        query = f"""
            WITH input_data AS (
                SELECT * FROM unnest(
                    $2::text[], $3::vector[], $4::timestamptz[], $5::timestamptz[], $6::timestamptz[], $7::timestamptz[],
                    $8::text[], $9::text[], $10::jsonb[], $11::text[], $12::text[], $13::jsonb[], $14::jsonb[], $15::text[]
                ) AS t(text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                       context, fact_type, metadata, chunk_id, document_id, tags_json,
                       observation_scopes_json, text_signals)
            )
            INSERT INTO {table} (bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                                 context, fact_type, metadata, chunk_id, document_id, tags,
                                 observation_scopes, text_signals{sv_insert_col})
            SELECT
                $1,
                text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                context, fact_type, metadata, chunk_id, document_id,
                COALESCE(
                    (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                    '{{}}'::varchar[]
                ),
                observation_scopes_json,
                text_signals{sv_select_val}
            FROM input_data
            RETURNING id
        """

        results = await conn.fetch(
            query,
            bank_id,
            fact_texts,
            embeddings,
            event_dates,
            occurred_starts,
            occurred_ends,
            mentioned_ats,
            contexts,
            fact_types,
            metadata_jsons,
            chunk_ids,
            document_ids,
            tags_list,
            observation_scopes_list,
            text_signals_list,
        )
        return [str(row["id"]) for row in results]

    async def bulk_insert_links(
        self,
        conn: DatabaseConnection,
        table: str,
        sorted_links: list[tuple],
        bank_id: str,
        nil_entity_uuid: str,
        exists_clause: str,
        chunk_size: int = 5000,
    ) -> None:
        # exists_clause is unused on PostgreSQL: the memory_links → memory_units
        # FKs are DEFERRABLE INITIALLY DEFERRED, so an INSERT takes no lock on the
        # referenced parent rows until COMMIT — a concurrent committed DELETE in
        # that window (consolidation pruning observations, document re-tracking)
        # trips fk_memory_links_{to,from}_unit_id_memory_units at COMMIT (#1882),
        # and a WHERE EXISTS guard can't prevent it (the row passes the check,
        # then is deleted before the deferred check runs). Instead a CTE locks the
        # referenced units FOR KEY SHARE in the *same statement*: the lock blocks a
        # concurrent DELETE until our transaction commits and is held through the
        # deferred check, and the INSERT only takes links whose endpoints are in
        # the locked set, so rows that already vanished are dropped. Folding it
        # into the one INSERT keeps this to a single round-trip — no extra query
        # and no surrounding transaction needed. (Oracle's immediate FK has no
        # such window and uses exists_clause via its own bulk_insert_links.)
        from ..schema import fq_table

        mu_table = fq_table("memory_units")
        from_ids = [lnk[0] for lnk in sorted_links]
        to_ids = [lnk[1] for lnk in sorted_links]
        types = [lnk[2] for lnk in sorted_links]
        weights = [lnk[3] for lnk in sorted_links]
        entity_ids = [lnk[4] for lnk in sorted_links]

        for chunk_start in range(0, len(sorted_links), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sorted_links))
            chunk_from = from_ids[chunk_start:chunk_end]
            chunk_to = to_ids[chunk_start:chunk_end]
            # Distinct referenced parents, sorted so concurrent inserters acquire
            # the row-share locks in a consistent order (avoids deadlocks; same
            # convention as the (from, to) link sort).
            referenced = sorted({str(x) for x in chunk_from} | {str(x) for x in chunk_to})
            await conn.execute(
                f"""
                WITH locked AS (
                    SELECT id FROM {mu_table}
                    WHERE id = ANY($7::uuid[])
                    ORDER BY id
                    FOR KEY SHARE
                )
                INSERT INTO {table}
                    (from_unit_id, to_unit_id, link_type, weight, entity_id, bank_id)
                SELECT f, t, tp, w, e, $6
                FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::float8[], $5::uuid[])
                    AS u(f, t, tp, w, e)
                WHERE f IN (SELECT id FROM locked) AND t IN (SELECT id FROM locked)
                ON CONFLICT (from_unit_id, to_unit_id, link_type,
                             COALESCE(entity_id, '{nil_entity_uuid}'::uuid))
                DO NOTHING
                """,
                chunk_from,
                chunk_to,
                types[chunk_start:chunk_end],
                weights[chunk_start:chunk_end],
                entity_ids[chunk_start:chunk_end],
                bank_id,
                referenced,
                timeout=300,
            )

    async def bulk_insert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_names: list[str],
        entity_dates: list,
        entity_kinds: list[str],
    ) -> dict[str, str]:
        # ORDER BY LOWER(name) so every concurrent batch inserts in the same order
        # as the conflict target (bank_id, LOWER(canonical_name)). ON CONFLICT DO
        # NOTHING takes a ShareLock on the inserting transaction of any speculative
        # row it collides with, so two batches with overlapping names inserting in
        # different orders deadlock. The caller already sorts by Python's
        # ``str.lower()``, which agrees with the index for ASCII but not for every
        # locale (see the Turkish-İ note in entity_resolver) — ordering in SQL makes
        # the database's own collation the single arbiter for all writers.
        inserted_rows = await conn.fetch(
            f"""
            INSERT INTO {table} (bank_id, canonical_name, first_seen, last_seen, mention_count, entity_kind)
            SELECT $1, name, COALESCE(event_date, now()), COALESCE(event_date, now()), 0, kind
            FROM unnest($2::text[], $3::timestamptz[], $4::text[]) AS t(name, event_date, kind)
            ORDER BY LOWER(name)
            ON CONFLICT (bank_id, LOWER(canonical_name))
            DO NOTHING
            RETURNING id, LOWER(canonical_name) AS name_lower
            """,
            bank_id,
            entity_names,
            entity_dates,
            entity_kinds,
        )
        return {row["name_lower"]: row["id"] for row in inserted_rows}

    async def fetch_missing_entity_ids(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        missing_names: list[str],
    ) -> list[ResultRow]:
        return await conn.fetch(
            f"""
            SELECT e.id, e.canonical_name, LOWER(e.canonical_name) AS name_lower, inputs.input_name
            FROM {table} e
            JOIN (
                SELECT LOWER(n) AS input_name_lower, n AS input_name
                FROM unnest($2::text[]) AS n
            ) AS inputs ON LOWER(e.canonical_name) = inputs.input_name_lower
            WHERE e.bank_id = $1
            """,
            bank_id,
            missing_names,
        )

    async def bulk_reassert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_ids: list[str],
        canonical_names: list[str],
        entity_kinds: list[str],
    ) -> None:
        # One statement, one round-trip (same shape as bulk_insert_links):
        #   * the CTE takes FOR KEY SHARE on every parent that still exists,
        #     held to COMMIT, so a concurrent prune_orphan_entities DELETE blocks
        #     until the caller's unit_entities insert has committed;
        #   * the INSERT re-creates only the parents that were already pruned
        #     (NOT IN locked), carrying the canonical_name resolved in Phase 1.
        # ON CONFLICT DO NOTHING (no target) keeps the rare case where another
        # worker recreated the name under a new id from raising — that row stays
        # absent and its unit link is the sole casualty, never the whole batch.
        await conn.execute(
            f"""
            WITH locked AS (
                SELECT id FROM {table}
                WHERE id = ANY($2::uuid[])
                ORDER BY id
                FOR KEY SHARE
            )
            INSERT INTO {table} (id, bank_id, canonical_name, entity_kind)
            SELECT t.entity_id, $1, t.canonical_name, t.entity_kind
            FROM unnest($2::uuid[], $3::text[], $4::text[]) AS t(entity_id, canonical_name, entity_kind)
            WHERE t.entity_id NOT IN (SELECT id FROM locked)
            ON CONFLICT DO NOTHING
            """,
            bank_id,
            entity_ids,
            canonical_names,
            entity_kinds,
        )

    async def bulk_insert_unit_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        unit_ids: list,
        entity_ids: list,
    ) -> None:
        await conn.execute(
            f"""
            INSERT INTO {table} (unit_id, entity_id)
            SELECT u, e FROM unnest($1::uuid[], $2::uuid[]) AS t(u, e)
            ON CONFLICT DO NOTHING
            """,
            unit_ids,
            entity_ids,
        )

    async def enqueue_graph_maintenance(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        unit_ids: list,
    ) -> None:
        if not unit_ids:
            return
        # Sort to enforce a global lock-acquisition order on the
        # (bank_id, unit_id) unique-key. Without this, two concurrent
        # transactions inserting overlapping unit_id sets in different
        # orders can deadlock on the ON CONFLICT row locks — Postgres
        # acquires a short-lived lock per row being checked, and cycle
        # detection then aborts one transaction. Sorting gives every
        # concurrent caller the same lock order, so conflicting inserts
        # queue cleanly instead of cycling.
        sorted_unit_ids = sorted(unit_ids)
        # DO UPDATE (not DO NOTHING) on a duplicate enqueue — #3034. The SET is a
        # deliberate no-op that preserves enqueued_at; its only purpose is to take
        # the existing row's lock. DO NOTHING does NOT lock the conflicting row, so
        # a mutation that re-enqueues an already-queued unit could not block a
        # worker from concurrently claiming (deleting) that row and processing the
        # unit's pre-mutation state; the re-enqueue signal was then silently lost
        # and the unit's derived links stayed stale with an empty queue. Locking
        # the row serialises the mutation against the worker's claim for that
        # (bank_id, unit_id): the worker either waits for the committed post-mutation
        # state, or (if it claimed first) this INSERT lands a fresh row after the
        # worker's delete commits. Row locks are acquired in sorted unit_id order,
        # matching claim_graph_maintenance_batch, so the two never cycle.
        await conn.execute(
            f"""
            INSERT INTO {table} (bank_id, unit_id)
            SELECT $1, v FROM unnest($2::uuid[]) AS t(v)
            ON CONFLICT (bank_id, unit_id)
                DO UPDATE SET enqueued_at = {table}.enqueued_at
            """,
            bank_id,
            sorted_unit_ids,
        )

    async def claim_graph_maintenance_batch(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        limit: int,
    ) -> list[str]:
        # Ordered locking (#3034). Choose the oldest batch by enqueued_at, but
        # acquire the row locks in (bank_id, unit_id) order — the same order the
        # enqueue upsert takes them — so a foreground mutation re-enqueueing an
        # overlapping unit set can never cycle against a worker draining it. The
        # `chosen` CTE is MATERIALIZED so the enqueued_at pick is fenced from the
        # locking clause; `FOR UPDATE OF q ... ORDER BY q.unit_id` then puts
        # LockRows above the Sort, so locks are taken ascending by unit_id (same
        # idiom as prune_stale_cooccurrences' #2529 ordered lock). A concurrent
        # enqueue holding one of these rows blocks this claim until it commits, at
        # which point the worker deletes and processes the committed state.
        rows = await conn.fetch(
            f"""
            WITH chosen AS MATERIALIZED (
                SELECT bank_id, unit_id FROM {table}
                WHERE bank_id = $1
                ORDER BY enqueued_at
                LIMIT $2
            ),
            locked AS (
                SELECT q.bank_id, q.unit_id
                FROM {table} q
                JOIN chosen c ON c.bank_id = q.bank_id AND c.unit_id = q.unit_id
                ORDER BY q.unit_id
                FOR UPDATE OF q
            )
            DELETE FROM {table} q
            USING locked l
            WHERE q.bank_id = l.bank_id AND q.unit_id = l.unit_id
            RETURNING q.unit_id
            """,
            bank_id,
            limit,
        )
        return [str(row["unit_id"]) for row in rows]

    async def enqueue_entity_maintenance(
        self,
        conn: DatabaseConnection,
        table: str,
        ue_table: str,
        bank_id: str,
        unit_ids: list,
    ) -> int:
        # Read the candidates straight out of unit_entities rather than making
        # callers pass entity ids: every caller runs this immediately before the
        # rows go, and the join they'd have to write is this one.
        #
        # The inner ORDER BY is load-bearing, not cosmetic: it makes the INSERT
        # take the (bank_id, entity_id) row locks ascending, the same order
        # claim_entity_maintenance_batch takes them, so a mutation enqueueing an
        # overlapping candidate set cannot cycle against a worker draining it.
        # (Same protocol as enqueue_graph_maintenance, which sorts in Python
        # because its ids arrive as a bind array.)
        #
        # DO UPDATE (not DO NOTHING) on a duplicate — #3034. The SET is a
        # deliberate no-op preserving enqueued_at; its only purpose is to lock
        # the conflicting row. DO NOTHING does not lock it, so a delete
        # re-enqueueing an already-queued entity could not block a worker from
        # claiming that row and evaluating the entity's pre-delete state — it
        # would find the entity still referenced, keep it, and the re-enqueue
        # signal would be lost, stranding the orphan until some later delete
        # happened to name it again.
        result = await conn.execute(
            f"""
            INSERT INTO {table} (bank_id, entity_id)
            SELECT $1, s.entity_id
            FROM (
                SELECT DISTINCT ue.entity_id
                FROM {ue_table} ue
                WHERE ue.unit_id = ANY($2::uuid[])
                ORDER BY 1
            ) s
            ON CONFLICT (bank_id, entity_id)
                DO UPDATE SET enqueued_at = {table}.enqueued_at
            """,
            bank_id,
            unit_ids,
        )
        return int(result.split()[-1]) if isinstance(result, str) and result.startswith("INSERT") else 0

    async def claim_entity_maintenance_batch(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        limit: int,
    ) -> list:
        # Same claim shape as claim_graph_maintenance_batch: pick the oldest
        # batch by enqueued_at, but acquire the row locks in (bank_id, entity_id)
        # order — the order enqueue_entity_maintenance takes them — so a
        # concurrent enqueue can never cycle against this claim. `chosen` is
        # MATERIALIZED so the enqueued_at pick is fenced from the locking clause,
        # and `FOR UPDATE OF q ... ORDER BY q.entity_id` puts LockRows above the
        # Sort.
        rows = await conn.fetch(
            f"""
            WITH chosen AS MATERIALIZED (
                SELECT bank_id, entity_id FROM {table}
                WHERE bank_id = $1
                ORDER BY enqueued_at
                LIMIT $2
            ),
            locked AS (
                SELECT q.bank_id, q.entity_id
                FROM {table} q
                JOIN chosen c ON c.bank_id = q.bank_id AND c.entity_id = q.entity_id
                ORDER BY q.entity_id
                FOR UPDATE OF q
            )
            DELETE FROM {table} q
            USING locked l
            WHERE q.bank_id = l.bank_id AND q.entity_id = l.entity_id
            RETURNING q.entity_id
            """,
            bank_id,
            limit,
        )
        return [row["entity_id"] for row in rows]

    async def prune_orphan_entities(
        self,
        conn: DatabaseConnection,
        entities_table: str,
        ue_table: str,
        bank_id: str,
        entity_ids: list,
    ) -> int:
        # Scoped to the claimed candidates: primary-key lookups, with the
        # NOT EXISTS backed by idx_unit_entities_entity_unit. Cost tracks the
        # batch, not the bank (#3222) — the bank-wide form this replaces probed
        # once per entity in the bank on every single run.
        #
        # Victims are locked in id order before the delete so the locks are
        # acquired the same way retain's entity upsert takes them
        # (bulk_upsert_entities locks `ORDER BY id FOR KEY SHARE`), which is what
        # keeps a prune and a concurrent re-assert from cycling.
        result = await conn.execute(
            f"""
            WITH victims AS (
                SELECT e.id
                FROM {entities_table} e
                WHERE e.bank_id = $1
                  AND e.id = ANY($2::uuid[])
                  AND NOT EXISTS (
                      SELECT 1 FROM {ue_table} ue WHERE ue.entity_id = e.id
                  )
                ORDER BY e.id
                FOR UPDATE
            )
            DELETE FROM {entities_table} e
            USING victims v
            WHERE e.id = v.id
            """,
            bank_id,
            entity_ids,
        )
        # asyncpg returns "DELETE N"
        return int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0

    async def prune_stale_cooccurrences(
        self,
        conn: DatabaseConnection,
        ec_table: str,
        ue_table: str,
        entity_ids: list,
    ) -> int:
        # Scoped to cooccurrence rows incident to the claimed candidates. The
        # two arms are a UNION rather than
        # `WHERE entity_id_1 = ANY(...) OR entity_id_2 = ANY(...)`: an OR across
        # two columns of the same table cannot be driven from either index, so
        # the planner would make entity_cooccurrences the outer relation and
        # scan it whole — the #3387 shape. As a UNION each arm is an index scan
        # (the PK for arm 1, idx_entity_cooccurrences_entity2 for arm 2).
        #
        # No bank predicate: entities don't span banks and the candidates came
        # off a bank-scoped queue, so both endpoints are already this bank's.
        #
        # Ordered locking (deadlock avoidance, #2529): retain's concurrent
        # cooccurrence upsert (entity_resolver._flush_pending) locks rows in
        # sorted (entity_id_1, entity_id_2) order — sorted specifically to give
        # every writer one consistent lock-acquisition order. A plain
        # `DELETE ... USING` scans/locks in whatever order the join plan picks,
        # so it could lock the same rows in the opposite order and cycle. We
        # instead select the victims in that same sorted order `FOR UPDATE`
        # first — the locking clause materialises the CTE and places LockRows
        # above the Sort, so locks are acquired ascending, matching the upsert —
        # then delete the already-locked rows. Same order on both sides ⇒ no
        # cycle (the deadlock is prevented, not merely retried). The Pass 2/3
        # retry wrap in run_graph_maintenance_job stays as a backstop for the
        # residual paths (FK cascade from prune_orphan_entities, Oracle).
        #
        # Staleness is decided against a SET of currently-live pairs, not with a
        # per-cooccurrence-row check (#3367). The old form ran a correlated
        # `NOT EXISTS (… INTERSECT …)` per row, and each evaluation re-scanned a
        # hub entity's full membership set — cost scaled as (rows judged) x (hub
        # degree), 88-140s on a real bank with a ~22K-degree hub. #2473 had
        # swapped an earlier hub-rescanning self-join to that INTERSECT, but only
        # made each per-row check cheaper; it kept the per-row structure, so the
        # product blew up again at scale.
        #
        # `live` groups unit_entities by unit (self-join on unit_id) to emit every
        # co-occurring (e1<e2) pair in one materialised pass: its cost is driven
        # by unit degree (entities per unit — small), never by entity degree, so a
        # hub contributes only its per-unit membership rather than a rescan per
        # edge. The victims anti-join then hashes against it. MATERIALIZED keeps
        # the planner from inlining `live` back into a per-row correlated plan.
        #
        # `live` is seeded from the *candidates'* units rather than the whole
        # bank's (#3222 composed with #3367): graph maintenance is queue-driven,
        # so this only has to decide about pairs in `incident`, and every such
        # pair has a candidate as at least one endpoint. Any unit still
        # witnessing such a pair therefore references a candidate, and so is in
        # the seeded set — the scoped build cannot miss a live pair. That keeps
        # the whole statement proportional to the batch instead of re-deriving
        # every pair in the bank on every run.
        result = await conn.execute(
            f"""
            WITH incident AS MATERIALIZED (
                SELECT c.entity_id_1, c.entity_id_2
                FROM {ec_table} c
                WHERE c.entity_id_1 = ANY($1::uuid[])
                UNION
                SELECT c.entity_id_1, c.entity_id_2
                FROM {ec_table} c
                WHERE c.entity_id_2 = ANY($1::uuid[])
            ),
            live AS MATERIALIZED (
                SELECT u1.entity_id AS e1, u2.entity_id AS e2
                FROM {ue_table} seed
                JOIN {ue_table} u1 ON u1.unit_id = seed.unit_id
                JOIN {ue_table} u2 ON u2.unit_id = u1.unit_id
                                  AND u2.entity_id > u1.entity_id
                WHERE seed.entity_id = ANY($1::uuid[])
            ),
            victims AS (
                SELECT c.entity_id_1, c.entity_id_2
                FROM incident i
                JOIN {ec_table} c
                  ON c.entity_id_1 = i.entity_id_1 AND c.entity_id_2 = i.entity_id_2
                WHERE NOT EXISTS (
                    SELECT 1 FROM live l
                    WHERE l.e1 = c.entity_id_1 AND l.e2 = c.entity_id_2
                )
                ORDER BY c.entity_id_1, c.entity_id_2
                FOR UPDATE OF c
            )
            DELETE FROM {ec_table} c
            USING victims v
            WHERE c.entity_id_1 = v.entity_id_1
              AND c.entity_id_2 = v.entity_id_2
            """,
            entity_ids,
        )
        return int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0

    async def fetch_unit_dates(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        unit_ids: list[str],
    ) -> list[ResultRow]:
        # Cast only canonical UUID text inputs, never the indexed column. The old
        # ``id::text`` predicate silently ignored malformed, uppercase, braced,
        # and unhyphenated inputs; filtering before the cast preserves that
        # behavior while allowing the primary-key index to serve the lookup.
        return await conn.fetch(
            f"""
            SELECT id, event_date, fact_type
            FROM {mu_table}
            WHERE id = ANY(
                ARRAY(
                    SELECT input.unit_id::uuid
                    FROM unnest($1::text[]) AS input(unit_id)
                    WHERE input.unit_id ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
                )
            )
            """,
            unit_ids,
        )

    async def fetch_temporal_neighbors(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        bank_id: str,
        lateral_unit_ids: list,
        lateral_event_dates: list,
        lateral_fact_types: list,
        half_limit: int,
        batch_size: int = 500,
    ) -> list[ResultRow]:
        rows: list[ResultRow] = []
        for start in range(0, len(lateral_unit_ids), batch_size):
            end = min(start + batch_size, len(lateral_unit_ids))
            # Exact v0.5.6 query shape: src.unit_id::text AS from_id,
            # combined.*, ABS(EXTRACT(...)), ROW_NUMBER PARTITION BY src.unit_id.
            batch_rows = await conn.fetch(
                f"""
                SELECT from_id, id, event_date, time_diff_hours FROM (
                    SELECT src.unit_id::text AS from_id, combined.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY src.unit_id
                               ORDER BY combined.time_diff_hours
                           ) AS rn
                    FROM unnest($1::uuid[], $2::timestamptz[], $3::text[])
                         AS src(unit_id, event_date, fact_type)
                    CROSS JOIN LATERAL (
                        (SELECT mu.id, mu.event_date,
                                ABS(EXTRACT(EPOCH FROM mu.event_date - src.event_date)) / 3600.0 AS time_diff_hours
                         FROM {mu_table} mu
                         WHERE mu.bank_id = $4
                           AND mu.fact_type = src.fact_type
                           AND mu.event_date <= src.event_date
                           AND mu.id != src.unit_id
                         ORDER BY mu.event_date DESC
                         LIMIT $5)
                        UNION ALL
                        (SELECT mu.id, mu.event_date,
                                ABS(EXTRACT(EPOCH FROM mu.event_date - src.event_date)) / 3600.0 AS time_diff_hours
                         FROM {mu_table} mu
                         WHERE mu.bank_id = $4
                           AND mu.fact_type = src.fact_type
                           AND mu.event_date > src.event_date
                           AND mu.id != src.unit_id
                         ORDER BY mu.event_date ASC
                         LIMIT $5)
                    ) combined
                ) ranked
                WHERE rn <= $5
                """,
                lateral_unit_ids[start:end],
                lateral_event_dates[start:end],
                lateral_fact_types[start:end],
                bank_id,
                half_limit,
            )
            rows.extend(batch_rows)
        return rows

    def build_entity_expansion_cte(
        self,
        mu_table: str,
        ue_table: str,
        per_entity_limit: int,
        window: UpdatedWindow,
    ) -> str:
        return f"""
            seed_entities AS (
                SELECT DISTINCT ue.entity_id
                FROM {ue_table} ue
                WHERE ue.unit_id = ANY($1::uuid[])
            ),
            entity_expanded AS (
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       COUNT(DISTINCT se.entity_id)::float AS score,
                       'entity'::text AS source
                FROM seed_entities se
                CROSS JOIN LATERAL (
                    SELECT ue_target.unit_id
                    FROM {ue_table} ue_target
                    WHERE ue_target.entity_id = se.entity_id
                      AND ue_target.unit_id != ALL($1::uuid[])
                      -- Filter before applying the cap: candidates from other fact
                      -- types, or outside the recall window, must not consume this
                      -- entity's bounded fan-out.
                      AND EXISTS (
                          SELECT 1
                          FROM {mu_table} mu_target
                          WHERE mu_target.id = ue_target.unit_id
                            AND mu_target.fact_type = $2
                            {window.clause("mu_target")}
                      )
                    ORDER BY ue_target.unit_id DESC
                    LIMIT {per_entity_limit}
                ) t
                JOIN {mu_table} mu ON mu.id = t.unit_id
                GROUP BY mu.id
                ORDER BY score DESC
                LIMIT $3
            )"""

    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
        window: UpdatedWindow,
    ) -> str:
        # Exact v0.5.6 query shape: GROUP BY + MAX(weight) for semantic,
        # DISTINCT ON for causal.
        return f"""
            semantic_expanded AS (
                SELECT
                    id, text, context, event_date, occurred_start,
                    occurred_end, mentioned_at,
                    fact_type, document_id, chunk_id, tags, proof_count,
                    MAX(weight) AS score,
                    'semantic'::text AS source
                FROM (
                    SELECT
                        mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                        mu.occurred_end, mu.mentioned_at,
                        mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                        ml.weight
                    FROM {ml_table} ml
                    JOIN {mu_table} mu ON mu.id = ml.to_unit_id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                      AND mu.fact_type = $2
                      AND mu.id != ALL($1::uuid[])
                      {window.clause("mu")}
                    UNION ALL
                    SELECT
                        mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                        mu.occurred_end, mu.mentioned_at,
                        mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                        ml.weight
                    FROM {ml_table} ml
                    JOIN {mu_table} mu ON mu.id = ml.from_unit_id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                      AND mu.fact_type = $2
                      AND mu.id != ALL($1::uuid[])
                      {window.clause("mu")}
                ) sem_raw
                GROUP BY id, text, context, event_date, occurred_start,
                         occurred_end, mentioned_at,
                         fact_type, document_id, chunk_id, tags, proof_count
                ORDER BY score DESC
                LIMIT $3
            ),
            causal_expanded AS (
                SELECT DISTINCT ON (mu.id)
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at,
                    mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                    ml.weight AS score,
                    'causal'::text AS source
                FROM {ml_table} ml
                JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = $2
                  {window.clause("mu")}
                ORDER BY mu.id, ml.weight DESC
                LIMIT $3
            )"""

    async def expand_observations(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        ue_table: str,
        ml_table: str,
        seed_ids: list,
        budget: int,
        per_entity_limit: int,
        window: UpdatedWindow,
    ) -> LinkExpansionRows:
        # v0.5.6 array ops: unnest, &&, COUNT(DISTINCT) on source_memory_ids.
        #
        # The window bounds the observations that come *back*, not the source facts
        # traversed to reach them: an observation is in the window when it was itself
        # written or refreshed there, regardless of how old the facts underneath it are.
        #
        # The shared-source count is scored set-wise (`scored`), not per candidate row.
        # It used to be a correlated subquery — COUNT(DISTINCT s) over
        # unnest(mu.source_memory_ids) filtered by `= ANY(ca.source_ids)` — which
        # re-scanned the connected-source array linearly for every element of every
        # candidate's array. Because consolidation appends to source_memory_ids and
        # never prunes it (issue #1725), that product grows with the bank's age: at
        # 5k observations averaging 113 sources against ~3k connected sources it was
        # ~1.7B element comparisons, 2.6s of one saturated backend (issue #3085).
        # Unnesting once and hash-joining connected_sources makes the work linear in
        # the number of source ids instead.
        #
        # `connected_sources` caps each entity with row_number() rather than the
        # LATERAL + LIMIT that reads more naturally. Do not "simplify" it back
        # (issue #3510). The scoring join above is O(C + U) when planned as a hash
        # join and O(U x C) when planned as a nested loop — 15s and ~15M rejected
        # rows on a realistically-shaped bank — and PostgreSQL picks between them
        # from its row estimate for this CTE. Out of a LATERAL + LIMIT subquery the
        # capped column carries no n_distinct statistic, so DISTINCT over it was
        # estimated at 2 and the NOT EXISTS took that to 1 against an actual ~3,700;
        # a 1-row inner side makes the nested loop look free, so it won on cost and
        # lost by four orders of magnitude at runtime. Ranking with a window keeps
        # the column traceable to unit_entities.unit_id, so the estimate comes from
        # real statistics (207-3,449 against 2,242-4,193 actual) and the nested loop
        # is priced honestly.
        #
        # The trade is that this reads every unit_entities row of a matched entity
        # to rank it, where the LATERAL stopped at per_entity_limit off the index:
        # O(sum of degree) rather than O(entities x per_entity_limit). Measured at
        # parity up to ~12k-degree hubs and +50% traversal cost at 38k. If banks
        # grow hubs far past that, re-measure before assuming this is still the
        # right shape.
        #
        # Entity/source traversal and semantic/causal expansion run as ONE query
        # (#3857): the observation entity arm is fused into the semantic/causal CTE
        # query behind an 'entity' source discriminator, like the non-observation
        # combined expansion, so a normal call performs one fetch instead of two.
        # Every predicate, score expression, ordering, limit, and the window bind
        # positions are exactly the two previous statements', now sharing one
        # snapshot. The caller splits the unioned rows by `source`.
        all_rows = await conn.fetch(
            f"""
            WITH seed_sources AS (
                SELECT DISTINCT unnest(source_memory_ids) AS source_id
                FROM {mu_table}
                WHERE id = ANY($1::uuid[])
                  AND source_memory_ids IS NOT NULL
            ),
            source_entities AS (
                SELECT DISTINCT ue_seed.entity_id
                FROM seed_sources ss
                JOIN {ue_table} ue_seed ON ue_seed.unit_id = ss.source_id
            ),
            connected_sources AS (
                SELECT DISTINCT t.unit_id AS source_id
                FROM (
                    SELECT
                        ue_target.unit_id,
                        row_number() OVER (
                            PARTITION BY ue_target.entity_id
                            ORDER BY ue_target.unit_id DESC
                        ) AS rn
                    FROM {ue_table} ue_target
                    JOIN source_entities se ON se.entity_id = ue_target.entity_id
                ) t
                WHERE t.rn <= {per_entity_limit}
                  AND NOT EXISTS (
                      SELECT 1 FROM seed_sources ss WHERE ss.source_id = t.unit_id
                  )
            ),
            connected_array AS (
                SELECT array_agg(source_id) AS source_ids FROM connected_sources
            ),
            candidates AS (
                SELECT
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at,
                    mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                    mu.source_memory_ids
                FROM {mu_table} mu, connected_array ca
                WHERE mu.fact_type = 'observation'
                  AND mu.id != ALL($1::uuid[])
                  AND ca.source_ids IS NOT NULL
                  AND mu.source_memory_ids && ca.source_ids
                  {window.clause("mu")}
            ),
            scored AS (
                SELECT c.id, COUNT(DISTINCT cs.source_id)::float AS score
                FROM candidates c
                CROSS JOIN LATERAL unnest(c.source_memory_ids) AS s(source_id)
                JOIN connected_sources cs ON cs.source_id = s.source_id
                GROUP BY c.id
            ),
            observation_entity_expanded AS (
                SELECT
                    c.id, c.text, c.context, c.event_date, c.occurred_start,
                    c.occurred_end, c.mentioned_at,
                    c.fact_type, c.document_id, c.chunk_id, c.tags, c.proof_count,
                    sc.score,
                    'entity'::text AS source
                FROM candidates c
                JOIN scored sc ON sc.id = c.id
                ORDER BY sc.score DESC
                LIMIT $2
            ),
            -- Exact v0.5.6 query shape: GROUP BY + MAX(weight) for semantic,
            -- DISTINCT ON for causal, hardcoded to fact_type='observation'.
            semantic_expanded AS (
                SELECT
                    id, text, context, event_date, occurred_start,
                    occurred_end, mentioned_at,
                    fact_type, document_id, chunk_id, tags, proof_count,
                    MAX(weight) AS score,
                    'semantic'::text AS source
                FROM (
                    SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                           mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id,
                           mu.chunk_id, mu.tags, mu.proof_count, ml.weight
                    FROM {ml_table} ml JOIN {mu_table} mu ON mu.id = ml.to_unit_id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic' AND mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
                      {window.clause("mu")}
                    UNION ALL
                    SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                           mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id,
                           mu.chunk_id, mu.tags, mu.proof_count, ml.weight
                    FROM {ml_table} ml JOIN {mu_table} mu ON mu.id = ml.from_unit_id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic' AND mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
                      {window.clause("mu")}
                ) sem_raw
                GROUP BY id, text, context, event_date, occurred_start, occurred_end,
                         mentioned_at, fact_type, document_id, chunk_id, tags, proof_count
                ORDER BY score DESC LIMIT $2
            ),
            causal_expanded AS (
                SELECT DISTINCT ON (mu.id)
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id,
                    mu.chunk_id, mu.tags, mu.proof_count, ml.weight AS score, 'causal'::text AS source
                FROM {ml_table} ml JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = 'observation'
                  {window.clause("mu")}
                ORDER BY mu.id, ml.weight DESC LIMIT $2
            )
            SELECT * FROM observation_entity_expanded
            UNION ALL
            SELECT * FROM semantic_expanded
            UNION ALL
            SELECT * FROM causal_expanded
            """,
            seed_ids,
            budget,
            *window.params,
        )

        return LinkExpansionRows(
            entity=[r for r in all_rows if r["source"] == "entity"],
            semantic=[r for r in all_rows if r["source"] == "semantic"],
            causal=[r for r in all_rows if r["source"] == "causal"],
        )

    def build_tag_listing_parts(self, mu_table: str) -> TagListingParts:
        return TagListingParts(
            tag_source=f"{mu_table}, unnest(tags) AS tag",
            non_empty_check="AND tags IS NOT NULL AND tags != '{}'",
            tag_col="tag",
            bank_prefix="",
        )

    async def create_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        internal_id: str,
        index_clause: str,
        fact_types: dict[str, str],
    ) -> None:
        # Plain CREATE INDEX, not CONCURRENTLY: this runs inside the bank-create
        # transaction, and CONCURRENTLY cannot. It is instant — the bank is empty
        # — but it still takes a ShareLock on the shared memory_units table and
        # so can deadlock with a concurrent writer. The caller retries the whole
        # transaction for that reason; the body is idempotent.
        escaped = bank_id.replace("'", "''")
        async with self._index_ddl_lock(table):
            for ft, suffix in fact_types.items():
                uid = str(internal_id).replace("-", "")[:16]
                idx = f"idx_mu_emb_{suffix}_{uid}"
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} "
                    f"ON {table} {index_clause} "
                    f"WHERE fact_type = '{ft}' AND bank_id = '{escaped}'"
                )

    async def drop_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        schema: str,
        internal_id: str,
        fact_types: dict[str, str],
    ) -> None:
        # CONCURRENTLY so the drop takes ShareUpdateExclusive, not ACCESS
        # EXCLUSIVE, on the shared memory_units table. A plain DROP INDEX blocks
        # (and deadlocks with) every other bank's concurrent reads/writes on the
        # table; CONCURRENTLY does not conflict with DML. The caller
        # (delete_bank) runs this on an autocommit connection after its delete
        # transaction has committed — CONCURRENTLY cannot run inside a tx.
        #
        # The in-process lock serializes concurrent bank creates and deletes
        # against each other; its key must match create_bank_vector_indexes',
        # whose `table` is the fq name this reconstructs from `schema`. It does
        # not cover the maintenance operation, which reconciles the same indexes
        # over its own raw connection: an in-process lock could not help there
        # anyway, since that runs in every process and the real contention is
        # cross-process. Both paths retry the transient deadlock (40P01)
        # instead, which is the only lock-free option available — the project
        # forbids advisory locks (unreliable behind poolers, #2817).
        async with self._index_ddl_lock(f"{schema}.memory_units"):
            for ft, suffix in fact_types.items():
                uid = str(internal_id).replace("-", "")[:16]
                idx = f"idx_mu_emb_{suffix}_{uid}"
                await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{idx}")

    def get_entity_resolution_strategy(self) -> str:
        return "trigram"

    # -- Webhook operations ------------------------------------------------

    async def create_webhook(
        self,
        conn,
        table,
        webhook_id,
        bank_id,
        url,
        secret,
        event_types,
        enabled,
        http_config_json,
    ):
        return await conn.fetchrow(
            f"""
            INSERT INTO {table}
            (id, bank_id, url, secret, event_types, enabled, http_config, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW(), NOW())
            RETURNING id, bank_id, url, secret, event_types, enabled,
                      http_config::text, created_at::text, updated_at::text
            """,
            webhook_id,
            bank_id,
            url,
            secret,
            event_types,
            enabled,
            http_config_json,
        )

    async def list_webhooks_for_bank(self, conn, table, bank_id):
        return await conn.fetch(
            f"""
            SELECT id, bank_id, url, secret, event_types, enabled,
                   http_config::text, created_at::text, updated_at::text
            FROM {table}
            WHERE bank_id = $1
            ORDER BY created_at
            """,
            bank_id,
        )

    async def get_webhooks_for_dispatch(self, conn, webhook_table, bank_id):
        return await conn.fetch(
            f"""
            SELECT id, bank_id, url, secret, event_types, enabled, http_config::text
            FROM {webhook_table}
            WHERE (bank_id = $1 OR bank_id IS NULL) AND enabled = true
            """,
            bank_id,
        )

    async def update_webhook(self, conn, table, webhook_id, bank_id, set_clauses, params):
        set_clauses_with_ts = set_clauses + ["updated_at = NOW()"]
        return await conn.fetchrow(
            f"""
            UPDATE {table}
            SET {", ".join(set_clauses_with_ts)}
            WHERE id = $1 AND bank_id = $2
            RETURNING id, bank_id, url, secret, event_types, enabled,
                      http_config::text, created_at::text, updated_at::text
            """,
            *params,
        )

    async def delete_webhook(self, conn, table, webhook_id, bank_id):
        result = await conn.execute(
            f"DELETE FROM {table} WHERE id = $1 AND bank_id = $2",
            webhook_id,
            bank_id,
        )
        return int(result.split()[-1]) > 0 if result else False

    async def list_webhook_deliveries(self, conn, ops_table, webhook_id, bank_id, limit, cursor):
        fetch_limit = limit + 1
        if cursor:
            return await conn.fetch(
                f"""
                SELECT operation_id, status, retry_count, next_retry_at::text,
                       error_message, task_payload, result_metadata::text, created_at::text, updated_at::text
                FROM {ops_table}
                WHERE operation_type = 'webhook_delivery'
                  AND bank_id = $1
                  AND task_payload->>'webhook_id' = $2
                  AND created_at < $3::timestamptz
                ORDER BY created_at DESC
                LIMIT $4
                """,
                bank_id,
                webhook_id,
                cursor,
                fetch_limit,
            )
        return await conn.fetch(
            f"""
            SELECT operation_id, status, retry_count, next_retry_at::text,
                   error_message, task_payload, result_metadata::text, created_at::text, updated_at::text
            FROM {ops_table}
            WHERE operation_type = 'webhook_delivery'
              AND bank_id = $1
              AND task_payload->>'webhook_id' = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            bank_id,
            webhook_id,
            fetch_limit,
        )

    async def insert_webhook_delivery_task(self, conn, ops_table, operation_id, bank_id, payload_json, timestamp):
        await conn.execute(
            f"""
            INSERT INTO {ops_table}
              (operation_id, bank_id, operation_type, status, task_payload, result_metadata, created_at, updated_at)
            VALUES ($1, $2, 'webhook_delivery', 'pending', $3::jsonb, '{{}}'::jsonb, $4, $4)
            """,
            operation_id,
            bank_id,
            payload_json,
            timestamp,
        )

    # -- Task claiming operations ------------------------------------------

    async def prune_terminal_operations(
        self,
        conn: DatabaseConnection,
        table: str,
        cutoff: datetime,
        *,
        batch_size: int,
    ) -> int:
        # Lock only the bounded candidate set. SKIP LOCKED lets multiple
        # workers prune disjoint batches without waiting or double-deleting.
        # Cancelled children cannot complete parent aggregation, so retain the
        # parent guard only for completed/failed children. Before removing a
        # cancelled child, preserve its signal by cancelling a pending parent
        # in this transaction and refreshing the parent's retention window.
        candidates = await conn.fetch(
            f"""
            SELECT candidate_operation.operation_id
            FROM {table} candidate_operation
            WHERE candidate_operation.status IN ('completed', 'failed', 'cancelled')
              AND candidate_operation.updated_at < $1
              AND (
                  candidate_operation.status = 'cancelled'
                  OR NOT EXISTS (
                      SELECT 1
                      FROM {table} parent
                      WHERE parent.operation_id = CASE
                          WHEN candidate_operation.result_metadata->>'parent_operation_id'
                              ~* '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
                          THEN (candidate_operation.result_metadata->>'parent_operation_id')::uuid
                          ELSE NULL
                      END
                        AND parent.bank_id = candidate_operation.bank_id
                  )
              )
            ORDER BY candidate_operation.updated_at, candidate_operation.operation_id
            LIMIT $2
            FOR UPDATE OF candidate_operation SKIP LOCKED
            """,
            cutoff,
            batch_size,
        )
        if not candidates:
            return 0

        candidate_ids = [row["operation_id"] for row in candidates]
        await conn.execute(
            f"""
            UPDATE {table} parent
            SET status = 'cancelled',
                updated_at = now(),
                completed_at = COALESCE(parent.completed_at, now()),
                error_message = COALESCE(
                    parent.error_message,
                    'Cancelled because a child operation was cancelled'
                )
            WHERE parent.status = 'pending'
              AND EXISTS (
                  SELECT 1
                  FROM {table} candidate_operation
                  WHERE candidate_operation.operation_id = ANY($1)
                    AND candidate_operation.status = 'cancelled'
                    AND candidate_operation.updated_at < $2
                    AND candidate_operation.bank_id = parent.bank_id
                    AND parent.operation_id = CASE
                        WHEN candidate_operation.result_metadata->>'parent_operation_id'
                            ~* '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
                        THEN (candidate_operation.result_metadata->>'parent_operation_id')::uuid
                        ELSE NULL
                    END
              )
            """,
            candidate_ids,
            cutoff,
        )
        rows = await conn.fetch(
            f"""
            DELETE FROM {table}
            WHERE operation_id = ANY($1)
              AND status IN ('completed', 'failed', 'cancelled')
              AND updated_at < $2
            RETURNING operation_id
            """,
            candidate_ids,
            cutoff,
        )
        return len(rows)

    async def _claim_consolidation_tasks(
        self,
        conn,
        table: str,
        claimed_ids: list,
        limit: int,
        priority_map: dict[str, int] | None,
    ) -> list:
        """Claim consolidation tasks with optional priority-based tiered ordering.

        When *priority_map* is ``None``, uses the default ``ORDER BY created_at``.
        When set, claims in priority tiers — highest-priority banks first.
        Specific patterns always take precedence over the catch-all ``*`` entry.

        Every query here carries :func:`bank_serialization_sql`, so whichever
        tier a bank is claimed in, at most one of its consolidations is taken.
        """
        if limit <= 0:
            return []

        # --- Fast path: no priority map -> current behavior ---
        if not priority_map:
            return await self._claim_consolidation_plain(conn, table, claimed_ids, limit)

        # --- Tiered claiming ---
        # Separate specific patterns from catch-all.
        # Specific patterns always take precedence: a bank matching ``shadow-*``
        # uses that entry's priority even if the catch-all ``*`` has a higher
        # value.  The catch-all only applies to banks not matching any specific
        # pattern.
        specific_by_priority: dict[int, list[str]] = {}
        all_specific_sql: list[str] = []
        catch_all_priority = 1  # default when no ``*`` entry

        for pattern, priority in priority_map.items():
            if pattern == "*":
                catch_all_priority = priority
            else:
                sql_pat = pattern.replace("*", "%")
                specific_by_priority.setdefault(priority, []).append(sql_pat)
                all_specific_sql.append(sql_pat)

        # Collect all priority levels (specific tiers + catch-all) sorted desc.
        all_priorities = sorted(set(specific_by_priority.keys()) | {catch_all_priority}, reverse=True)

        remaining = limit
        result: list = []

        for pri in all_priorities:
            if remaining <= 0:
                break

            # Specific-pattern tier at this priority level
            if pri in specific_by_priority:
                rows = await self._claim_consolidation_like(
                    conn,
                    table,
                    claimed_ids,
                    remaining,
                    specific_by_priority[pri],
                )
                for row in rows:
                    claimed_ids.append(row["operation_id"])
                    result.append(row)
                remaining -= len(rows)

            # Catch-all tier at this priority level
            if pri == catch_all_priority and remaining > 0:
                rows = await self._claim_consolidation_not_like(
                    conn,
                    table,
                    claimed_ids,
                    remaining,
                    all_specific_sql,
                )
                for row in rows:
                    claimed_ids.append(row["operation_id"])
                    result.append(row)
                remaining -= len(rows)

        return result

    async def _claim_consolidation_plain(
        self,
        conn,
        table,
        claimed_ids,
        limit,
    ) -> list:
        """Claim consolidation tasks with default created_at ordering."""
        if claimed_ids:
            return await conn.fetch(
                f"""
                SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count, o.serialization_key, o.bank_id
                FROM {table} o
                WHERE o.status = 'pending'
                  AND o.task_payload IS NOT NULL
                  AND o.operation_type = 'consolidation'
                  AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
                  AND o.operation_id != ALL($1::uuid[])
                  AND {bank_serialization_sql(table, "o", "consolidation")}
                ORDER BY o.created_at
                LIMIT $2
                FOR UPDATE SKIP LOCKED
                """,
                claimed_ids,
                limit,
            )
        return await conn.fetch(
            f"""
            SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count, o.serialization_key, o.bank_id
            FROM {table} o
            WHERE o.status = 'pending'
              AND o.task_payload IS NOT NULL
              AND o.operation_type = 'consolidation'
              AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
              AND {bank_serialization_sql(table, "o", "consolidation")}
            ORDER BY o.created_at
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            """,
            limit,
        )

    async def _claim_consolidation_like(
        self,
        conn,
        table,
        claimed_ids,
        limit,
        sql_patterns,
    ) -> list:
        """Claim consolidation tasks from banks matching LIKE patterns."""
        # bank_id is deliberately unqualified in the LIKE ANY / NOT LIKE ALL
        # conditions here and in _claim_consolidation_not_like: db/oracle.py
        # rewrites those forms by regex on a bare column name, so an "o."
        # prefix would survive the rewrite as "o.(bank_id LIKE :p0 OR ...)".
        # One table is in scope, so unqualified resolves to o.bank_id anyway.
        params: list = [sql_patterns]
        conditions = ["bank_id LIKE ANY($1::text[])"]
        idx = 2

        if claimed_ids:
            conditions.append(f"o.operation_id != ALL(${idx}::uuid[])")
            params.append(claimed_ids)
            idx += 1

        params.append(limit)
        extra = " AND ".join(conditions)
        return await conn.fetch(
            f"""
            SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count, o.serialization_key, o.bank_id
            FROM {table} o
            WHERE o.status = 'pending'
              AND o.task_payload IS NOT NULL
              AND o.operation_type = 'consolidation'
              AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
              AND {bank_serialization_sql(table, "o", "consolidation")}
              AND {extra}
            ORDER BY o.created_at
            LIMIT ${idx}
            FOR UPDATE SKIP LOCKED
            """,
            *params,
        )

    async def _claim_consolidation_not_like(
        self,
        conn,
        table,
        claimed_ids,
        limit,
        exclude_patterns,
    ) -> list:
        """Claim consolidation tasks from banks NOT matching any specific pattern (catch-all tier)."""
        params: list = []
        conditions: list[str] = []
        idx = 1

        if exclude_patterns:
            conditions.append(f"bank_id NOT LIKE ALL(${idx}::text[])")
            params.append(exclude_patterns)
            idx += 1

        if claimed_ids:
            conditions.append(f"o.operation_id != ALL(${idx}::uuid[])")
            params.append(claimed_ids)
            idx += 1

        params.append(limit)
        extra_clause = (" AND " + " AND ".join(conditions)) if conditions else ""
        return await conn.fetch(
            f"""
            SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count, o.serialization_key, o.bank_id
            FROM {table} o
            WHERE o.status = 'pending'
              AND o.task_payload IS NOT NULL
              AND o.operation_type = 'consolidation'
              AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
              AND {bank_serialization_sql(table, "o", "consolidation")}{extra_clause}
            ORDER BY o.created_at
            LIMIT ${idx}
            FOR UPDATE SKIP LOCKED
            """,
            *params,
        )

    async def _claim_reserved_tasks(self, conn, table, limit, op_type) -> list:
        """Claim rows of one operation type against that type's reserved pool.

        No bank rotation here: reservations are a floor for an operation *type*,
        and every non-consolidation type defaults to 0 (``WORKER_SLOT_TYPE_DEFAULTS``),
        so retains are claimed from the shared pool where the rotation lives. An
        operator who reserves the whole of ``WORKER_MAX_SLOTS`` by type leaves no
        shared pool for it to rotate in — deliberate, and the reason the defaults
        keep a shared pool.
        """
        return await conn.fetch(
            f"""
            SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count, o.serialization_key, o.bank_id
            FROM {table} o
            WHERE o.status = 'pending'
              AND o.task_payload IS NOT NULL
              AND o.operation_type = $1
              AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
              AND {bank_serialization_sql(table, "o")}
              AND {document_serialization_sql(table, "o")}
            ORDER BY o.created_at
            LIMIT $2
            FOR UPDATE SKIP LOCKED
            """,
            op_type,
            limit,
        )

    async def _claim_shared_tasks(self, conn, table, claimed_ids, limit, bank_cursor) -> list:
        """Claim the shared pool: one row for the bank whose turn it is, rest by age.

        Deficit round robin (Shreedhar & Varghese) with a quantum of one slot.
        Every operation costs exactly one slot, so the deficit counter the
        algorithm carries for variable-size packets is always zero and drops out
        — what is left is the round-robin walk. ``bank_id > $1`` *is* that walk:
        a cursor over the bank id space, advanced by an index seek rather than by
        maintaining a list of active queues. A list could not do it anyway — the
        starved bank is the one this worker has never claimed for, so only a
        range can discover it.

        Both tiers are one statement on purpose. ``rot`` is a bounded seek and
        ``fifo`` is the claim query that was always here, so the fairness costs
        no extra round trip; measured at 50k pending rows, 0.10 ms against
        0.02 ms for the bare FIFO claim. Splitting them cost a statement on every
        claim, for every schema, on every poll.

        ``fifo`` is also what makes the rotation safe at both ends:

        * It is the wrap. Once the cursor passes the last bank with work, ``rot``
          matches nothing and ``fifo`` claims the whole pool by itself — no
          second query, no reset round-trip, and a single-bank deployment (where
          the cursor is *always* past the only bank) pays nothing but the empty
          seek.
        * It is what keeps this work-conserving. The rotation can never hold a
          slot open for an idle bank, because ``fifo`` always offers a full
          pool's worth of candidates behind it. A bank alone with work still
          takes every slot.

        The predicates are repeated in the outer query, not just in the CTEs:
        the CTEs read rows unlocked, and re-checking under ``FOR UPDATE`` is what
        makes a row another worker claimed in between fall out instead of being
        claimed twice.

        Two ways that re-check costs a little precision, both self-correcting on
        the next poll and both preferred to a second statement: a candidate lost
        to ``SKIP LOCKED`` makes this return fewer rows than the pool has free
        even though other rows are claimable, and losing the *rotation* row that
        way reads as an empty tier, so the cursor restarts the round early.

        ``tier`` comes back in the rows so the caller can read the rotation's new
        cursor off the result. Within the rotated bank the row is index-ordered
        rather than that bank's oldest — sorting by ``created_at`` there would
        need an index on ``(bank_id, created_at)`` and cost 12 s without one, and
        per-document order is held by ``document_serialization_sql`` regardless.
        """
        params: list = [bank_cursor]
        exclusion = ""
        idx = 2

        if claimed_ids:
            exclusion = f" AND o.operation_id != ALL(${idx}::uuid[])"
            params.append(claimed_ids)
            idx += 1

        params.append(limit)
        claimable = f"""o.status = 'pending'
              AND o.task_payload IS NOT NULL
              AND o.operation_type != 'consolidation'
              AND (o.next_retry_at IS NULL OR o.next_retry_at <= NOW())
              AND {bank_serialization_sql(table, "o")}
              AND {document_serialization_sql(table, "o")}{exclusion}"""

        return await conn.fetch(
            f"""
            WITH rot AS (
                SELECT o.operation_id, 0 AS tier, o.created_at
                FROM {table} o
                WHERE {claimable}
                  AND o.bank_id > $1
                ORDER BY o.bank_id
                LIMIT 1
            ), fifo AS (
                SELECT o.operation_id, 1 AS tier, o.created_at
                FROM {table} o
                WHERE {claimable}
                  AND NOT EXISTS (SELECT 1 FROM rot WHERE rot.operation_id = o.operation_id)
                ORDER BY o.created_at
                LIMIT ${idx}
            ), cand AS (
                SELECT * FROM rot UNION ALL SELECT * FROM fifo
            )
            SELECT o.operation_id, o.operation_type, o.task_payload, o.retry_count,
                   o.serialization_key, o.bank_id, c.tier
            FROM cand c
            JOIN {table} o ON o.operation_id = c.operation_id
            WHERE {claimable}
            ORDER BY c.tier, c.created_at
            LIMIT ${idx}
            FOR UPDATE OF o SKIP LOCKED
            """,
            *params,
        )

    async def claim_tasks(
        self,
        conn,
        table,
        worker_id,
        reserved_limits,
        shared_limit,
        *,
        bank_cursor="",
        consolidation_bank_priority=None,
    ):
        all_rows = []
        claimed_ids = []
        next_bank_cursor = bank_cursor

        def _collect(rows) -> int:
            """Record a claim query's rows, and report how many it returned."""
            for row in rows:
                claimed_ids.append(row["operation_id"])
                all_rows.append(row)
            return len(rows)

        # --- Phase 1: claim from reserved pools ---
        for op_type, limit in reserved_limits.items():
            if limit <= 0:
                continue

            if op_type == "consolidation":
                _collect(
                    await self._claim_consolidation_tasks(
                        conn,
                        table,
                        claimed_ids,
                        limit,
                        consolidation_bank_priority,
                    )
                )
            else:
                _collect(await self._claim_reserved_tasks(conn, table, limit, op_type))

        # --- Phase 2: claim from shared pool ---
        remaining_shared = shared_limit
        if remaining_shared > 0:
            # 2a. Non-consolidation tasks: one row for the bank the rotation is
            # up to (#3861), the rest oldest-first, in one statement.
            # graph_maintenance stays in this created_at-ordered query — see
            # bank_serialization_sql for why it is a predicate rather than a
            # phase of its own.
            shared_rows = await self._claim_shared_tasks(conn, table, claimed_ids, remaining_shared, bank_cursor)
            # An empty rotation tier means no bank sorts after the cursor: the
            # round is over, so the next claim starts from the beginning.
            next_bank_cursor = next((row["bank_id"] for row in shared_rows if row["tier"] == 0), "")
            remaining_shared -= _collect(shared_rows)

            # 2b. Consolidation tasks (with bank-serialization + optional priority)
            if remaining_shared > 0:
                _collect(
                    await self._claim_consolidation_tasks(
                        conn,
                        table,
                        claimed_ids,
                        remaining_shared,
                        consolidation_bank_priority,
                    )
                )

        if not all_rows:
            return ClaimedOperations(rows=[], next_bank_cursor=next_bank_cursor)

        # Mark all claimed rows as processing
        operation_ids = [row["operation_id"] for row in all_rows]
        await self.mark_operations_processing(conn, table, worker_id, operation_ids)

        return ClaimedOperations(rows=all_rows, next_bank_cursor=next_bank_cursor)

    async def mark_operations_processing(
        self,
        conn: DatabaseConnection,
        table: str,
        worker_id: str,
        operation_ids: list,
    ) -> None:
        if not operation_ids:
            return
        await conn.execute(
            f"""
            UPDATE {table}
            SET status = 'processing', worker_id = $1, claimed_at = now(), updated_at = now()
            WHERE operation_id = ANY($2)
            """,
            worker_id,
            operation_ids,
        )

    async def fetch_foldable_retain_peers(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        serialization_key: str,
        limit: int,
    ) -> list[ResultRow]:
        if limit <= 0:
            return []
        return await conn.fetch(
            f"""
            SELECT operation_id, task_payload, retry_count
            FROM {table}
            WHERE status = 'pending'
              AND task_payload IS NOT NULL
              AND operation_type = 'retain'
              AND bank_id = $1
              AND serialization_key = $2
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
            ORDER BY created_at, operation_id
            LIMIT $3
            FOR UPDATE SKIP LOCKED
            """,
            bank_id,
            serialization_key,
            limit,
        )
