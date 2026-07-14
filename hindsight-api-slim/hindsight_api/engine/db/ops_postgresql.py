"""PostgreSQL implementation of DataAccessOps.

Uses unnest(), LATERAL, DISTINCT ON, and native array operations for
efficient batch operations.
"""

from datetime import datetime

from .base import DatabaseConnection
from .ops import DataAccessOps, TagListingParts
from .result import ResultRow


def pg_search_vector_expr(
    config,
    *,
    text_col: str = "text",
    context_col: str = "context",
    signals_col: str = "text_signals",
) -> str | None:
    """SQL expression that builds ``search_vector`` for the configured PG text-search backend.

    Single source of truth shared by the batch insert (over the ``input_data``
    CTE columns) and the curation revert recompute (over a ``memory_units`` row),
    so the two can never drift. Returns ``None`` for backends that leave
    ``search_vector`` unpopulated — pgroonga / pg_textsearch / pg_search index the
    base text columns directly and keep only a dummy column, so there is nothing
    to build.

    ``text_search_extension_native_language`` is validated as a PG identifier in
    ``HindsightConfig.validate()``, so embedding it as a SQL literal is safe.
    """
    combined = f"COALESCE({text_col}, '') || ' ' || COALESCE({context_col}, '') || ' ' || COALESCE({signals_col}, '')"
    if config.text_search_extension == "vchord":
        return f"tokenize({combined}, 'llmlingua2')::bm25_catalog.bm25vector"
    if config.text_search_extension == "native":
        return f"to_tsvector('{config.text_search_extension_native_language}'::regconfig, {combined})"
    return None


class PostgreSQLOps(DataAccessOps):
    """PostgreSQL-specific data access operations using unnest and LATERAL."""

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
    ) -> dict[str, str]:
        inserted_rows = await conn.fetch(
            f"""
            INSERT INTO {table} (bank_id, canonical_name, first_seen, last_seen, mention_count)
            SELECT $1, name, COALESCE(event_date, now()), COALESCE(event_date, now()), 0
            FROM unnest($2::text[], $3::timestamptz[]) AS t(name, event_date)
            ON CONFLICT (bank_id, LOWER(canonical_name))
            DO NOTHING
            RETURNING id, LOWER(canonical_name) AS name_lower
            """,
            bank_id,
            entity_names,
            entity_dates,
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
            SELECT e.id, LOWER(e.canonical_name) AS name_lower, inputs.input_name
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
        await conn.execute(
            f"""
            INSERT INTO {table} (bank_id, unit_id)
            SELECT $1, v FROM unnest($2::uuid[]) AS t(v)
            ON CONFLICT (bank_id, unit_id) DO NOTHING
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
        rows = await conn.fetch(
            f"""
            DELETE FROM {table}
            WHERE (bank_id, unit_id) IN (
                SELECT bank_id, unit_id FROM {table}
                WHERE bank_id = $1
                ORDER BY enqueued_at
                LIMIT $2
            )
            RETURNING unit_id
            """,
            bank_id,
            limit,
        )
        return [str(row["unit_id"]) for row in rows]

    async def prune_orphan_entities(
        self,
        conn: DatabaseConnection,
        entities_table: str,
        ue_table: str,
        bank_id: str,
    ) -> int:
        # Scoped by entities.bank_id (indexed). The NOT EXISTS subquery is
        # backed by idx_ue_entity on unit_entities(entity_id), so this stays
        # linear in the number of entities in the bank — not in the size of
        # unit_entities globally.
        result = await conn.execute(
            f"""
            DELETE FROM {entities_table} e
            WHERE e.bank_id = $1
              AND NOT EXISTS (
                  SELECT 1 FROM {ue_table} ue WHERE ue.entity_id = e.id
              )
            """,
            bank_id,
        )
        # asyncpg returns "DELETE N"
        return int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0

    async def prune_stale_cooccurrences(
        self,
        conn: DatabaseConnection,
        ec_table: str,
        ue_table: str,
        entities_table: str,
        bank_id: str,
    ) -> int:
        # Scope by joining through entities.bank_id (entity_cooccurrences itself
        # has no bank_id column — entities don't span banks, so scoping via
        # entity_id_1 is sufficient).
        result = await conn.execute(
            f"""
            DELETE FROM {ec_table} c
            USING {entities_table} e
            WHERE e.id = c.entity_id_1
              AND e.bank_id = $1
              AND NOT EXISTS (
                  SELECT 1
                  FROM {ue_table} u1
                  JOIN {ue_table} u2 ON u1.unit_id = u2.unit_id
                  WHERE u1.entity_id = c.entity_id_1
                    AND u2.entity_id = c.entity_id_2
              )
            """,
            bank_id,
        )
        return int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0

    async def fetch_unit_dates(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        unit_ids: list[str],
    ) -> list[ResultRow]:
        return await conn.fetch(
            f"""
            SELECT id, event_date, fact_type
            FROM {mu_table}
            WHERE id::text = ANY($1)
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
                      -- types must not consume this entity's bounded fan-out.
                      AND EXISTS (
                          SELECT 1
                          FROM {mu_table} mu_target
                          WHERE mu_target.id = ue_target.unit_id
                            AND mu_target.fact_type = $2
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
    ) -> tuple[list[ResultRow], list[ResultRow], list[ResultRow]]:
        # v0.5.6 array ops: unnest, &&, COUNT(DISTINCT) on source_memory_ids.

        entity_rows = await conn.fetch(
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
                FROM source_entities se
                CROSS JOIN LATERAL (
                    SELECT ue_target.unit_id
                    FROM {ue_table} ue_target
                    WHERE ue_target.entity_id = se.entity_id
                    ORDER BY ue_target.unit_id DESC
                    LIMIT {per_entity_limit}
                ) t
                WHERE NOT EXISTS (
                    SELECT 1 FROM seed_sources ss WHERE ss.source_id = t.unit_id
                )
            ),
            connected_array AS (
                SELECT array_agg(source_id) AS source_ids FROM connected_sources
            )
            SELECT
                mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                mu.occurred_end, mu.mentioned_at,
                mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                (SELECT COUNT(DISTINCT s) FROM unnest(mu.source_memory_ids) s WHERE s = ANY(ca.source_ids))::float AS score
            FROM {mu_table} mu, connected_array ca
            WHERE mu.fact_type = 'observation'
              AND mu.id != ALL($1::uuid[])
              AND ca.source_ids IS NOT NULL
              AND mu.source_memory_ids && ca.source_ids
            ORDER BY score DESC
            LIMIT $2
            """,
            seed_ids,
            budget,
        )

        # Exact v0.5.6 query shape: GROUP BY + MAX(weight) for semantic,
        # DISTINCT ON for causal, hardcoded to fact_type='observation'.
        sem_causal_rows = await conn.fetch(
            f"""
            WITH semantic_expanded AS (
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
                    UNION ALL
                    SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                           mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id,
                           mu.chunk_id, mu.tags, mu.proof_count, ml.weight
                    FROM {ml_table} ml JOIN {mu_table} mu ON mu.id = ml.from_unit_id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic' AND mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
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
                ORDER BY mu.id, ml.weight DESC LIMIT $2
            )
            SELECT * FROM semantic_expanded
            UNION ALL
            SELECT * FROM causal_expanded
            """,
            seed_ids,
            budget,
        )

        semantic_rows = [r for r in sem_causal_rows if r["source"] == "semantic"]
        causal_rows = [r for r in sem_causal_rows if r["source"] == "causal"]
        return list(entity_rows), semantic_rows, causal_rows

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
        escaped = bank_id.replace("'", "''")
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
        for ft, suffix in fact_types.items():
            uid = str(internal_id).replace("-", "")[:16]
            idx = f"idx_mu_emb_{suffix}_{uid}"
            await conn.execute(f"DROP INDEX IF EXISTS {schema}.{idx}")

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
        busy_bank_ids: list[str],
        claimed_ids: list,
        limit: int,
        priority_map: dict[str, int] | None,
    ) -> list:
        """Claim consolidation tasks with optional priority-based tiered ordering.

        When *priority_map* is ``None``, uses the default ``ORDER BY created_at``
        with bank-serialization (exclude busy banks).  When set, claims in
        priority tiers — highest-priority banks first.  Specific patterns always
        take precedence over the catch-all ``*`` entry.
        """
        if limit <= 0:
            return []

        # --- Fast path: no priority map -> current behavior ---
        if not priority_map:
            return await self._claim_consolidation_plain(conn, table, busy_bank_ids, claimed_ids, limit)

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
                    busy_bank_ids,
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
                    busy_bank_ids,
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
        busy_bank_ids,
        claimed_ids,
        limit,
    ) -> list:
        """Claim consolidation tasks with default created_at ordering."""
        exclude_ids = claimed_ids if claimed_ids else None
        if busy_bank_ids:
            if exclude_ids:
                return await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type = 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                      AND bank_id != ALL($1::text[])
                      AND operation_id != ALL($2::uuid[])
                    ORDER BY created_at
                    LIMIT $3
                    FOR UPDATE SKIP LOCKED
                    """,
                    busy_bank_ids,
                    exclude_ids,
                    limit,
                )
            else:
                return await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type = 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                      AND bank_id != ALL($1::text[])
                    ORDER BY created_at
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    busy_bank_ids,
                    limit,
                )
        else:
            if exclude_ids:
                return await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type = 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                      AND operation_id != ALL($1::uuid[])
                    ORDER BY created_at
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    exclude_ids,
                    limit,
                )
            else:
                return await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type = 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    limit,
                )

    async def _claim_consolidation_like(
        self,
        conn,
        table,
        busy_bank_ids,
        claimed_ids,
        limit,
        sql_patterns,
    ) -> list:
        """Claim consolidation tasks from banks matching LIKE patterns."""
        params: list = [sql_patterns]
        conditions = ["bank_id LIKE ANY($1::text[])"]
        idx = 2

        if busy_bank_ids:
            conditions.append(f"bank_id != ALL(${idx}::text[])")
            params.append(busy_bank_ids)
            idx += 1

        if claimed_ids:
            conditions.append(f"operation_id != ALL(${idx}::uuid[])")
            params.append(claimed_ids)
            idx += 1

        params.append(limit)
        extra = " AND ".join(conditions)
        return await conn.fetch(
            f"""
            SELECT operation_id, operation_type, task_payload, retry_count
            FROM {table}
            WHERE status = 'pending'
              AND task_payload IS NOT NULL
              AND operation_type = 'consolidation'
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
              AND {extra}
            ORDER BY created_at
            LIMIT ${idx}
            FOR UPDATE SKIP LOCKED
            """,
            *params,
        )

    async def _claim_consolidation_not_like(
        self,
        conn,
        table,
        busy_bank_ids,
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

        if busy_bank_ids:
            conditions.append(f"bank_id != ALL(${idx}::text[])")
            params.append(busy_bank_ids)
            idx += 1

        if claimed_ids:
            conditions.append(f"operation_id != ALL(${idx}::uuid[])")
            params.append(claimed_ids)
            idx += 1

        params.append(limit)
        extra_clause = (" AND " + " AND ".join(conditions)) if conditions else ""
        return await conn.fetch(
            f"""
            SELECT operation_id, operation_type, task_payload, retry_count
            FROM {table}
            WHERE status = 'pending'
              AND task_payload IS NOT NULL
              AND operation_type = 'consolidation'
              AND (next_retry_at IS NULL OR next_retry_at <= NOW()){extra_clause}
            ORDER BY created_at
            LIMIT ${idx}
            FOR UPDATE SKIP LOCKED
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
        consolidation_bank_priority=None,
    ):
        all_rows = []
        claimed_ids = []

        # --- Phase 1: claim from reserved pools ---
        for op_type, limit in reserved_limits.items():
            if limit <= 0:
                continue

            if op_type == "consolidation":
                busy_banks = await conn.fetch(
                    f"""
                    SELECT DISTINCT bank_id FROM {table}
                    WHERE operation_type = 'consolidation' AND status = 'processing'
                    """,
                )
                busy_bank_ids = [r["bank_id"] for r in busy_banks]

                rows = await self._claim_consolidation_tasks(
                    conn,
                    table,
                    busy_bank_ids,
                    claimed_ids,
                    limit,
                    consolidation_bank_priority,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type = $1
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    op_type,
                    limit,
                )

            for row in rows:
                claimed_ids.append(row["operation_id"])
                all_rows.append(row)

        # --- Phase 2: claim from shared pool ---
        remaining_shared = shared_limit
        if remaining_shared > 0:
            # 2a. Non-consolidation tasks
            if claimed_ids:
                rows = await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type != 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                      AND operation_id != ALL($1::uuid[])
                    ORDER BY created_at
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    claimed_ids,
                    remaining_shared,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT operation_id, operation_type, task_payload, retry_count
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type != 'consolidation'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    remaining_shared,
                )

            for row in rows:
                claimed_ids.append(row["operation_id"])
                all_rows.append(row)
            remaining_shared -= len(rows)

            # 2b. Consolidation tasks (with bank-serialization + optional priority)
            if remaining_shared > 0:
                busy_banks_2 = await conn.fetch(
                    f"""
                    SELECT DISTINCT bank_id FROM {table}
                    WHERE operation_type = 'consolidation' AND status = 'processing'
                    """,
                )
                busy_bank_ids_2 = [r["bank_id"] for r in busy_banks_2]

                rows = await self._claim_consolidation_tasks(
                    conn,
                    table,
                    busy_bank_ids_2,
                    claimed_ids,
                    remaining_shared,
                    consolidation_bank_priority,
                )

                for row in rows:
                    claimed_ids.append(row["operation_id"])
                    all_rows.append(row)

        if not all_rows:
            return []

        # Mark all claimed rows as processing
        operation_ids = [row["operation_id"] for row in all_rows]
        await conn.execute(
            f"""
            UPDATE {table}
            SET status = 'processing', worker_id = $1, claimed_at = now(), updated_at = now()
            WHERE operation_id = ANY($2)
            """,
            worker_id,
            operation_ids,
        )

        return all_rows
