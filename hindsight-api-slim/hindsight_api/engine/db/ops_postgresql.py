"""PostgreSQL implementation of DataAccessOps.

Uses unnest(), LATERAL, DISTINCT ON, and native array operations for
efficient batch operations.
"""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from .base import DatabaseConnection
from .ops import DataAccessOps, TagListingParts
from .result import ResultRow


class PostgreSQLOps(DataAccessOps):
    """PostgreSQL-specific data access operations using unnest and LATERAL."""

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

        if config.text_search_extension == "vchord":
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
                                     observation_scopes, text_signals, search_vector)
                SELECT
                    $1,
                    text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                    context, fact_type, metadata, chunk_id, document_id,
                    COALESCE(
                        (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                        '{{}}'::varchar[]
                    ),
                    observation_scopes_json,
                    text_signals,
                    tokenize(
                        COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''),
                        'llmlingua2'
                    )::bm25_catalog.bm25vector
                FROM input_data
                RETURNING id
            """
        else:
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
                                     observation_scopes, text_signals)
                SELECT
                    $1,
                    text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                    context, fact_type, metadata, chunk_id, document_id,
                    COALESCE(
                        (SELECT array_agg(elem) FROM jsonb_array_elements_text(tags_json) AS elem),
                        '{{}}'::varchar[]
                    ),
                    observation_scopes_json,
                    text_signals
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
        from_ids = [lnk[0] for lnk in sorted_links]
        to_ids = [lnk[1] for lnk in sorted_links]
        types = [lnk[2] for lnk in sorted_links]
        weights = [lnk[3] for lnk in sorted_links]
        entity_ids = [lnk[4] for lnk in sorted_links]

        for chunk_start in range(0, len(sorted_links), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sorted_links))
            await conn.execute(
                f"""
                INSERT INTO {table}
                    (from_unit_id, to_unit_id, link_type, weight, entity_id, bank_id)
                SELECT f, t, tp, w, e, $6
                FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::float8[], $5::uuid[])
                    AS t(f, t, tp, w, e)
                {exists_clause}
                ON CONFLICT (from_unit_id, to_unit_id, link_type,
                             COALESCE(entity_id, '{nil_entity_uuid}'::uuid))
                DO NOTHING
                """,
                from_ids[chunk_start:chunk_end],
                to_ids[chunk_start:chunk_end],
                types[chunk_start:chunk_end],
                weights[chunk_start:chunk_end],
                entity_ids[chunk_start:chunk_end],
                bank_id,
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

    async def fetch_entity_unit_fanout(
        self,
        conn: DatabaseConnection,
        ue_table: str,
        entity_id_list: list[UUID],
        limit_per_entity: int,
    ) -> list[ResultRow]:
        return await conn.fetch(
            f"""
            SELECT e.entity_id, n.unit_id
            FROM unnest($1::uuid[]) AS e(entity_id)
            CROSS JOIN LATERAL (
                SELECT ue.unit_id
                FROM {ue_table} ue
                WHERE ue.entity_id = e.entity_id
                ORDER BY ue.unit_id DESC
                LIMIT $2
            ) n
            """,
            entity_id_list,
            limit_per_entity,
        )

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
            # Each LATERAL arm fetches up to half_limit in each direction
            # (backward + forward = 2×half_limit).  The outer ROW_NUMBER
            # picks the half_limit closest overall per source unit, matching
            # the original query behavior.
            batch_rows = await conn.fetch(
                f"""
                SELECT from_id, id, event_date, time_diff_hours FROM (
                    SELECT sub.from_id, sub.id, sub.event_date, sub.time_diff_hours,
                           ROW_NUMBER() OVER (
                               PARTITION BY sub.from_id
                               ORDER BY sub.time_diff_hours
                           ) AS rn
                    FROM unnest($1::uuid[], $2::timestamptz[], $3::text[]) AS inp(uid, edate, ftype)
                    CROSS JOIN LATERAL (
                        (
                            SELECT inp.uid AS from_id, mu.id, mu.event_date,
                                   EXTRACT(EPOCH FROM (inp.edate - mu.event_date)) / 3600.0 AS time_diff_hours
                            FROM {mu_table} mu
                            WHERE mu.bank_id = $4
                              AND mu.fact_type = inp.ftype
                              AND mu.event_date <= inp.edate
                              AND mu.id != inp.uid
                            ORDER BY mu.event_date DESC
                            LIMIT $5
                        )
                        UNION ALL
                        (
                            SELECT inp.uid AS from_id, mu.id, mu.event_date,
                                   EXTRACT(EPOCH FROM (mu.event_date - inp.edate)) / 3600.0 AS time_diff_hours
                            FROM {mu_table} mu
                            WHERE mu.bank_id = $4
                              AND mu.fact_type = inp.ftype
                              AND mu.event_date > inp.edate
                              AND mu.id != inp.uid
                            ORDER BY mu.event_date ASC
                            LIMIT $5
                        )
                    ) sub
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
                    ORDER BY ue_target.unit_id DESC
                    LIMIT {per_entity_limit}
                ) t
                JOIN {mu_table} mu ON mu.id = t.unit_id
                WHERE mu.fact_type = $2
                GROUP BY mu.id
                ORDER BY score DESC
                LIMIT $3
            )"""

    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
    ) -> str:
        # DISTINCT ON deduplicates by mu.id (keeping the best weight), but
        # forces ORDER BY mu.id first.  Wrap in a subquery to re-sort by score
        # so the LIMIT keeps the globally highest-scored rows (matching the
        # original GROUP BY + MAX(weight) + ORDER BY score DESC behavior).
        return f"""
            sem_deduped AS (
                SELECT DISTINCT ON (mu.id)
                       mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ml.weight::float AS score,
                       'semantic'::text AS source
                FROM (
                    SELECT ml.to_unit_id AS id, ml.weight
                    FROM {ml_table} ml
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                    UNION ALL
                    SELECT ml.from_unit_id AS id, ml.weight
                    FROM {ml_table} ml
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                ) ml
                JOIN {mu_table} mu ON mu.id = ml.id
                WHERE mu.fact_type = $2
                  AND mu.id != ALL($1::uuid[])
                ORDER BY mu.id, ml.weight DESC
            ),
            semantic_expanded AS (
                SELECT * FROM sem_deduped
                ORDER BY score DESC
                LIMIT $3
            ),
            causal_deduped AS (
                SELECT DISTINCT ON (mu.id)
                       mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ml.weight::float AS score,
                       'causal'::text AS source
                FROM {ml_table} ml
                JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = $2
                ORDER BY mu.id, ml.weight DESC
            ),
            causal_expanded AS (
                SELECT * FROM causal_deduped
                ORDER BY score DESC
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
        # Entity expansion via observation_sources junction table.
        # Previously used PG-specific unnest(source_memory_ids) and array
        # overlap (&&). The junction table approach is portable across backends.
        from ..schema import fq_table

        obs_sources_table = fq_table("observation_sources")
        entity_rows = await conn.fetch(
            f"""
            WITH source_ids AS (
                SELECT DISTINCT os.source_id
                FROM {obs_sources_table} os
                WHERE os.observation_id = ANY($1::uuid[])
            ),
            source_entities AS (
                SELECT DISTINCT ue_seed.entity_id
                FROM source_ids si
                JOIN {ue_table} ue_seed ON ue_seed.unit_id = si.source_id
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
                WHERE t.unit_id NOT IN (SELECT source_id FROM source_ids)
            )
            SELECT
                mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                mu.occurred_end, mu.mentioned_at,
                mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                (SELECT COUNT(*)
                 FROM {obs_sources_table} os2
                 WHERE os2.observation_id = mu.id
                   AND os2.source_id IN (SELECT source_id FROM connected_sources)
                )::float AS score
            FROM {mu_table} mu
            WHERE mu.fact_type = 'observation'
              AND mu.id != ALL($1::uuid[])
              AND EXISTS (
                  SELECT 1 FROM {obs_sources_table} os3
                  WHERE os3.observation_id = mu.id
                    AND os3.source_id IN (SELECT source_id FROM connected_sources)
              )
            ORDER BY score DESC
            LIMIT $2
            """,
            seed_ids,
            budget,
        )

        # Semantic + causal expansion (same pattern as build_semantic_causal_cte
        # but hardcoded to fact_type='observation').
        # Wrap DISTINCT ON in subqueries to re-sort by score before LIMIT.
        sem_causal_rows = await conn.fetch(
            f"""
            WITH
            sem_deduped AS (
                SELECT DISTINCT ON (mu.id)
                       mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ml.weight::float AS score,
                       'semantic'::text AS source
                FROM (
                    SELECT ml.to_unit_id AS id, ml.weight
                    FROM {ml_table} ml
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                    UNION ALL
                    SELECT ml.from_unit_id AS id, ml.weight
                    FROM {ml_table} ml
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                ) ml
                JOIN {mu_table} mu ON mu.id = ml.id
                WHERE mu.fact_type = 'observation'
                  AND mu.id != ALL($1::uuid[])
                ORDER BY mu.id, ml.weight DESC
            ),
            semantic_expanded AS (
                SELECT * FROM sem_deduped ORDER BY score DESC LIMIT $2
            ),
            causal_deduped AS (
                SELECT DISTINCT ON (mu.id)
                       mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ml.weight::float AS score,
                       'causal'::text AS source
                FROM {ml_table} ml
                JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = 'observation'
                ORDER BY mu.id, ml.weight DESC
            ),
            causal_expanded AS (
                SELECT * FROM causal_deduped ORDER BY score DESC LIMIT $2
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

    async def claim_tasks(self, conn, table, worker_id, reserved_limits, shared_limit):
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

                if busy_bank_ids:
                    rows = await conn.fetch(
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
                    rows = await conn.fetch(
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

            # 2b. Consolidation tasks (with bank-serialization)
            if remaining_shared > 0:
                busy_banks_2 = await conn.fetch(
                    f"""
                    SELECT DISTINCT bank_id FROM {table}
                    WHERE operation_type = 'consolidation' AND status = 'processing'
                    """,
                )
                busy_bank_ids_2 = [r["bank_id"] for r in busy_banks_2]

                if claimed_ids:
                    if busy_bank_ids_2:
                        rows = await conn.fetch(
                            f"""
                            SELECT operation_id, operation_type, task_payload, retry_count
                            FROM {table}
                            WHERE status = 'pending'
                              AND task_payload IS NOT NULL
                              AND operation_type = 'consolidation'
                              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                              AND operation_id != ALL($1::uuid[])
                              AND bank_id != ALL($2::text[])
                            ORDER BY created_at
                            LIMIT $3
                            FOR UPDATE SKIP LOCKED
                            """,
                            claimed_ids,
                            busy_bank_ids_2,
                            remaining_shared,
                        )
                    else:
                        rows = await conn.fetch(
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
                            claimed_ids,
                            remaining_shared,
                        )
                else:
                    if busy_bank_ids_2:
                        rows = await conn.fetch(
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
                            busy_bank_ids_2,
                            remaining_shared,
                        )
                    else:
                        rows = await conn.fetch(
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
                            remaining_shared,
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
