"""Oracle 23ai implementation of DataAccessOps.

Uses executemany, per-row queries, JSON_TABLE, and ROW_NUMBER() workarounds
for Oracle-specific syntax requirements (no unnest, no DISTINCT ON, CLOB
columns can't appear in GROUP BY).
"""

import json
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from .base import DatabaseConnection
from .ops import DataAccessOps, TagListingParts
from .result import ResultRow


class OracleOps(DataAccessOps):
    """Oracle-specific data access operations."""

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
        # Oracle's thin-client executemany with array binds is already well-optimized —
        # it batches network round-trips into a single call, so INSERT ALL or other
        # patterns would not provide a meaningful improvement.
        await conn.bulk_insert_from_arrays(
            table,
            ["chunk_id", "document_id", "bank_id", "chunk_text", "chunk_index", "content_hash"],
            [
                chunk_ids,
                document_ids,
                bank_ids,
                chunk_texts,
                chunk_indices,
                content_hashes,
            ],
            column_types=["text[]", "text[]", "text[]", "text[]", "integer[]", "text[]"],
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
        table = self._get_mu_table()
        # Generate UUIDs client-side so we can use executemany (single network
        # round-trip) instead of N individual INSERT+RETURNING calls.
        unit_ids = [str(uuid_mod.uuid4()) for _ in range(len(fact_texts))]
        rows_data = []
        for i in range(len(fact_texts)):
            tags_value = json.loads(tags_list[i]) if tags_list[i] else []
            rows_data.append(
                (
                    unit_ids[i],
                    bank_id,
                    fact_texts[i],
                    embeddings[i],
                    event_dates[i],
                    occurred_starts[i],
                    occurred_ends[i],
                    mentioned_ats[i],
                    contexts[i],
                    fact_types[i],
                    metadata_jsons[i],
                    chunk_ids[i],
                    document_ids[i],
                    tags_value,
                    observation_scopes_list[i],
                    text_signals_list[i],
                )
            )
        await conn.executemany(
            f"""
            INSERT INTO {table} (id, bank_id, text, embedding, event_date, occurred_start,
                occurred_end, mentioned_at, context, fact_type, metadata, chunk_id, document_id,
                tags, observation_scopes, text_signals)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
            rows_data,
        )
        return unit_ids

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
        # The backend rewrites ON CONFLICT DO NOTHING for duplicate suppression.
        # WHERE EXISTS checks are intentionally skipped: executemany does not support
        # correlated subqueries in this form, and callers guarantee unit validity.
        from_ids = [lnk[0] for lnk in sorted_links]
        to_ids = [lnk[1] for lnk in sorted_links]
        types = [lnk[2] for lnk in sorted_links]
        weights = [lnk[3] for lnk in sorted_links]
        entity_ids = [lnk[4] for lnk in sorted_links]

        await conn.executemany(
            f"""
            INSERT INTO {table}
                (from_unit_id, to_unit_id, link_type, weight, entity_id, bank_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (from_unit_id, to_unit_id, link_type,
                         COALESCE(entity_id, '{nil_entity_uuid}'::uuid))
            DO NOTHING
            """,
            [(from_ids[i], to_ids[i], types[i], weights[i], entity_ids[i], bank_id) for i in range(len(sorted_links))],
        )

    async def bulk_insert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_names: list[str],
        entity_dates: list,
    ) -> dict[str, str]:
        # Row-by-row insert with duplicate suppression.
        # Can't use RETURNING with ON CONFLICT DO NOTHING reliably,
        # so INSERT (ignoring dups) then SELECT all IDs at the end.
        id_by_name: dict[str, str] = {}
        for name, event_date in zip(entity_names, entity_dates):
            ts = event_date if event_date else datetime.now(UTC)
            await conn.execute(
                f"""
                INSERT INTO {table} (bank_id, canonical_name, first_seen, last_seen, mention_count)
                VALUES ($1, $2, $3, $3, 0)
                ON CONFLICT (bank_id, LOWER(canonical_name)) DO NOTHING
                """,
                bank_id,
                name,
                ts,
            )
        # Now SELECT all the entities we just inserted (or that already existed)
        for name in entity_names:
            row = await conn.fetchrow(
                f"""
                SELECT id, LOWER(canonical_name) AS name_lower
                FROM {table}
                WHERE bank_id = $1 AND LOWER(canonical_name) = LOWER($2)
                """,
                bank_id,
                name,
            )
            if row:
                id_by_name[row["name_lower"]] = row["id"]
        return id_by_name

    async def fetch_missing_entity_ids(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        missing_names: list[str],
    ) -> list[ResultRow]:
        # Query each missing entity individually
        results: list[ResultRow] = []
        for orig_name in missing_names:
            row = await conn.fetchrow(
                f"""
                SELECT id, LOWER(canonical_name) AS name_lower
                FROM {table}
                WHERE bank_id = $1 AND LOWER(canonical_name) = LOWER($2)
                """,
                bank_id,
                orig_name,
            )
            if row:
                # Wrap in a dict-like to include input_name for downstream compat
                results.append(row)
        return results

    async def bulk_insert_unit_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        unit_ids: list,
        entity_ids: list,
    ) -> None:
        await conn.executemany(
            f"""
            INSERT INTO {table} (unit_id, entity_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            """,
            list(zip(unit_ids, entity_ids)),
        )

    async def fetch_entity_unit_fanout(
        self,
        conn: DatabaseConnection,
        ue_table: str,
        entity_id_list: list[UUID],
        limit_per_entity: int,
    ) -> list[ResultRow]:
        # Query each entity individually
        rows: list[ResultRow] = []
        for eid in entity_id_list:
            entity_rows = await conn.fetch(
                f"""
                SELECT $1 AS entity_id, ue.unit_id
                FROM {ue_table} ue
                WHERE ue.entity_id = $1
                ORDER BY ue.unit_id DESC
                LIMIT $2
                """,
                eid,
                limit_per_entity,
            )
            rows.extend(entity_rows)
        return rows

    async def fetch_unit_dates(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        unit_ids: list[str],
    ) -> list[ResultRow]:
        # No ANY() array binding; query each unit individually
        rows: list[ResultRow] = []
        for uid in unit_ids:
            row = await conn.fetchrow(
                f"""
                SELECT id, event_date, fact_type
                FROM {mu_table}
                WHERE id = $1
                """,
                uid,
            )
            if row:
                rows.append(row)
        return rows

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
        # Per-unit queries (no unnest/LATERAL in Oracle).
        # Fetch up to half_limit in each direction, then combine and keep the
        # half_limit closest overall via ROW_NUMBER — matching the PG behavior.
        rows: list[ResultRow] = []
        for uid, edate, ftype in zip(lateral_unit_ids, lateral_event_dates, lateral_fact_types):
            uid_str = str(uid) if not isinstance(uid, str) else uid
            unit_rows = await conn.fetch(
                f"""
                SELECT from_id, id, event_date, time_diff_hours FROM (
                    SELECT combined.*, ROW_NUMBER() OVER (ORDER BY combined.time_diff_hours) AS rn
                    FROM (
                        SELECT * FROM (
                            SELECT $1 AS from_id, mu.id, mu.event_date,
                                   ABS(EXTRACT(DAY FROM (mu.event_date - $2)) * 24
                                       + EXTRACT(HOUR FROM (mu.event_date - $2))) AS time_diff_hours
                            FROM {mu_table} mu
                            WHERE mu.bank_id = $4
                              AND mu.fact_type = $3
                              AND mu.event_date <= $2
                              AND mu.id != $6
                            ORDER BY mu.event_date DESC
                            FETCH FIRST $5 ROWS ONLY
                        ) bwd
                        UNION ALL
                        SELECT * FROM (
                            SELECT $1 AS from_id, mu.id, mu.event_date,
                                   ABS(EXTRACT(DAY FROM (mu.event_date - $2)) * 24
                                       + EXTRACT(HOUR FROM (mu.event_date - $2))) AS time_diff_hours
                            FROM {mu_table} mu
                            WHERE mu.bank_id = $4
                              AND mu.fact_type = $3
                              AND mu.event_date > $2
                              AND mu.id != $6
                            ORDER BY mu.event_date ASC
                            FETCH FIRST $5 ROWS ONLY
                        ) fwd
                    ) combined
                ) ranked
                WHERE rn <= $5
                """,
                uid_str,
                edate,
                ftype,
                bank_id,
                half_limit,
                uid,
            )
            rows.extend(unit_rows)
        return rows

    def build_entity_expansion_cte(
        self,
        mu_table: str,
        ue_table: str,
        per_entity_limit: int,
    ) -> str:
        # Oracle: can't GROUP BY CLOB columns (text, context).
        # Restructure: count entities per unit_id in a subquery, then join to get full columns.
        return f"""
            seed_entities AS (
                SELECT DISTINCT ue.entity_id
                FROM {ue_table} ue
                WHERE ue.unit_id = ANY($1::uuid[])
            ),
            entity_scores AS (
                SELECT t.unit_id, COUNT(DISTINCT se.entity_id) AS score
                FROM seed_entities se
                CROSS JOIN LATERAL (
                    SELECT ue_target.unit_id
                    FROM {ue_table} ue_target
                    WHERE ue_target.entity_id = se.entity_id
                      AND ue_target.unit_id != ALL($1::uuid[])
                    ORDER BY ue_target.unit_id DESC
                    FETCH FIRST {per_entity_limit} ROWS ONLY
                ) t
                GROUP BY t.unit_id
            ),
            entity_expanded AS (
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       es.score, 'entity' AS source
                FROM entity_scores es
                JOIN {mu_table} mu ON mu.id = es.unit_id
                WHERE mu.fact_type = $2
                ORDER BY es.score DESC
                FETCH FIRST $3 ROWS ONLY
            )"""

    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
    ) -> str:
        # Non-PG: can't GROUP BY CLOB columns, no DISTINCT ON.
        # Restructure semantic: compute max weight per id, then join for full columns.
        return f"""
            sem_scores AS (
                SELECT id, MAX(weight) AS score
                FROM (
                    SELECT mu.id, ml.weight
                    FROM {ml_table} ml
                    JOIN {mu_table} mu ON mu.id = ml.to_unit_id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                      AND mu.fact_type = $2
                      AND mu.id != ALL($1::uuid[])
                    UNION ALL
                    SELECT mu.id, ml.weight
                    FROM {ml_table} ml
                    JOIN {mu_table} mu ON mu.id = ml.from_unit_id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic'
                      AND mu.fact_type = $2
                      AND mu.id != ALL($1::uuid[])
                ) sem_raw
                GROUP BY id
            ),
            semantic_expanded AS (
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ss.score, 'semantic' AS source
                FROM sem_scores ss
                JOIN {mu_table} mu ON mu.id = ss.id
                ORDER BY ss.score DESC
                FETCH FIRST $3 ROWS ONLY
            ),
            causal_ranked AS (
                SELECT
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at,
                    mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                    ml.weight AS score,
                    'causal' AS source,
                    ROW_NUMBER() OVER (PARTITION BY mu.id ORDER BY ml.weight DESC) AS rn_
                FROM {ml_table} ml
                JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = $2
            ),
            causal_expanded AS (
                SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at,
                       fact_type, document_id, chunk_id, tags, proof_count, score, source
                FROM causal_ranked WHERE rn_ = 1
                ORDER BY score DESC
                FETCH FIRST $3 ROWS ONLY
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
        import logging

        logger = logging.getLogger(__name__)

        # Entity expansion via observation_sources junction table.
        # Previously used JSON_TABLE to explode source_memory_ids CLOB. The junction
        # table approach uses standard SQL joins, identical to the PG backend.
        from ..schema import fq_table

        obs_sources_table = fq_table("observation_sources")
        entity_rows = await conn.fetch(
            f"""
            WITH seed_sources AS (
                SELECT DISTINCT os.source_id
                FROM {obs_sources_table} os
                WHERE os.observation_id = ANY($1::uuid[])
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
                    FETCH FIRST {per_entity_limit} ROWS ONLY
                ) t
                WHERE NOT EXISTS (
                    SELECT 1 FROM seed_sources ss WHERE ss.source_id = t.unit_id
                )
            )
            SELECT
                mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                mu.occurred_end, mu.mentioned_at,
                mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                (SELECT COUNT(*)
                 FROM {obs_sources_table} os2
                 WHERE os2.observation_id = mu.id
                   AND os2.source_id IN (SELECT source_id FROM connected_sources)
                ) AS score
            FROM {mu_table} mu
            WHERE mu.fact_type = 'observation'
              AND mu.id != ALL($1::uuid[])
              AND EXISTS (
                  SELECT 1 FROM {obs_sources_table} os3
                  WHERE os3.observation_id = mu.id
                    AND os3.source_id IN (SELECT source_id FROM connected_sources)
              )
            ORDER BY score DESC
            FETCH FIRST $2 ROWS ONLY
            """,
            seed_ids,
            budget,
        )
        logger.debug(f"[LinkExpansion] observation graph (Oracle): found {len(entity_rows)} connected observations")

        # Semantic + causal for observations (Oracle path)
        # Avoids GROUP BY CLOB and DISTINCT ON — mirrors _expand_world_facts Oracle strategy.
        sem_causal_rows = await conn.fetch(
            f"""
            WITH sem_scores AS (
                SELECT id, MAX(weight) AS score
                FROM (
                    SELECT mu.id, ml.weight
                    FROM {ml_table} ml JOIN {mu_table} mu ON mu.id = ml.to_unit_id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic' AND mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
                    UNION ALL
                    SELECT mu.id, ml.weight
                    FROM {ml_table} ml JOIN {mu_table} mu ON mu.id = ml.from_unit_id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type = 'semantic' AND mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
                ) sem_raw
                GROUP BY id
            ),
            semantic_expanded AS (
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                       mu.occurred_end, mu.mentioned_at,
                       mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.proof_count,
                       ss.score, 'semantic' AS source
                FROM sem_scores ss
                JOIN {mu_table} mu ON mu.id = ss.id
                ORDER BY ss.score DESC
                FETCH FIRST $2 ROWS ONLY
            ),
            causal_ranked AS (
                SELECT
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id,
                    mu.chunk_id, mu.tags, mu.proof_count, ml.weight AS score,
                    'causal' AS source,
                    ROW_NUMBER() OVER (PARTITION BY mu.id ORDER BY ml.weight DESC) AS rn_
                FROM {ml_table} ml
                JOIN {mu_table} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND mu.fact_type = 'observation'
            ),
            causal_expanded AS (
                SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at,
                       fact_type, document_id, chunk_id, tags, proof_count, score, source
                FROM causal_ranked WHERE rn_ = 1
                ORDER BY score DESC
                FETCH FIRST $2 ROWS ONLY
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
            tag_source=(
                f"{mu_table} mu CROSS APPLY JSON_TABLE(mu.tags, '$[*]' COLUMNS (tag VARCHAR2(256) PATH '$')) jt"
            ),
            non_empty_check="AND mu.tags IS NOT NULL AND DBMS_LOB.GETLENGTH(mu.tags) > 2",
            tag_col="jt.tag",
            bank_prefix="mu.",
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
        # Oracle 23ai supports HNSW vector indexes but does NOT support partial
        # indexes (WHERE clause on CREATE INDEX for vector indexes). Uses a single
        # global HNSW index with ORGANIZATION NEIGHBOR PARTITIONS created during
        # migrations. memory_units is partitioned by LIST (bank_id) AUTOMATIC,
        # so Oracle creates partitions per bank on INSERT and the optimizer can
        # prune partitions on bank_id-scoped queries.
        return

    async def drop_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        schema: str,
        internal_id: str,
        fact_types: dict[str, str],
    ) -> None:
        # Oracle uses a single global vector index (no per-bank indexes to drop).
        return

    def get_entity_resolution_strategy(self) -> str:
        return "oracle_fuzzy"

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
        """Oracle two-step claiming to avoid ORA-02014 with NOT EXISTS + FOR UPDATE."""
        all_rows = []
        claimed_ids = []

        # --- Phase 1: claim from reserved pools ---
        for op_type, limit in reserved_limits.items():
            if limit <= 0:
                continue

            if op_type == "consolidation":
                # Two-step: find busy banks first, then claim excluding them
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
