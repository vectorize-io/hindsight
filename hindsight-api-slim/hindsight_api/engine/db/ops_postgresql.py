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
            batch_rows = await conn.fetch(
                f"""
                SELECT sub.from_id, sub.id, sub.event_date, sub.time_diff_hours
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
        return f"""
            semantic_expanded AS (
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
            causal_expanded AS (
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
                  AND ml.weight >= $4
                  AND mu.fact_type = $2
                ORDER BY mu.id, ml.weight DESC
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
        causal_weight_threshold: float,
    ) -> tuple[list[ResultRow], list[ResultRow], list[ResultRow]]:
        # Entity expansion via source_memory_ids (native array ops)
        entity_rows = await conn.fetch(
            f"""
            WITH source_ids AS (
                SELECT DISTINCT unnest(source_memory_ids) AS source_id
                FROM {mu_table}
                WHERE id = ANY($1::uuid[])
                  AND source_memory_ids IS NOT NULL
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
                 FROM unnest(mu.source_memory_ids) s
                 WHERE s = ANY(ARRAY(SELECT source_id FROM connected_sources))
                )::float AS score
            FROM {mu_table} mu
            WHERE mu.fact_type = 'observation'
              AND mu.id != ALL($1::uuid[])
              AND mu.source_memory_ids && ARRAY(SELECT source_id FROM connected_sources)::uuid[]
            ORDER BY score DESC
            LIMIT $2
            """,
            seed_ids,
            budget,
        )

        # Semantic + causal expansion (same as non-observation)
        sem_causal_rows = await conn.fetch(
            f"""
            WITH
            semantic_expanded AS (
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
            causal_expanded AS (
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
                  AND ml.weight >= $3
                  AND mu.fact_type = 'observation'
                ORDER BY mu.id, ml.weight DESC
            )
            SELECT * FROM semantic_expanded
            UNION ALL
            SELECT * FROM causal_expanded
            LIMIT $2
            """,
            seed_ids,
            budget,
            causal_weight_threshold,
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

    def _get_mu_table(self) -> str:
        """Get the fully-qualified memory_units table name."""
        from ..memory_engine import fq_table

        return fq_table("memory_units")
