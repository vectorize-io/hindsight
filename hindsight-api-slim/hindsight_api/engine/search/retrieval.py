"""
Retrieval module for 4-way parallel search.

Implements:
1. Semantic retrieval (vector similarity)
2. BM25 retrieval (keyword/full-text search)
3. Graph retrieval (via pluggable GraphRetriever interface)
4. Temporal retrieval (time-aware search with spreading)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

from ...config import DEFAULT_BM25_MAX_QUERY_TERMS, get_config
from ..db.ops import UpdatedWindow
from ..memory_engine import fq_table, get_current_schema
from ..sql import create_sql_dialect
from .bm25_term_selection import select_selective_bm25_tokens
from .graph_retrieval import GraphRetriever
from .link_expansion_retrieval import GRAPH_SEED_LIMIT, LinkExpansionRetriever
from .tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause_simple
from .types import GraphRetrievalTimings, RetrievalResult

if TYPE_CHECKING:
    from ..query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


def tokenize_query(query_text: str) -> list[str]:
    """Normalize query text and split into BM25 tokens.

    Strips punctuation, lowercases, and splits on whitespace.
    Returns an empty list when the query contains no word characters.
    """
    return re.sub(r"[^\w\s]", " ", query_text.lower()).split()


@dataclass
class ParallelRetrievalResult:
    """Result from parallel retrieval across all methods."""

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph: list[RetrievalResult]
    temporal: list[RetrievalResult] | None
    timings: dict[str, float] = field(default_factory=dict)
    temporal_constraint: tuple | None = None  # (start_date, end_date)
    graph_timings: list[GraphRetrievalTimings] = field(
        default_factory=list
    )  # Graph retrieval sub-step timings per fact type
    max_conn_wait: float = 0.0  # Maximum connection acquisition wait time across all methods


@dataclass
class MultiFactTypeRetrievalResult:
    """Result from retrieval across all fact types."""

    # Results per fact type
    results_by_fact_type: dict[str, ParallelRetrievalResult]
    # Aggregate timings
    timings: dict[str, float] = field(default_factory=dict)
    # Max connection wait across all operations
    max_conn_wait: float = 0.0


@dataclass
class SemanticBm25Result:
    """Per-fact-type candidates returned by the shared semantic/BM25 query."""

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph_seeds: list[RetrievalResult] | None


# Default graph retriever instance (can be overridden)
_default_graph_retriever: GraphRetriever | None = None


def get_default_graph_retriever() -> GraphRetriever:
    """Get or create the default graph retriever.

    The memories store gets first refusal: the SQL retrievers walk `memory_links`
    and `unit_entities`, so a store that keeps its links elsewhere has to supply
    its own or the graph arm would silently return nothing. A store whose links
    are in Postgres returns None and ``config.graph_retriever`` decides, as ever.
    """
    global _default_graph_retriever
    if _default_graph_retriever is None:
        from ..memories import get_memories

        from_store = get_memories().graph_retriever()
        if from_store is not None:
            _default_graph_retriever = from_store
            logger.info("Using the memories store's graph retriever")
            return _default_graph_retriever

        config = get_config()
        retriever_type = config.graph_retriever.lower()
        if retriever_type == "link_expansion":
            _default_graph_retriever = LinkExpansionRetriever()
            logger.info("Using LinkExpansion graph retriever")
        else:
            logger.warning(f"Unknown graph retriever '{retriever_type}', falling back to link_expansion")
            _default_graph_retriever = LinkExpansionRetriever()
    return _default_graph_retriever


def set_default_graph_retriever(retriever: GraphRetriever | None) -> None:
    """Set the default graph retriever (for configuration/testing).

    ``None`` clears the cache so the next call re-resolves it — used when the
    memories store changes, since the retriever is chosen from it.
    """
    global _default_graph_retriever
    _default_graph_retriever = retriever


async def retrieve_semantic_bm25_combined_sql(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_semantic: float | None = None,
    min_keyword: float | None = None,
    graph_seed_min_similarity: float | None = None,
) -> dict[str, SemanticBm25Result]:
    """
    Combined semantic + BM25 retrieval for multiple fact types in a single query.

    Uses UNION ALL of per-fact_type subqueries so that each arm has its own
    ORDER BY ... LIMIT, enabling the partial HNSW indexes per fact_type instead
    of forcing a full sequential scan (which the previous window-function approach
    caused by using PARTITION BY inside ROW_NUMBER()).

    Requires partial HNSW indexes per fact_type (idx_mu_emb_world,
    idx_mu_emb_observation, idx_mu_emb_experience), created automatically by
    Alembic migration a3b4c5d6e7f8_add_partial_hnsw_indexes.py.

    HNSW is approximate — semantic arms over-fetch by 5x (min 100) and trim to
    limit in Python to compensate.  ef_search=200 is set globally on pool
    connections at init time (see memory_engine.py) to improve recall on sparse
    graphs.

    fact_type values are inlined as literals (safe: they come from a controlled
    internal enum, never from user input).

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        query_text: Query text for BM25
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        limit: Maximum results per method per fact type
        tags: Optional tags to filter by
        tags_match: Tag matching mode

    Returns:
        Candidate groups for each fact type. ``graph_seeds`` is ``None`` when
        the semantic query's threshold is too strict to cover graph entry points.
    """
    result_dict = {ft: SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None) for ft in fact_types}

    config = get_config()
    tokens = tokenize_query(query_text)

    # Per-request retrieval-level score floors (recall min_scores.semantic / .keyword)
    # override the global config defaults for this query, pruning weak matches in
    # the SQL arms before fusion.
    sem_min = min_semantic if min_semantic is not None else config.semantic_min_similarity
    bm25_min = min_keyword if min_keyword is not None else config.bm25_min_score

    # Over-fetch for HNSW approximation; semantic results trimmed to limit in Python.
    hnsw_fetch = max(limit * 5, 100)

    cols = (
        "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
        "fact_type, document_id, chunk_id, tags, metadata, proof_count"
    )
    table = fq_table("memory_units")

    # Use the SQL dialect to build backend-specific query arms, avoiding
    # inline if/else branches for each database.
    # Use getattr for backward compat: raw asyncpg connections (used in some
    # tests) lack backend_type; default to "postgresql".
    dialect = create_sql_dialect(getattr(conn, "backend_type", "postgresql"))

    # --- Parameter layout ---
    # $1 = query_emb_str  (semantic arms)
    # $2 = bank_id
    # When tokens present:
    #   $3 = limit          (BM25 LIMIT; semantic uses inlined hnsw_fetch literal)
    #   $4 = bm25_text
    #   $5 = tags           (if present)
    #   $6+ = tag_groups params (one per leaf)
    # When no tokens:
    #   $3 = tags           (if present)
    #   $4+ = tag_groups params (one per leaf)
    _include_bm25 = bool(tokens)
    tags_param_idx = 5 if _include_bm25 else 3
    tags_clause = build_tags_where_clause_simple(tags, tags_param_idx, match=tags_match)

    # tag_groups params start immediately after the tags param slot
    tag_groups_param_start = tags_param_idx + (1 if tags else 0)
    groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

    # --- created_after/created_before time range filter (appended after tags/groups) ---
    # The bounds are named for creation but filter `updated_at` — "memories that changed
    # in this window", so an edited fact re-enters it. That is what the mental-model delta
    # refresh needs from its watermark; see META_UPDATED_AT in engine/memories/base.py.
    # Param indices are computed relative to the final params list built below,
    # so we pre-compute the next available index after all preceding params.
    _next_idx = tag_groups_param_start + len(groups_params)
    updated_range_clause = ""
    updated_range_params: list[Any] = []
    if created_after is not None:
        updated_range_params.append(created_after)
        updated_range_clause += f" AND updated_at > ${_next_idx}"
        _next_idx += 1
    if created_before is not None:
        updated_range_params.append(created_before)
        updated_range_clause += f" AND updated_at < ${_next_idx}"
        _next_idx += 1

    # --- Semantic UNION ALL arms (one per fact_type) ---
    # Each arm has its own ORDER BY ... LIMIT, enabling the partial HNSW indexes
    # per fact_type instead of forcing a full sequential scan.
    arms = [
        dialect.build_semantic_arm(
            table=table,
            cols=cols,
            fact_type=ft,
            embedding_param="$1",
            bank_id_param="$2",
            fetch_limit=hnsw_fetch,
            min_similarity=sem_min,
            tags_clause=tags_clause,
            groups_clause=groups_clause,
            extra_where=updated_range_clause,
        )
        for ft in fact_types
    ]

    # --- BM25 UNION ALL arms (one per fact_type, only when tokens present) ---
    if _include_bm25:
        text_ext = config.text_search_extension
        max_query_terms = getattr(config, "bm25_max_query_terms", DEFAULT_BM25_MAX_QUERY_TERMS)
        bm25_tokens = tokens
        # Native tsvector has no IDF and ranks every `@@` match, so a long OR
        # query over common terms scans and ranks a large fraction of the bank
        # (the +60s prod timeout). Keep only the most selective terms — lowest
        # tenant-wide document frequency, read for free from pg_stats — which
        # bounds both the match set and the per-row rank cost while preserving
        # the high-signal terms a blunt first-N cap would discard. PG-native
        # only; best-effort (falls back to first-N when stats are unavailable).
        # Opt out via bm25_selective_terms to cap by position instead.
        if (
            text_ext == "native"
            and max_query_terms > 0
            and len(tokens) > max_query_terms
            and getattr(config, "bm25_selective_terms", True)
            and getattr(conn, "backend_type", "postgresql") == "postgresql"
        ):
            bm25_tokens = await select_selective_bm25_tokens(
                conn,
                tokens,
                schema=get_current_schema(),
                table="memory_units",
                language=config.text_search_extension_native_language,
                max_terms=max_query_terms,
            )
        bm25_text_param: str = dialect.prepare_bm25_text(
            bm25_tokens,
            query_text,
            text_search_extension=text_ext,
            max_query_terms=max_query_terms,
        )
        for i, ft in enumerate(fact_types):
            arms.append(
                dialect.build_bm25_arm(
                    table=table,
                    cols=cols,
                    fact_type=ft,
                    bank_id_param="$2",
                    limit_param="$3",
                    text_param="$4",
                    tags_clause=tags_clause,
                    groups_clause=groups_clause,
                    arm_index=i,
                    text_search_extension=text_ext,
                    bm25_language=config.text_search_extension_native_language,
                    bm25_min_score=bm25_min,
                    extra_where=updated_range_clause,
                )
            )

    query = "\nUNION ALL\n".join(arms)

    params: list = [query_emb_str, bank_id]
    if _include_bm25:
        params.append(limit)  # $3: BM25 LIMIT (only referenced when tokens are present)
        params.append(bm25_text_param)  # $4
    if tags:
        params.append(tags)
    params.extend(groups_params)
    params.extend(updated_range_params)

    try:
        rows = await conn.fetch(query, *params)
    except Exception as e:
        # Oracle Text CONTAINS can fail with DRG-10599 ("column is not indexed")
        # if the CTXSYS text index hasn't synced yet or is unavailable.  Fall
        # back to semantic-only so the search still returns results.
        # We must rebuild the semantic arms with no-BM25 param indices because
        # Oracle requires every bind param to be referenced in the query (DPY-4008).
        err_str = str(e)
        if _include_bm25 and ("DRG-10599" in err_str or "ORA-30600" in err_str or "ORA-29902" in err_str):
            logger.warning("Oracle Text CONTAINS failed (%s), falling back to semantic-only search", err_str[:120])
            # Rebuild with no-BM25 param layout: $1=embedding, $2=bank_id, $3=tags, ...
            fb_tags_idx = 3
            fb_tags_clause = build_tags_where_clause_simple(tags, fb_tags_idx, match=tags_match)
            fb_groups_start = fb_tags_idx + (1 if tags else 0)
            fb_groups_clause, _, _ = build_tag_groups_where_clause(tag_groups, fb_groups_start)
            fb_next_idx = fb_groups_start + len(groups_params)
            fb_updated_clause = ""
            if created_after is not None:
                fb_updated_clause += f" AND updated_at > ${fb_next_idx}"
                fb_next_idx += 1
            if created_before is not None:
                fb_updated_clause += f" AND updated_at < ${fb_next_idx}"
                fb_next_idx += 1
            fb_arms = [
                dialect.build_semantic_arm(
                    table=table,
                    cols=cols,
                    fact_type=ft,
                    embedding_param="$1",
                    bank_id_param="$2",
                    fetch_limit=hnsw_fetch,
                    min_similarity=sem_min,
                    tags_clause=fb_tags_clause,
                    groups_clause=fb_groups_clause,
                    extra_where=fb_updated_clause,
                )
                for ft in fact_types
            ]
            fb_query = "\nUNION ALL\n".join(fb_arms)
            fb_params: list = [query_emb_str, bank_id]
            if tags:
                fb_params.append(tags)
            fb_params.extend(groups_params)
            fb_params.extend(updated_range_params)
            rows = await conn.fetch(fb_query, *fb_params)
        else:
            raise

    # Group results. The semantic SQL deliberately over-fetches for HNSW recall;
    # when that pool also covers the graph threshold, derive graph entry points
    # from the same ordered rows instead of issuing one duplicate ANN query per
    # fact type. Convert only the prefix either consumer can observe, not the
    # entire HNSW over-fetch pool.
    graph_seed_threshold = (
        graph_seed_min_similarity
        if graph_seed_min_similarity is not None and sem_min <= graph_seed_min_similarity
        else None
    )
    semantic_candidate_limit = max(limit, GRAPH_SEED_LIMIT if graph_seed_threshold is not None else 0)
    semantic_candidates: dict[str, list[RetrievalResult]] = {ft: [] for ft in fact_types}
    for r in rows:
        row = dict(r)
        source = row.pop("source")
        ft = row.get("fact_type")
        if ft not in result_dict:
            continue
        if source == "semantic":
            if len(semantic_candidates[ft]) < semantic_candidate_limit:
                semantic_candidates[ft].append(RetrievalResult.from_db_row(row))
        else:
            result_dict[ft].bm25.append(RetrievalResult.from_db_row(row))

    for ft, candidates in semantic_candidates.items():
        result_dict[ft].semantic.extend(candidates[:limit])
        if graph_seed_threshold is not None:
            result_dict[ft].graph_seeds = [
                candidate
                for candidate in candidates
                if candidate.similarity is not None and candidate.similarity >= graph_seed_threshold
            ][:GRAPH_SEED_LIMIT]

    return result_dict


# Temporal entry-point selection tuning.
_TEMPORAL_POOL_SIZE = 60  # ANN candidates fetched per fact_type before coverage selection
_TEMPORAL_ENTRY_POINTS = 10  # entry points kept per fact_type after coverage selection
_TEMPORAL_COVERAGE_BUCKETS = 8  # time-buckets the window is divided into for coverage


def _coalesce_date(row: Any) -> datetime | None:
    """The unit's effective time — matches COALESCE(occurred_start, mentioned_at, occurred_end)."""
    return row["occurred_start"] or row["mentioned_at"] or row["occurred_end"]


def _select_with_temporal_coverage(
    pool: list,
    start_date: datetime,
    end_date: datetime,
    limit: int,
    n_buckets: int,
) -> list:
    """Pick `limit` entry points from a similarity-ranked pool, spread across the window.

    The window [start_date, end_date] is split into `n_buckets` equal time-buckets.
    Candidates are taken round-robin across the buckets that contain them — the
    best-similarity item from each populated bucket first, then the second-best from each,
    and so on — so every populated slice of the window is represented before any slice
    contributes a second item. Within a tier, higher-similarity items lead. When the
    in-window dates are degenerate (all in one bucket — e.g. a batch stamped with a single
    date) this collapses to plain similarity order.
    """
    if len(pool) <= limit:
        return list(pool)

    ranked = sorted(pool, key=lambda r: r["similarity"], reverse=True)
    span = (end_date - start_date).total_seconds()

    def _bucket(row: Any) -> int:
        d = _coalesce_date(row)
        if d is None or span <= 0:
            return 0
        if d.tzinfo is None:
            d = d.replace(tzinfo=UTC)
        frac = (d - start_date).total_seconds() / span
        return max(0, min(int(frac * n_buckets), n_buckets - 1))

    buckets: dict[int, list] = {}
    for row in ranked:  # ranked is similarity-desc, so each bucket list inherits that order
        buckets.setdefault(_bucket(row), []).append(row)

    selected: list = []
    tier = 0
    while len(selected) < limit and any(len(b) > tier for b in buckets.values()):
        # The tier-th best item from every bucket that still has one, strongest first.
        tier_rows = [b[tier] for b in buckets.values() if len(b) > tier]
        tier_rows.sort(key=lambda r: r["similarity"], reverse=True)
        for row in tier_rows:
            if len(selected) < limit:
                selected.append(row)
        tier += 1
    return selected


async def retrieve_temporal_combined_sql(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_types: list[str],
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> dict[str, list[RetrievalResult]]:
    """
    Temporal retrieval for multiple fact types in a single query.

    Batches the entry point query using window functions to get top-N per fact type,
    then runs spreading for each fact type.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        start_date: Start of time range
        end_date: End of time range
        budget: Node budget for spreading per fact type
        semantic_threshold: Minimum semantic similarity to include

    Returns:
        Dict mapping fact_type -> list of RetrievalResult
    """
    from ..memory_engine import fq_table

    # Ensure dates are timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    # Build tags clause
    # Entry-point query: fixed params are $1-$5 (emb, bank, start, end, threshold), tags at $6.
    # fact_type is inlined as a literal per UNION ALL arm (not a bind) — this avoids `unnest`,
    # which has no Oracle equivalent (the `<=>` operator and LIMIT are translated to Oracle by
    # the backend on execute, but `unnest` is not). Mirrors retrieve_semantic_bm25_combined_sql.
    tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)
    tag_groups_param_start = 6 + (1 if tags else 0)
    groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

    # created_after/created_before time range filter (after tags/groups) — filters
    # `updated_at`, as above.
    _next_idx = tag_groups_param_start + len(groups_params)
    updated_range_clause = ""
    updated_range_params: list[Any] = []
    if created_after is not None:
        updated_range_params.append(created_after)
        updated_range_clause += f" AND updated_at > ${_next_idx}"
        _next_idx += 1
    if created_before is not None:
        updated_range_params.append(created_before)
        updated_range_clause += f" AND updated_at < ${_next_idx}"
        _next_idx += 1

    params: list = [query_emb_str, bank_id, start_date, end_date, semantic_threshold]
    if tags:
        params.append(tags)
    params.extend(groups_params)
    params.extend(updated_range_params)

    # Entry-point selection: similarity-gated, window-filtered, then narrowed for coverage.
    #
    # For each fact_type, ANN-rank the units whose time overlaps the window
    # (ORDER BY embedding <=> query) and keep a pool of the most relevant
    # (_TEMPORAL_POOL_SIZE). The planner serves this from the per-(bank, fact_type) vector
    # index when the window is broad — the dense-metadata case, where the window matches
    # most rows — and from the partial date indexes plus an exact sort when the window is
    # narrow. Either way the work is bounded; neither path is a scan-and-sort of the whole
    # match set.
    #
    # Selecting by *similarity* (not recency) is deliberate. The earlier form ranked the
    # entire match set by COALESCE(occurred_start, mentioned_at, occurred_end) and kept the
    # 50 most recent: that biased results toward the end of the window and, on banks with
    # dense/near-uniform dates (e.g. a retain batch stamped with one date), the date key was
    # degenerate so the "50 most recent" became a near-random sample that could drop the
    # single most relevant in-window memory — and it degraded to a full scan + disk-spilling
    # sort (30s+ on a 660k-row bank). The pool is then narrowed to _TEMPORAL_ENTRY_POINTS per
    # fact_type by _select_with_temporal_coverage so the entry points span the window's range
    # rather than clustering in one slice.
    if not fact_types:
        return {}

    # One similarity-ranked, window-filtered arm per fact_type, UNION ALL'd — each arm has its
    # own ORDER BY ... LIMIT so the per-(bank, fact_type) vector index can serve it. fact_type
    # is inlined as a literal (controlled internal enum, never user input), matching
    # retrieve_semantic_bm25_combined_sql; this keeps the query free of `unnest`/LATERAL, which the
    # Oracle backend cannot translate.
    pool_cols = (
        "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
        "fact_type, proof_count, document_id, chunk_id, tags, metadata"
    )
    table = fq_table("memory_units")
    arms = [
        f"""(
        SELECT {pool_cols}, 1 - (embedding <=> $1::vector) AS similarity
        FROM {table}
        WHERE bank_id = $2
          AND fact_type = '{ft}'
          AND embedding IS NOT NULL
          AND (
              (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
               AND occurred_start <= $4 AND occurred_end >= $3)
              OR
              (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $3 AND $4)
              OR
              (occurred_start IS NOT NULL AND occurred_start BETWEEN $3 AND $4)
              OR
              (occurred_end IS NOT NULL AND occurred_end BETWEEN $3 AND $4)
          )
          AND (1 - (embedding <=> $1::vector)) >= $5
          {tags_clause}
          {groups_clause}
          {updated_range_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT {_TEMPORAL_POOL_SIZE}
        )"""
        for ft in fact_types
    ]
    pool_rows = await conn.fetch("\nUNION ALL\n".join(arms), *params)

    if not pool_rows:
        return {ft: [] for ft in fact_types}

    # Group the ANN pool by fact type, then narrow each to coverage-spread entry points.
    pool_by_ft: dict[str, list] = {ft: [] for ft in fact_types}
    for row in pool_rows:
        ft = row["fact_type"]
        if ft in pool_by_ft:
            pool_by_ft[ft].append(row)

    entries_by_ft: dict[str, list] = {
        ft: _select_with_temporal_coverage(
            rows, start_date, end_date, _TEMPORAL_ENTRY_POINTS, _TEMPORAL_COVERAGE_BUCKETS
        )
        for ft, rows in pool_by_ft.items()
    }

    # Calculate shared temporal parameters
    total_days = (end_date - start_date).total_seconds() / 86400
    mid_date = start_date + (end_date - start_date) / 2

    # Process each fact type (spreading needs to stay per fact type due to link filtering)
    results_by_ft: dict[str, list[RetrievalResult]] = {}

    for ft in fact_types:
        ft_entry_points = entries_by_ft.get(ft, [])
        if not ft_entry_points:
            results_by_ft[ft] = []
            continue

        results = []
        visited = set()
        node_scores = {}

        # Process entry points
        for ep in ft_entry_points:
            unit_id = str(ep["id"])
            visited.add(unit_id)

            # Calculate temporal proximity
            best_date = None
            if ep["occurred_start"] is not None and ep["occurred_end"] is not None:
                best_date = ep["occurred_start"] + (ep["occurred_end"] - ep["occurred_start"]) / 2
            elif ep["occurred_start"] is not None:
                best_date = ep["occurred_start"]
            elif ep["occurred_end"] is not None:
                best_date = ep["occurred_end"]
            elif ep["mentioned_at"] is not None:
                best_date = ep["mentioned_at"]

            if best_date:
                if best_date.tzinfo is None:
                    best_date = best_date.replace(tzinfo=UTC)
                days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
                temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
            else:
                temporal_proximity = 0.5

            ep_result = RetrievalResult.from_db_row(dict(ep))
            ep_result.temporal_score = temporal_proximity
            ep_result.temporal_proximity = temporal_proximity
            results.append(ep_result)
            node_scores[unit_id] = (ep["similarity"], 1.0)

        # Spreading through temporal links (same as single-fact-type version)
        frontier = list(node_scores.keys())
        budget_remaining = budget - len(ft_entry_points)
        batch_size = 20
        # Per-source neighbor limit: lets the planner use the composite index
        # (from_unit_id, link_type, weight DESC) with early termination, avoiding
        # a full scan of all links from all source nodes before sorting.
        per_source_limit = 10
        # Safety cap on BFS iterations to prevent runaway spreading in dense graphs.
        max_iterations = 5
        iteration = 0

        # Build tags clause for spreading (use param 7 since 1-6 are used)
        spreading_tags_clause = build_tags_where_clause_simple(tags, 7, table_alias="mu.", match=tags_match)
        spreading_groups_param_start = 7 + (1 if tags else 0)
        spreading_groups_clause, spreading_groups_params, _ = build_tag_groups_where_clause(
            tag_groups, spreading_groups_param_start, table_alias="mu."
        )
        # The window has to be repeated here, not just on the entry-point query
        # above: spreading walks temporal/causal links outward, so an in-window
        # entry point would otherwise pull out-of-window neighbours into results.
        spreading_window = UpdatedWindow(
            after=created_after,
            before=created_before,
            first_param_index=spreading_groups_param_start + len(spreading_groups_params),
        )

        # Multi-hop temporal spreading expands a batch of seed ids with
        # ``FROM unnest($2::uuid[])``, which has no Oracle equivalent. On backends
        # without unnest, skip the spread: the temporal entry points are still
        # returned above, and the semantic/keyword/graph retrievers cover the rest.
        supports_unnest = getattr(conn, "backend_type", "postgresql") != "oracle"

        while frontier and budget_remaining > 0 and iteration < max_iterations and supports_unnest:
            iteration += 1
            batch_ids = frontier[:batch_size]
            frontier = frontier[batch_size:]

            # $1=query_emb, $2=batch_ids, $3=fact_type, $4=threshold, $5=per_source_limit, $6=bank_id, $7=tags, $M+=tag_groups
            spreading_params = [query_emb_str, batch_ids, ft, semantic_threshold, per_source_limit, bank_id]
            if tags:
                spreading_params.append(tags)
            spreading_params.extend(spreading_groups_params)
            spreading_params.extend(spreading_window.params)

            # LATERAL join: for each source node, fetch top-K neighbors by weight using
            # the existing idx_memory_links_from_type_weight index with early-exit semantics.
            # This avoids scanning all temporal links from all source nodes before sorting.
            # bank_id on memory_units lets the planner use idx_memory_units_bank_fact_type.
            neighbors = await conn.fetch(
                f"""
                SELECT src.from_unit_id, mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.metadata, mu.proof_count,
                       l.weight, l.link_type,
                       1 - (mu.embedding <=> $1::vector) AS similarity
                FROM unnest($2::uuid[]) AS src(from_unit_id)
                CROSS JOIN LATERAL (
                    SELECT ml.to_unit_id, ml.weight, ml.link_type
                    FROM {fq_table("memory_links")} ml
                    WHERE ml.from_unit_id = src.from_unit_id
                      AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
                      AND ml.weight >= 0.1
                    ORDER BY ml.weight DESC
                    LIMIT $5
                ) l
                JOIN {fq_table("memory_units")} mu ON mu.id = l.to_unit_id
                WHERE mu.bank_id = $6
                  AND mu.fact_type = $3
                  AND mu.embedding IS NOT NULL
                  AND (1 - (mu.embedding <=> $1::vector)) >= $4
                  {spreading_tags_clause}
                  {spreading_groups_clause}
                  {spreading_window.clause("mu")}
                """,
                *spreading_params,
            )

            for n in neighbors:
                neighbor_id = str(n["id"])
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                budget_remaining -= 1

                parent_id = str(n["from_unit_id"])
                _, parent_temporal_score = node_scores.get(parent_id, (0.5, 0.5))

                neighbor_best_date = None
                if n["occurred_start"] is not None and n["occurred_end"] is not None:
                    neighbor_best_date = n["occurred_start"] + (n["occurred_end"] - n["occurred_start"]) / 2
                elif n["occurred_start"] is not None:
                    neighbor_best_date = n["occurred_start"]
                elif n["occurred_end"] is not None:
                    neighbor_best_date = n["occurred_end"]
                elif n["mentioned_at"] is not None:
                    neighbor_best_date = n["mentioned_at"]

                if neighbor_best_date:
                    if neighbor_best_date.tzinfo is None:
                        neighbor_best_date = neighbor_best_date.replace(tzinfo=UTC)
                    days_from_mid = abs((neighbor_best_date - mid_date).total_seconds() / 86400)
                    neighbor_temporal_proximity = (
                        1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
                    )
                else:
                    neighbor_temporal_proximity = 0.3

                link_type = n["link_type"]
                if link_type in ("causes", "caused_by"):
                    causal_boost = 2.0
                elif link_type in ("enables", "prevents"):
                    causal_boost = 1.5
                else:
                    causal_boost = 1.0

                propagated_temporal = parent_temporal_score * n["weight"] * causal_boost * 0.7
                combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)

                neighbor_result = RetrievalResult.from_db_row(dict(n))
                neighbor_result.temporal_score = combined_temporal
                neighbor_result.temporal_proximity = neighbor_temporal_proximity
                results.append(neighbor_result)

                if budget_remaining > 0 and combined_temporal > 0.2:
                    node_scores[neighbor_id] = (n["similarity"], combined_temporal)
                    frontier.append(neighbor_id)

                if budget_remaining <= 0:
                    break

        results_by_ft[ft] = results

    return results_by_ft


async def retrieve_all_fact_types_parallel(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_types: list[str],
    thinking_budget: int,
    question_date: datetime | None = None,
    query_analyzer: Optional["QueryAnalyzer"] = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_semantic: float | None = None,
    min_keyword: float | None = None,
    enable_temporal_retrieval: bool = True,
    enable_graph_retrieval: bool = True,
) -> MultiFactTypeRetrievalResult:
    """
    Retrieve every recall arm for all fact types, through the memories store.

    Extracts the temporal constraint (CPU-only), then hands the whole recall off to the
    store's single ``recall_unified`` method — the one recall interface. How the arms are
    run (a per-arm SQL orchestration for Postgres, a single index query for a store that
    owns its index) is the store's business; this only assembles the per-arm result it
    returns into :class:`MultiFactTypeRetrievalResult`. Fusion/rerank happen downstream.

    Args:
        pool: Database connection pool, handed to the store as its connection handle.
        query_text: Query text
        query_embedding_str: Query embedding as string
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        thinking_budget: Budget for graph traversal and retrieval limits
        question_date: Optional date when question was asked (for temporal filtering)
        query_analyzer: Query analyzer to use (defaults to TransformerQueryAnalyzer)
        enable_temporal_retrieval: Run the temporal arm. False also skips the date-aware
            query analysis that feeds it (no constraint means nothing to filter on).
        enable_graph_retrieval: Run the entity/link graph arm. False skips those queries
            and returns no graph results.

    Returns:
        MultiFactTypeRetrievalResult with results organized by fact type
    """
    import time

    config = get_config()
    start_time = time.time()
    timings: dict[str, float] = {}

    # Step 1: Extract temporal constraint first (CPU work, no DB)
    # Do this before the store call so we know whether the temporal arm is needed at all.
    temporal_extraction_start = time.time()
    temporal_constraint = None
    if enable_temporal_retrieval:
        from .temporal_extraction import extract_temporal_constraint_async

        # Off the event loop: this is pure CPU and would otherwise stall every
        # other in-flight request in the process, not just this recall.
        temporal_constraint = await extract_temporal_constraint_async(
            query_text, reference_date=question_date, analyzer=query_analyzer
        )
    temporal_extraction_time = time.time() - temporal_extraction_start
    timings["temporal_extraction"] = temporal_extraction_time

    # Step 2: Run every arm for every fact type through the store's single recall method.
    from ..memories import RecallArms, get_memories

    unified = await get_memories().recall_unified(
        conn=pool,
        bank_id=bank_id,
        fact_types=fact_types,
        query_embedding=query_embedding_str,
        query_text=query_text,
        limit=thinking_budget,
        temporal_window=temporal_constraint,
        temporal_semantic_threshold=config.temporal_semantic_min_similarity,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        min_semantic=min_semantic,
        min_keyword=min_keyword,
        enable_graph=enable_graph_retrieval,
    )

    results_by_fact_type: dict[str, ParallelRetrievalResult] = {}
    for ft in fact_types:
        arms = unified.get(ft) or RecallArms()
        # An empty temporal list collapses to None — the "no temporal arm" signal downstream.
        temporal_arm = arms.temporal or None
        results_by_fact_type[ft] = ParallelRetrievalResult(
            semantic=arms.semantic,
            bm25=arms.bm25,
            graph=arms.graph,
            temporal=temporal_arm,
            timings={
                "semantic": 0.0,
                "bm25": 0.0,
                "graph": 0.0,
                "temporal": 0.0,
                "temporal_extraction": temporal_extraction_time,
            },
            temporal_constraint=temporal_constraint,
            graph_timings=[],
            max_conn_wait=0.0,
        )

    timings["total"] = time.time() - start_time
    return MultiFactTypeRetrievalResult(
        results_by_fact_type=results_by_fact_type,
        timings=timings,
        max_conn_wait=0.0,
    )
