"""IDF-aware BM25 query-term selection for the native tsvector backend.

A long recall query is tokenized and OR-joined into one ``tsquery``
(``tok1 | tok2 | ...``). On the native backend the ``@@`` gate then matches a
large fraction of the bank and ``ts_rank_cd`` — which has no IDF and is not
index-backed — is computed for *every* matched row, so ``ORDER BY ... LIMIT``
cannot prune before ranking. A query that matches thousands of memories times
out (the production +60s BM25 hang this module addresses).

Blindly truncating to the first N tokens is the wrong cut: it keeps whichever
terms happen to come first, which are usually the common, low-signal words that
drive the fan-out, and drops the discriminative ones. Instead we keep the N
tokens with the *lowest* corpus document frequency — the most selective,
highest-signal terms, which is exactly what BM25's IDF weighting favours.

The document frequencies come from ``pg_stats.most_common_elems`` for
``memory_units.search_vector``, a statistic PostgreSQL's ``ANALYZE`` maintains
for free (autovacuum-refreshed). No new table, no index change, no reindex.

Caveats, by design:
- The stats are per *table*, i.e. tenant-global across all banks in the schema,
  not per bank. A term rare tenant-wide but hot in a single bank is not caught
  here; that residual case is what a statement-timeout backstop is for.
- Only the most-common lexemes are tracked, so a query term absent from the
  stats is treated as rare (df 0) and kept — precisely what we want.
- On any failure (stats absent on a freshly loaded table, permissions, an
  unexpected catalog shape) we fall back to keeping the first N tokens, so
  selection can never block recall.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..memory_engine import get_current_schema

if TYPE_CHECKING:
    from ...config import HindsightConfig
    from ..sql import SQLDialect

logger = logging.getLogger(__name__)

# Map each query token to the maximum document frequency among its lexemes (as
# produced by the same text-search config that built ``search_vector``), reading
# the per-lexeme frequencies ANALYZE stored in ``pg_stats.most_common_elems``.
#
# ``most_common_elem_freqs`` carries the per-element frequencies followed by
# trailing summary values (min/max/null frequency), so it is sliced to the
# length of ``most_common_elems``. The ``::text::text[]`` round-trip is the
# standard way to unnest the view's ``anyarray`` element column.
#
# Tokens whose text yields no lexeme (stopwords) come back with
# ``has_lexeme = false`` and are dropped by the caller; tokens whose lexemes are
# absent from the stats get df 0 (rare → kept first).
_TOKEN_DF_SQL = """
WITH toks AS (
    SELECT ord, tok
    FROM unnest($1::text[]) WITH ORDINALITY AS u(tok, ord)
),
lex AS (
    SELECT t.ord, l.lexeme
    FROM toks t
    LEFT JOIN LATERAL unnest(tsvector_to_array(to_tsvector($4::regconfig, t.tok))) AS l(lexeme) ON true
),
stats AS (
    SELECT unnest(most_common_elems::text::text[]) AS lexeme,
           unnest((most_common_elem_freqs)[1:array_length(most_common_elems::text::text[], 1)]) AS freq
    FROM pg_stats
    WHERE schemaname = $2 AND tablename = $3 AND attname = 'search_vector'
)
SELECT lex.ord AS ord,
       bool_or(lex.lexeme IS NOT NULL) AS has_lexeme,
       COALESCE(MAX(s.freq), 0)::float8 AS df
FROM lex
LEFT JOIN stats s ON s.lexeme = lex.lexeme
GROUP BY lex.ord
ORDER BY lex.ord
"""


async def select_selective_bm25_tokens(
    conn,
    tokens: list[str],
    *,
    schema: str,
    table: str,
    language: str,
    max_terms: int,
) -> list[str]:
    """Return at most ``max_terms`` tokens, preferring the lowest-df (most selective).

    The returned tokens keep their original relative order (the OR ``tsquery`` is
    order-insensitive; preserving order keeps traces and logs readable). Falls
    back to the first ``max_terms`` tokens whenever document-frequency stats
    cannot be read, so recall is never blocked by a stats problem.
    """
    if max_terms <= 0 or len(tokens) <= max_terms:
        return tokens

    try:
        rows = await conn.fetch(_TOKEN_DF_SQL, tokens, schema, table, language)
    except Exception:
        logger.debug("BM25 term-df lookup failed; falling back to first-N cap", exc_info=True)
        return tokens[:max_terms]

    if not rows:
        return tokens[:max_terms]

    # ``ord`` is 1-based into ``tokens``. Keep only real search terms (drop
    # stopwords), then take the ``max_terms`` lowest-df, ties broken by position.
    scored = [(row["df"], row["ord"]) for row in rows if row["has_lexeme"]]
    if not scored:
        return tokens[:max_terms]

    scored.sort(key=lambda df_ord: df_ord)
    kept_ords = sorted(ord_ for _, ord_ in scored[:max_terms])
    return [tokens[ord_ - 1] for ord_ in kept_ords]


async def build_bm25_query_text(
    conn,
    dialect: SQLDialect,
    *,
    tokens: list[str],
    query_text: str,
    table: str,
    language: str,
    config: HindsightConfig,
) -> str:
    """Turn query tokens into the text parameter a BM25 arm binds.

    Every BM25 arm — memory recall over ``memory_units`` and knowledge search over
    ``mental_models`` — goes through here, so the two cannot drift apart again: the
    native backend gets one disjunctive ``tok | tok`` tsquery on both paths, and the
    other backends get the raw query text their own parsers expect.

    ``table`` and ``language`` name the corpus the selectivity stats are read from;
    they differ per call site because ``mental_models.search_vector`` is generated
    with a hard-coded ``'english'`` config while ``memory_units`` follows the
    configured native language.
    """
    text_search_extension = config.text_search_extension
    max_query_terms = config.bm25_max_query_terms
    bm25_tokens = tokens
    # Native tsvector has no IDF and ranks every `@@` match, so a long OR query over
    # common terms scans and ranks a large fraction of the corpus (the +60s prod
    # timeout). Keep only the most selective terms — lowest document frequency, read
    # for free from pg_stats — which bounds both the match set and the per-row rank
    # cost while preserving the high-signal terms a blunt first-N cap would discard.
    # PG-native only; best-effort (falls back to first-N when stats are unavailable).
    # Opt out via bm25_selective_terms to cap by position instead.
    if (
        text_search_extension == "native"
        and max_query_terms > 0
        and len(tokens) > max_query_terms
        and config.bm25_selective_terms
        and getattr(conn, "backend_type", "postgresql") == "postgresql"
    ):
        bm25_tokens = await select_selective_bm25_tokens(
            conn,
            tokens,
            schema=get_current_schema(),
            table=table,
            language=language,
            max_terms=max_query_terms,
        )
    return dialect.prepare_bm25_text(
        bm25_tokens,
        query_text,
        text_search_extension=text_search_extension,
        max_query_terms=max_query_terms,
    )
