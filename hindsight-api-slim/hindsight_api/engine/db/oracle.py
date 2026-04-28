"""Oracle 23ai backend implementation using python-oracledb.

Wraps oracledb's async pool and cursor objects behind the DatabaseBackend
and DatabaseConnection interfaces.

Includes transparent query rewriting so that PostgreSQL-style SQL ($1 params,
::type casts) works against Oracle without requiring callers to change their
query strings. Also auto-converts Python UUID objects to bytes for RAW(16)
columns and rewrites PG-specific SQL patterns.

Requires: python-oracledb (thin mode — pure Python, no Oracle client needed).

Supports multi-tenant schema isolation via ALTER SESSION SET CURRENT_SCHEMA.
"""

import datetime
import json
import logging
import re
import uuid as _uuid_mod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, NamedTuple


class _OracleJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and UUID objects."""

    def default(self, o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, _uuid_mod.UUID):
            return str(o)
        return super().default(o)


from .base import DatabaseBackend, DatabaseConnection
from .result import ResultRow

logger = logging.getLogger(__name__)


class RewriteResult(NamedTuple):
    """Result of rewriting a PostgreSQL query to Oracle SQL."""

    query: str
    ignore_dup: bool
    returning_cols: list[str] | None


# ---------------------------------------------------------------------------
# Regex patterns for PostgreSQL → Oracle query rewriting
# ---------------------------------------------------------------------------

_PG_PARAM_RE = re.compile(r"\$(\d+)")

_PG_CAST_RE = re.compile(
    r"::(?:jsonb|json|text\[\]|text|uuid\[\]|uuid|varchar\[\]|varchar|"
    r"timestamptz\[\]|timestamptz|interval|vector\[\]|vector|"
    r"integer\[\]|integer|int|bigint|float|numeric|boolean)"
)

# Match ON CONFLICT [(...)] DO NOTHING — column list is optional
_ON_CONFLICT_DO_NOTHING_RE = re.compile(
    r"\bON\s+CONFLICT\s*(?:\((?:[^()]*|\([^()]*\))*\)\s*)?DO\s+NOTHING\b", re.IGNORECASE
)
_ON_CONFLICT_DO_UPDATE_RE = re.compile(
    r"\bON\s+CONFLICT\s*\((?:[^()]*|\([^()]*\))*\)\s*DO\s+UPDATE\s+SET\b", re.IGNORECASE
)

_RETURNING_RE = re.compile(r"\bRETURNING\s+(.+)", re.IGNORECASE | re.DOTALL)

_ANY_RE = re.compile(r"=\s*ANY\s*\(\s*:(\d+)\s*\)", re.IGNORECASE)
_NOT_ALL_RE = re.compile(r"!=\s*ALL\s*\(\s*:(\d+)\s*\)", re.IGNORECASE)

_JSON_ARROW_TEXT_RE = re.compile(r'("?\w+"?)\s*->>\s*\'(\w+)\'')  # handles both col and "col"
_JSON_HAS_KEY_RE = re.compile(r"(\w+)\s*\?\s*'(\w+)'")
_JSONB_CONTAINS_RE = re.compile(r"(\w+)\s*@>\s*:(\d+)")

# ---------------------------------------------------------------------------
# Argument conversion helpers
# ---------------------------------------------------------------------------


_UUID_STR_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _convert_arg(value: Any) -> Any:
    """Convert a single Python value to an Oracle-compatible bind value.

    Handles:
    - uuid.UUID → bytes (RAW(16))
    - UUID-formatted strings (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) → bytes
    - dict → JSON string
    - list of UUIDs → list of bytes
    - list → JSON string
    """
    if isinstance(value, _uuid_mod.UUID):
        return value.bytes
    if isinstance(value, str) and _UUID_STR_RE.match(value):
        return _uuid_mod.UUID(value).bytes
    if isinstance(value, dict):
        return json.dumps(value, cls=_OracleJSONEncoder)
    if isinstance(value, list):
        # Always serialize lists as JSON strings for Oracle compatibility.
        # JSON_TABLE (used by && and @> rewrites) needs JSON strings,
        # and _expand_any_lists handles re-parsing for IN () expansion.
        if value and isinstance(value[0], _uuid_mod.UUID):
            return json.dumps([str(v) for v in value])
        if value and isinstance(value[0], str) and _UUID_STR_RE.match(value[0]):
            return json.dumps([str(v) for v in value])
        return json.dumps(value, cls=_OracleJSONEncoder)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value
    return value


def _convert_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert a tuple of Python values to Oracle-compatible bind values."""
    return tuple(_convert_arg(a) for a in args)


def _convert_args_list(args_list: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Convert a list of tuples for executemany."""
    return [_convert_args(row) for row in args_list]


# ---------------------------------------------------------------------------
# Row conversion helpers (Oracle → Python)
# ---------------------------------------------------------------------------

# Column names that are known to contain UUIDs stored as RAW(16)
_UUID_COL_SUFFIXES = ("_id", "_uuid")
_UUID_COL_EXACT = {"id", "internal_id"}

# Columns that store JSON arrays/objects as CLOB in Oracle and need deserialization
_JSON_COL_NAMES = {
    "tags",
    "metadata",
    "disposition",
    "retain_params",
    "config",
    "observation_scopes",
    "source_memory_ids",
    "trigger",
    "http_config",
    "event_types",
    "request",
    "response",
    "reflect_response",
    "result_metadata",
    "task_payload",
    "history",
}


def _is_uuid_column(col: str) -> bool:
    """Heuristic: does this column name likely hold a UUID?"""
    return col in _UUID_COL_EXACT or any(col.endswith(s) for s in _UUID_COL_SUFFIXES)


def _convert_row_from_oracle(columns: list[str], row: tuple) -> tuple:
    """Convert Oracle result row values back to Python-friendly types.

    JSON columns (tags, metadata, etc.) are parsed from their CLOB string
    representation into Python objects so that Pydantic models and engine code
    receive the same types as asyncpg provides for PG jsonb columns.
    """
    result = []
    for i, val in enumerate(row):
        col = columns[i] if i < len(columns) else ""
        if isinstance(val, bytes) and len(val) == 16:
            if _is_uuid_column(col):
                try:
                    val = _uuid_mod.UUID(bytes=val)
                except Exception:
                    pass
        elif isinstance(val, str) and col in _JSON_COL_NAMES:
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(val)
    return tuple(result)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------


# Match INSERT ... VALUES (...) ON CONFLICT (...) DO UPDATE SET ...
# The VALUES group supports up to two levels of nested parentheses to handle
# expressions like COALESCE($7, NOW()) inside the values list.
_UPSERT_RE = re.compile(
    r"INSERT\s+INTO\s+(\S+)\s*\(([^)]+)\)\s*"
    r"VALUES\s*\(((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*)\)\s*"
    r"ON\s+CONFLICT\s*\(((?:[^()]*|\([^()]*\))*)\)\s*DO\s+UPDATE\s+SET\s+"
    r"(.+?)(?:\s*RETURNING\s+.+)?$",
    re.IGNORECASE | re.DOTALL,
)


def _split_respecting_parens(s: str) -> list[str]:
    """Split a comma-separated string while respecting parenthesised groups.

    e.g. "$1, COALESCE($2, NOW()), $3" → ["$1", "COALESCE($2, NOW())", "$3"]
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _rewrite_upsert_to_merge(query: str) -> str | None:
    """Rewrite INSERT ... ON CONFLICT ... DO UPDATE SET ... to Oracle MERGE INTO.

    Returns the rewritten query or None if the pattern doesn't match.
    """
    m = _UPSERT_RE.search(query)
    if not m:
        return None

    table = m.group(1).strip()
    insert_cols_str = m.group(2).strip()
    values_str = m.group(3).strip()
    conflict_cols_str = m.group(4).strip()
    update_set_str = m.group(5).strip()

    insert_cols = [c.strip() for c in insert_cols_str.split(",")]
    values = _split_respecting_parens(values_str)
    conflict_cols = [c.strip() for c in conflict_cols_str.split(",")]

    # Build USING clause: SELECT :1 AS col1, :2 AS col2, ... FROM DUAL
    using_parts = []
    for col, val in zip(insert_cols, values):
        using_parts.append(f"{val} AS {col}")
    using_clause = ", ".join(using_parts)

    # Build ON clause: t.conflict_col = s.conflict_col AND ...
    on_parts = [f"t.{c} = s.{c}" for c in conflict_cols]
    on_clause = " AND ".join(on_parts)

    # Build WHEN MATCHED THEN UPDATE SET clause
    # Replace EXCLUDED.col_name with s.col_name
    update_set = re.sub(r"\bEXCLUDED\.(\w+)", r"s.\1", update_set_str)
    # Replace table_name.col_name with t.col_name (table is aliased as 't' in MERGE)
    # Escape table name for use in regex (handle schema-qualified names like schema.table)
    escaped_table = re.escape(table)
    update_set = re.sub(rf"\b{escaped_table}\.(\w+)", r"t.\1", update_set)
    # Strip trailing semicolons or whitespace
    update_set = update_set.rstrip().rstrip(";")

    # Build WHEN NOT MATCHED THEN INSERT
    insert_col_list = ", ".join(insert_cols)
    insert_val_list = ", ".join(f"s.{c}" for c in insert_cols)

    merge_query = (
        f"MERGE INTO {table} t "
        f"USING (SELECT {using_clause} FROM DUAL) s "
        f"ON ({on_clause}) "
        f"WHEN MATCHED THEN UPDATE SET {update_set} "
        f"WHEN NOT MATCHED THEN INSERT ({insert_col_list}) VALUES ({insert_val_list})"
    )

    return merge_query


def _rewrite_pg_to_oracle(query: str) -> RewriteResult:
    """Rewrite PostgreSQL-style SQL to Oracle-compatible SQL.

    Returns (rewritten_query, ignore_dup, returning_cols).
    - ignore_dup: True if ON CONFLICT DO NOTHING was stripped (caller should catch ORA-00001)
    - returning_cols: column names from RETURNING clause (None if no RETURNING)
    """
    ignore_dup = False
    returning_cols: list[str] | None = None

    # $N → :N
    query = _PG_PARAM_RE.sub(r":\1", query)

    # JSONB merge operator: col || :N::jsonb → JSON_MERGEPATCH(col, :N)
    # Must happen BEFORE cast strip so we can detect ::jsonb
    query = re.sub(r"(\w+)\s*\|\|\s*(:\w+)::jsonb", r"JSON_MERGEPATCH(\1, \2)", query, flags=re.IGNORECASE)

    # JSONB merge with complex left-hand expression (e.g. COALESCE(...)):
    #   COALESCE(col, '[]'::jsonb) || :N::jsonb
    #   → JSON_MERGEPATCH(COALESCE(col, TO_CLOB('[]')), :N)
    # The simple \w+ regex above won't match a closing paren.  We also
    # wrap any JSON string literals inside the COALESCE with TO_CLOB to
    # prevent ORA-00932 (CHAR vs CLOB type mismatch with CLOB columns).
    def _rewrite_coalesce_merge(m: re.Match) -> str:
        coalesce_expr = m.group(1)
        bind_param = m.group(2)
        # Wrap any 'literal'::jsonb inside COALESCE with TO_CLOB
        coalesce_expr = re.sub(r"'([^']*)'::(jsonb|json)", r"TO_CLOB('\1')", coalesce_expr, flags=re.IGNORECASE)
        return f"JSON_MERGEPATCH({coalesce_expr}, {bind_param})"

    query = re.sub(
        r"(COALESCE\([^)]+\))\s*\|\|\s*(:\w+)::jsonb",
        _rewrite_coalesce_merge,
        query,
        flags=re.IGNORECASE,
    )

    # JSONB text extract + boolean cast + comparison (must run BEFORE cast strip)
    # (trigger->>'refresh_after_consolidation')::boolean = true → JSON_VALUE("trigger", '$.key') = 'true'
    def _rewrite_json_bool(m: re.Match) -> str:
        col = m.group(1)
        key = m.group(2)
        val = m.group(3).lower()
        # Quote Oracle reserved words
        if col.lower() in ("trigger", "comment", "order", "group", "index"):
            col = f'"{col}"'
        return f"JSON_VALUE({col}, '$.{key}') = '{val}'"

    query = re.sub(
        r"""\((\w+)\s*->>\s*'(\w+)'\)::boolean\s*=\s*(true|false)""",
        _rewrite_json_bool,
        query,
        flags=re.IGNORECASE,
    )

    # Strip ::type casts (including bare ::jsonb on literals in generic contexts)
    query = _PG_CAST_RE.sub("", query)

    # PG-specific SET SESSION commands → skip (Oracle doesn't need these)
    if re.match(r"\s*SET\s+SESSION\s+CHARACTERISTICS\b", query, re.IGNORECASE):
        return RewriteResult("SELECT 1 FROM DUAL", False, None)

    # NOW() → SYSTIMESTAMP
    query = re.sub(r"\bNOW\(\)", "SYSTIMESTAMP", query, flags=re.IGNORECASE)
    # gen_random_uuid() → SYS_GUID()
    query = re.sub(r"\bgen_random_uuid\(\)", "SYS_GUID()", query, flags=re.IGNORECASE)
    # Boolean literals: Oracle uses NUMBER(1) for booleans
    query = re.sub(r"\b=\s*TRUE\b", "= 1", query, flags=re.IGNORECASE)
    query = re.sub(r"\b=\s*FALSE\b", "= 0", query, flags=re.IGNORECASE)
    # FOR SHARE → FOR UPDATE (Oracle doesn't support FOR SHARE)
    query = re.sub(r"\bFOR\s+SHARE\b", "FOR UPDATE", query, flags=re.IGNORECASE)

    # Oracle reserved word: quote "trigger" column name.
    # Use negative lookbehind/lookahead to skip already-quoted occurrences.
    query = re.sub(r'(?<!")\btrigger\b(?!")', '"trigger"', query)

    # date_trunc('interval', expr) → TRUNC(expr, 'fmt')
    # The second capture group is a balanced expression (not just a column name)
    # to handle e.g. date_trunc('hour', created_at AT TIME ZONE 'UTC').
    _DATE_TRUNC_MAP = {"day": "DD", "hour": "HH24", "month": "MM", "week": "IW", "year": "YYYY", "minute": "MI"}

    def _rewrite_date_trunc(m):
        interval = m.group(1).lower()
        expr = m.group(2).strip()
        # Strip AT TIME ZONE — Oracle timestamps are already in the session timezone.
        expr = re.sub(r"\s+AT\s+TIME\s+ZONE\s+'[^']*'", "", expr, flags=re.IGNORECASE)
        fmt = _DATE_TRUNC_MAP.get(interval, "DD")
        return f"TRUNC(CAST({expr} AS DATE), '{fmt}')"

    query = re.sub(r"date_trunc\(\s*'(\w+)'\s*,\s*(.+?)\s*\)", _rewrite_date_trunc, query, flags=re.IGNORECASE)

    # interval 'N units' → NUMTODSINTERVAL(N, 'UNIT')
    # Handles PG interval literals like interval '7 days', interval '1 hour', etc.
    _INTERVAL_MAP = {
        "second": "SECOND",
        "seconds": "SECOND",
        "minute": "MINUTE",
        "minutes": "MINUTE",
        "hour": "HOUR",
        "hours": "HOUR",
        "day": "DAY",
        "days": "DAY",
    }

    def _rewrite_interval(m):
        num = m.group(1)
        unit = m.group(2).lower()
        ora_unit = _INTERVAL_MAP.get(unit)
        if ora_unit:
            return f"NUMTODSINTERVAL({num}, '{ora_unit}')"
        return m.group(0)  # leave as-is if unknown unit

    query = re.sub(r"interval\s+'(\d+)\s+(\w+)'", _rewrite_interval, query, flags=re.IGNORECASE)

    # JSON operators
    query = _JSON_ARROW_TEXT_RE.sub(r"JSON_VALUE(\1, '$.\2')", query)
    query = _JSON_HAS_KEY_RE.sub(r"JSON_EXISTS(\1, '$.\2')", query)
    query = _JSONB_CONTAINS_RE.sub(r"JSON_EXISTS(\1, '$' PASSING :\2 AS cond)", query)

    # pgvector distance operator: col <=> :N → VECTOR_DISTANCE(col, :N, COSINE)
    # Use [\w.]+ to capture table-qualified columns like mu.embedding
    query = re.sub(
        r"([\w.]+)\s*<=>\s*(:\w+)",
        r"VECTOR_DISTANCE(\1, \2, COSINE)",
        query,
    )

    # LIMIT/OFFSET rewriting: PG uses "LIMIT N OFFSET M" or "LIMIT N" or "OFFSET M LIMIT N"
    # Oracle uses "OFFSET M ROWS FETCH FIRST N ROWS ONLY" (OFFSET before FETCH FIRST)
    #
    # IMPORTANT: Oracle does NOT allow FETCH FIRST with FOR UPDATE (ORA-02014 — treats
    # the row-limiting clause as an inline view). When FOR UPDATE is present, we must use
    # ROWNUM in the WHERE clause instead. This is safe because FOR UPDATE + LIMIT queries
    # in the poller are simple single-table SELECTs with no OFFSET.
    has_for_update = bool(re.search(r"\bFOR\s+UPDATE\b", query, re.IGNORECASE))

    if has_for_update:
        # FOR UPDATE path: use ROWNUM instead of FETCH FIRST.
        # Extract and remove LIMIT clause, inject ROWNUM into WHERE.
        def _limit_to_rownum(m):
            return ""  # Remove the LIMIT clause; we'll add ROWNUM below

        limit_val = None
        limit_match = re.search(r"\bLIMIT\s+(\d+|:\w+)\b", query, re.IGNORECASE)
        if limit_match:
            limit_val = limit_match.group(1)
            query = re.sub(r"\bLIMIT\s+(\d+|:\w+)\b", "", query, flags=re.IGNORECASE)

        # Remove OFFSET if present (not expected with FOR UPDATE, but be safe)
        query = re.sub(r"\bOFFSET\s+(\d+|:\w+)\b(?!\s+ROWS)", "", query, flags=re.IGNORECASE)

        # Inject ROWNUM constraint into WHERE clause
        if limit_val is not None:
            # Insert ROWNUM <= N right after WHERE
            query = re.sub(
                r"\bWHERE\b",
                f"WHERE ROWNUM <= {limit_val} AND",
                query,
                count=1,
                flags=re.IGNORECASE,
            )
    else:
        # No FOR UPDATE: use standard FETCH FIRST / OFFSET ROWS syntax
        # First handle "LIMIT N OFFSET M" → "OFFSET M ROWS FETCH FIRST N ROWS ONLY"
        query = re.sub(
            r"\bLIMIT\s+(\d+|:\w+)\s+OFFSET\s+(\d+|:\w+)\b",
            r"OFFSET \2 ROWS FETCH FIRST \1 ROWS ONLY",
            query,
            flags=re.IGNORECASE,
        )
        # Handle "OFFSET M LIMIT N" → "OFFSET M ROWS FETCH FIRST N ROWS ONLY"
        query = re.sub(
            r"\bOFFSET\s+(\d+|:\w+)\s+LIMIT\s+(\d+|:\w+)\b",
            r"OFFSET \1 ROWS FETCH FIRST \2 ROWS ONLY",
            query,
            flags=re.IGNORECASE,
        )
        # Handle standalone "LIMIT N" (no OFFSET)
        query = re.sub(
            r"\bLIMIT\s+(\d+|:\w+)\b",
            r"FETCH FIRST \1 ROWS ONLY",
            query,
            flags=re.IGNORECASE,
        )
        # Handle standalone "OFFSET N" (no LIMIT, less common)
        query = re.sub(
            r"\bOFFSET\s+(\d+|:\w+)\b(?!\s+ROWS)",
            r"OFFSET \1 ROWS",
            query,
            flags=re.IGNORECASE,
        )

    # PG non-empty array check: tags != '{}' → Oracle: NOT (DBMS_LOB empty check)
    query = re.sub(
        r"(\w+)\s*!=\s*'\{\}'",
        r"NOT (DBMS_LOB.GETLENGTH(\1) IS NULL OR DBMS_LOB.GETLENGTH(\1) <= 2)",
        query,
    )

    # PG empty array check: tags = '{}' in comparisons → Oracle: DBMS_LOB empty check
    # Avoid matching SET assignments (preceded by SET keyword or comma in SET list)
    # Use a function to check context
    def _rewrite_empty_eq(m):
        col = m.group(1)
        return f"(DBMS_LOB.GETLENGTH({col}) IS NULL OR DBMS_LOB.GETLENGTH({col}) <= 2)"

    # Match col = '{}' preceded by OR/AND/WHERE or opening paren (comparison context)
    query = re.sub(r"(?<=\bOR\s)(\w+)\s*=\s*'\{\}'", _rewrite_empty_eq, query, flags=re.IGNORECASE)
    query = re.sub(r"(?<=\bAND\s)(\w+)\s*=\s*'\{\}'", _rewrite_empty_eq, query, flags=re.IGNORECASE)
    query = re.sub(r"(?<=\bWHERE\s)(\w+)\s*=\s*'\{\}'", _rewrite_empty_eq, query, flags=re.IGNORECASE)

    # PG array overlap: tags && :N → Oracle: JSON array overlap check using JSON_TABLE
    def _rewrite_array_overlap(m):
        col = m.group(1)
        param = m.group(2)
        return f"EXISTS (SELECT 1 FROM JSON_TABLE({param}, '$[*]' COLUMNS (val VARCHAR2(256) PATH '$')) jt WHERE JSON_EXISTS({col}, '$[*]?(@ == $v)' PASSING jt.val AS \"v\"))"

    query = re.sub(r"(\w+)\s*&&\s*(:\w+)", _rewrite_array_overlap, query)

    # PG array containment: tags @> :N → Oracle: all elements from param exist in col
    # (Override the JSONB contains regex which doesn't work for array containment)
    # Already handled by _JSONB_CONTAINS_RE above, but that generates wrong SQL for arrays.
    # Re-detect and fix: JSON_EXISTS(col, '$' PASSING :N AS cond) → proper array containment
    def _rewrite_array_contains(m):
        col = m.group(1)
        param = m.group(2)
        return (
            f"(SELECT COUNT(*) FROM JSON_TABLE({param}, '$[*]' COLUMNS (val VARCHAR2(256) PATH '$')) jt "
            f"WHERE JSON_EXISTS({col}, '$[*]?(@ == $v)' PASSING jt.val AS \"v\")) = "
            f"JSON_VALUE({param}, '$.size()' RETURNING NUMBER)"
        )

    # Fix the already-rewritten @> pattern if it was handled by _JSONB_CONTAINS_RE
    query = re.sub(
        r"JSON_EXISTS\((\w+),\s*'\$'\s*PASSING\s*(:\w+)\s*AS\s*cond\)",
        _rewrite_array_contains,
        query,
    )

    # ILIKE → UPPER(...) LIKE UPPER(...)
    query = re.sub(
        r"(\w+)\s+ILIKE\s+(:\w+)",
        r"UPPER(\1) LIKE UPPER(\2)",
        query,
        flags=re.IGNORECASE,
    )

    # = ANY(:N) → placeholder that _make_bind_params will expand
    # Mark these for expansion: the bind var is a list that needs to be expanded
    # into multiple individual bind vars.
    query = _ANY_RE.sub(r"IN (/*EXPAND:\1*/)", query)

    # != ALL(:N) → NOT IN (expanded list) — the negative counterpart of = ANY
    query = _NOT_ALL_RE.sub(r"NOT IN (/*EXPAND:\1*/)", query)

    # CTE AS MATERIALIZED (...) → AS (...) — Oracle doesn't support MATERIALIZED CTE hint
    query = re.sub(r"\bAS\s+MATERIALIZED\s*\(", "AS (", query, flags=re.IGNORECASE)

    # COALESCE(:N, <numeric_literal>) — Oracle defaults None bind vars to VARCHAR2,
    # causing type mismatch.  Wrap the bind var in TO_NUMBER so types align.
    query = re.sub(
        r"COALESCE\(\s*(:\w+)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
        r"COALESCE(TO_NUMBER(\1), \2)",
        query,
        flags=re.IGNORECASE,
    )

    # CROSS JOIN LATERAL → CROSS APPLY (Oracle 12c+)
    query = re.sub(r"\bCROSS\s+JOIN\s+LATERAL\b", "CROSS APPLY", query, flags=re.IGNORECASE)

    # Note: DISTINCT ON is PG-specific and NOT handled here — it's too complex
    # to rewrite transparently in a multi-CTE query. Consumers that use DISTINCT ON
    # must provide Oracle-specific alternatives.

    # ON CONFLICT DO NOTHING → strip it, suppress ORA-00001
    if _ON_CONFLICT_DO_NOTHING_RE.search(query):
        query = _ON_CONFLICT_DO_NOTHING_RE.sub("", query)
        ignore_dup = True
        # Also strip any RETURNING clause (can't return from a dup-suppressed insert)
        query = _RETURNING_RE.sub("", query)
        query = query.strip()

    # ON CONFLICT ... DO UPDATE SET → rewrite to MERGE INTO
    if _ON_CONFLICT_DO_UPDATE_RE.search(query):
        merged = _rewrite_upsert_to_merge(query)
        if merged is not None:
            query = merged
        else:
            logger.warning("ON CONFLICT DO UPDATE rewrite failed — query may not work: %s", query[:200])

    # RETURNING clause → rewrite to RETURNING ... INTO :ret_0, :ret_1, ...
    # Only applies to DML (INSERT/UPDATE/DELETE) — SELECT queries should never
    # have RETURNING rewritten, even if they contain the word in a string literal
    # or CTE alias.
    if not ignore_dup and re.match(r"\s*(INSERT|UPDATE|DELETE)\b", query, re.IGNORECASE):
        m = _RETURNING_RE.search(query)
        if m:
            ret_cols_str = m.group(1).strip()
            returning_cols = [c.strip() for c in ret_cols_str.split(",") if c.strip()]
            into_vars = ", ".join(f":ret_{i}" for i in range(len(returning_cols)))
            query = query[: m.start()] + f"RETURNING {ret_cols_str} INTO {into_vars}" + query[m.end() :]

    return RewriteResult(query, ignore_dup, returning_cols)


# ---------------------------------------------------------------------------
# oracledb lazy import
# ---------------------------------------------------------------------------


def _import_oracledb():
    """Lazy import oracledb to avoid hard dependency."""
    try:
        import oracledb  # type: ignore[import-not-found]

        oracledb.defaults.fetch_lobs = False
        return oracledb
    except ImportError:
        raise ImportError(
            "python-oracledb is required for Oracle backend. Install it with: pip install oracledb"
        ) from None


# ---------------------------------------------------------------------------
# OracleConnection
# ---------------------------------------------------------------------------


class OracleConnection(DatabaseConnection):
    """DatabaseConnection wrapper around an oracledb async connection.

    Transparently rewrites PostgreSQL-style SQL ($N params, ::type casts)
    to Oracle syntax so that existing consumer code works without modification.
    """

    __slots__ = ("_conn",)

    @property
    def backend_type(self) -> str:
        return "oracle"

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # -- helpers ----------------------------------------------------------

    def _make_bind_params(
        self, cursor: Any, args: tuple[Any, ...], returning_cols: list[str] | None
    ) -> dict[str, Any] | None:
        """Build bind params as a dict (named binding).

        Always uses dict-based named binding (:1, :2, ...) because oracledb's
        positional tuple binding fails when a placeholder like :1 appears
        multiple times in the query (counts references, not distinct placeholders).

        JSON-serialized strings (from lists/dicts) are explicitly typed as CLOB
        via setinputsizes so Oracle doesn't reject short JSON like '[]' or '{}'
        that would otherwise bind as VARCHAR2.
        """
        oracledb = _import_oracledb()
        converted = _convert_args(args) if args else ()

        if not converted and returning_cols is None:
            return None

        # Always use named params dict for Oracle
        params: dict[str, Any] = {}
        for i, val in enumerate(converted):
            params[str(i + 1)] = val

        if returning_cols is not None:
            # Column names that are known to be numeric
            _NUMERIC_COLS = {
                "max_tokens",
                "priority",
                "proof_count",
                "access_count",
                "importance_score",
                "decay_factor",
                "chunk_index",
                "progress",
                "active",
            }
            for i, col in enumerate(returning_cols):
                clean = col.strip().lower()
                if " as " in clean:
                    clean = clean.split(" as ")[-1].strip()
                # Use appropriate type for timestamp columns
                if clean.endswith("_at") or clean in ("started_at", "ended_at", "last_updated", "last_refreshed_at"):
                    params[f"ret_{i}"] = cursor.var(oracledb.DB_TYPE_TIMESTAMP_TZ, arraysize=1)
                elif clean in _NUMERIC_COLS:
                    params[f"ret_{i}"] = cursor.var(oracledb.DB_TYPE_NUMBER, arraysize=1)
                else:
                    params[f"ret_{i}"] = cursor.var(oracledb.DB_TYPE_VARCHAR, arraysize=1)

        return params

    @staticmethod
    def _apply_clob_input_sizes(cursor: Any, query: str, params: dict[str, Any] | None) -> None:
        """Tell oracledb to bind typed input sizes for ambiguous parameters.

        Must be called AFTER _expand_any_lists so we only register sizes for
        params that still exist in the final query.  Handles two cases:

        1. JSON strings: Oracle's thin driver defaults short strings like '[]'
           to VARCHAR2 which fails with ORA-00932 when the column is CLOB.
        2. NULL datetime params: When a None param appears in COALESCE with
           SYSTIMESTAMP, Oracle defaults it to VARCHAR2 causing a type mismatch.
           We detect COALESCE(:N, SYSTIMESTAMP) patterns and hint the param as
           TIMESTAMP WITH TIME ZONE.
        """
        if not params:
            return
        oracledb = _import_oracledb()
        sizes: dict[str, Any] = {}
        for key, val in params.items():
            if isinstance(val, str) and val and val[0] in ("{", "[") and f":{key}" in query:
                sizes[key] = oracledb.DB_TYPE_CLOB
            # None params in COALESCE/GREATEST/LEAST with timestamp columns need
            # explicit timestamp type to avoid ORA-00932 (VARCHAR2 NULL vs
            # TIMESTAMP WITH TIME ZONE mismatch). Match patterns like:
            #   COALESCE(:N, SYSTIMESTAMP)
            #   COALESCE(:N, occurred_end)  -- timestamp column
            #   GREATEST(col, COALESCE(:N, col))
            elif val is None:
                coalesce_match = re.search(rf"COALESCE\s*\(:{key}\s*,\s*(\w+)\)", query, re.IGNORECASE)
                if coalesce_match:
                    fallback = coalesce_match.group(1).lower()
                    # Heuristic: column names ending in _at, or known timestamp cols
                    _TS_NAMES = {
                        "systimestamp",
                        "occurred_start",
                        "occurred_end",
                        "mentioned_at",
                        "last_seen",
                        "event_date",
                    }
                    if fallback in _TS_NAMES or fallback.endswith("_at"):
                        sizes[key] = oracledb.DB_TYPE_TIMESTAMP_TZ
        if sizes:
            cursor.setinputsizes(**sizes)

    _expand_counter = 0

    @staticmethod
    def _expand_any_lists(query: str, params: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
        """Expand /*EXPAND:N*/ markers into individual bind vars for IN clauses.

        Converts: IN (/*EXPAND:1*/) with params["1"] = [a, b, c]
        Into:     IN (:any_0, :any_1, :any_2) with params["any_0"]=a, etc.

        Uses a unique prefix to avoid name collisions with other bind vars.
        The original param is kept (for other references to :N in the query).
        """
        if params is None or "/*EXPAND:" not in query:
            return query, params

        expand_re = re.compile(r"/\*EXPAND:(\d+)\*/")
        keys_to_remove: set[str] = set()

        def _replace(m):
            param_key = m.group(1)
            val = params.get(param_key)
            # Handle JSON string lists (from _convert_arg serialization)
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        val = parsed
                        # Convert UUID strings back to bytes for RAW(16) columns
                        converted = []
                        for item in val:
                            if isinstance(item, str) and _UUID_STR_RE.match(item):
                                converted.append(_uuid_mod.UUID(item).bytes)
                            else:
                                converted.append(item)
                        val = converted
                except (json.JSONDecodeError, TypeError):
                    return f":{param_key}"
            if val is None or not isinstance(val, (list, tuple)):
                return f":{param_key}"
            if len(val) == 0:
                return "NULL"
            OracleConnection._expand_counter += 1
            prefix = f"any{OracleConnection._expand_counter}"
            expanded_keys = []
            for i, item in enumerate(val):
                k = f"{prefix}_{i}"
                params[k] = item
                expanded_keys.append(f":{k}")
            keys_to_remove.add(param_key)
            return ", ".join(expanded_keys)

        query = expand_re.sub(_replace, query)

        # Remove original list params that were expanded — their placeholder
        # (:N) no longer exists in the query, and leaving them causes DPY-4008.
        # Only remove if the key's placeholder is truly gone from the query.
        for key in keys_to_remove:
            if not re.search(rf":{key}\b", query):
                del params[key]

        return query, params

    def _read_returning_values(self, returning_cols: list[str], params: dict[str, Any]) -> dict[str, Any] | None:
        """Read values from RETURNING INTO output variables after execute."""
        row: dict[str, Any] = {}
        for i, col in enumerate(returning_cols):
            var = params[f"ret_{i}"]
            values = var.getvalue()
            if not values:
                return None
            val = values[0] if isinstance(values, list) else values

            # Clean alias: "LOWER(canonical_name) AS name_lower" → "name_lower"
            clean_col = col.strip()
            upper = clean_col.upper()
            if " AS " in upper:
                idx = upper.index(" AS ")
                clean_col = clean_col[idx + 4 :].strip().lower()
            else:
                clean_col = clean_col.lower()

            # Convert hex strings back to UUID for RAW(16) columns
            if isinstance(val, str) and len(val) == 32 and _is_uuid_column(clean_col):
                try:
                    val = _uuid_mod.UUID(val)
                except Exception:
                    pass

            # Auto-parse JSON strings for known JSON columns
            if isinstance(val, str) and clean_col in _JSON_COL_NAMES:
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Auto-parse timestamp strings for columns ending in _at.
            # Oracle returns timestamps as strings; ensure they are
            # timezone-aware (UTC) to avoid naive/aware mismatches
            # downstream (e.g. entity_resolver temporal scoring).
            if isinstance(val, str) and (
                clean_col.endswith("_at") or clean_col in ("started_at", "ended_at", "last_seen", "event_date")
            ):
                try:
                    val = datetime.datetime.fromisoformat(val)
                    if val.tzinfo is None:
                        val = val.replace(tzinfo=datetime.timezone.utc)
                except (ValueError, TypeError):
                    pass

            row[clean_col] = val

        return row if any(v is not None for v in row.values()) else None

    # -- transaction ------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["OracleConnection"]:
        sp_name = f"sp_{_uuid_mod.uuid4().hex[:12]}"
        cursor = self._conn.cursor()
        await cursor.execute(f"SAVEPOINT {sp_name}")
        cursor.close()
        try:
            yield self
        except Exception:
            cursor = self._conn.cursor()
            await cursor.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            cursor.close()
            raise

    # -- DML methods ------------------------------------------------------

    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        orig_query = query
        query, ignore_dup, ret_cols = _rewrite_pg_to_oracle(query)
        cursor = self._conn.cursor()
        try:
            params = self._make_bind_params(cursor, args, ret_cols)
            query, params = self._expand_any_lists(query, params)
            self._apply_clob_input_sizes(cursor, query, params)
            if ignore_dup:
                try:
                    await cursor.execute(query, params)
                except Exception as e:
                    if "ORA-00001" in str(e):
                        return "INSERT 0 0"
                    raise
            else:
                try:
                    await cursor.execute(query, params)
                except Exception as e:
                    logger.error(
                        "Oracle execute failed. Query: %s\nParams keys: %s\nError: %s",
                        query[:500],
                        list(params.keys()) if params else None,
                        e,
                    )
                    raise
            # Return PG-compatible status string
            cmd = orig_query.strip().split()[0].upper() if orig_query.strip() else "OK"
            rowcount = cursor.rowcount
            if cmd == "INSERT":
                return f"INSERT 0 {rowcount}"
            elif cmd in ("DELETE", "UPDATE", "SELECT", "MERGE"):
                return f"{cmd} {rowcount}"
            return f"{cmd} {rowcount}"
        finally:
            cursor.close()

    async def executemany(self, query: str, args: list[tuple[Any, ...]], *, timeout: float | None = None) -> None:
        query, ignore_dup, _ = _rewrite_pg_to_oracle(query)
        converted = _convert_args_list(args)
        cursor = self._conn.cursor()
        try:
            if ignore_dup:
                # Row-by-row with individual dup suppression
                for row in converted:
                    params = {str(i + 1): v for i, v in enumerate(row)}
                    try:
                        await cursor.execute(query, params)
                    except Exception as e:
                        if "ORA-00001" not in str(e):
                            raise
            else:
                # Convert tuples to dicts for named binding (:1, :2, ...)
                converted_dicts = [{str(i + 1): v for i, v in enumerate(row)} for row in converted]
                try:
                    await cursor.executemany(query, converted_dicts)
                except Exception as e:
                    logger.error("Oracle executemany failed. Query: %s\nError: %s", query[:500], e)
                    raise
        finally:
            cursor.close()

    async def bulk_insert_from_arrays(
        self,
        table: str,
        columns: list[str],
        arrays: list[list],
        *,
        column_types: list[str] | None = None,
        returning: str | None = None,
    ) -> list[ResultRow] | str:
        """Oracle override: uses executemany for bulk inserts (no unnest support).

        The non-RETURNING path uses executemany() which batches network round-trips
        efficiently via oracledb's array bind optimization. The RETURNING path must
        use row-by-row inserts because Oracle's RETURNING ... INTO requires individual
        cursor execution to read back output bind variables.
        """
        col_list = ", ".join(columns)
        n_cols = len(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(n_cols))
        n_rows = len(arrays[0]) if arrays else 0

        if returning:
            results: list[ResultRow] = []
            for row_idx in range(n_rows):
                row_args = tuple(arrays[col_idx][row_idx] for col_idx in range(n_cols))
                query = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) RETURNING {returning}"
                rows = await self.fetch(query, *row_args)
                results.extend(rows)
            return results

        rows_data = [tuple(arrays[col_idx][row_idx] for col_idx in range(n_cols)) for row_idx in range(n_rows)]
        query = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        await self.executemany(query, rows_data)
        return f"INSERT 0 {n_rows}"

    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[ResultRow]:
        query, ignore_dup, ret_cols = _rewrite_pg_to_oracle(query)
        cursor = self._conn.cursor()
        try:
            params = self._make_bind_params(cursor, args, ret_cols)
            query, params = self._expand_any_lists(query, params)
            self._apply_clob_input_sizes(cursor, query, params)
            if ignore_dup:
                try:
                    await cursor.execute(query, params)
                except Exception as e:
                    if "ORA-00001" in str(e):
                        return []
                    raise
                # INSERT succeeded but RETURNING was stripped — return empty
                # (callers using fetch with ON CONFLICT DO NOTHING typically
                # only care about rowcount, not the returned rows)
                return []

            try:
                await cursor.execute(query, params)
            except Exception as e:
                logger.error(
                    "Oracle fetch failed. Query: %s\nParams keys: %s\nError: %s",
                    query[:500],
                    list(params.keys()) if params else None,
                    e,
                )
                raise

            if ret_cols is not None:
                row_dict = self._read_returning_values(ret_cols, params)
                return [ResultRow(row_dict)] if row_dict else []

            columns = [col[0].lower() for col in cursor.description or []]
            rows = await cursor.fetchall()
            return [ResultRow(dict(zip(columns, _convert_row_from_oracle(columns, row)))) for row in rows]
        finally:
            cursor.close()

    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> ResultRow | None:
        query, ignore_dup, ret_cols = _rewrite_pg_to_oracle(query)
        cursor = self._conn.cursor()
        try:
            params = self._make_bind_params(cursor, args, ret_cols)
            query, params = self._expand_any_lists(query, params)
            self._apply_clob_input_sizes(cursor, query, params)
            if ignore_dup:
                try:
                    await cursor.execute(query, params)
                except Exception as e:
                    if "ORA-00001" in str(e):
                        return None
                    raise
                # INSERT succeeded but RETURNING was stripped
                return None

            try:
                await cursor.execute(query, params)
            except Exception as e:
                logger.error(
                    "Oracle fetchrow failed. Query: %s\nParams keys: %s\nError: %s",
                    query[:500],
                    list(params.keys()) if params else None,
                    e,
                )
                raise

            if ret_cols is not None:
                row_dict = self._read_returning_values(ret_cols, params)
                return ResultRow(row_dict) if row_dict else None

            columns = [col[0].lower() for col in cursor.description or []]
            row = await cursor.fetchone()
            if row is None:
                return None
            return ResultRow(dict(zip(columns, _convert_row_from_oracle(columns, row))))
        finally:
            cursor.close()

    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: float | None = None) -> Any:
        query, ignore_dup, ret_cols = _rewrite_pg_to_oracle(query)
        cursor = self._conn.cursor()
        try:
            params = self._make_bind_params(cursor, args, ret_cols)
            query, params = self._expand_any_lists(query, params)
            self._apply_clob_input_sizes(cursor, query, params)
            if ignore_dup:
                try:
                    await cursor.execute(query, params)
                except Exception as e:
                    if "ORA-00001" in str(e):
                        return None
                    raise
                # INSERT succeeded (no dup). RETURNING was stripped, so we can't
                # fetch rows. Return the first positional arg as a best-effort
                # stand-in (callers typically RETURNING the PK they just inserted).
                return args[0] if args else True

            await cursor.execute(query, params)

            if ret_cols is not None:
                row_dict = self._read_returning_values(ret_cols, params)
                if row_dict is None:
                    return None
                vals = list(row_dict.values())
                val = vals[column] if column < len(vals) else None
                # Try to convert RAW(16) hex back to UUID
                if isinstance(val, str) and len(val) == 32:
                    try:
                        return _uuid_mod.UUID(val)
                    except Exception:
                        pass
                return val

            row = await cursor.fetchone()
            if row is None:
                return None
            val = row[column]
            if isinstance(val, bytes) and len(val) == 16:
                try:
                    return _uuid_mod.UUID(bytes=val)
                except Exception:
                    return val
            return val
        finally:
            cursor.close()


# ---------------------------------------------------------------------------
# OracleBackend
# ---------------------------------------------------------------------------


class OracleBackend(DatabaseBackend):
    """DatabaseBackend implementation wrapping an oracledb async connection pool."""

    # -- Capability overrides (Oracle differs from the PG defaults) ------
    @property
    def backend_type(self) -> str:
        return "oracle"

    @property
    def supports_partial_indexes(self) -> bool:
        return False

    @property
    def supports_bm25(self) -> bool:
        return False

    @property
    def supports_unnest(self) -> bool:
        return False

    @property
    def supports_pg_trgm(self) -> bool:
        return False

    @property
    def supports_worker_poller(self) -> bool:
        return True

    def normalize_schema(self, schema: str | None) -> str | None:
        """Oracle uses users as schemas; ``"public"`` is PG-specific."""
        if schema == "public":
            return None
        return schema

    def run_migrations(self, dsn: str, *, schema: str | None = None) -> None:
        """Run Oracle DDL migrations."""
        from ...migrations_oracle import run_oracle_migrations

        run_oracle_migrations(dsn, schema=schema)

    def create_task_backend(self, *, pool_getter: Any = None, schema_getter: Any = None) -> Any:
        """Oracle now uses BrokerTaskBackend — worker/poller is backend-agnostic."""
        from ..task_backend import BrokerTaskBackend

        return BrokerTaskBackend(pool_getter=pool_getter, schema_getter=schema_getter)

    def __init__(self) -> None:
        self._pool: Any = None
        self._oracledb: Any = None

    async def initialize(
        self,
        dsn: str,
        *,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 300,
        acquire_timeout: float = 30,
        statement_cache_size: int = 0,
        init_callback: Any | None = None,
    ) -> None:
        oracledb = _import_oracledb()
        self._oracledb = oracledb

        # Parse URL-format DSN (oracle://user:pass@host:port/service)
        from urllib.parse import urlparse

        parsed = urlparse(dsn)
        pool_kwargs: dict[str, Any] = {"min": min_size, "max": max_size, "stmtcachesize": statement_cache_size}
        if parsed.scheme in ("oracle", "oracle+oracledb"):
            pool_kwargs["user"] = parsed.username
            pool_kwargs["password"] = parsed.password
            host = parsed.hostname or "localhost"
            port = parsed.port or 1521
            service = parsed.path.lstrip("/") if parsed.path else "FREEPDB1"
            pool_kwargs["dsn"] = f"{host}:{port}/{service}"
        else:
            pool_kwargs["dsn"] = dsn

        self._pool = oracledb.create_pool_async(**pool_kwargs)

        logger.info(f"Oracle pool created (min={min_size}, max={max_size})")

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close(force=True)
            self._pool = None
            logger.info("Oracle pool closed")

    async def _set_session_schema(self, conn: Any) -> None:
        """Set the session schema on an Oracle connection.

        Uses ALTER SESSION SET CURRENT_SCHEMA so that all unqualified table
        references resolve to the tenant's schema. This is Oracle's equivalent
        of PostgreSQL's SET search_path — fq_table() returns bare table names
        for Oracle, relying on this session-level setting for isolation.
        """
        # Lazy import to avoid circular dependency (memory_engine → db → memory_engine).
        from ..memory_engine import get_current_schema

        schema = get_current_schema()
        if schema and schema != "public":
            cursor = conn.cursor()
            await cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA = "{schema}"')
            await cursor.close()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[OracleConnection]:
        pool = self._ensure_pool()
        conn = await pool.acquire()
        try:
            await self._set_session_schema(conn)
            yield OracleConnection(conn)
            # Auto-commit on clean exit (asyncpg uses autocommit by default;
            # oracledb does not, so we must commit explicitly)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await pool.release(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[OracleConnection]:
        pool = self._ensure_pool()
        conn = await pool.acquire()
        try:
            await self._set_session_schema(conn)
            yield OracleConnection(conn)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await pool.release(conn)

    def get_pool(self) -> Any:
        return self._ensure_pool()

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("OracleBackend is not initialized. Call initialize() first.")
        return self._pool
