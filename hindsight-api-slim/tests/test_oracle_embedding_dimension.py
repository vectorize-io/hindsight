"""Embedding-dimension management on the Oracle backend.

The Oracle baseline declares ``VECTOR(384, FLOAT32)`` and, unlike PostgreSQL, nothing used to
reconcile it with the configured embeddings model: the startup path never ran the check and
``run-db-migration --embedding-dimension`` reached PostgreSQL-only SQL. These unit tests drive
``_ensure_oracle_table_embedding_dimension`` with a scripted cursor; the ``oracle``-marked tests
at the bottom exercise the same code against a live database.
"""

import array
import json
import re
import uuid

import pytest

from hindsight_api.migrations import _ORACLE_PENDING_INDEXES_MARKER, _ensure_oracle_table_embedding_dimension


class _Crash(Exception):
    """The process dying between two DDL statements."""


def _crashes_on(sql: str, step: str | None) -> bool:
    # The pending-index marker (a COMMENT) quotes the CREATE statements, so it is never the step.
    return bool(step) and step in sql and not sql.startswith("COMMENT ON")


class _ScriptedCursor:
    """Answers the dictionary/data queries the dimension check issues; records every statement."""

    def __init__(
        self,
        *,
        vector_info: str | None,
        stored_dimensions: list[int] | None = None,
        vector_indexes: list[tuple[str, str, str]] | None = None,  # (name, INDEX_SUBTYPE, PARTITIONED)
        vector_index_params: dict[str, tuple[str, int]] | None = None,  # name -> (DISTANCE, ACCURACY)
        has_legacy: bool = False,
        legacy_rows: int = 0,  # rows sitting in EMBEDDING_LEGACY (a write that raced the resize)
        comments: dict[str, str] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.vector_info = vector_info
        self.stored_dimensions = stored_dimensions or []
        self.vector_indexes = vector_indexes or []
        self.vector_index_params = vector_index_params or {}
        self.has_legacy = has_legacy
        self.legacy_rows = legacy_rows
        self.comments: dict[str, str] = dict(comments or {})  # column -> comment
        self.fail_on = fail_on  # raise when a statement contains this text (simulated crash)
        self.statements: list[str] = []
        self._result: list[tuple] = []

    def execute(self, sql: str, binds: dict | None = None) -> None:
        if _crashes_on(sql, self.fail_on):
            raise _Crash(sql)
        self.statements.append(sql)
        # Column DDL changes what the dictionary reports next, like the real catalog.
        if "RENAME COLUMN embedding TO embedding_legacy" in sql:
            self.vector_info, self.has_legacy = None, True
            if "EMBEDDING" in self.comments:  # Oracle keeps a column's comment across a rename
                self.comments["EMBEDDING_LEGACY"] = self.comments.pop("EMBEDDING")
        elif added := re.search(r"ADD \(embedding VECTOR\((\d+), FLOAT32\)\)", sql):
            self.vector_info = f"VECTOR({added.group(1)},FLOAT32,DENSE)"
        elif "DROP COLUMN embedding_legacy" in sql:
            self.has_legacy = False
            self.comments.pop("EMBEDDING_LEGACY", None)
        elif comment := re.match(r"COMMENT ON COLUMN \w+\.(\w+) IS '(.*)'$", sql, re.DOTALL):
            self.comments[comment.group(1).upper()] = comment.group(2).replace("''", "'")
        elif dropped := re.match(r'DROP INDEX "(\w+)"', sql):
            self.vector_indexes = [i for i in self.vector_indexes if i[0] != dropped.group(1)]
        elif created := re.match(r'CREATE VECTOR INDEX "(\w+)" .*ORGANIZATION (INMEMORY )?', sql):
            if any(i[0] == created.group(1) for i in self.vector_indexes):
                raise Exception("ORA-00955: name is already used by an existing object")
            subtype = "INMEMORY_NEIGHBOR_GRAPH_HNSW" if created.group(2) else "NEIGHBOR_PARTITIONS_IVF"
            self.vector_indexes.append((created.group(1), subtype, "YES" if sql.endswith(" LOCAL") else "NO"))
        if "all_col_comments" in sql:
            wants_one = "column_name = :column_name" in sql
            if wants_one and not (binds or {}).get("column_name"):
                raise AssertionError(f"single-column comment read missing :column_name bind: {sql}")
            wanted = (binds or {}).get("column_name")
            values = [self.comments.get(str(wanted).upper())] if wants_one else self.comments.values()
            self._result = [(c,) for c in values if c is not None]
        elif "vector_info" in sql.lower():
            self._result = ([("EMBEDDING", self.vector_info)] if self.vector_info is not None else []) + (
                [("EMBEDDING_LEGACY", None)] if self.has_legacy else []
            )
        elif "embedding_legacy IS NOT NULL" in sql:
            self._result = [(1,)] * min(self.legacy_rows, 1)
        elif "VECTOR_DIMENSION_COUNT" in sql:
            # With :dim bound the query looks for a row of any OTHER dimension; without it, any row.
            wanted = (binds or {}).get("dim")
            matches = [d for d in self.stored_dimensions if wanted is None or d != wanted]
            self._result = [(d,) for d in matches[:1]]
        elif "index_type = 'VECTOR'" in sql:
            self._result = list(self.vector_indexes)
        elif "all_vector_indexes" in sql or "v$vector_index" in sql or "vecsys.vector$index" in sql:
            params = self.vector_index_params.get((binds or {}).get("index_name", ""))
            self._result = [params] if params else []
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


def _ddl(cursor: _ScriptedCursor) -> list[str]:
    return [s for s in cursor.statements if re.match(r"\s*(ALTER TABLE|DROP INDEX|CREATE)", s)]


def test_matching_fixed_dimension_changes_nothing():
    cursor = _ScriptedCursor(vector_info="VECTOR(1536,FLOAT32,DENSE)", stored_dimensions=[1536])
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


@pytest.mark.parametrize("required", [384, 768, 1536, 3072])
def test_empty_table_is_resized_and_its_vector_index_rebuilt(required):
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)" if required != 384 else "VECTOR(1024,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF", "NO")],
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", required)
    # MODIFY is not an option: Oracle rejects any VECTOR dimension change with ORA-51859.
    assert _ddl(cursor) == [
        'DROP INDEX "IDX_MU_EMBEDDING_HNSW"',
        "ALTER TABLE MEMORY_UNITS RENAME COLUMN embedding TO embedding_legacy",
        f"ALTER TABLE MEMORY_UNITS ADD (embedding VECTOR({required}, FLOAT32))",
        "ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy",
        'CREATE VECTOR INDEX "IDX_MU_EMBEDDING_HNSW" ON MEMORY_UNITS (embedding) '
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95",
    ]


def test_resize_keeps_a_local_hnsw_index_local():
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("MU_HNSW", "INMEMORY_NEIGHBOR_GRAPH_HNSW", "YES")],
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor)[-1] == (
        'CREATE VECTOR INDEX "MU_HNSW" ON MEMORY_UNITS (embedding) '
        "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95 LOCAL"
    )


def test_resize_keeps_a_custom_index_distance_and_accuracy():
    """A non-baseline DISTANCE/TARGET ACCURACY must survive the rebuild, not revert to COSINE/95."""
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EUCLIDEAN", "NEIGHBOR_PARTITIONS_IVF", "NO")],
        vector_index_params={"IDX_MU_EUCLIDEAN": ("EUCLIDEAN", 80)},
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor)[-1] == (
        'CREATE VECTOR INDEX "IDX_MU_EUCLIDEAN" ON MEMORY_UNITS (embedding) '
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE EUCLIDEAN WITH TARGET ACCURACY 80"
    )


def test_resize_defaults_when_the_index_params_catalog_is_unreadable():
    """ALL_VECTOR_INDEXES/V$VECTOR_INDEX/VECSYS.VECTOR$INDEX may all be unreadable or empty."""
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF", "NO")],
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert "DISTANCE COSINE WITH TARGET ACCURACY 95" in _ddl(cursor)[-1]


def test_resize_locks_the_table_around_the_emptiness_check():
    """A writer must not slip an insert between the empty check and the column rename."""
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)")
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert "LOCK TABLE MEMORY_UNITS IN EXCLUSIVE MODE WAIT 30" in cursor.statements


def test_resize_aborts_instead_of_dropping_a_row_that_raced_in():
    """A write landing mid-resize ends up in embedding_legacy; dropping it would lose the row."""
    cursor = _ScriptedCursor(vector_info="VECTOR(1536,FLOAT32,DENSE)", has_legacy=True, legacy_rows=1)
    with pytest.raises(RuntimeError, match="embedding_legacy"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert not any("DROP COLUMN embedding_legacy" in s for s in cursor.statements)


def test_resize_restores_a_prior_column_comment():
    """The pending-index marker borrows the column comment; the operator's own text must survive."""
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU", "NEIGHBOR_PARTITIONS_IVF", "NO")],
        comments={"EMBEDDING": "operator note"},
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert cursor.comments["EMBEDDING"] == "operator note"


def test_resize_without_indexes_still_restores_a_prior_column_comment():
    """The rename carries the comment onto the soon-dropped legacy column even with no marker."""
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)", comments={"EMBEDDING": "operator note"})
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert cursor.comments["EMBEDDING"] == "operator note"


def test_resume_restores_the_prior_comment_the_marker_carried():
    """A crash after the marker overwrote the comment loses the local variable; the marker holds it."""
    marker = _ORACLE_PENDING_INDEXES_MARKER + json.dumps(
        {
            "ddl": [
                'CREATE VECTOR INDEX "IDX_MU_EMBEDDING_HNSW" ON MEMORY_UNITS (embedding) '
                "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95"
            ],
            "comment": "operator note",
        }
    )
    cursor = _ScriptedCursor(
        vector_info="VECTOR(1536,FLOAT32,DENSE)",
        comments={"EMBEDDING": marker},
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert [i[0] for i in cursor.vector_indexes] == ["IDX_MU_EMBEDDING_HNSW"]
    assert cursor.comments["EMBEDDING"] == "operator note"


def test_a_comment_too_long_for_the_marker_still_survives_the_resize():
    """Marker + comment would exceed Oracle's 4000-byte comment cap, so only the DDL rides in it."""
    note = "x" * 3900
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU", "NEIGHBOR_PARTITIONS_IVF", "NO")],
        comments={"EMBEDDING": note},
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert cursor.comments["EMBEDDING"] == note
    markers = [s for s in cursor.statements if _ORACLE_PENDING_INDEXES_MARKER in s]
    assert markers and all(note not in s for s in markers)


def test_resize_refuses_an_index_it_cannot_rebuild():
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)", vector_indexes=[("X", "SOMETHING_NEW", "NO")])
    with pytest.raises(RuntimeError, match="unknown subtype 'SOMETHING_NEW'"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_resize_interrupted_after_the_rename_is_finished():
    cursor = _ScriptedCursor(vector_info=None, has_legacy=True)
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == [
        "ALTER TABLE MEMORY_UNITS ADD (embedding VECTOR(1536, FLOAT32))",
        "ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy",
    ]


def test_resize_interrupted_after_the_new_column_is_finished():
    cursor = _ScriptedCursor(vector_info="VECTOR(1536,FLOAT32,DENSE)", has_legacy=True)
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == ["ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy"]


@pytest.mark.parametrize(
    "crash_at",
    [
        "DROP INDEX",
        "RENAME COLUMN",
        "ADD (embedding",
        "DROP COLUMN embedding_legacy",
        "CREATE VECTOR INDEX",
    ],
)
def test_resize_interrupted_at_any_step_still_ends_with_the_vector_index(crash_at):
    """Oracle DDL is not transactional; a crash between steps must not lose the baseline index."""
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF", "NO")],
        fail_on=crash_at,
    )
    with pytest.raises(_Crash):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)

    cursor.fail_on = None  # the next boot
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)

    assert cursor.vector_info == "VECTOR(1536,FLOAT32,DENSE)"
    assert not cursor.has_legacy
    assert [i[:2] for i in cursor.vector_indexes] == [("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF")]
    assert not any(cursor.comments.values()), "the pending-index marker is cleared once rebuilt"


def test_a_worker_that_lost_the_race_to_create_the_index_finishes_cleanly():
    """Another worker rebuilt the index between our DROP and CREATE: ORA-00955 is not an error."""
    cursor = _ScriptedCursor(
        vector_info="VECTOR(1536,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF", "NO")],
        comments={
            "EMBEDDING": 'hindsight:pending-vector-indexes:["CREATE VECTOR INDEX \\"IDX_MU_EMBEDDING_HNSW\\" '
            'ON MEMORY_UNITS (embedding) ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95"]'
        },
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert len(cursor.vector_indexes) == 1
    assert not any(cursor.comments.values())


def test_table_with_embeddings_of_another_dimension_fails_explicitly():
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)", stored_dimensions=[384])
    with pytest.raises(RuntimeError, match=r"from 384 to 1536.*MEMORY_UNITS"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_flexible_column_holding_only_the_model_dimension_is_accepted():
    """A VECTOR(*, *) column (as found on a live Autonomous deployment) is left as is."""
    cursor = _ScriptedCursor(vector_info="VECTOR(*,*,DENSE)", stored_dimensions=[1536])
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_flexible_column_holding_another_dimension_fails_explicitly():
    cursor = _ScriptedCursor(vector_info="VECTOR(*,*,DENSE)", stored_dimensions=[384])
    with pytest.raises(RuntimeError, match=r"MEMORY_UNITS.*384.*1536"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_missing_table_is_skipped():
    cursor = _ScriptedCursor(vector_info=None)
    _ensure_oracle_table_embedding_dimension(cursor, "MENTAL_MODELS", 1536)
    assert _ddl(cursor) == []


@pytest.fixture()
def oracle_cursor(_oracle_admin_dsn):
    """A live cursor on the test schema plus a scratch table name, dropped afterwards."""
    oracledb = pytest.importorskip("oracledb")
    conn = oracledb.connect(**_oracle_admin_dsn)
    conn.autocommit = True
    cursor = conn.cursor()
    table = f"HS_DIM_{uuid.uuid4().hex[:8].upper()}"
    try:
        yield cursor, table
    finally:
        try:
            cursor.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        conn.close()


def _vector_info(cursor, table: str) -> str:
    cursor.execute(
        "SELECT vector_info FROM user_tab_columns WHERE table_name = :t AND column_name = 'EMBEDDING'", {"t": table}
    )
    return cursor.fetchone()[0]


def _vector_indexes(cursor, table: str) -> list[str]:
    cursor.execute(
        "SELECT index_name FROM user_indexes WHERE table_name = :t AND index_type = 'VECTOR' ORDER BY 1", {"t": table}
    )
    return [r[0] for r in cursor.fetchall()]


def _vector_index_params(cursor, table: str) -> dict[str, tuple[str, int]]:
    """DISTANCE_METRIC/ACCURACY of the table's vector indexes, as ALL_VECTOR_INDEXES reports them."""
    try:
        cursor.execute(
            "SELECT index_name, distance_metric, accuracy FROM all_vector_indexes "
            "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AND target_table = :t",
            {"t": table},
        )
    except Exception:
        return {}
    return {name: (metric, acc) for name, metric, acc in cursor.fetchall()}


@pytest.mark.oracle
def test_live_resize_replaces_the_column_and_keeps_the_vector_index(oracle_cursor):
    cursor, table = oracle_cursor
    # Same shape as the baseline: VECTOR(384, FLOAT32) with an IVF (neighbor partitions) index.
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, text CLOB, embedding VECTOR(384, FLOAT32))")
    cursor.execute(
        f"CREATE VECTOR INDEX {table}_IVF ON {table} (embedding) ORGANIZATION NEIGHBOR PARTITIONS "
        "DISTANCE COSINE WITH TARGET ACCURACY 95"
    )

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)

    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"
    assert _vector_indexes(cursor, table) == [f"{table}_IVF"]
    cursor.execute(
        "SELECT comments FROM user_col_comments WHERE table_name = :t AND comments IS NOT NULL", {"t": table}
    )
    assert cursor.fetchall() == [], "the pending-index marker is cleared once the index is rebuilt"

    # Idempotent: a second run with the same model changes nothing.
    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)
    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"

    # Once embeddings are stored, a model with another dimension is refused, not mixed in.
    cursor.execute(f"INSERT INTO {table} VALUES (1, 'x', :v)", {"v": array.array("f", [0.1] * 1536)})
    with pytest.raises(RuntimeError, match="from 1536 to 768"):
        _ensure_oracle_table_embedding_dimension(cursor, table, 768)
    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"


@pytest.mark.oracle
def test_live_resize_preserves_a_custom_index_distance_and_accuracy(oracle_cursor):
    """The rebuilt index must keep the original DISTANCE/TARGET ACCURACY, not revert to COSINE/95."""
    cursor, table = oracle_cursor
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, embedding VECTOR(384, FLOAT32))")
    cursor.execute(
        f"CREATE VECTOR INDEX {table}_IVF ON {table} (embedding) ORGANIZATION NEIGHBOR PARTITIONS "
        "DISTANCE EUCLIDEAN WITH TARGET ACCURACY 80"
    )
    if _vector_index_params(cursor, table).get(f"{table}_IVF") is None:
        pytest.skip("no vector-index catalog is readable from this account")

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)

    assert _vector_index_params(cursor, table).get(f"{table}_IVF") == ("EUCLIDEAN", 80)


class _CrashingCursor:
    """Wraps a live cursor and dies once, just before the first statement containing ``crash_at``."""

    def __init__(self, cursor, crash_at: str | None) -> None:
        self._cursor, self._crash_at = cursor, crash_at

    def execute(self, sql: str, binds: dict | None = None) -> None:
        if _crashes_on(sql, self._crash_at):
            self._crash_at = None
            raise _Crash(sql)
        if binds is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql, binds)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


@pytest.mark.oracle
@pytest.mark.parametrize("crash_at", ["ADD (embedding", "CREATE VECTOR INDEX"])
def test_live_resize_interrupted_mid_way_rebuilds_the_vector_index(oracle_cursor, crash_at):
    """Crashing after the rename only works out if Oracle keeps the column comment across RENAME COLUMN."""
    cursor, table = oracle_cursor
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, embedding VECTOR(384, FLOAT32))")
    cursor.execute(
        f"CREATE VECTOR INDEX {table}_IVF ON {table} (embedding) ORGANIZATION NEIGHBOR PARTITIONS "
        "DISTANCE COSINE WITH TARGET ACCURACY 95"
    )
    crashing = _CrashingCursor(cursor, crash_at)
    with pytest.raises(_Crash):
        _ensure_oracle_table_embedding_dimension(crashing, table, 1536)
    assert _vector_indexes(cursor, table) == []

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)

    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"
    assert _vector_indexes(cursor, table) == [f"{table}_IVF"]
    cursor.execute(
        "SELECT comments FROM user_col_comments WHERE table_name = :t AND comments IS NOT NULL", {"t": table}
    )
    assert cursor.fetchall() == []


@pytest.mark.oracle
def test_live_flexible_column_is_validated_not_altered(oracle_cursor):
    cursor, table = oracle_cursor
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, embedding VECTOR)")
    cursor.execute(f"INSERT INTO {table} VALUES (1, :v)", {"v": array.array("f", [0.1] * 1536)})

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)
    assert _vector_info(cursor, table) == "VECTOR(*,*,DENSE)"

    with pytest.raises(RuntimeError, match="flexible VECTOR column holding 1536"):
        _ensure_oracle_table_embedding_dimension(cursor, table, 384)


class _FakeOracleDriver:
    """Stands in for python-oracledb: a connection whose cursor accepts the session setup."""

    class DatabaseError(Exception):
        pass

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return _ScriptedCursor(vector_info=None)

    def connect(self, **_):
        return self._Connection()


def _patch_reconcile(monkeypatch, outcomes: list[Exception | None]) -> list[str]:
    """Replace the per-table reconcile with one that raises (or not) per call, in order."""
    from hindsight_api import migrations
    from hindsight_api.engine.db import oracle

    driver = _FakeOracleDriver()
    monkeypatch.setattr(oracle, "_import_oracledb", lambda: driver)
    monkeypatch.setattr(oracle, "_oracle_connect_params", lambda url: {})
    calls: list[str] = []

    def reconcile(cursor, table_name, required_dimension):
        calls.append(table_name)
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(migrations, "_ensure_oracle_table_embedding_dimension", reconcile)
    return calls


def test_a_worker_that_raced_another_ddl_re_checks_and_converges(monkeypatch):
    from hindsight_api import migrations

    race = _FakeOracleDriver.DatabaseError("ORA-01430: column being added already exists in table")
    calls = _patch_reconcile(monkeypatch, [race, None, None])
    migrations._ensure_embedding_dimension_oracle("oracle://x", 1536, None, store_owned_memories=False)
    assert calls == ["MEMORY_UNITS", "MEMORY_UNITS", "MENTAL_MODELS"]


def test_reconcile_gives_up_after_repeated_ddl_failures_and_never_retries_a_data_mismatch(monkeypatch):
    from hindsight_api import migrations

    error = _FakeOracleDriver.DatabaseError("ORA-00054: resource busy")
    _patch_reconcile(monkeypatch, [error, error, error])
    with pytest.raises(_FakeOracleDriver.DatabaseError):
        migrations._ensure_embedding_dimension_oracle("oracle://x", 1536, None, store_owned_memories=False)

    calls = _patch_reconcile(monkeypatch, [RuntimeError("Cannot change embedding dimension")])
    with pytest.raises(RuntimeError):
        migrations._ensure_embedding_dimension_oracle("oracle://x", 1536, None, store_owned_memories=False)
    assert calls == ["MEMORY_UNITS"]


def test_admin_migration_of_an_oracle_url_skips_the_postgresql_steps(monkeypatch):
    """`hindsight-admin run-db-migration --embedding-dimension N` on Oracle.

    It used to go through the PostgreSQL per-schema unit (libpq URL, information_schema,
    pgvector extension checks) and fail; it now runs Alembic and the Oracle dimension step,
    as the migration user, and maps PG's "public" default to the connecting user's schema.
    """
    from hindsight_api import migrations

    calls: list[str] = []
    monkeypatch.setattr(migrations, "_should_isolate_migrations", lambda: False)
    monkeypatch.setattr(
        migrations,
        "run_migrations",
        lambda url, *, schema=None, migration_database_url=None, **_: calls.append(
            f"migrate {url} schema={schema} as={migration_database_url}"
        ),
    )
    monkeypatch.setattr(
        migrations,
        "ensure_embedding_dimension",
        lambda url, dim, *, schema=None, store_owned_memories=False, **_: calls.append(
            f"dimension {url} {dim} schema={schema}"
        ),
    )
    monkeypatch.setattr(migrations, "_migrate_one_schema_pg", lambda *a, **k: calls.append("postgresql path"))

    migrations.run_migrations_for_schemas(
        "oracle+oracledb://app@/?dsn=x",
        ["public", "TENANT_B"],
        migration_database_url="oracle+oracledb://owner@/?dsn=x",
        embedding_dimension=1536,
    )

    assert calls == [
        "migrate oracle+oracledb://app@/?dsn=x schema=None as=oracle+oracledb://owner@/?dsn=x",
        "dimension oracle+oracledb://owner@/?dsn=x 1536 schema=None",
        "migrate oracle+oracledb://app@/?dsn=x schema=TENANT_B as=oracle+oracledb://owner@/?dsn=x",
        "dimension oracle+oracledb://owner@/?dsn=x 1536 schema=TENANT_B",
    ]
