"""Oracle semantic recall arm against a live database: exactness, index usage and bank isolation.

Builds a scratch table shaped like ``memory_units`` (LIST AUTOMATIC partitions on ``bank_id``),
indexes its embeddings, and runs the SQL ``OracleDialect.build_semantic_arm`` produces. With the
default ``exact`` mode the plan must not read the vector index and the rows must be the true top-k
— on Autonomous Database a bare ``FETCH FIRST`` is answered from the index whenever one exists.
With ``approx`` the plan must read the index. No row of another bank may come back in either mode.
"""

import array
import random
import uuid

import numpy as np
import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.sql.oracle import OracleDialect

pytestmark = pytest.mark.oracle

DIMENSION = 64
TOP_K = 20
IVF_GLOBAL = "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95"
HNSW_LOCAL = "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95 LOCAL"


class _IndexedTable:
    def __init__(self, cursor, name: str, vectors: dict[int, list[float]], query: list[float]) -> None:
        self.cursor = cursor
        self.name = name
        self.vectors = vectors
        self.query = query

    def true_top_k(self, bank_parity: int) -> set[int]:
        ids = [i for i in self.vectors if i % 2 == bank_parity]
        matrix = np.array([self.vectors[i] for i in ids], dtype=np.float32)
        query = np.array(self.query, dtype=np.float32)
        similarity = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        return {ids[j] for j in np.argsort(-similarity)[:TOP_K]}


def _indexed_table(oracle_admin_dsn, index_clause: str):
    oracledb = pytest.importorskip("oracledb")
    conn = oracledb.connect(**oracle_admin_dsn)
    conn.autocommit = True
    cursor = conn.cursor()
    name = f"HS_VS_{uuid.uuid4().hex[:8].upper()}"
    cursor.execute(
        f"CREATE TABLE {name} (id NUMBER PRIMARY KEY, bank_id VARCHAR2(256) NOT NULL, "
        f"fact_type VARCHAR2(64) NOT NULL, embedding VECTOR({DIMENSION}, FLOAT32)) "
        "PARTITION BY LIST (bank_id) AUTOMATIC (PARTITION p_default VALUES ('__default__'))"
    )
    rng = random.Random(42)
    vectors = {i: [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)] for i in range(1, 2001)}
    cursor.executemany(
        f"INSERT INTO {name} VALUES (:1, :2, 'world', :3)",
        [(i, f"bank-{i % 2}", array.array("f", v)) for i, v in vectors.items()],
    )
    cursor.execute(f"CREATE VECTOR INDEX {name}_VI ON {name} (embedding) {index_clause}")
    query = [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]
    return conn, _IndexedTable(cursor, name, vectors, query)


@pytest.fixture(params=[pytest.param(IVF_GLOBAL, id="ivf-global"), pytest.param(HNSW_LOCAL, id="hnsw-local")])
def indexed_table(request, _oracle_admin_dsn):
    conn, table = _indexed_table(_oracle_admin_dsn, request.param)
    try:
        yield table
    finally:
        table.cursor.execute(f"DROP TABLE {table.name} PURGE")
        conn.close()


@pytest.fixture()
def ivf_table(_oracle_admin_dsn):
    conn, table = _indexed_table(_oracle_admin_dsn, IVF_GLOBAL)
    try:
        yield table
    finally:
        table.cursor.execute(f"DROP TABLE {table.name} PURGE")
        conn.close()


def _arm(table: str, mode: str, monkeypatch) -> str:
    monkeypatch.setenv("HINDSIGHT_API_ORACLE_VECTOR_SEARCH", mode)
    clear_config_cache()
    try:
        return OracleDialect().build_semantic_arm(
            table=table,
            cols="id, bank_id",
            fact_type="world",
            embedding_param=":1",
            bank_id_param=":2",
            fetch_limit=TOP_K,
            min_similarity=-1.0,
        )
    finally:
        clear_config_cache()


def _reads_vector_index(cursor, sql: str, binds: dict) -> bool:
    """HNSW shows as a VECTOR INDEX operation; IVF as scans of its VECTOR$... centroid tables."""
    cursor.execute("EXPLAIN PLAN FOR " + sql, binds)
    cursor.execute("SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY(NULL, NULL, 'BASIC'))")
    plan = "\n".join(row[0] for row in cursor.fetchall())
    return "VECTOR INDEX" in plan or "VECTOR$" in plan


def _run(table: _IndexedTable, sql: str) -> list:
    # Named binds, as OracleConnection sends them: the arm repeats :1 three times.
    binds = {"1": array.array("f", table.query), "2": "bank-0"}
    table.cursor.execute(sql, binds)
    return table.cursor.fetchall()


def test_exact_mode_ignores_the_vector_index_and_returns_the_true_top_k(indexed_table, monkeypatch):
    sql = _arm(indexed_table.name, "exact", monkeypatch)
    binds = {"1": array.array("f", indexed_table.query), "2": "bank-0"}

    assert not _reads_vector_index(indexed_table.cursor, sql, binds)
    rows = _run(indexed_table, sql)
    assert {bank for _id, bank, *_ in rows} == {"bank-0"}
    assert {row[0] for row in rows} == indexed_table.true_top_k(bank_parity=0)


def test_approx_mode_reads_the_vector_index(ivf_table, monkeypatch):
    # Global IVF only. On a partitioned table with a LOCAL HNSW index the optimizer plans
    # TABLE ACCESS FULL for FETCH APPROX on Oracle Free 23.26.3 even with the index populated,
    # so the plan assertion cannot hold there; bank isolation for that shape is checked below.
    indexed_table = ivf_table
    sql = _arm(indexed_table.name, "approx", monkeypatch)
    binds = {"1": array.array("f", indexed_table.query), "2": "bank-0"}

    assert _reads_vector_index(indexed_table.cursor, sql, binds)
    rows = _run(indexed_table, sql)
    assert rows
    assert {bank for _id, bank, *_ in rows} == {"bank-0"}


def test_approx_mode_stays_inside_the_bank_for_every_index_shape(indexed_table, monkeypatch):
    # No plan assertion: whether the optimizer reads the index is shape- and edition-dependent,
    # but approximate results must never cross banks regardless.
    sql = _arm(indexed_table.name, "approx", monkeypatch)
    rows = _run(indexed_table, sql)
    assert rows
    assert {bank for _id, bank, *_ in rows} == {"bank-0"}
