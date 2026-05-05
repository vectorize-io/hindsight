from hindsight_api._vector_index import (
    bootstrap_extension,
    index_type_keyword,
    index_using_clause,
    pg_extension_name,
    validate_extension,
)


class RecordingConn:
    def __init__(self):
        self.statements = []

    def execute(self, statement, *args, **kwargs):
        self.statements.append(str(statement))


def test_validate_extension_accepts_scann():
    assert validate_extension("scann") == "scann"
    assert validate_extension("ScaNN") == "scann"


def test_pg_extension_name_maps_scann_to_alloydb_extension():
    assert pg_extension_name("scann") == "alloydb_scann"


def test_index_using_clause_scann_uses_cosine_auto_mode():
    clause = index_using_clause("scann")

    assert "USING scann (embedding cosine)" in clause
    assert "mode = 'AUTO'" in clause


def test_index_using_clause_pgvector_matches_existing_clause():
    assert index_using_clause("pgvector") == "USING hnsw (embedding vector_cosine_ops)"


def test_index_type_keyword_scann_round_trips_pg_indexes_indexdef():
    keyword = index_type_keyword("scann")
    indexdef = "CREATE INDEX idx ON memory_units USING scann (embedding cosine) WITH (mode='AUTO')"

    assert keyword == "scann"
    assert keyword in indexdef.lower()


def test_bootstrap_extension_scann_installs_vector_before_alloydb_scann():
    conn = RecordingConn()

    bootstrap_extension(conn, "scann")

    assert conn.statements == [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS alloydb_scann CASCADE",
    ]
