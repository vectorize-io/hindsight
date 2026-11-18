"""Oracle: every foreign key has an index that leads with its columns (#6434).

Oracle treats a foreign key as indexed only when an index's leading columns are exactly
the key's columns (in any order). An unindexed one makes every delete of a parent row
take a table-level share lock on the child for the statement, so retains and deletes in
*different* banks block each other and deadlock (ORA-00060, an HTTP 500). Migration
d3f8a1c6e9b2 indexed the keys that had the gap; this keeps a new table from adding one.
"""

from urllib.parse import urlparse

import pytest

_UNINDEXED_FOREIGN_KEYS = """
SELECT c.table_name, c.constraint_name,
       LISTAGG(cc.column_name, ',') WITHIN GROUP (ORDER BY cc.position) AS cols
FROM user_constraints c
JOIN user_cons_columns cc ON cc.constraint_name = c.constraint_name
WHERE c.constraint_type = 'R'
  AND c.table_name NOT LIKE 'BIN$%'
  AND NOT EXISTS (
    SELECT 1 FROM user_indexes i
    WHERE i.table_name = c.table_name
      AND (SELECT COUNT(*) FROM user_cons_columns k WHERE k.constraint_name = c.constraint_name)
        = (SELECT COUNT(*) FROM user_ind_columns ic
           JOIN user_cons_columns k ON k.constraint_name = c.constraint_name AND k.column_name = ic.column_name
           WHERE ic.index_name = i.index_name
             AND ic.column_position <= (SELECT COUNT(*) FROM user_cons_columns k2
                                        WHERE k2.constraint_name = c.constraint_name))
  )
GROUP BY c.table_name, c.constraint_name
ORDER BY c.table_name, c.constraint_name
"""


@pytest.mark.oracle
def test_every_foreign_key_has_a_leading_index(oracle_db_url: str) -> None:
    import oracledb

    url = urlparse(oracle_db_url)
    with oracledb.connect(user=url.username, password=url.password, dsn=f"{url.hostname}:{url.port}{url.path}") as conn:
        rows = conn.cursor().execute(_UNINDEXED_FOREIGN_KEYS).fetchall()

    assert rows == [], f"foreign keys with no index leading with their columns: {rows}"
