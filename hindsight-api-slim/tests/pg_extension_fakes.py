"""A fake PostgreSQL connection for the extension-installation helpers.

Records every statement, answers the two catalog queries the helpers issue
(``current_setting('search_path')`` and the ``pg_extension`` lookup), and can be
told to fail a specific statement.
"""

import re


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value

    def fetchone(self):
        return self._value


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakePgConnection:
    """Minimal stand-in for a SQLAlchemy ``Connection`` in extension tests.

    Args:
        search_path: what ``current_setting('search_path')`` returns.
        extensions: installed extensions as ``{name: (schema, relocatable)}``.
        fail_on: substring of a statement that should raise instead of running.
    """

    def __init__(
        self,
        search_path: str = '"$user", public',
        extensions: dict[str, tuple[str, bool]] | None = None,
        fail_on: str | None = None,
    ):
        self.search_path = search_path
        self.extensions = dict(extensions or {})
        self.fail_on = fail_on
        self.statements: list[str] = []
        self.params: list[dict] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement, params=None, *args, **kwargs):
        sql = str(statement)
        self.statements.append(sql)
        self.params.append(params or {})
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError(f"simulated failure: {sql}")
        if "current_setting('search_path')" in sql:
            return _Result(self.search_path)
        if "set_config('search_path'" in sql:
            self.search_path = (params or {}).get("schema") or (params or {}).get("previous") or ""
            return _Result(self.search_path)
        if "= ANY(:names)" in sql:
            wanted = (params or {}).get("names") or []
            return _Rows(
                [
                    (name,)
                    for name, (schema, relocatable) in self.extensions.items()
                    if name in wanted and schema != "public" and relocatable
                ]
            )
        if "FROM pg_extension" in sql:
            name = (params or {}).get("name")
            return _Result(self.extensions.get(name))
        match = re.match(r"CREATE EXTENSION IF NOT EXISTS (\w+)", sql)
        if match:
            name = match.group(1)
            # PostgreSQL installs into the first schema on the search_path.
            self.extensions.setdefault(name, (self.search_path.split(",")[0].strip().strip('"'), True))
            return _Result(None)
        match = re.match(r'ALTER EXTENSION (\w+) SET SCHEMA "([^"]+)"', sql)
        if match:
            name, schema = match.group(1), match.group(2)
            self.extensions[name] = (schema, self.extensions[name][1])
            return _Result(None)
        return _Result(None)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def created_extensions(self) -> list[str]:
        """Extension names this connection was asked to create, in order."""
        return [
            m.group(1) for m in (re.match(r"CREATE EXTENSION IF NOT EXISTS (\w+)", s) for s in self.statements) if m
        ]
