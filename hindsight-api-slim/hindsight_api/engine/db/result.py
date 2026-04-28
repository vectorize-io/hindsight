"""Uniform row wrapper over heterogeneous database drivers.

ResultRow provides dict-like access to database rows regardless of whether
the underlying driver returns asyncpg.Record, oracledb rows, or plain dicts.
"""

from typing import Any


class ResultRow:
    """Dict-like wrapper over database rows.

    Supports both key-based access (row["col"]) and attribute access (row.col).
    Wraps asyncpg.Record, oracledb named-tuple rows, or plain dicts.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:
        """Wrap a row from any database driver.

        Args:
            data: The raw row object (asyncpg.Record, dict, named tuple, etc.)
        """
        object.__setattr__(self, "_data", data)

    # -- dict-like access ------------------------------------------------

    def __getitem__(self, key: str | int) -> Any:
        """Get a value by column name or index."""
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return data[key]
        return data[key]

    def __getattr__(self, key: str) -> Any:
        """Get a value by attribute name (for convenience)."""
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            try:
                return data[key]
            except KeyError:
                raise AttributeError(key) from None
        # asyncpg.Record and named tuples support key-based access
        try:
            return data[key]
        except (KeyError, TypeError):
            raise AttributeError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default (like dict.get)."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self) -> list[str]:
        """Return column names."""
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.keys())
        # asyncpg.Record has .keys()
        if hasattr(data, "keys"):
            return list(data.keys())
        return []

    def values(self) -> list[Any]:
        """Return column values."""
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.values())
        if hasattr(data, "values"):
            return list(data.values())
        return []

    def items(self) -> list[tuple[str, Any]]:
        """Return (key, value) pairs."""
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.items())
        if hasattr(data, "items"):
            return list(data.items())
        return list(zip(self.keys(), self.values()))

    # -- representation --------------------------------------------------

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"ResultRow({data!r})"

    def __contains__(self, key: str) -> bool:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return key in data
        if hasattr(data, "keys"):
            return key in data.keys()
        return False

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    def __bool__(self) -> bool:
        return True
