"""Uniform row interface over heterogeneous database drivers.

ResultRow is a Protocol that describes the dict-like access pattern all
database rows must support.  asyncpg.Record already satisfies this protocol
natively (key-based access, .keys(), .values(), etc.) so the PostgreSQL
backend returns raw Records — zero wrapping overhead.

Only backends whose native row type does NOT satisfy the protocol (e.g.,
Oracle named-tuple rows) need the concrete DictResultRow wrapper.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResultRow(Protocol):
    """Dict-like row interface returned by all database operations."""

    def __getitem__(self, key: str | int) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def keys(self) -> Any: ...
    def values(self) -> Any: ...
    def items(self) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...


class DictResultRow:
    """Concrete wrapper for backends whose native rows lack dict-like access.

    Used by Oracle (named-tuple rows, plain dicts) and tests.
    asyncpg.Record satisfies ResultRow natively — do NOT wrap it.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key: str | int) -> Any:
        data = object.__getattribute__(self, "_data")
        return data[key]

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            try:
                return data[key]
            except KeyError:
                raise AttributeError(key) from None
        try:
            return data[key]
        except (KeyError, TypeError):
            raise AttributeError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self) -> list[str]:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.keys())
        if hasattr(data, "keys"):
            return list(data.keys())
        return []

    def values(self) -> list[Any]:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.values())
        if hasattr(data, "values"):
            return list(data.values())
        return []

    def items(self) -> list[tuple[str, Any]]:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return list(data.items())
        if hasattr(data, "items"):
            return list(data.items())
        return list(zip(self.keys(), self.values()))

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"DictResultRow({data!r})"

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
        data = object.__getattribute__(self, "_data")
        return bool(data)
