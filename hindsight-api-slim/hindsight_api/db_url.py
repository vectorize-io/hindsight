"""Database URL normalization.

Hindsight accepts SQLAlchemy-style URLs like ``postgresql+asyncpg://...?ssl=require``
for its async engine, but the same string cannot be handed directly to synchronous
SQLAlchemy (psycopg2) or to :func:`asyncpg.create_pool`, which both expect a
libpq-compatible URL (``postgresql://...?sslmode=require``).

:func:`to_libpq_url` performs that translation. It is idempotent and safe to
apply to URLs that are already libpq-compatible, to the ``pg0`` embedded-PG
marker, or to any non-PostgreSQL string (returned unchanged).
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_ASYNCPG_SCHEMES = ("postgresql+asyncpg", "postgres+asyncpg")
_POSTGRES_SCHEMES = ("postgresql", "postgres") + _ASYNCPG_SCHEMES


def to_libpq_url(url: str) -> str:
    """Normalize a PostgreSQL URL for libpq-style consumers.

    Accepts a SQLAlchemy URL (``postgresql+asyncpg://...``) or a plain libpq
    URL and returns a form suitable for:

    - :func:`sqlalchemy.create_engine` (sync / psycopg2)
    - :func:`asyncpg.create_pool`

    Transformations:

    - ``postgresql+asyncpg`` / ``postgres+asyncpg`` / ``postgres`` → ``postgresql``
    - Query param ``ssl=<mode>`` → ``sslmode=<mode>`` (SQLAlchemy's asyncpg
      dialect uses ``ssl=``; libpq uses ``sslmode=``)

    Any non-PostgreSQL input (e.g. the ``pg0`` embedded-PG marker, a sqlite
    URL, an empty string) is returned unchanged. Already-normalized URLs are
    returned unchanged.
    """
    if not url or "://" not in url:
        return url

    parts = urlsplit(url)
    if parts.scheme not in _POSTGRES_SCHEMES:
        return url

    new_scheme = "postgresql"

    new_query_pairs = [
        ("sslmode", v) if k == "ssl" else (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    new_query = urlencode(new_query_pairs)

    if new_scheme == parts.scheme and new_query == parts.query:
        return url

    return urlunsplit((new_scheme, parts.netloc, parts.path, new_query, parts.fragment))
