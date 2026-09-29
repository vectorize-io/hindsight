"""Text-search SQL shapes shared by index DDL and the queries that must hit it.

A PostgreSQL expression index is only selectable when the query repeats the
indexed expression verbatim, so the DDL (``migrations.py`` and the Alembic
versions) and the read arms (``engine/sql/postgresql.py``) cannot be allowed to
drift. Both sides call the helpers here — same idea as
``_pg_search.pg_search_bm25_columns``.
"""


def mental_models_text_document(alias: str | None = None) -> str:
    """The ``mental_models`` full-text document: model/page name + content.

    Mirrors the generating expression of the native tsvector column created by
    the ``n9i0j1k2l3m4`` (learnings / pinned_reflections) migration, so every
    backend indexes and queries the exact same document. ``content`` is NOT NULL,
    hence the deliberate lack of a ``COALESCE`` around it.

    ``alias`` qualifies the columns for queries that join the table (``mm``);
    leave it unset for DDL, where the expression is already table-scoped.
    """
    prefix = f"{alias}." if alias else ""
    return f"(COALESCE({prefix}name, '') || ' ' || {prefix}content)"
