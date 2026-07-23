"""The Postgres memories implementation, split by what calls it.

:class:`~hindsight_api.engine.memories.postgres.PostgresMemories` is a thin class
over these modules; the queries live here, grouped by concern rather than piled
behind one object:

* :mod:`counts`   — the stats/admin aggregates (freshness, per-doc, timeseries, scopes)
* :mod:`curation` — the memory/entity list and detail views
* :mod:`graph`    — the graph view, entity postings, and the maintenance passes
* :mod:`reads`    — addressed reads: get, scan, count, tags, consolidation state
* :mod:`writes`   — inserts, deletes, and observation invalidation

Every function here takes the live connection and Hindsight's ``fq_table``
resolver rather than reaching for globals, so each is callable from a
transaction the caller already owns.
"""

from __future__ import annotations

__all__ = ["counts", "curation", "graph", "reads", "writes"]
