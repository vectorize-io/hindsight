"""The Postgres memories implementation, split by what calls it.

:class:`~hindsight_api.engine.memories.postgres.PostgresMemories` is a thin class
over these modules; the queries live here, grouped by concern rather than piled
behind one object:

* :mod:`counts`   — the stats/admin aggregates (freshness, per-doc, timeseries, scopes)
* :mod:`curation` — the memory/entity list and detail views
* :mod:`graph`    — the graph view, entity postings, and the maintenance passes
* :mod:`reads`    — addressed reads: get, scan, count, tags, consolidation state
* :mod:`writes`   — inserts, deletes, and observation invalidation
* :mod:`retain`   — retain's document-row, chunk and per-document memory statements
* :mod:`banks`    — the per-bank vector indexes and the bank list's watermarks and counts
* :mod:`transfer` — bank export / import: documents, facts, observations, the curation archive
* :mod:`admin`    — operator-side SQL: consolidation gauges, index sizing, the admin CLI's schema walks
* :mod:`documents` — the document / chunk routes, attachment lookups and recall's chunk / source-fact enrichment
* :mod:`engine_curation` — memory and bank deletes, consolidation requeues, observation history, entity views
* :mod:`consolidation` — consolidation's liveness checks, observation writes and dedup folds
* :mod:`expand`   — the reads behind reflect's ``expand`` tool: memories, then their chunks and documents
* :mod:`recall`   — the dense + BM25 and temporal recall arms
* :mod:`link_expansion` — the graph recall arm (:class:`LinkExpansionRetriever`)
* :mod:`entity_resolver` — the SQL entity registry retain resolves names against
* :mod:`links`    — retain's temporal / semantic / causal link writes (and its entity-resolution front)

Every function here takes the live connection and Hindsight's ``fq_table``
resolver rather than reaching for globals, so each is callable from a
transaction the caller already owns. The last four predate the store and keep
their own signatures; they resolve store tables through ``fq_store_table``.
"""

from __future__ import annotations

__all__ = [
    "admin",
    "banks",
    "counts",
    "consolidation",
    "curation",
    "documents",
    "engine_curation",
    "entity_resolver",
    "expand",
    "graph",
    "link_expansion",
    "links",
    "reads",
    "recall",
    "retain",
    "transfer",
    "writes",
]
