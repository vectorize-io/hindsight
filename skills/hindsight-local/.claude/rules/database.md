---
description: Database schema design, normalization, indexing, and integrity
  standards for SQL, NoSQL, and vector databases
paths: "**/*.{sql,prisma}"
---


# Database Standards

## Core Rules

1. Choose model first, engine second: workload, access pattern, consistency, and scale drive DB selection.
2. For relational workloads, enforce at least **3NF** by default. Break 3NF only with explicit performance justification.
3. For distributed/non-relational workloads, model around aggregates and access paths; document **BASE** and consistency tradeoffs.
4. Document **ACID** expectations for relational transactions. For distributed tradeoffs, document consistency compromises explicitly.
5. Always document the three schema layers: **external schema**, **conceptual schema**, **internal schema**.
6. Treat integrity as first-class: entity, domain, referential, and business-rule integrity must be explicit.
7. Concurrency is never implicit: define transaction boundaries, locking strategy, and isolation level per critical flow.
8. Data standards are mandatory: naming, definition, format, allowed values, and validation rules.
9. Maintain living artifacts: glossary, schema decision log, and capacity estimation — update whenever the model changes.
10. Proactively flag anti-patterns and insecure shortcuts instead of silently implementing them.
11. Vector DBs are retrieval infrastructure, not source-of-truth databases. Store embeddings and metadata there; keep canonical documents elsewhere.
12. Never treat vector search as a drop-in for lexical search. Default to hybrid retrieval when exact match or explainability matters.
13. Embeddings are schema-like assets: version model, dimension, chunking, and preprocessing. Plan re-embedding migrations explicitly.
