# DB Agent - Self-Verification Checklist

Run through every item before submitting your work.

## Modeling
- [ ] DB type selection justified from workload, not preference
- [ ] Relational models normalized to at least 3NF, or denormalization explicitly justified
- [ ] NoSQL models aligned to concrete access patterns and aggregate boundaries
- [ ] Vector retrieval is used only where semantic similarity is actually required
- [ ] Keys, cardinality, and ownership clearly defined
- [ ] No obvious anti-patterns remain: CSV-in-column, repeated columns, EAV, polymorphic FK, missing FK

## Schema Documentation
- [ ] External schema documented
- [ ] Conceptual schema documented
- [ ] Internal schema documented
- [ ] Naming standards and field definitions documented
- [ ] Glossary / terminology dictionary updated

## Integrity & Concurrency
- [ ] Entity, domain, referential, and business-rule integrity addressed
- [ ] Transaction boundaries documented
- [ ] Isolation level selected for critical flows
- [ ] Locking / optimistic concurrency strategy documented
- [ ] Transparency assumptions called out when using distributed systems

## Performance & Operations
- [ ] Indexes match hot queries and join/filter patterns
- [ ] Partitioning / sharding strategy justified if introduced
- [ ] Hot vs cold data separation considered
- [ ] Full and incremental backup strategy documented
- [ ] Restore test or recovery validation approach documented
- [ ] Query anti-patterns reviewed: `SELECT *`, random sort, ambiguous grouping, full wildcard search without proper tooling

## Vector Retrieval
- [ ] Vector DB is not being used as the canonical document store
- [ ] Hybrid retrieval considered where exact match, explainability, or compliance filtering matter
- [ ] Embedding model, dimension, normalization, chunking, and version metadata documented
- [ ] Re-embedding / re-index migration plan exists for model or chunking changes
- [ ] ANN index choice and tuning parameters benchmarked on production-like data
- [ ] Metadata filtering behavior and high-cardinality filter impact understood
- [ ] Similarity score thresholds calibrated per domain or reranking added for precision-critical paths
- [ ] Retrieval observability defined: golden queries, recall drift, semantic drift, token waste

## Capacity
- [ ] Capacity sheet updated by object
- [ ] Tablespace/storage estimate updated
- [ ] Disk estimate includes online, batch, backup, and growth assumptions
- [ ] Data type choices reviewed for storage efficiency

## Delivery
- [ ] Assumptions, tradeoffs, and risks are explicit
- [ ] Migration impact or rollout steps included when schema changes are proposed
- [ ] ISO 27001 / 27002-friendly suggestions included where design affects access control, logging, encryption, backup, recovery, or change management
- [ ] ISO 22301-friendly suggestions included where design affects resilience, failover, restore testing, RTO, or RPO
