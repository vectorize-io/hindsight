# DB Agent - Deliverable Templates

Use this structure when the user asks for modeling or DB architecture outputs.

## 1. Decision Summary

```md
## Database Decision Summary
- Workload type:
- Recommended engine/model:
- Reason:
- Consistency target:
- Availability target:
- Retention target:
- RPO / RTO:
```

## 2. Schema Documentation

```md
## External Schema
- Consumer / view:
- Required fields:
- Access pattern:

## Conceptual Schema
- Entity / aggregate:
- Description:
- Relationships:
- Cardinality:

## Internal Schema
- Object name:
- Type: table / collection / index / partition
- Primary key / shard key:
- Constraints:
- Access path:
- Storage / tablespace:
```

## 3. Data Standards

```md
| Standard Item | Value |
| --- | --- |
| Naming convention | |
| Definition rule | |
| Data type / format rule | |
| Allowed value rule | |
| Nullability rule | |
| Default value rule | |
```

## 4. Glossary

```md
| Term | Definition | Synonym / Forbidden Term | Owner |
| --- | --- | --- | --- |
```

## 5. Capacity Estimation

```md
| Object | Daily Txn | Online Workload | Batch Workload | Row / Doc Size | Growth | Retention | Primary Storage | Backup Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

Add these summary sections after the table:
- tablespace/storage allocation
- disk total with growth buffer
- hot vs cold data split
- backup type and cadence

## 6. Concurrency & Integrity

```md
## Integrity
- Entity integrity:
- Domain integrity:
- Referential integrity:
- Business-rule integrity:

## Concurrency
- Critical transaction:
- Isolation level:
- Locking method:
- Retry / idempotency strategy:
```

## 7. Vector Retrieval Design

```md
## Vector Retrieval Design
- Use case:
- Why vector retrieval is needed:
- Why lexical search alone is insufficient:
- Hybrid retrieval approach:
- Canonical document store:
- Vector store contents:
- Embedding model / dimension / normalization:
- Chunking policy / overlap:
- ANN index type / key tuning params:
- Metadata filtering strategy:
- Reranking strategy:
- Re-embedding migration plan:
- Retrieval evaluation metrics:
```
