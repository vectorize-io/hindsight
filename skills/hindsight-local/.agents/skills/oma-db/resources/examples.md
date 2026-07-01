# DB Agent - Examples

## Example 1: Relational Order Management

**Input**: "Design the database for ordering, payment, shipment, and refund flows."

**Output**:
```md
## Model Choice
- PostgreSQL chosen for strong integrity, cross-entity transactions, and reporting joins

## Conceptual Schema
- Customer 1:N Order
- Order 1:N OrderItem
- Order 1:N Payment
- Order 1:N Shipment
- Order 1:N Refund

## Internal Schema
- orders(id PK, customer_id FK, status, ordered_at, total_amount, currency_code)
- order_items(id PK, order_id FK, product_id FK, qty, unit_price, line_amount)
- payments(id PK, order_id FK, payment_method_code, status, approved_at, amount)
- shipments(id PK, order_id FK, warehouse_id FK, status, shipped_at)
- refunds(id PK, payment_id FK, reason_code, refunded_at, amount)

## Integrity / Concurrency
- 3NF satisfied
- FKs enforce referential integrity
- Unique(order_id, product_id) for line uniqueness
- Payment capture uses READ COMMITTED with idempotency key
- Inventory reservation uses SELECT ... FOR UPDATE

## Operations
- Hot data: last 12 months in primary partition
- Cold data: monthly archive partitions after 12 months
- Daily full backup + 15-minute incremental/WAL archiving
```

## Example 2: NoSQL Activity Feed

**Input**: "Model a high-write social feed with fan-out reads."

**Output**:
```md
## Model Choice
- NoSQL chosen because feed reads follow aggregate access by user timeline and tolerate eventual consistency

## Conceptual Schema
- User aggregate
- Post aggregate
- TimelineEntry aggregate

## Internal Schema
- users collection keyed by user_id
- posts collection keyed by post_id
- timelines collection keyed by owner_user_id + created_at

## Consistency / BASE
- Timeline fan-out is eventually consistent
- Post delete creates compensating cleanup event
- Source-of-truth post document retained separately from denormalized timeline entry

## Optimization
- Partition key: owner_user_id
- TTL on ephemeral recommendation cache
- Cold feed archive after 180 days
- Nightly full snapshot + hourly incremental backup
```

## Example 3: Data Standards & Capacity Update

**Input**: "Add subscription billing and keep the glossary and sizing docs current."

**Output**:
```md
## Data Standards
| Name | Definition | Format | Rule |
| --- | --- | --- | --- |
| subscription_status_code | Subscription lifecycle state | VARCHAR(20) | ACTIVE, PAUSED, CANCELED only |
| billing_cycle_day | Day of month for billing | SMALLINT | 1-28 only |

## Glossary Additions
- Subscription: recurring billing agreement between customer and service
- Invoice: billable statement generated for a subscription cycle

## Capacity Update
- subscriptions: 1.2M rows / year
- invoices: 14.4M rows / year
- payments: 16M rows / year
- Estimated primary storage: 180 GB / year
- Backup storage multiplier: 2.5x including retention
```

## Example 4: Enterprise RAG Retrieval

**Input**: "Design a policy search platform using a vector DB for enterprise RAG."

**Output**:
```md
## Model Choice
- PostgreSQL remains source of truth for document metadata and lifecycle
- Object storage holds canonical documents
- Vector DB is used only as semantic retrieval index

## Retrieval Architecture
- Hybrid retrieval = BM25 + ANN + metadata filter + cross-encoder rerank
- Filters applied for tenant_id, region, policy_type, effective_date
- Exact compliance terms still supported by lexical search

## Embedding Governance
- embedding_model: text-embedding-3-large
- embedding_version: v2026_03_policy_chunks_v2
- chunking: heading-aware, 350-600 tokens, 80-token overlap
- Shadow index used during re-embedding migration

## ANN / Operations
- HNSW selected after recall-latency benchmark
- search cluster separated from ingestion cluster
- Hot vectors: last 90 days of updated policies
- Cold vectors: older embeddings compacted to lower-cost tier

## Observability
- Golden query set maintained for recall regression
- Similarity score distribution monitored by policy domain
- Prompt token waste tracked for bad retrieval
```
