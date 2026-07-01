---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
specs:
  - "W3C Trace Context: Level 1 Recommendation 2020-02-06"
---

# Distributed Traces Signal

## 1. Scope

Traces are the "T" in MELT+P; the correlation backbone of distributed observability. A single trace links every span produced by every service that participated in one request, job, or message.

Covers: OTel span data model, SpanKind, W3C Trace Context, DB patterns (N+1, slow query, pool exhaustion, `db.*` Stable), messaging patterns (Kafka, Flink, Spark, DLQ, `messaging.*`), RPC (`rpc.*` RC), sampling, backends, exemplars, baggage security.

Cross-references:
- `../boundaries/cross-application.md`: propagator matrix per cloud and mesh vendor
- `../layers/mesh.md`: zero-code auto-instrumentation; propagator headers

---

## 2. OTel Span Data Model (Stable)

Source: <https://opentelemetry.io/docs/specs/otel/trace/api/>

Every span carries the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `TraceId` | 16-byte hex | Globally unique identifier for the entire trace |
| `SpanId` | 8-byte hex | Unique identifier for this span |
| `ParentSpanId` | 8-byte hex | SpanId of the parent; absent for root spans |
| `Name` | string | Operation name (e.g., `HTTP GET /api/orders`) |
| `Kind` | enum | SpanKind; see table below |
| `StartTime` | timestamp | Monotonic clock at span start |
| `EndTime` | timestamp | Monotonic clock at span end |
| `Attributes` | key-value map | Semantic convention attributes describing the operation |
| `Events` | list | Timestamped structured log records within the span |
| `Links` | list | References to other spans (used for async messaging) |
| `Status` | enum + string | `UNSET`, `OK`, or `ERROR` with optional description |

### SpanKind

SpanKind describes the role of the span in the larger distributed operation. Backends use it to render topology maps and assign latency accountability.

| SpanKind | Role | Typical origin |
|----------|------|----------------|
| `INTERNAL` | Default; in-process operation with no network boundary | Business logic, cache lookup, CPU computation |
| `SERVER` | Incoming synchronous request handled by this service | HTTP server, gRPC server handler |
| `CLIENT` | Outgoing synchronous call from this service to another | HTTP client, DB driver, gRPC stub |
| `PRODUCER` | Async message sent to a queue or topic | Kafka producer, RabbitMQ publisher |
| `CONSUMER` | Async message received from a queue or topic | Kafka consumer, worker pulling from queue |

Rule of thumb: sync RPC produces a `CLIENT`+`SERVER` pair sharing the same `TraceId`. Async messaging produces a `PRODUCER`+`CONSUMER` pair connected by a span link (not parent-child), because the two operations run independently.

---

## 3. W3C Trace Context Propagation

Source: <https://www.w3.org/TR/trace-context/>

### Header format

```
traceparent: 00-<trace-id>-<span-id>-<flags>
```

- `00`: version (currently always `00`)
- `<trace-id>`: 32 hex chars (128-bit TraceId)
- `<span-id>`: 16 hex chars (64-bit SpanId of the sending span)
- `<flags>`: 8-bit flags; `01` = sampled, `00` = not sampled

Example:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate:  vendor1=abc123,vendor2=xyz789
```

`tracestate` carries vendor-specific opaque state as a comma-separated `key=value` list forwarded unchanged by intermediaries that do not own the key.

### Propagation rules

- **Inject** `traceparent` on every outbound HTTP/gRPC call; **extract** at every inbound call before creating the root span.
- Stripping `traceparent` silently is anti-pattern #1 in `../anti-patterns.md`.
- Cross-cloud/mesh vendor header bridging: `../boundaries/cross-application.md §Propagators`.
- Mesh-injected headers (Envoy, Linkerd): `../layers/mesh.md §Propagator headers`.

---

## 4. Database Tracing Patterns (db.* semconv Stable)

Source: <https://opentelemetry.io/docs/specs/semconv/database/>

All DB spans use `SpanKind = CLIENT`. The span name follows the pattern `<db.operation.name> <db.collection.name>` (e.g., `SELECT orders`).

### Core db.* attributes

| Attribute | Stability | Example | Notes |
|-----------|-----------|---------|-------|
| `db.system` | Stable | `postgresql`, `mysql`, `mongodb`, `redis` | Required; identifies the DB technology |
| `db.operation.name` | Stable | `SELECT`, `INSERT`, `HMGET` | SQL verb or command name |
| `db.query.text` | Stable | `SELECT * FROM orders WHERE id = $1` | Parameterized query text; PII risk; see caveat |
| `db.namespace` | Stable | `mydb` | Database name / schema |
| `db.collection.name` | Stable | `orders` | Table or collection name |
| `db.client.connections.used` | Development | `14` | Active connections in pool (attribute name varies by semconv draft; pin to version in `../standards.md`) |
| `db.client.connection.pool.utilization` | Development | `0.875` | Ratio: used / max (current semconv draft name; older drafts use `db.client.connections.usage`) |

`db.query.text` caveat: WHERE clause literals may contain PII (email addresses, phone numbers). Always use parameterized queries so that literals are replaced by `$1`, `?`, or `:name` placeholders. If the ORM or driver captures the raw query, apply redaction at the OTel SDK layer or in the Collector processor. Cross-ref `../signals/privacy.md`.

### 4.1 N+1 Query Detection

Symptom: a single parent `SERVER` span contains N child `CLIENT` spans that all share the same `db.operation.name` and `db.collection.name`, each taking 2-30 ms, producing an additive latency of N × latency_per_query.

Trace waterfall example:

```
[SERVER] POST /api/orders            .................... 480ms
  [CLIENT] SELECT orders WHERE ...   ..  18ms
  [CLIENT] SELECT orders WHERE ...   ..  17ms
  [CLIENT] SELECT orders WHERE ...   ..  17ms
  [CLIENT] SELECT orders WHERE ...   ..  18ms
  ...  (× 24 more)
```

Detection:

- OTel trace visualizers (Honeycomb BubbleUp, Datadog Watchdog, Grafana Tempo TraceQL) can flag traces where `count(db.CLIENT spans per SERVER span) > N_threshold`.
- TraceQL example (Grafana Tempo): `{ span.db.system = "postgresql" } | count() > 10`
- Honeycomb: group by `trace.parent_id`, count `db.*` child spans, alert on p95 count > 10.

Remediation: replace per-row lookups with eager loading (ORM `include`/`join`), batch queries (`SELECT ... WHERE id = ANY($1)`), or the DataLoader pattern for GraphQL.

### 4.2 Slow Query Analysis

Slow queries surface as `CLIENT` spans with duration above the p99 threshold.

1. Index `db.query.text` + `db.operation.name` against a `span.duration` histogram in the trace backend.
2. Flag spans where `span.duration > 100ms` at p99 (adjust threshold per SLO).
3. Correlate `db.query.text` with `pg_stat_statements` / MySQL `slow_query_log` to obtain execution plans.
4. Apply `EXPLAIN ANALYZE` and add the missing index or rewrite the query.

Privacy: if `db.query.text` captures literal values, redact before export; see `../signals/privacy.md §query text redaction`.

### 4.3 Connection Pool Exhaustion

Symptom: the blocking span is on the **client side** (waiting to acquire a pool connection), not inside the DB. `CLIENT` span duration grows while actual DB response time stays flat.

```
[SERVER] POST /api/orders            .............. 5200ms
  [CLIENT] acquire pool connection   ............  5190ms  ← pool wait, no DB call yet
  [CLIENT] SELECT orders WHERE ...   .           10ms
```

Attributes to monitor: `db.client.connections.used` (Development), `db.client.connections.usage` (Development, alert at > 0.85). Cross-ref `../signals/metrics.md` for `pool_wait_count` and `pool_wait_duration_seconds` histograms.

Remediation: tune pool size to expected concurrency (10-20 connections per pod); add circuit breaker upstream; split read/write pools for long-running read transactions.

---

## 5. Messaging Tracing Patterns (messaging.* semconv)

Source: <https://opentelemetry.io/docs/specs/semconv/messaging/>

### Core messaging.* attributes

| Attribute | Example | Notes |
|-----------|---------|-------|
| `messaging.system` | `kafka`, `rabbitmq`, `nats` | Required |
| `messaging.destination.name` | `orders-topic` | Topic or queue name |
| `messaging.operation` | `publish`, `create`, `receive`, `deliver`, `settle`, `process` | Operation type per semconv 1.27.0 (note: the attribute is spelled `messaging.operation.type` in latest drafts; pin via `../standards.md`). `settle` covers ack/nack; `ack` alone is NOT a valid enum value |
| `messaging.message.id` | `msg-uuid-1234` | Message identifier for dedup |
| `messaging.kafka.message.key` | `order-9988` | Kafka partition key |
| `messaging.kafka.consumer.group` | `payment-consumer-group` | Kafka consumer group |

### SpanKind assignment

- `PRODUCER`: set when calling `producer.send()` or equivalent.
- `CONSUMER`: set when calling `consumer.poll()` or the message handler.
- Connect the two with a **span link** (not parent-child): async messaging breaks the synchronous call chain; span links preserve trace continuity across the queue boundary.

```
[PRODUCER] send orders-topic         ..  3ms    TraceId: aaa
  Link → [CONSUMER] receive orders-topic  ...  220ms  TraceId: aaa
    [CLIENT] INSERT INTO payments    ..  18ms
```

### 5.1 Kafka-Specific Patterns

Inject W3C `traceparent` into Kafka message headers at the producer; extract at the consumer before creating the `CONSUMER` span. OTel Kafka instrumentation handles this automatically (`opentelemetry-instrumentation-kafka` for Java/Python).

Kafka consumer lag is an external metric, not a trace attribute: MSK (`kafka.consumer_lag` CloudWatch), Confluent Cloud (built-in), self-managed (`kafka_consumer_group_lag` via JMX + Prometheus). Cross-ref `../signals/metrics.md §messaging`.

**Primary-topic lag alert threshold (recommended starting point):**
- Warn: `kafka_consumer_group_lag > 10_000 for 5m` (queue growing faster than consumer can drain)
- Page: `kafka_consumer_group_lag > 100_000 for 10m` (backlog will breach SLO if unaddressed)
- Calibrate per topic throughput and consumer group parallelism; high-throughput topics routinely run at 50k–500k lag without impact.

### 5.2 Flink and Spark Streaming

Create one span per pipeline stage; the trace covers end-to-end job from source read to sink write. Flink OTel instrumentation is emerging (not stable as of semconv 1.27.0); use `messaging.system = kafka` for source/sink spans and `INTERNAL` for operator chains. Spark: instrument at the `foreachBatch` boundary; each micro-batch is one `INTERNAL` span. Checkpoint metrics: cross-ref `../signals/metrics.md §streaming`.

### 5.3 Dead Letter Queue (DLQ) Observability

A DLQ receives messages that failed all retries. Losing trace context at DLQ ingestion prevents diagnosis.

1. **Trace propagation**: copy the original `traceparent` from the failed message headers into the DLQ message headers; do not generate a new trace ID.
2. **DLQ depth alert**: alert on `kafka_consumer_group_lag{topic="orders-dlq"}` > 0 for critical queues.
3. **DLQ arrival span**: create a `CONSUMER` span at DLQ arrival with `messaging.destination.name = orders-dlq` and the original `trace_id`.
4. **Replay tooling**: re-inject the original trace context into replayed messages. A new `trace_id` at replay orphans the original failure span.

---

## 6. RPC Tracing (rpc.* semconv RC)

Source: <https://opentelemetry.io/docs/specs/semconv/rpc/grpc/>

### gRPC

| Attribute | Example |
|-----------|---------|
| `rpc.system` | `grpc` |
| `rpc.service` | `com.example.OrderService` |
| `rpc.method` | `CreateOrder` |

SpanKind: `CLIENT` on the caller; `SERVER` on the callee. The `traceparent` header is propagated via gRPC metadata.

### HTTP RPC

Use `http.*` attributes for REST and JSON-RPC over HTTP. `rpc.*` is for binary RPC protocols only.

### Error handling

Set `status.code = ERROR` and `status.description` on RPC failures. Use `span.recordException(e)` to atomically populate `exception.type`, `exception.message`, and `exception.stacktrace`; required by MRA in `../incident-forensics.md §2.2`.

---

## 7. Trace Sampling

Cross-ref `../transport/sampling-recipes.md` for full configuration recipes.

| Strategy | Pros | Cons |
|----------|------|------|
| Head-based | Low overhead; decision at trace root | Cannot retain traces that only show problems at a downstream hop |
| Tail-based | Trace-complete decision; keeps all error traces | Requires buffering the full trace before deciding; higher Collector memory |

Recommended recipe:

- 100% of traces where `status.code = ERROR`
- 100% of traces where business cost > $0.50 (custom attribute)
- 5-10% tail sampling of remaining traces for baseline coverage

For multi-replica tail sampling, use consistent hashing on `trace_id` via the OTel Collector `loadbalancing` exporter to route all spans of one trace to the same Collector instance.

---

## 8. Trace Backends

Cross-ref `../vendor-categories.md` for full vendor category taxonomy.

| Backend | Category | Key differentiator |
|---------|----------|--------------------|
| Jaeger (CNCF Graduated) | OSS full-stack | Kubernetes-native; Badger or Cassandra storage |
| Grafana Tempo | OSS full-stack | Low-cost object storage (S3/GCS); TraceQL query language |
| Zipkin | OSS (legacy) | Older wire format; superseded by OTel/Jaeger for new deployments |
| Honeycomb | High-cardinality specialist | BubbleUp auto-analysis; best for high-attribute-count traces |
| Datadog APM | Commercial SaaS | Deep integration with Datadog metrics and logs |
| Elastic APM | Commercial SaaS | Tight Kibana/Elasticsearch integration |
| SigNoz | OSS full-stack | ClickHouse-backed; cost-efficient at high ingestion rates |

---

## 9. Trace Exemplars

Exemplars link a metric data point to a specific trace, enabling direct navigation from a PromQL alert to the trace that caused the spike.

- Prometheus exemplar support is available since v2.31.0. Histograms and summaries can carry a `trace_id` label on selected samples.
- OTel SDK Java, Go, and Python emit exemplars automatically when the current span context is sampled.
- Usage: click a spike in a Grafana panel backed by a Prometheus histogram → the exemplar tooltip shows a `trace_id` link → click to open the trace in the configured trace backend (Tempo, Jaeger, etc.).

```promql
# Example: p99 latency histogram with exemplar-capable query
histogram_quantile(0.99,
  sum by (le, service_name) (
    rate(http_server_request_duration_seconds_bucket{
      deployment_environment="prod"
    }[5m])
  )
)
```

Cross-ref `../incident-forensics.md §Step 2` for the exemplar-based trace acquisition flow.

---

## 10. Propagating Baggage (Security and Privacy)

W3C Baggage (`baggage` header) carries key-value pairs alongside the trace context. It is not part of the trace itself; it is a side-channel for business attributes that downstream services need.

### Allowed vs. prohibited baggage content

| Allowed | Prohibited |
|---------|-----------|
| `tenant.id` | User email address |
| `user.tier` (tier name, not PII) | Session tokens or auth cookies |
| `deployment.sha` | Passwords or API keys |
| `feature.flag.name` | Any field classified as PII under GDPR/PIPA |

W3C Baggage Recommendation §3.1: baggage propagates to all downstream services within the same distributed operation. Any data placed in baggage is visible to every service on the call path, including third-party services. Data that is proprietary, confidential, or personally identifiable MUST NOT be placed in baggage without explicit downstream trust agreement.

Enforcement rule: strip or validate incoming baggage at the API gateway / ingress before forwarding to internal services. The mesh ingress gateway is the correct enforcement point; see `../layers/mesh.md §Baggage scrubbing`.

Cross-ref:

- `../boundaries/cross-application.md`: trust-boundary baggage validation rules per cloud and mesh
- `../signals/privacy.md`: PII classification and redaction for baggage content

---

## 11. Matrix Coverage (traces column)

These cells from `../matrix.md` have trace-specific behavior worth noting:

| Layer | Boundary | Status | Rationale |
|-------|----------|--------|-----------|
| L3-network | cross-application | PARTIAL | L3 packets carry no trace context natively; use trace-ID tagging at egress only |
| L4-transport | any | PARTIAL | TCP is not trace-native; trace context begins at mesh or L7 |
| mesh | cross-application | PASS | Primary trace origin; Envoy injects spans with zero application code changes |
| L7-application | cross-application | PASS | Primary; W3C `traceparent` on all outbound HTTP/gRPC calls |
| L7-application | multi-tenant | PASS | `tenant.id` propagated in W3C Baggage on every span |
| L7-application | release | PASS | `service.version` resource attribute on all spans; canary routing by version |
| L7-application | slo | PARTIAL | Tail sampling must retain error traces within the error budget window; traces are forensic input, not the SLI computation source |

---

## 12. Anti-patterns

These candidates extend `../anti-patterns.md`. Each pattern breaks trace continuity or introduces compliance risk.

| Anti-pattern | Consequence | Remediation |
|---|---|---|
| Missing parent spans (orphan spans) | Trace visualizer shows disconnected spans; root cause chain is broken | Verify `traceparent` extraction at every inbound call; check SDK auto-instrumentation coverage |
| `db.query.text` captured with untrimmed PII | Email or phone literal in WHERE clause stored in trace backend; GDPR/PIPA violation | Parameterize all queries; add OTel SDK `db.sanitize_query` option or Collector redaction processor |
| Async messaging without span links | `PRODUCER` and `CONSUMER` spans in separate disconnected traces; cannot follow a message end-to-end | Use OTel SDK `span.addLink()` with the producer span context when creating the consumer span |
| DLQ drain creates a new `trace_id` | Original failure span is orphaned; forensic link to the root cause is lost | Re-inject original `traceparent` from DLQ message headers into the replayed message headers |
| Head-based sampling on multi-service paths | A decision to not sample at the entry service drops all downstream spans, including error spans that only appear deep in the call graph | Use tail-based sampling with 100% error retention; see §7 |
| Kafka consumer lag unmonitored | DLQ silently fills; processing backlog not detected until downstream systems starve | Alert on `kafka_consumer_group_lag` per consumer group; cross-ref `../signals/metrics.md §messaging` |

---

## References

- OTel Trace API specification: <https://opentelemetry.io/docs/specs/otel/trace/api/>
- OTel Database semconv: <https://opentelemetry.io/docs/specs/semconv/database/>
- OTel Messaging semconv: <https://opentelemetry.io/docs/specs/semconv/messaging/>
- OTel RPC/gRPC semconv: <https://opentelemetry.io/docs/specs/semconv/rpc/grpc/>
- W3C Trace Context Level 1 Recommendation: <https://www.w3.org/TR/trace-context/>
- W3C Baggage Recommendation: <https://www.w3.org/TR/baggage/>
- Kafka producer/consumer instrumentation: <https://kafka.apache.org/documentation/>
- Confluent consumer lag metrics: <https://docs.confluent.io/platform/current/kafka/monitoring.html>
