---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
notes:
  - "OpenMetrics: IETF RFC 9416 (2023-08)"
---

# Metrics Signal

## 1. Scope

Metrics are the "M" in MELT+P (Metrics, Events, Logs, Traces + Profiles).

This file covers:
- OTel metric instrument types and their sync/async variants
- Prometheus exposition format and OpenMetrics (RFC 9416) formalization
- SLI computation patterns (Golden Signals, RED, USE) and healthcheck integration
- `hostmetrics` receiver for host-level collection
- Kubernetes metric sources
- Application metric integration via OTel SDK
- OpenCost metric surface for cost attribution
- Cardinality budget enforcement
- Long-term storage backend options

Out of scope:
- Full SLO framework, error budget policy, and burn-rate alert rules: see `../boundaries/slo.md`
- Full FinOps unit economics and cost allocation model: see `cost.md`
- Cardinality guardrail detail and pipeline remediation: see `../meta-observability.md §Section C`

---

## 2. OTel Metric Instrument Types (Stable)

Source: <https://opentelemetry.io/docs/specs/otel/metrics/api/>

Five instrument types are stable in the OTel API/SDK 1.x. Choose based on measurement semantics.

### 2.1 Counter

Monotonic, cumulative-only. Value never decreases. Measures cumulative totals.

```
Use for: requests_total, errors_total, bytes_sent_total
Never use for: values that can decrease (use UpDownCounter)
```

Sync variant: `counter.Add(ctx, delta, attrs...)`; caller increments explicitly.
Async variant: `ObservableCounter`; callback provides absolute cumulative value.

### 2.2 UpDownCounter

Monotonic: no. Can increment or decrement. Measures quantities that fluctuate.

```
Use for: queue_length, active_connections, in_flight_requests
Never use for: strictly increasing values (use Counter)
```

Sync variant: `upDownCounter.Add(ctx, delta, attrs...)` where delta is positive or negative.
Async variant: `ObservableUpDownCounter`; callback provides current value.

### 2.3 Gauge

Point-in-time observation of a current value. No accumulation.

```
Use for: cpu_usage, memory_used_bytes, temperature_celsius, cache_hit_ratio
```

Sync variant: `gauge.Record(ctx, value, attrs...)`.
Async variant: `ObservableGauge`; preferred for values read from an external source (OS, hardware sensor).

### 2.4 Histogram

Distribution of values. Captures bucket counts, sum, and count. Supports quantile computation via `histogram_quantile` at query time.

```
Use for: request_duration_seconds, response_size_bytes, db_query_duration_seconds
Prefer over Summary for all new instrumentation
```

Sync variant only: `histogram.Record(ctx, value, attrs...)`.
Configure explicit bucket boundaries per instrument to control cardinality. Default OTel SDK buckets: `[0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]` (milliseconds; override for seconds-based metrics).

### 2.5 Summary (not recommended for new code)

Client-side quantile computation. Produces quantile values at SDK level.

Limitations:
- Quantiles are computed per-process; they cannot be aggregated across replicas.
- `p99` from three replicas cannot be merged into a fleet-level `p99`.
- Use Histogram + `histogram_quantile` instead for any distributed system.

When to keep existing Summary metrics: only if the downstream consumer cannot be migrated and the measurement is single-process.

### 2.6 Temporality: Delta vs Cumulative

OTel SDK default: `delta` (each export contains only the measurements since the last collection).
Prometheus wire format requires: `cumulative` (monotonic, ever-increasing values).

Set the preference via environment variable:

```
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=cumulative
```

Or configure per-instrument in the OTLP exporter. The `prometheusreceiver` in the OTel Collector handles delta-to-cumulative conversion if the SDK exports delta and Prometheus is the backend.

---

## 3. Prometheus Exposition Format

Source: <https://prometheus.io/docs/instrumenting/exposition_formats/>
OpenMetrics: <https://github.com/OpenObservability/OpenMetrics/blob/main/specification/OpenMetrics.md> (IETF RFC 9416)

### 3.1 Text Format Structure

```
# HELP http_requests_total Total HTTP requests received
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200",route="/api/v1/users"} 1234 1714521600000
http_requests_total{method="POST",status="422",route="/api/v1/users"} 7 1714521600000
```

Rules:
- `# HELP` line: one per metric family, human-readable description.
- `# TYPE` line: one per metric family, declares instrument type.
- Metric line: `name{label="value",...} value [timestamp_ms]`. Timestamp is optional.
- Blank line separates metric families.

### 3.2 Naming Convention

Pattern: `{component}_{operation}_{unit}_{suffix}`

| Suffix | When to use | Example |
|--------|-------------|---------|
| `_total` | Counters (monotonic) | `http_requests_total` |
| `_seconds` | Duration (always seconds, not ms) | `http_request_duration_seconds` |
| `_bytes` | Size in bytes | `response_size_bytes` |
| `_ratio` | Fraction 0.0–1.0 | `cache_hit_ratio` |
| `_bucket` | Histogram bucket (auto-appended) | `http_request_duration_seconds_bucket` |
| `_count` | Histogram observation count (auto) | `http_request_duration_seconds_count` |
| `_sum` | Histogram observation sum (auto) | `http_request_duration_seconds_sum` |

### 3.3 Label Rules

Low-cardinality labels only. Cross-ref `../meta-observability.md §Section C Cardinality Guardrails`.

Safe label examples: `method`, `status_code`, `route` (normalized), `service`, `region`, `env`.
Forbidden label examples: `user.id`, `request.id`, `trace.id`, `http.url` (raw), `error.message`.

### 3.4 OpenMetrics Snippet (RFC 9416)

OpenMetrics is the IETF formalization of Prometheus text format. Key difference: `# EOF` terminator is required, and `_total` suffix is mandatory for counters in the `# TYPE counter` family.

```
# HELP http_requests_total Total HTTP requests received.
# TYPE http_requests_total counter
http_requests_total_total{method="GET",status="200"} 1234.0 1714521600.000
# EOF
```

The OTel Prometheus exporter supports both Prometheus text format and OpenMetrics format; select via `Accept: application/openmetrics-text` request header.

---

## 4. SLI / Healthcheck Basics

### 4.1 Google SRE Golden Signals

Four signals for monitoring any request-serving system:

| Signal | Description | Primary instrument |
|--------|-------------|--------------------|
| Latency | Time to serve a request (good vs error latency tracked separately) | Histogram |
| Traffic | Volume of demand (requests/sec, queries/sec) | Counter rate |
| Errors | Rate of failed requests (explicit 5xx, implicit latency budget exceeded) | Counter rate |
| Saturation | How "full" the service is; utilization as fraction of capacity | Gauge or Histogram |

### 4.2 RED Method (request-serving services)

| Signal | PromQL sketch |
|--------|---------------|
| Rate | `sum(rate(http_requests_total[5m])) by (service)` |
| Errors | `sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)` |
| Duration | `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))` |

### 4.3 USE Method (resource-level: CPU, memory, disk)

| Signal | PromQL sketch |
|--------|---------------|
| Utilization | `avg(system.cpu.utilization) by (host)` |
| Saturation | `system.cpu.load_average.1m / count(system.cpu.utilization) by (host)` |
| Errors | `rate(system.network.errors{direction="transmit"}[5m])` |

### 4.4 SLI Examples (PromQL)

Availability SLI (fraction of successful requests over the measurement window):

```promql
sum(rate(http_requests_total{status=~"2..|3.."}[5m]))
  /
sum(rate(http_requests_total[5m]))
```

Latency SLI (p99 from histogram, 5-minute rolling window):

```promql
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
)
```

Cross-ref `../boundaries/slo.md` for SLO target definitions, error-budget burn-rate computation, and multi-window alert rules.

### 4.5 Healthcheck Endpoints and Metrics

Kubernetes readiness and liveness probes (`/readyz`, `/healthz`) are HTTP checks; they are not metrics themselves. However, the underlying health state MUST be exposed as a metric so that dashboards and SLO computation can consume it continuously:

```promql
# Health metric emitted by application (1 = healthy, 0 = unhealthy)
up{job="my-service"}   # Prometheus scrape synthetic metric

# Custom health gauge from OTel SDK
service_health_status{service="checkout", check="database"} 1
```

Anti-pattern: a healthcheck that returns 200 without reflecting actual dependency health. If the database is down, the readiness probe MUST return a non-2xx status, and the `service_health_status` gauge MUST reflect 0. A probe that always returns 200 is not tied to an underlying SLI metric and produces false-positive availability signals.

---

## 5. hostmetrics Receiver (OTel Collector)

Source: <https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/hostmetricsreceiver>

The `hostmetrics` receiver collects host-level OS metrics without requiring a separate agent. It runs inside the OTel Collector process and reads from OS interfaces.

### 5.1 Deployment

Deploy as DaemonSet (one Collector per node) to collect per-node metrics. Cross-ref `../transport/collector-topology.md` for the two-tier topology (DaemonSet agent + gateway).

```yaml
receivers:
  hostmetrics:
    collection_interval: 30s
    root_path: /hostfs    # required when running in container; mount host /proc and /sys
    scrapers:
      cpu: {}
      memory: {}
      disk: {}
      filesystem: {}
      network: {}
      load: {}
      paging: {}
      process:
        mute_process_name_error: true
```

### 5.2 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system.cpu.time` | Counter | CPU time by state (user, system, idle, iowait, irq, softirq, steal, nice) |
| `system.cpu.utilization` | Gauge | CPU utilization 0.0–1.0 by state |
| `system.memory.usage` | Gauge | Memory bytes by state (used, free, cached, buffered) |
| `system.memory.utilization` | Gauge | Memory utilization 0.0–1.0 by state |
| `system.disk.io` | Counter | Disk bytes read/written by device |
| `system.disk.io_time` | Counter | Time disk is active by device (seconds) |
| `system.filesystem.usage` | Gauge | Filesystem bytes by mount point and state |
| `system.network.io` | Counter | Network bytes transmitted/received by interface |
| `system.network.errors` | Counter | Network errors by interface and direction |
| `system.load.average.1m` | Gauge | 1-minute load average |
| `system.load.average.5m` | Gauge | 5-minute load average |
| `system.load.average.15m` | Gauge | 15-minute load average |
| `system.paging.usage` | Gauge | Swap/paging space by state |
| `process.cpu.time` | Counter | Per-process CPU time by state |
| `process.memory.physical_usage` | Gauge | Per-process RSS in bytes |

### 5.3 Alternative: Prometheus Node Exporter

For environments where the OTel Collector cannot run on the host, Prometheus Node Exporter is the mature alternative (CNCF Graduated). Scrape with `prometheusreceiver` and convert to OTel metrics via the Collector pipeline. Metric names differ from OTel semconv (e.g., `node_cpu_seconds_total` vs `system.cpu.time`). Prefer `hostmetrics` receiver for new deployments.

---

## 6. Kubernetes Metric Sources

Three complementary sources cover cluster, node, pod, and container metrics:

| Source | Scope | Collector component |
|--------|-------|---------------------|
| `kubeletstats` receiver | Pod and container metrics from kubelet API | `kubeletstatsreceiver` |
| `k8scluster` receiver | Cluster-level state (node conditions, pod phases, deployment replicas) | `k8sclusterreceiver` |
| Prometheus operator + ServiceMonitor CRD | Scrape application Prometheus endpoints declaratively | `prometheusreceiver` |

Cross-ref `../transport/collector-topology.md §Component preferences` for the recommended configuration of each source in a DaemonSet + gateway topology.

---

## 7. Application Metric Integration

### 7.1 Auto-instrumentation

OTel SDK auto-instrumentation provides RED metrics out-of-the-box for common frameworks:
- HTTP servers: `http.server.request.duration` (Histogram), `http.server.active_requests` (UpDownCounter)
- HTTP clients: `http.client.request.duration` (Histogram)
- gRPC: `rpc.server.duration` (Histogram), `rpc.client.duration` (Histogram)
- Database clients: `db.client.operation.duration` (Histogram)

These use OTel semconv stable attributes (`http.*`, `rpc.*`, `db.*` core groups). Verify your framework's OTel instrumentation library supports the attribute groups before relying on them in SLO computation.

### 7.2 Custom Metrics via Meter API

```python
# Python SDK example — creating a histogram for a custom operation
from opentelemetry import metrics

meter = metrics.get_meter("com.example.checkout", version="1.0.0")

payment_duration = meter.create_histogram(
    name="payment.processing.duration",
    unit="s",
    description="Time to process a payment request",
)

# Recording a measurement
payment_duration.record(0.42, {"payment.method": "card", "currency": "USD"})
```

### 7.3 Attribute Allow-listing for Cardinality Control

Explicitly allow-list attributes on each instrument view to prevent label explosion. Cross-ref `../meta-observability.md §Section C`:

```python
from opentelemetry.sdk.metrics.view import View

view = View(
    instrument_name="payment.processing.duration",
    attribute_keys={"payment.method", "currency"},  # only these two labels emitted
)
```

---

## 8. OpenCost Metric Surface

Source: <https://www.opencost.io/docs/configuration/prometheus>

OpenCost (CNCF Incubating) exposes cost attribution in Prometheus format at its `/metrics` endpoint. Scrape via `prometheusreceiver` to bring cost into the same TSDB as latency and error metrics, making cost a dimension alongside performance.

### 8.1 Key OpenCost Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `opencost_namespace_cost_total` | Counter | USD | Cumulative cost by Kubernetes namespace |
| `opencost_workload_cost_total` | Counter | USD | Cumulative cost by workload (deployment/statefulset) |
| `opencost_cpu_cost` | Gauge | USD/hr | Current CPU cost attribution |
| `opencost_ram_cost` | Gauge | USD/hr | Current memory cost attribution |
| `opencost_network_cost` | Gauge | USD/hr | Current network egress cost attribution |
| `opencost_storage_cost` | Gauge | USD/hr | Current persistent storage cost attribution |

### 8.2 Scrape Configuration

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: opencost
          scrape_interval: 60s
          static_configs:
            - targets: ["opencost.opencost.svc.cluster.local:9003"]
```

Cross-ref `cost.md` for the full FinOps unit-economics model, per-tenant cost attribution, and chargeback/showback patterns.

---

## 9. Cardinality Budget

Cross-ref `../meta-observability.md §Section C Cardinality Guardrails` for full detail and remediation procedures.

### 9.1 Per-Service Budget

Set a hard budget per service, enforced via alert at 80% of budget:

```promql
# Alert: service approaching 5000-series budget
count({job="my-service"}) > 4000
```

### 9.2 Forbidden Labels (Never use as metric labels)

| Label | Reason |
|-------|--------|
| `user.id` | Unbounded; one series per user |
| `request.id` | One series per request; instant explosion |
| `trace.id` | One series per trace |
| `http.url` (raw) | Query strings are unbounded |
| `error.message` | Free-text; unbounded cardinality |

### 9.3 Tenant Cap

`tenant.id` label is allowed with a hard cap: top-N explicit tenants (e.g., 1000), all others mapped to label value `"other"`. Never create a new metric name per tenant; this bypasses TSDB cardinality controls entirely (anti-pattern listed in Section 12).

---

## 10. Long-Term Storage Backends

Prometheus-compatible TSDBs for metrics retention beyond 15 days:

| Backend | CNCF Status | Key Characteristic | Typical Retention |
|---------|-------------|--------------------|-------------------|
| Prometheus | CNCF Graduated | Local storage, no HA by default | 15d (short-term) |
| Thanos | CNCF Graduated | Object storage long-term; Prometheus sidecar model | 1y+ |
| Cortex | CNCF Incubating | Multi-tenant Prometheus; horizontally scalable | 1y+ |
| Grafana Mimir | Not CNCF | Grafana Labs fork of Cortex; production-grade | 1y+ |
| VictoriaMetrics | Not CNCF | High-performance; efficient storage compression | 1y+ |

Cross-ref `../vendor-categories.md §TSDB / Long-term Metrics` for selection guidance per workload size and multi-tenancy requirements.

---

## 11. Matrix Coverage (metrics column)

This table maps the metrics signal cells from `../matrix.md` for quick navigation.

| Layer | Boundary | Coverage | Detail |
|-------|----------|----------|--------|
| L3-network | multi-tenant | PASS | Per-tenant VPC flow byte/packet counters |
| L3-network | cross-application | PASS | Inter-VPC/peering flow metrics |
| L4-transport | multi-tenant | PASS | Per-tenant TCP retransmit rate, RTT histograms via eBPF |
| L4-transport | cross-application | PASS | Cross-service TCP RTT and retransmit metrics via eBPF |
| mesh | multi-tenant | PASS | Per-tenant RED from Envoy telemetry; `tenant.id` via baggage |
| mesh | cross-application | PASS | Cross-service RED metrics at mesh proxy; topology mapping |
| mesh | slo | PASS | Request rate and error rate as primary SLI sources |
| mesh | release | PASS | Request split metrics for canary traffic (Flagger/Argo Rollouts) |
| L7-application | multi-tenant | PASS | Per-tenant RED + custom business metrics with `tenant.id` |
| L7-application | cross-application | PASS | Inter-service histograms with `service.name` + `peer.service` |
| L7-application | slo | PASS | SLI metric computation; SLO targets in OpenSLO YAML |
| L7-application | release | PASS | Release marker; `service.version` label for before/after delta |

---

## 12. Anti-Patterns

These are candidates for `../anti-patterns.md §Section B Cardinality & Cost`:

| Anti-pattern | Description | Correction |
|--------------|-------------|------------|
| `user.id` as metric label | Creates one series per user; TSDB storage explosion | Use `user.tier` or aggregated cohort label |
| New metric name per tenant | `http_requests_total_tenant_acme` bypasses cardinality controls | Use `http_requests_total{tenant="acme"}` with top-N cap |
| Summary for cross-service aggregation | Client-side quantiles cannot be merged across replicas | Replace with Histogram + `histogram_quantile` at query time |
| Healthcheck not tied to underlying SLI | `/healthz` returns 200 regardless of dependency health | Emit `service_health_status{check="db"} 0` when unhealthy; wire into SLI |
| Raw `http.url` label | Query strings are unbounded; `?token=...` leaks secrets | Use `http.route` (normalized); apply `replace_pattern` in Collector |
| Histogram with default bucket boundaries | Default millisecond buckets are meaningless for second-scale requests | Set explicit bucket boundaries per instrument matching the expected value range |

---

## Cross-References

| Topic | File |
|-------|------|
| SLO policy, error budget, burn-rate alerts | `../boundaries/slo.md` |
| Full FinOps / unit economics | `cost.md` |
| Cardinality guardrails and remediation | `../meta-observability.md §Section C` |
| Collector two-tier topology (DaemonSet + gateway) | `../transport/collector-topology.md` |
| OTel spec and semconv stability tiers | `../standards.md` |
| Anti-patterns (Section B) | `../anti-patterns.md` |
| Vendor TSDB selection | `../vendor-categories.md §TSDB / Long-term Metrics` |
| Full matrix coverage (all signals) | `../matrix.md` |
| Logs signal | `logs.md` |
| Traces signal | `traces.md` |
| Profiles signal | `profiles.md` |
