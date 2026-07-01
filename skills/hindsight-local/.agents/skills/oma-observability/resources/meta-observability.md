---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# Meta-Observability

## Why Meta-Observability

"Observing the observer"; if the OTel Collector is silently dropping 10% of traces, every SLO
dashboard, every alerting rule, and every incident forensics query is built on incomplete data.
You will not know the pipeline is degraded unless you instrument the pipeline itself.

Other files in this skill assume a reliable telemetry pipeline. This file forces you to verify
that assumption. Four disciplines close the gap:

| Discipline | Risk if ignored | Section |
|---|---|---|
| Pipeline self-health | Silent data loss, metric gaps, memory crashes | A |
| Clock skew / NTP | Waterfall charts lie; phantom race conditions chased | B |
| Cardinality guardrails | TSDB storage explosion, query latency, vendor bill spikes | C |
| Retention matrix | Compliance violations, over-spend on raw data, data unavailable at audit | D |

Cross-cutting failure modes and recovery paths are in Section E. Alert/dashboard scaffolding
that feeds `resources/observability-as-code.md` is in Section F.

---

## Section A: Pipeline Self-Health

### A1. OTel Collector Self-Metrics

The Collector exposes its own telemetry via a Prometheus scrape endpoint (default `:8888/metrics`).
Enable it explicitly:

```yaml
# otelcol-config.yaml — telemetry block
service:
  telemetry:
    metrics:
      level: detailed          # normal | detailed | none
      address: 0.0.0.0:8888
    logs:
      level: info
```

Scrape this endpoint from a separate Prometheus instance (or a second Collector) so that
Collector failures do not destroy their own observability.

### A2. Key otelcol_* Metrics

| Metric | What it measures | Alert threshold |
|---|---|---|
| `otelcol_receiver_accepted_spans` | Spans accepted from upstream | Baseline drop > 10% |
| `otelcol_receiver_refused_spans` | Spans refused (parse failure, queue full) | > 0 sustained for 2m |
| `otelcol_exporter_sent_spans` | Spans successfully shipped to backend | Baseline drop > 10% |
| `otelcol_exporter_send_failed_spans` | Export failures (network, auth, rate-limit) | > 1% of sent for 5m |
| `otelcol_processor_queued_retry_send_queue_length` | Backpressure queue depth | > 80% of queue capacity |
| `otelcol_process_runtime_heap_alloc_bytes` | Collector heap usage | > 75% of container limit |
| `otelcol_processor_memory_limiter_refused_spans` | Forced drops due to memory limit | > 0 |
| `otelcol_processor_memory_limiter_refused_metric_points` | Same, for metrics | > 0 |
| `otelcol_processor_memory_limiter_refused_log_records` | Same, for logs | > 0 |

The same `*_accepted_*`, `*_refused_*`, `*_sent_*`, and `*_send_failed_*` counter families exist
for metric points (`metric_points`) and log records (`log_records`). Apply identical thresholds.

### A3. Memory Limiter Processor (Required)

Add `memory_limiter` as the first processor in every pipeline to prevent OOM crashes:

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_percentage: 75          # hard limit: refuse new data above 75% heap
    spike_limit_percentage: 20    # headroom for burst above limit_percentage

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, ...]
      exporters: [otlp]
    metrics:
      receivers: [otlp, prometheus]
      processors: [memory_limiter, batch, ...]
      exporters: [prometheusremotewrite]
    logs:
      receivers: [otlp, fluentforward]
      processors: [memory_limiter, batch, ...]
      exporters: [loki]
```

### A4. Self-Scrape Configuration

Scrape Collector self-metrics into the same metrics pipeline so they land in the same TSDB
alongside application metrics:

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: otelcol-self
          scrape_interval: 15s
          static_configs:
            - targets: ["localhost:8888"]
          metric_relabel_configs:
            - source_labels: [__name__]
              regex: otelcol_.*
              action: keep
```

### A5. Agent Self-Resource Consumption

| Agent | Typical RAM | Typical CPU | Source |
|---|---|---|---|
| Fluent Bit | 5–15 MB | < 1% (idle), < 5% (burst) | Fluent Bit 2.x benchmarks |
| OTel Collector (DaemonSet) | 30–100 MB | < 5% (moderate load) | VictoriaMetrics 2026 benchmark |
| OTel Collector (gateway) | 100–500 MB | scales with throughput | VictoriaMetrics 2026 benchmark |

Collect host-level CPU and memory for agent processes via `hostmetrics` receiver:

```yaml
receivers:
  hostmetrics:
    collection_interval: 30s
    scrapers:
      process:
        include:
          names: ["otelcol", "fluent-bit"]
          match_type: regexp
        mute_process_name_error: true
```

Fluent Bit exposes its own Prometheus metrics at `:2020/api/v1/metrics/prometheus`:

- `fluentbit_input_records_total`: records ingested per input plugin
- `fluentbit_output_proc_records_total`: records successfully processed per output plugin
- `fluentbit_output_errors_total`: output failures

### A6. Golden Signal: End-to-End Delivery Ratio

The single most important pipeline health metric:

```promql
# Delivery ratio for traces (1.0 = 100% delivered)
sum(rate(otelcol_exporter_sent_spans[5m]))
  /
sum(rate(otelcol_receiver_accepted_spans[5m]))
```

Alert when this ratio drops below 0.99 for 5 minutes (see Section F, Alert 1).

---

## Section B: Clock Skew & NTP Discipline

### B1. Why Trace Waterfalls Lie

Distributed traces use wall-clock timestamps from the node where each span is recorded.
If two nodes have diverging clocks, the waterfall view in any tracing backend will show:

- A child span appearing to start before its parent span started.
- A child span ending after its parent span ended (`child.end_time > parent.end_time`).
- Negative durations on synthetic computed spans.

These are not code bugs. They are clock-drift artifacts. Engineers chase phantom race conditions
or assume broken instrumentation, losing hours of incident investigation time.

Typical NTP accuracy on well-connected cloud VMs: < 50 ms. On baremetal with good NTP: < 10 ms.
PTP (IEEE 1588) achieves sub-millisecond accuracy for financial and telco workloads.

Reference: `resources/standards.md §Clock Discipline` for the span timestamp validation rule.

### B2. NTP Requirements

All host VMs, container hosts, and Kubernetes nodes MUST run a time synchronization daemon:

- **Linux (systemd)**: `systemd-timesyncd` (lightweight) or `chrony` (recommended for accuracy)
- **Kubernetes**: Node time sync is the Linux host's responsibility. The kubelet and containers
  inherit the host clock. The path is: hypervisor NTP → node OS → container runtime → container.
  Confirm this chain with your cloud provider's documentation.

Cloud-provider time sources:

| Cloud | NTP endpoint | Notes |
|---|---|---|
| AWS | `169.254.169.123` (Amazon Time Sync Service) | PTP-backed, link-local; add to `chrony.conf`; verify with `chronyc sources -v` |
| GCP | `metadata.google.internal` | Internal hypervisor sync. `timedatectl show` only shows daemon state; use `chronyc sources -v` for actual offset |
| Azure | Hyper-V IC timesync primary; `time.windows.com` external NTP is fallback only | Azure Linux VMs with IC tools use host time. External NTP as a parallel peer can conflict; configure as fallback. Docs: <https://learn.microsoft.com/azure/virtual-machines/linux/time-sync> |

For sub-millisecond requirements (financial trading, high-frequency event processing):
use **PTP (IEEE 1588)** with hardware timestamping. Cloud support varies; AWS supports PTP on
Nitro instances via the Amazon Time Sync Service; verify availability before committing.

### B3. Span-Level Drift Detection

Flag spans where the child-ends-after-parent invariant is violated:

```promql
# PromQL: no native span-level metric; implement as a Collector transform rule
# In otelcol, use the transform processor to emit a counter on violation:
# counter: otelcol_span_clock_violation_total{service="...", direction="child_outlives_parent"}
```

OTel Collector `transform` processor example (append to spans pipeline):

```yaml
processors:
  transform/clock_check:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - set(attributes["clock.violation"], true)
              where IsRootSpan() == false
              and end_time > parent_end_time   # pseudo-field; requires OTTL extension
```

Until native OTTL parent-time access is stable, implement this check in your tracing backend's
query layer (e.g., Tempo TraceQL, Jaeger query API) as a periodic job that emits a metric.

### B4. chrony Offset Metric

Emit `node.clock.drift_ms` from each host using a cron job or node exporter textfile collector:

```bash
#!/bin/sh
# /etc/cron.d/chrony-offset — runs every minute
OFFSET=$(chronyc tracking | awk '/System time/ {gsub(/[^0-9.-]/, "", $4); print $4 * 1000}')
echo "node_clock_drift_ms $OFFSET" > /var/lib/node_exporter/textfile/chrony_offset.prom
```

Alert rule (see Section F, Alert 3):

```promql
node_clock_drift_ms > 100
```

---

## Section C: Cardinality Guardrails

### C1. Why Cardinality Matters

Every unique label combination in a TSDB (Prometheus, VictoriaMetrics, Thanos, Mimir) creates a
separate time series. A metric with 3 labels each having 100 values creates 1,000,000 series.

Consequences of label explosion:
- TSDB storage grows quadratically with cardinality.
- Query latency increases: each query fans out across more series.
- Vendor bill spikes: SaaS TSDBs charge per active series.
- Ingestion falls behind; scrape intervals are missed.

### C2. Hard Rules

| Label | Rule | Reason |
|---|---|---|
| `user.id` | **NEVER** as metric label | Unbounded; one series per user |
| `request.id` | **NEVER** as metric label | One series per request; instant explosion |
| `trace.id` | **NEVER** as metric label | One series per trace |
| `user.email` | **NEVER** as metric label | PII + cardinality double violation |
| `tenant.id` | Cap at top-N (e.g., top-1000); overflow → label value `"other"` | Bounded set of known tenants |
| `endpoint` / `route` | Normalize high-cardinality routes: `/users/42` → `/users/_` | URL parameters are unbounded |
| `http.url` | **NEVER** raw; use `http.route` (normalized) | Query strings are unbounded |
| `error.message` | **NEVER** as label; use error type/code only | Free-text strings are unbounded |

Implement route normalization in the Collector `transform` processor:

```yaml
processors:
  transform/normalize_routes:
    error_mode: ignore
    metric_statements:
      - context: datapoint
        statements:
          - replace_pattern(attributes["http.route"], "/[0-9]+", "/_")
          - replace_pattern(attributes["http.route"], "/[0-9a-f-]{36}", "/_")  # UUID
```

### C3. Cardinality Measurement

Measure active series per metric name in Prometheus-compatible TSDBs:

```promql
# Count distinct series for a specific metric
count(http_request_duration_seconds_bucket) by (job)

# Top 10 highest-cardinality metrics in the entire TSDB
topk(10, count({__name__!=""}) by (__name__))
```

VictoriaMetrics exposes `/api/v1/status/tsdb` for cardinality breakdown by metric name
and label value; use this for bulk auditing.

OTel metric SDK cardinality limit (Development feature as of SDK 1.x):

```python
# Python SDK example
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import View

view = View(
    instrument_name="http_request_duration",
    attribute_keys={"http.method", "http.status_code", "http.route"},  # explicit allow-list
)
provider = MeterProvider(views=[view])
```

Explicitly allow-listing attributes is the most effective cardinality control.

### C4. Series Budget

Set a per-service series budget. The 5,000-series figure below is an illustrative starting baseline; high-throughput services legitimately operate at 20k-100k series. Calibrate to your service's traffic shape and TSDB ingestion cost.

```promql
# Alert: service approaching cardinality budget (example baseline: 5000 series/service)
count({job="my-service"}) > 4000
```

Cardinality explosion is an anti-pattern. See `resources/anti-patterns.md §Section B` for
the full cardinality anti-pattern list.

---

## Section D: Retention Matrix

Retention policy governs how long raw and aggregated data is kept, at what resolution,
and under what storage class. Failing to set explicit retention leads to either:
- Over-spend: keeping full-resolution data for years.
- Compliance violation: flushing audit logs before mandatory retention periods expire.

### D1. Unified Per-Signal Retention Policy

| Signal | Raw resolution | Aggregated / downsampled | Archive |
|---|---|---|---|
| Metrics | 15d full-res | 90d @ 5m resolution | 2y @ 1h resolution |
| Logs (operational) | 7d | 30d | 90d |
| Logs (audit; SOC2/ISO 27001) | 90d | 1y | **7y WORM** |
| Traces (sampled, tail-based) | 30d | N/A | N/A |
| Traces (full 100% sample) | 3d | N/A | N/A |
| Profiles | 14d | N/A | N/A |
| Events | 30d | 90d | N/A |

### D2. Rationale by Signal

**Metrics (15d / 90d / 2y)**: Full-resolution metrics are needed for hourly/daily incident
investigation (15d covers most post-incident reviews). Downsampled 5m aggregates cover quarterly
business reviews and SLO trend reporting (90d). 1h aggregates for 2y support year-over-year
capacity planning without raw-data storage cost.

**Logs (operational, 7d / 30d / 90d)**: Operational logs (app errors, access logs) are used
within days of generation. 30d aggregated (e.g., error-count rollups) covers sprint-level
incident retrospectives. 90d archive is a common compliance floor for operational data.

**Logs (audit, 90d / 1y / 7y WORM)**: SOC2 Type II requires audit evidence covering the audit
period (typically 1y). ISO 27001 Annex A.12.4 requires log retention that satisfies legal and
regulatory requirements. GDPR Article 17 right-to-erasure does not apply to audit logs where
retention is required by another legal obligation (recital 65). 7-year WORM aligns with
financial audit requirements (e.g., SOX) and is the safe upper bound.
Cross-ref: `signals/audit.md` for WORM immutability requirements and hash-chain tamper evidence.

**Traces (sampled, 30d)**: Sampled traces (tail-based, typically 1-10% of production traffic)
are the primary debugging artifact. 30d covers cross-sprint incident investigations. No
aggregation applies; traces are point-in-time artifacts, not time-series.

**Traces (full 100% sample, 3d)**: 100% sampling is expensive. Keep for 3 days only; long
enough to cover a release rollout + initial stability period. Use head-based or tail-based
sampling to reduce to 1-10% thereafter.

**Profiles (14d)**: Continuous profiling data (Parca, Pyroscope) is used for performance
regression detection immediately after a deploy and for on-call debugging. 14d covers
two sprint cycles. No aggregation standard exists for profiling data yet.

**Events (30d / 90d)**: Structured events (deployment markers, feature flag changes, SLO
breaches) are retained for 30d raw (covers post-release review). Aggregated event counts
(deploys per day, flag toggles per week) are retained for 90d to support quarterly reviews.

### D3. Storage Class Configuration Example (Loki)

```yaml
# loki-config.yaml — example retention per stream selector
# NOTE: table_manager was removed in Loki 3.x. For Loki 3+, configure retention
# via compactor only. The block below applies to Loki 2.8-2.9 legacy.

# Legacy (Loki 2.x): table_manager
table_manager:
  retention_deletes_enabled: true
  retention_period: 720h    # default 30d

# Per-stream retention via compactor (Loki 2.8+ and required on Loki 3.x)
compactor:
  retention_enabled: true
  retention_delete_delay: 2h

# Stream-level policy in ruler or via label matchers:
# {log_type="audit"} → 61320h (7y)
# {log_type="operational"} → 2160h (90d)
# {log_type="debug"} → 168h (7d)
```

Cross-ref: `signals/audit.md` for WORM immutable storage configuration (S3 Object Lock,
GCS Object Hold, Azure Immutable Blob Storage). Cross-ref: `signals/privacy.md` for
GDPR Article 5(1)(e) storage limitation (data not kept longer than necessary for the purpose).

---

## Section E: Pipeline Failure Modes

| Failure mode | Symptom | Remediation |
|---|---|---|
| Upstream receiver overflow | `otelcol_receiver_refused_*` rising; `send_queue_length` climbing | Add `memory_limiter` processor; scale gateway Collector replicas horizontally |
| Exporter backpressure | `otelcol_processor_queued_retry_send_queue_length` rising; `send_failed_*` incrementing | Check vendor rate limits; tune exporter `retry_on_failure` with exponential backoff; reduce batch size |
| Clock drift spike | Waterfall inversion; `node_clock_drift_ms > 100` alert fires | Run `chronyc makestep` to force immediate re-sync; for persistent drift, check NTP source reachability; escalate to PTP for sub-ms requirements |
| Cardinality bomb | TSDB ingestion lag; query timeouts; vendor bill alert | Identify offending metric with `topk(10, count({__name__!=""}) by (__name__))`, add `filter` processor to drop or `transform` to remove offending label |
| Memory OOM crash | Collector process exits; gap in all signals | `memory_limiter` prevents this if configured; if reached anyway, increase container memory limit or reduce `limit_percentage` |
| Audit log retention violation | Audit records purged before WORM period | Verify S3/GCS Object Lock policy; confirm `retention_period` in log backend matches 7y; cross-ref `signals/audit.md §WORM` |

### E1. Exporter Retry Configuration

```yaml
exporters:
  otlp:
    endpoint: backend:4317
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s    # give up after 5 minutes; prevents infinite queue growth
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 1000
```

### E2. Filter Processor for Cardinality Drop

```yaml
processors:
  filter/drop_high_cardinality:
    error_mode: ignore
    metrics:
      datapoint:
        - 'attributes["user.id"] != nil'      # drop any datapoint carrying user.id
        - 'attributes["request.id"] != nil'   # drop request.id label
```

Cross-ref: `resources/checklist.md §7 Recovery` for the full recovery procedure checklist.
Cross-ref: `resources/anti-patterns.md §Section C Pipeline` for pipeline anti-patterns.

---

## Section F: Alerts & Dashboards

### F1. Five Golden Meta-Observability Alerts

These five alerts MUST be in place before any other observability alerting is considered
reliable. Without them, you cannot trust your alerts.

**Alert 1; Pipeline delivery ratio below threshold**

```promql
# otelcol_pipeline_delivery_ratio — traces
(
  sum(rate(otelcol_exporter_sent_spans[5m]))
  /
  sum(rate(otelcol_receiver_accepted_spans[5m]))
) < 0.99
```

```yaml
# PrometheusRule CRD
- alert: OtelcolDeliveryRatioBelowThreshold
  expr: |
    (
      sum(rate(otelcol_exporter_sent_spans[5m]))
      /
      (sum(rate(otelcol_receiver_accepted_spans[5m])) > 0)
    ) < 0.99
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "OTel Collector trace delivery ratio below 99%"
    description: "{{ $value | humanizePercentage }} of accepted spans are being delivered. Check exporter errors and backpressure queue."
```

**Alert 2; Exporter send failures above 1%**

```promql
# Ratio of failed exports to sent (any signal)
(
  sum(rate(otelcol_exporter_send_failed_spans[5m]))
  /
  (sum(rate(otelcol_exporter_sent_spans[5m])) > 0)
) > 0.01
```

```yaml
- alert: OtelcolExporterSendFailed
  expr: |
    (
      sum(rate(otelcol_exporter_send_failed_spans[5m]))
      /
      (sum(rate(otelcol_exporter_sent_spans[5m])) > 0)
    ) > 0.01
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "OTel Collector exporter failures above 1%"
    description: "Check vendor rate limits, network connectivity, and exporter authentication."
```

**Alert 3; Node clock drift above 100 ms**

```yaml
- alert: NodeClockDriftHigh
  expr: node_clock_drift_ms > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Node clock drift exceeds 100ms on {{ $labels.instance }}"
    description: "Trace waterfall ordering may be unreliable. Run chronyc makestep or check NTP source."
```

**Alert 4; Service approaching cardinality budget**

```yaml
- alert: MetricCardinalityBudgetExceeded
  expr: count({job=~".+"}) by (job) > 4000
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.job }} exceeds 80% of 5000-series cardinality budget"
    description: "Identify high-cardinality labels with topk(10, count({job='{{ $labels.job }}'}) by (__name__)). Apply attribute filter."
```

**Alert 5; Audit log retention policy violation**

This alert cannot be expressed in PromQL alone; implement it as a scheduled policy check in
your compliance tooling or as a metric emitted by a retention audit job:

```yaml
- alert: AuditLogRetentionViolation
  expr: audit_log_retention_days{log_type="audit"} < 2555   # 7 years = 2555 days
  for: 1h
  labels:
    severity: critical
  annotations:
    summary: "Audit log retention is below the 7-year WORM requirement"
    description: "Verify S3 Object Lock / GCS Object Hold policy on audit log bucket. Cross-ref signals/audit.md."
```

### F2. Grafana Dashboard Blueprint

A meta-observability Grafana dashboard MUST include the following panels:

| Panel | Query summary | Visualization |
|---|---|---|
| Pipeline delivery ratio (traces) | `sent / accepted` rate | Stat (green > 99%, red < 99%) |
| Pipeline delivery ratio (metrics) | same for metric_points | Stat |
| Pipeline delivery ratio (logs) | same for log_records | Stat |
| Receiver accepted vs refused | `rate(accepted)` and `rate(refused)` | Time series |
| Exporter sent vs failed | `rate(sent)` and `rate(failed)` | Time series |
| Queue depth | `send_queue_length` per exporter | Gauge |
| Collector heap usage | `heap_alloc_bytes` vs limit | Gauge |
| Node clock drift (all hosts) | `node_clock_drift_ms` | Heatmap by instance |
| Top cardinality metrics | `topk(10, count(...) by (__name__))` | Table (auto-refresh 5m) |
| Fluent Bit input vs output records | `fluentbit_input_records_total` vs `fluentbit_output_proc_records_total` | Time series |

The full Jsonnet/YAML implementation of this dashboard belongs in `resources/observability-as-code.md`
(Grafana-as-code section). This section provides the blueprint; that file provides the code.

---

## Cross-References

| Topic | File |
|---|---|
| Clock discipline normative requirements | `resources/standards.md §Clock Discipline` |
| WORM immutable storage for audit logs | `signals/audit.md` |
| GDPR storage limitation (Art. 5(1)(e)) | `signals/privacy.md` |
| Cardinality anti-patterns (Section B) | `resources/anti-patterns.md` |
| Pipeline anti-patterns (Section C) | `resources/anti-patterns.md` |
| Recovery checklist (§7) | `resources/checklist.md` |
| Dashboard/alert as code | `resources/observability-as-code.md` |
| Two-tier Collector topology | `transport/collector-topology.md` |
| Tail-based sampling configuration | `transport/sampling-recipes.md` |
| SLO burn-rate alerts | `boundaries/slo.md` |
| Incident forensics (6-dimension localization) | `resources/incident-forensics.md` |

---

## Review and Maintenance

- **Review cadence**: quarterly, aligned with `resources/standards.md`.
- On OTel Collector minor version bump: verify `otelcol_*` metric names have not changed.
  Metric names are stable within a major version but have changed between 0.x and 1.x.
- On cloud provider NTP endpoint changes: update Section B2 table.
- On TSDB migration (e.g., Prometheus → VictoriaMetrics): verify cardinality query syntax
  in Section C3; VictoriaMetrics uses `/api/v1/status/tsdb` instead of PromQL cardinality queries.
- Owner: SysE primary (see design document Ownership & Quality Gates table).
