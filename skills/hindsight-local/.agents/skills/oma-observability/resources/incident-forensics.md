---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# Incident Forensics Playbook

## 1. Purpose

**Goal**: given a production incident, locate root cause across 6 dimensions; code, service, layer, host, region, infra; in under 15 minutes, using the MELT+P+cost+audit+privacy signals defined across this skill tree.

The 6-dimension narrowing flow ties every signal file together through one executable playbook. No signal exists in isolation; correlation across signals is what separates diagnosis from guesswork.

**In scope**: any incident surfaced by an alert, a user report, or an anomaly detected in metrics, traces, or logs.  
**Out of scope**: incident response workflow (on-call rotation, escalation, postmortem tooling); use PagerDuty, OpsGenie, or Grafana OnCall for those.

Cross-skill entry point: `oma-debug` invokes this playbook on failure, pulling traces and logs by `request_id`. See the design document Integration table.

---

## 2. Minimum Required Attributes (MRA)

Every span, log record, and metric data point MUST carry the attributes below before reaching production. Missing attributes break the narrowing flow at the step that depends on them.

Semconv stability tiers follow `resources/standards.md §3`. Attributes marked Stable are safe for production SLOs and alerting. Attributes marked Development must not be used as SLO inputs.

### 2.1 Resource attributes: every signal

These attributes identify the workload that emitted the signal. They must be set on the OTel Resource, not on individual spans or log records.

| Attribute | Semconv group | Stability | Example value |
|-----------|--------------|-----------|---------------|
| `service.name` | `service.*` | Stable | `payment-service` |
| `service.namespace` | `service.*` | Stable | `checkout` |
| `service.version` | `service.*` | Stable | `v2.4.1` |
| `deployment.environment` | `deployment.*` | Stable | `prod` |
| `cloud.provider` | `cloud.*` | Stable | `aws` |
| `cloud.region` | `cloud.*` | Stable | `ap-northeast-2` |
| `cloud.availability_zone` | `cloud.*` | Stable | `ap-northeast-2a` |
| `host.id` | `host.*` | Stable | `i-0a1b2c3d4e5f` |
| `k8s.pod.name` | `k8s.*` | Stable | `payment-7d9f4b-xk2rp` |
| `k8s.node.name` | `k8s.*` | Stable | `ip-10-0-1-45.ec2.internal` |
| `k8s.cluster.name` | `k8s.*` | Stable | `prod-ap2-cluster` |
| `container.id` | `container.*` | Stable | `a3f7c2d1e9b0...` (first 12 chars) |

Note: for non-Kubernetes hosts use `host.id`; for Kubernetes workloads set both `k8s.pod.name` and `k8s.node.name`. `host.id` is not redundant; it maps to the underlying EC2/GCE instance for host-level pivot.

### 2.2 Error span additional attributes

Error spans (those where `status.code = ERROR`) MUST additionally carry code and exception attributes to enable the Code dimension pivot.

| Attribute | Semconv group | Stability | Example value |
|-----------|--------------|-----------|---------------|
| `code.function` | `code.*` | Stable | `processPayment` |
| `code.filepath` | `code.*` | Stable | `src/payments/processor.ts` |
| `code.lineno` | `code.*` | Stable | `287` |
| `exception.type` | `exception.*` | Stable | `TimeoutException` |
| `exception.message` | `exception.*` | Stable | `Redis pool exhausted after 5000ms` |
| `exception.stacktrace` | `exception.*` | Stable | full stack trace string |

Set these via the OTel SDK `span.recordException(e)` call, which populates all three `exception.*` attributes atomically.

### 2.3 Structured log mandatory fields

Every structured log record MUST carry these correlation keys. Without `trace_id`, log-trace join is impossible. Without `request_id`, user-support correlation is impossible.

| Field | Source | Example value |
|-------|--------|---------------|
| `trace_id` | OTel SDK (current span context) | `4bf92f3577b34da6a3ce929d0e0e4736` |
| `span_id` | OTel SDK (current span context) | `00f067aa0ba902b7` |
| `request_id` | `x-request-id` header (injected at API gateway) | `req_7Kp2mNxQr` |
| `tenant.id` | W3C Baggage propagation (if multi-tenant) | `tnt_A9cBx3` |

`tenant.id` is required only for multi-tenant services. Propagate via W3C Baggage per `resources/standards.md §W3C Baggage`. Do NOT carry user email or session tokens in baggage; see anti-patterns below.

### 2.4 Propagation requirement

W3C `traceparent` MUST propagate through every outbound HTTP and gRPC call. Stripping it silently is anti-pattern #1 in `resources/anti-patterns.md`. Verify propagation at every service boundary by checking that `trace_id` is consistent across services in a single request's log stream.

---

## 3. Six-Dimension Narrowing Flow

Execute the steps in order. Each step narrows the blast radius before the next. Do not skip steps; jumping to Code before confirming Region wastes time on red herrings.

**Time budget**: 15 minutes total. Suggested split: symptom capture (1 min), trace_id acquisition (2 min), dimension pivots (8 min), cross-signal validation (3 min), release correlation + action (1 min).

### Step 1: Symptom capture

Collect the following before touching any query tool:

- **Time window**: incident start (± 2 min buffer) and current time.
- **Initial clue**: which of these does the reporter provide?
  - `trace_id`: fastest path; jump directly to Step 3.
  - `request_id`: search logs: `request_id = <value>` → extract `trace_id`.
  - User/tenant identifier: search logs: `tenant.id = <value>` in the time window.
  - Alert only: proceed to Step 2 via metrics.
- **Symptom class**: error rate spike, latency spike, OOM/crash, or silent data corruption?

### Step 2: Acquire trace_id

If no `trace_id` was provided in Step 1, acquire one from metrics via exemplar:

1. Open the alerting metric (e.g., `http_server_request_duration_seconds` p99 or `http_requests_total` error rate) for the incident time window.
2. Locate a data point near the spike peak. Most modern metric backends (Grafana Mimir, Prometheus with exemplars, Datadog APM metrics) attach a trace exemplar to high-value data points.
3. Click the exemplar → copy the `trace_id`.
4. If the backend does not expose exemplars, filter error logs by `deployment.environment = prod` and the time window, then extract `trace_id` from the first matching log record.

### Step 3: Narrow by dimension (coarse to fine)

With `trace_id` in hand, execute pivots in this order:

#### 3a. Region / Infra (`cloud.region`, `k8s.cluster.name`)

Filter the trace or log stream by region first. Multi-region incidents are rarer than single-region ones; confirming scope is cheap.

- Query: filter spans or logs where `cloud.region != ap-northeast-2` (or whichever region the alert fired for). If traces from other regions are healthy, the blast radius is single-region.
- If the issue is infra-wide (all regions): pivot to `k8s.cluster.name` to check whether a specific cluster is affected.

#### 3b. Server (`host.id`, `k8s.pod.name`, `k8s.node.name`)

Within the confirmed region, check whether the failure is pod-wide or node-wide.

- A single pod failing repeatedly suggests a pod-level issue (OOM, file descriptor leak, bad canary rollout to that pod).
- Multiple pods on the same `k8s.node.name` suggests a node-level issue (disk pressure, noisy neighbor, kernel panic).
- Uniformly distributed across nodes suggests a service-wide or downstream issue.

#### 3c. Service (`service.name`, `service.version`)

With the server scope established, identify the service boundary where errors originate.

- Inspect the trace waterfall: find the earliest span where `status.code = ERROR`.
- Record `service.name` and `service.version` from that span's resource attributes.
- If `service.version` changed recently (see Step 5), this is the primary suspect.

#### 3d. Layer (`span.kind`, layer classification)

Classify the failing span's network layer using `span.kind` and span name patterns:

| `span.kind` value | Layer | Typical next pivot |
|---|---|---|
| `SERVER` or `CLIENT` with HTTP attributes | L7 application | code.function |
| `CLIENT` with `db.*` attributes | L7 DB call | traces.md DB patterns |
| `INTERNAL` with `messaging.*` | L7 messaging | check DLQ |
| eBPF-sourced span (Beyla/Pixie) | L4 transport | TCP retransmit metrics |
| Envoy/Istio proxy span | mesh | layers/mesh.md |

For L3/L4 root causes (PMTUD black hole, TCP retransmit storm), the L7 error manifests as a connection timeout. The trace will show a CLIENT span with no child spans and a `net.peer.name` pointing to the downstream. Pivot to L3/L4 metrics from there.

#### 3e. Code (`code.function`, `code.filepath`, `code.lineno`, `exception.stacktrace`)

On the failing span, read the error attributes set by MRA §2.2:

- `exception.type` and `exception.message` give the immediate cause.
- `exception.stacktrace` gives the call chain.
- `code.function` + `code.filepath` + `code.lineno` identify the exact source location without needing to search the codebase.

Cross-check: if `exception.stacktrace` points to a library call (e.g., a Redis client timeout), the root cause is likely downstream resource exhaustion, not a code bug; continue to cross-signal validation.

### Step 4: Cross-signal validation

With the failing span identified, validate the hypothesis across all available signals. Correlation is mandatory; without it, you may fix the wrong thing.

| Signal | What to check | File reference |
|--------|--------------|----------------|
| metrics | Error rate and latency p99 trend at the confirmed `service.name` + `cloud.region`. Does the scale match the alert? | `signals/metrics.md` |
| traces | Full trace waterfall: where does latency accumulate? Where does `status.code` first become ERROR? | `signals/traces.md` |
| logs | Filter by `trace_id` across all services. Look for WARNING or ERROR logs from downstream services within ±5 seconds of the failing span's start time. | `signals/logs.md` |
| profiles | If available (Parca/Pyroscope): pull the CPU flame graph for the pod in the incident window. Look for unexpectedly hot functions or heap growth. | `signals/profiles.md` |

All four signals should converge on the same root cause. If they diverge, re-examine the time window alignment (NTP drift is a common cause; see anti-patterns below).

### Step 5: Release correlation

Compare the incident start time to recent deployments of the suspected `service.name`:

| Window | Confidence of correlation |
|--------|--------------------------|
| `service.version` changed ≤ 30 minutes before incident | High; treat as primary hypothesis |
| ≤ 2 hours before incident | Medium; check changelog for risky changes |
| ≤ 24 hours before incident | Low; consider, but rule out other causes first |

Release markers must be present as structured log events or metric annotations carrying `service.version`, `deployment.environment`, and the deployment SHA (propagated via `oma-scm`). Without release markers, this step is blind; see anti-patterns below.

### Step 6: Recover or rollback

Based on Steps 1-5:

- **Root cause identified, fix available**: deploy hotfix → verify metric recovery in the same time window.
- **Root cause identified, fix not immediate**: rollback to previous `service.version` → verify metric recovery.
- **Root cause identified in downstream service**: escalate to owning team with the `trace_id`, the failing span attributes, and the cross-signal evidence bundle.
- **Root cause unresolved after 15 minutes**: escalate with the narrowed blast radius (region/host/service confirmed) and all evidence collected so far. Do not continue solo investigation beyond this threshold without escalation.

---

## 4. Vendor-Specific Query Examples

Given a `trace_id` of `4bf92f3577b34da6a3ce929d0e0e4736`, the equivalent query in each backend:

| Vendor | Interface | Query syntax |
|--------|-----------|-------------|
| Honeycomb | Query Builder | `trace:<id>` in the trace panel |
| Datadog APM | APM / Logs | `@trace_id:4bf92f3577b34da6a3ce929d0e0e4736` |
| Grafana Tempo | TraceQL | `{ trace:id = "4bf92f3577b34da6a3ce929d0e0e4736" }` |
| Jaeger | HTTP API | `GET /api/traces/4bf92f3577b34da6a3ce929d0e0e4736` |
| Sentry | Search | `trace_id:4bf92f3577b34da6a3ce929d0e0e4736` in Issues or Performance |
| SigNoz / OSS OTel | UI filter | trace_id filter field in Traces explorer |
| Elastic APM | Kibana | `trace.id: "4bf92f3577b34da6a3ce929d0e0e4736"` in Discover or APM |

```
# Honeycomb query builder (BubbleUp / Trace view)
trace:4bf92f3577b34da6a3ce929d0e0e4736

# Datadog logs query (also works in APM trace search)
@trace_id:4bf92f3577b34da6a3ce929d0e0e4736

# Grafana Tempo — TraceQL
{ trace:id = "4bf92f3577b34da6a3ce929d0e0e4736" }

# Jaeger HTTP API (replace <host> with your Jaeger query host)
curl http://<host>:16686/api/traces/4bf92f3577b34da6a3ce929d0e0e4736

# Elastic (KQL in Discover or APM)
trace.id: "4bf92f3577b34da6a3ce929d0e0e4736"
```

For metric-to-trace pivot via exemplars (Step 2):

```
# Grafana Mimir / Prometheus with exemplars — PromQL
histogram_quantile(0.99,
  sum by (le, service_name) (
    rate(http_server_request_duration_seconds_bucket{
      deployment_environment="prod",
      cloud_region="ap-northeast-2"
    }[5m])
  )
)
# Click a data point spike → exemplar panel shows trace_id link
```

```
# Datadog — metric to trace pivot
# In a metric graph, click a spike → "View related traces" opens APM with
# automatic time filter and service filter applied
```

---

## 5. Walkthrough Scenarios

### Scenario A: ap-northeast-2 payment service 5xx spike

Alert fires: `http_requests_total{status=~"5..", service_name="payment-service", cloud_region="ap-northeast-2"}` error rate crosses 5% burn-rate threshold.

1. **Symptom capture**: time window 14:22-14:35 UTC. Alert only; no trace_id provided. Symptom class: error rate spike.
2. **Acquire trace_id**: open the alerting metric in Grafana Mimir. Click a spike data point at 14:24. Exemplar attached: `trace_id = 9a3f1c8e2b7d4e50a1c3f9d2b8e7a4c0`.
3. **Region pivot**: filter spans by `cloud.region = ap-northeast-2`. Spans from `ap-southeast-1` are healthy. Blast radius: single region.
4. **Server pivot**: error spans distributed across 3 of 6 pods, all on `k8s.node.name = ip-10-0-2-11.ec2.internal`. Potential node-level issue; note but continue to service pivot first.
5. **Service pivot**: earliest ERROR span is `service.name = payment-service`, `service.version = v2.4.1`. Downstream span to `redis-cache` shows CLIENT timeout with no response.
6. **Layer pivot**: `span.kind = CLIENT`, `db.system = redis`. Layer: L7 DB call. No L3/L4 trace artifacts; this is an application-layer Redis client timeout.
7. **Code pivot**: `exception.type = TimeoutException`, `exception.message = "Redis pool exhausted after 5000ms"`, `code.function = processPayment`, `code.filepath = src/payments/processor.ts`, `code.lineno = 287`.
8. **Cross-signal validation**:
   - metrics: Redis connection pool saturation metric (`redis_pool_active_connections / redis_pool_max_connections`) at 100% from 14:21 UTC.
   - logs: filter `trace_id = 9a3f1c8e2b7d4e50a1c3f9d2b8e7a4c0`. Log at 14:22: `WARN: connection pool at capacity (256/256)` from `payment-service`.
   - traces: 8 concurrent `processPayment` spans all blocking on Redis CLIENT span; pool exhaustion pattern.
   - profiles: not available for this service (profiling not yet instrumented).
9. **Release correlation**: `service.version = v2.4.1` deployed at 13:57 UTC; 25 minutes before incident. Release marker log entry found. Changelog shows connection pool size reduced from 512 to 256 in this version.
10. **Action**: rollback to `v2.4.0`. Error rate drops to baseline within 3 minutes. Hotfix: restore pool size to 512, add `redis_pool_wait_duration_seconds` histogram alert.

### Scenario B: single tenant p99 latency spike

Alert fires: p99 latency SLO burn-rate for `api-gateway` crosses 2x threshold. Reported by tenant support: `tenant.id = tnt_A9cBx3` experiencing slow responses. All other tenants are healthy.

1. **Symptom capture**: time window 09:15-09:45 UTC. Tenant identifier provided: `tnt_A9cBx3`. Symptom class: latency spike, single tenant.
2. **Acquire trace_id**: search logs for `tenant.id = tnt_A9cBx3` in the window. First matching record: `trace_id = c7b2a1f4e8d3c9b0a2e4f1d7c3b8a5e2`.
3. **Region pivot**: `cloud.region = us-east-1` (this tenant's home region). Spans from other regions not affected. Blast radius: single region, single tenant.
4. **Server pivot**: error spans distributed evenly across all pods. Not a node-level issue.
5. **Service pivot**: latency spike originates in `service.name = tenant-config-service`, `service.version = v1.8.0` (unchanged for 2 weeks; not a release issue).
6. **Layer pivot**: `span.kind = CLIENT` with `db.system = postgresql`. Layer: L7 DB call.
7. **Code pivot**: no ERROR status; latency only. `span.duration = 812ms`. Span name: `SELECT tenant_config`. No exception attributes because the query succeeds, just slowly.
8. **Cross-signal validation**:
   - metrics: `db_client_operation_duration_seconds` p99 for `tenant-config-service` at 800ms. Other services normal.
   - traces: 47 sequential DB CLIENT spans within a single parent SERVER span. Each takes ~17ms. Total: 812ms. Classic N+1 query pattern.
   - logs: `trace_id` filter shows no errors. Log at request start: `tenant.id = tnt_A9cBx3, config_keys_requested = 47`.
   - profiles: Parca flame graph for `tenant-config-service` in the window. `tenantConfigLookup` accounts for 810ms of wall time. Call graph shows a loop issuing individual `findByKey` calls instead of a batch `findByKeys`.
9. **Release correlation**: no recent deployment. Root cause: N+1 query in `tenantConfigLookup` triggered by `tnt_A9cBx3` having 47 config keys (others have 5-12). The code path was never slow enough to surface with smaller tenants.
10. **Action**: fix `tenantConfigLookup` to use batch `SELECT ... WHERE key = ANY($1)`. Add a `db_query_count_per_request` histogram alert with threshold > 10 per span. Deploy and verify p99 for `tnt_A9cBx3` drops to < 50ms.

Reference: `boundaries/multi-tenant.md` for per-tenant pivot strategy; `signals/profiles.md` for Parca/Pyroscope flamegraph correlation.

### Scenario C: OOM crash after deployment

Alert fires: pod restart storm on `rendering-service` pods. `k8s.pod.name` shows 3 of 12 pods (25%) restarting with OOMKilled reason. The 3 affected pods are canary instances (Flagger progressive delivery).

1. **Symptom capture**: time window 16:40-17:00 UTC. Alert only. Symptom class: OOM/crash on canary pods.
2. **Acquire trace_id**: OOMKilled pods produce no useful traces after crash. Pivot to release event first.
3. **Release correlation** (Step 5 first, by exception): release marker log entry at 16:35 UTC: `service.name = rendering-service`, `service.version = v3.1.0`, `deployment.strategy = canary`. 5 minutes before OOM storm; high confidence correlation.
4. **Region/Server pivot**: `cloud.region = eu-west-1`, `k8s.node.name` spread across 3 different nodes; not a node issue. All 3 OOM pods are on `service.version = v3.1.0`. Stable pods on `v3.0.9` are healthy.
5. **Layer pivot**: OOM is a host/runtime event, not an application span. No span to inspect. Pivot directly to profiles.
6. **Cross-signal validation**:
   - metrics: `container_memory_working_set_bytes` for `v3.1.0` pods ramps from 200MB to 1.5GB (container limit) over 18 minutes. `v3.0.9` pods stable at 200MB.
   - traces: no useful traces from crashed pods. Pre-crash traces show `renderTemplate` CLIENT spans with duration increasing over time.
   - logs: pre-crash logs from `v3.1.0` pods: `WARN: template cache size = 412MB at 16:52`. Not present in `v3.0.9` pods.
   - profiles: Pyroscope heap diff between `v3.1.0` (pod alive, approaching limit) and `v3.0.9`. New allocation in `renderTemplate` retaining 400MB; a template object cache introduced in `v3.1.0` is not evicting entries.
7. **Flagger auto-rollback check**: Flagger's canary analysis `successRate` threshold is 99%. OOMKilled pods were restarting before requests failed; success rate stayed above threshold. Flagger did NOT auto-rollback.
8. **Action**: manual rollback of canary to `v3.0.9` via `kubectl argo rollouts undo rendering-service`. OOM storm stops within 2 minutes. Hotfix: add LRU eviction to template cache with `maxSize = 50MB`. Adjust Flagger analysis to include memory saturation metric as a custom metric gate.

Reference: `boundaries/release.md` for Flagger/Argo canary analysis configuration; `signals/profiles.md` for Pyroscope heap diff workflow; `layers/L7-application/crash-analytics.md` for mobile/native crash-specific forensics.

---

## 6. Integration Points

This file does not operate in isolation. The following skill files provide the detailed implementation guidance for each step of the narrowing flow.

| Step / Concern | Cross-reference |
|---|---|
| Per-tenant pivot (Scenario B, `tenant.id` baggage) | `boundaries/multi-tenant.md`; 4-tier isolation and per-tenant collector routing |
| Mesh trace continuity (mesh span injection, zero-code) | `layers/mesh.md`; Envoy span propagation, baggage scrubbing at ingress |
| Flamegraph acquisition (Scenario B profiles, Scenario C heap diff) | `signals/profiles.md`; Parca/Pyroscope per-tenant labeling, Pyroscope diff view |
| Canary/rollback gates (Scenario C Flagger analysis) | `boundaries/release.md`; Flagger + Argo Rollouts custom metric gates, GitOps |
| Mobile/native crash OOM forensics | `layers/L7-application/crash-analytics.md`; CFR, symbolication, crash-linked release tracking |
| Release marker injection | `boundaries/release.md`; deployment SHA → `service.version` + marker event via `oma-scm` |
| Semconv stability for MRA attributes | `resources/standards.md §3`; stability tier table |
| Full signal-layer-boundary coverage | `resources/matrix.md`; 112-cell coverage map |

---

## 7. Anti-patterns

Violating these patterns makes the narrowing flow impossible or misleading. Full list lives in `resources/anti-patterns.md`.

| Anti-pattern | Consequence | Dimension blocked |
|---|---|---|
| Missing `trace_id` in structured logs | Log-trace join impossible; Step 4 cross-signal validation fails | All |
| `request_id` not injected at API gateway | User-support correlation impossible; Step 1 has no starting point | All |
| No release markers in log/metric stream | Step 5 release correlation is blind; rollback decision is guesswork | Service / release |
| NTP drift unmonitored on nodes | Waterfall chart timestamps are wrong; parent-before-child ordering unreliable; Step 3 pivots mislead | Layer / Code |
| PII in W3C Baggage across trust boundaries | `tenant.id` is acceptable; user email or session token in baggage is a GDPR/PIPA violation that becomes evidence in incident investigation records | All (compliance risk) |
| `user.id` or `user.email` as a metric label | High-cardinality metric explosion + PII in metric storage; cardinality guardrails in `resources/meta-observability.md` | Service |
| Canary Flagger analysis missing memory saturation gate | OOMKilled pods do not fail the success rate threshold; auto-rollback never triggers (Scenario C) | Service / infra |

Normative anti-patterns reference: `resources/anti-patterns.md §Section Z` (items 17, 18 directly referenced above).
