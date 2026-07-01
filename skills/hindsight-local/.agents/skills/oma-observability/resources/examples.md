# oma-observability Examples

> End-to-end walkthroughs demonstrating the skill's value.
> Each scenario ties multiple files together into a single executable story.

---

## Scenario 1: Greenfield: OSS Full-Stack on Kubernetes

**Situation:** Series A startup, new k8s cluster. Team wants OTel-native metrics + logs + traces + Grafana
dashboards at zero licensing cost.

**Intent:** `setup`

**Files referenced:** `vendor-categories.md §(a)`, `transport/collector-topology.md §2`, `observability-as-code.md §5`, `checklist.md`

### Walkthrough

1. Invoke `/oma-observability "set up OTel stack on k8s"`.
2. Intent classifier routes to `setup`.
3. `vendor-categories.md §(a) OSS Full-Stack` selects **Grafana LGTM+** (Mimir, Loki, Tempo, Grafana)
   for a team already familiar with Grafana dashboards.
4. `transport/collector-topology.md §2 Two-Tier Hybrid` determines the deployment pattern:
   DaemonSet agent per node + Deployment gateway.

**Agent CRD; DaemonSet (node-level collection):**

```yaml
apiVersion: opentelemetry.io/v1beta1
kind: OpenTelemetryCollector
metadata:
  name: otel-agent
  namespace: observability
spec:
  mode: daemonset
  config:
    receivers:
      hostmetrics:
        collection_interval: 30s
        scrapers: { cpu: {}, memory: {}, filesystem: {} }
      filelog:
        include: [/var/log/pods/*/*/*.log]
        include_file_path: true
      kubeletstats:
        collection_interval: 20s
        auth_type: serviceAccount
    processors:
      memory_limiter:            # mandatory: first processor in every pipeline
        limit_mib: 400
        check_interval: 1s
      k8sattributes:
        extract:
          metadata: [k8s.pod.name, k8s.namespace.name, k8s.node.name]
    exporters:
      otlp:
        endpoint: otel-gateway.observability.svc:4317
    service:
      pipelines:
        metrics:
          receivers: [hostmetrics, kubeletstats]
          processors: [memory_limiter, k8sattributes]
          exporters: [otlp]
        logs:
          receivers: [filelog]
          processors: [memory_limiter, k8sattributes]
          exporters: [otlp]
```

**Gateway CRD; Deployment (batching, `mode: deployment`, 2 replicas):**

```yaml
# Key fields only — full config in observability-as-code.md §5
spec:
  mode: deployment
  replicas: 2
  config:
    receivers:
      otlp: { protocols: { grpc: { endpoint: 0.0.0.0:4317 } } }
    processors:
      memory_limiter: { check_interval: 1s, limit_percentage: 75 }
      batch: {}
    exporters:
      otlphttp/tempo: { endpoint: http://tempo.observability.svc:4418 }
      prometheusremotewrite: { endpoint: http://mimir.observability.svc:9009/api/v1/push }
      loki: { endpoint: http://loki.observability.svc:3100/loki/api/v1/push }
    service:
      pipelines:
        traces: { receivers: [otlp], processors: [memory_limiter, batch], exporters: [otlphttp/tempo] }
        metrics: { receivers: [otlp], processors: [memory_limiter, batch], exporters: [prometheusremotewrite] }
        logs: { receivers: [otlp], processors: [memory_limiter, batch], exporters: [loki] }
```

5. Collector config is committed to git and applied via Argo CD app-of-apps per `observability-as-code.md §7`.
6. Validation checklist (from `checklist.md §1, §3`):
   - [ ] `memory_limiter` is the first processor in every pipeline
   - [ ] NTP drift < 100 ms on all nodes (`chronyc tracking`)
   - [ ] Cardinality budget set before first data ingest
   - [ ] Gateway has `≥ 2` replicas with PodDisruptionBudget

**Outcome:** 9 pods, 2 services, first trace visible in Grafana Tempo within 15 minutes of `kubectl apply`.

---

## Scenario 2: Incident Forensics: Payment Service 5xx Spike

**Situation:** "ap-northeast-2 payment service 5xx spike at 14:20 UTC"; alert fires with no `trace_id`
provided. On-call SRE must localize root cause in under 15 minutes.

**Intent:** `investigate`

**Files referenced:** `incident-forensics.md §3–§5 Scenario A`, `boundaries/slo.md §7`, `signals/traces.md`, `signals/logs.md §7`, `boundaries/release.md`

### Walkthrough

1. **Alert source:** multi-window burn-rate alert from `boundaries/slo.md §7` fires; 2 % budget
   consumed in 1 h at 14.4× rate. `severity: critical`.
2. Invoke `/oma-observability --investigate "5xx spike in ap-northeast-2"`.
3. Router routes to `incident-forensics.md §3 Six-Dimension Narrowing Flow`.

**Step 1; Acquire `trace_id` via metric exemplar (`incident-forensics.md §2`):**

```promql
histogram_quantile(0.99,
  sum by (le, service_name) (
    rate(http_server_request_duration_seconds_bucket{
      deployment_environment="prod",
      cloud_region="ap-northeast-2"
    }[5m])
  )
)
# Click a spike data point → Grafana Mimir exemplar panel shows trace_id link
```

Exemplar retrieved: `trace_id = 9a3f1c8e2b7d4e50a1c3f9d2b8e7a4c0`.

**Step 2; Six-dimension narrowing:**

| Dimension | Finding |
|-----------|---------|
| Region | `cloud.region = ap-northeast-2` only. `ap-southeast-1` healthy. Single-region blast radius. |
| Server | Errors on 3 of 6 pods, all on `k8s.node.name = ip-10-0-2-11.ec2.internal`. |
| Service | Earliest ERROR span: `service.name = payments-checkout`, `service.version = v2.4.1`. |
| Layer | `span.kind = CLIENT`, `db.system = redis`; L7 DB call, not a network-layer issue. |
| Code | `exception.type = TimeoutException`, `exception.message = "Redis pool exhausted after 5000ms"`, `code.function = processPayment`, `code.lineno = 287`. |

**Step 3; Cross-signal validation (`incident-forensics.md §4`):**

```
# Grafana Tempo — TraceQL lookup
{ trace:id = "9a3f1c8e2b7d4e50a1c3f9d2b8e7a4c0" }

# Filter logs by trace_id to find correlated warnings
# Loki LogQL
{service_name="payments-checkout"} | json | trace_id="9a3f1c8e2b7d4e50a1c3f9d2b8e7a4c0"
```

Logs reveal: `WARN: connection pool at capacity (256/256)` at 14:21. Redis
`redis_pool_active_connections / redis_pool_max_connections` metric at 100 %.

**Step 4; Release correlation:**

`service.version = v2.4.1` deployed at 13:57 UTC; 25 minutes before incident.
Changelog: Redis connection pool reduced from 512 → 256 in this version.

**Step 5; Remediation:**

Flagger canary analysis `successRate` already < 99 % → auto-rollback to `v2.4.0` triggers
per `boundaries/release.md §Flagger Canary CR`. Error rate returns to baseline in 3 minutes.
Deployment event chain preserved in `signals/audit.md` for post-mortem.

---

## Scenario 3: Multi-Tenant Cost Attribution

**Situation:** B2B SaaS now has 50+ tenants. CFO asks: "Which tenants cost most to serve, and
how does our observability bill break down per customer?"

**Intent:** `route` + `tune`

**Files referenced:** `boundaries/multi-tenant.md §7`, `signals/cost.md §3–§4`, `vendor-categories.md §(f)`, `meta-observability.md §Cardinality Guardrails`

### Walkthrough

1. Tag every pod at deploy time with `tenant.id`; applied via Helm values or admission webhook
   per `boundaries/multi-tenant.md §7`:

```yaml
metadata:
  labels:
    tenant.id: "acme-corp"
    tenant.tier: "enterprise"
```

2. Deploy OpenCost (CNCF Incubating) with Prometheus scrape per `signals/cost.md §3`:

```yaml
# OTel Collector gateway — OpenCost scrape
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: opencost
          scrape_interval: 60s
          static_configs:
            - targets: ["opencost.opencost.svc.cluster.local:9003"]
```

3. Per-tenant cost PromQL (daily, `signals/cost.md §5`):

```promql
sum by (tenant_id) (
  increase(opencost_workload_cost_total[24h])
  * on(namespace) group_left(tenant_id)
  kube_namespace_labels
)
```

4. Cardinality guard: keep top-1 000 tenants labeled, bucket overflow as `"other"` to prevent
   TSDB series explosion (`meta-observability.md §Cardinality Guardrails`, `boundaries/multi-tenant.md §13`):

```yaml
# OTel Collector transform processor — cardinality cap
processors:
  transform/cardinality:
    metric_statements:
      - context: datapoint
        statements:
          - set(attributes["tenant_id"], "other")
              where attributes["tenant_id"] not in ${env:TOP_TENANT_LIST}
```

5. Per-tier sampling policy at the gateway to control observability bill per tenant
   (`boundaries/multi-tenant.md §5`): `enterprise` = 100 %, `pro` = 20 %, `free` = 2 %.
   Full `tail_sampling` YAML (with `and` sub-policies) lives in `transport/sampling-recipes.md §4`.

6. Include the observability bill itself: instrument the collector pipeline with throughput counters
   per `tenant.id` to apportion Grafana Cloud / Datadog usage proportionally
   (`signals/cost.md §4`, `vendor-categories.md §(f)`).

**Outcome:** Identified 3 free-tier tenants consuming 40 % of egress cost. Applied egress rate
limit at collector ingress (`boundaries/multi-tenant.md §11 Noisy Neighbor Protection`).
Those tenants were moved to a stricter free-tier cap, reducing monthly egress spend by 35 %.

---

## Scenario 4: Migrating off Fluentd to Fluent Bit

**Situation:** Legacy Fluentd DaemonSet is consuming 400 MB RAM per node vs. Fluent Bit's typical
80 MB. The CNCF 2025-10 migration guide is now the normative reference. Team needs zero-downtime
migration with log parity verification before decommission.

**Intent:** `migrate`

**Files referenced:** `vendor-categories.md §(h)`, `signals/logs.md §6`, `transport/collector-topology.md §2`, `observability-as-code.md §3`

### Walkthrough

1. Invoke `/oma-observability --migrate "Fluentd to Fluent Bit"`.
2. Router routes to `vendor-categories.md §(h) Log Pipeline` → **Fluent Bit** (CNCF Graduated,
   drop-in config compatibility with Fluentd, C runtime, 5–15 MB RAM typical).

**Migration phases (zero-downtime):**

**Phase 1; Dual-send:** Deploy Fluent Bit DaemonSet alongside the existing Fluentd DaemonSet.
Fluent Bit ships logs to the same backend; Fluentd continues as primary. Confirm both streams
reach the backend without duplicates by comparing log record counts.

```ini
# Fluent Bit DaemonSet — core pipeline (fluent-bit.conf)
[INPUT]
    Name           tail
    Path           /var/log/pods/*/*/*.log
    Parser         cri
    Tag            kube.*
    Mem_Buf_Limit  5MB

[FILTER]
    Name        kubernetes
    Match       kube.*
    Merge_Log   On
    Labels      On

[OUTPUT]
    Name      opentelemetry
    Match     kube.*
    Host      otel-gateway.observability.svc
    Port      4318
    Logs_uri  /v1/logs
```

**Phase 2; Parity verification:** query log record count per `service.name` in both pipelines
over a 1-hour window. Expect < 0.1 % divergence accounting for timing windows.

```promql
# Fluent Bit throughput
sum by (service_name) (rate(fluentbit_output_proc_records_total[1h]))

# Fluentd throughput (comparison)
sum by (service_name) (rate(fluentd_output_emit_records_total[1h]))
```

**Phase 3; Decommission Fluentd:** once parity is confirmed over 24 h, scale down Fluentd
DaemonSet to 0 and remove its resources.

```bash
kubectl scale daemonset fluentd -n logging --replicas=0
kubectl delete daemonset fluentd -n logging
```

3. Commit Fluent Bit ConfigMap and DaemonSet manifests to git; apply via Argo CD per
   `observability-as-code.md §7`.
4. Validate with `otelcol validate` on the gateway config; run `promtool check rules` on any
   log-based alerting rules.

**Outcome:** Node-level log collection RAM drops from 400 MB (Fluentd/Ruby) to ~80 MB (Fluent Bit/C),
a 5× reduction typical per the CNCF 2025-10 guide. CPU overhead at steady state also reduces by
~60 %. No log gaps observed during the dual-send window.

---

## Scenario 5: SLO with Burn-Rate Alerts (GitOps End-to-End)

**Situation:** Platform team adopts the SLO practice for the checkout service. Target: 99.9 %
availability on a 28-day rolling window. Everything must be versioned in git and applied
via Argo CD; no manual UI edits.

**Intent:** `alert`

**Files referenced:** `boundaries/slo.md §5, §7, §8`, `observability-as-code.md §4, §6, §7`, `boundaries/release.md`

### Walkthrough

1. Define the SLO as OpenSLO YAML and commit to the observability repo (`boundaries/slo.md §5`):

```yaml
# Key fields — full spec in boundaries/slo.md §5
apiVersion: openslo.com/v1
kind: SLO
metadata:
  name: checkout-availability
spec:
  service: checkout
  sloType: Request-Based
  objectives:
    - target: 0.999
      window: 28d
  indicator:
    spec:
      ratioMetric:
        good:
          metricSource:
            type: Prometheus
            spec:
              query: >
                sum(rate(http_requests_total{service="checkout",status=~"2..|3.."}[{{.Window}}]))
        total:
          metricSource:
            type: Prometheus
            spec:
              query: sum(rate(http_requests_total{service="checkout"}[{{.Window}}]))
```

2. Generate PrometheusRule CRD via Sloth (`observability-as-code.md §4.2`):

```bash
sloth generate -i sloth/checkout-slo.yaml | kubectl apply -f -
```

3. The generated `PrometheusRule` includes multi-window burn-rate alerts from `boundaries/slo.md §7`
   and `observability-as-code.md §6`:

```yaml
# Fast burn — 2 % budget in 1 h (14.4× rate), gated by 5 m short window → page
- alert: CheckoutSLOBurnRateFast
  expr: |
    (sum(rate(http_requests_total{service="checkout",status=~"5.."}[1h]))
      / sum(rate(http_requests_total{service="checkout"}[1h]))) / (1 - 0.999) > 14.4
    and
    (sum(rate(http_requests_total{service="checkout",status=~"5.."}[5m]))
      / sum(rate(http_requests_total{service="checkout"}[5m]))) / (1 - 0.999) > 14.4
  for: 2m
  labels: { severity: critical, slo: checkout-availability }
  annotations:
    summary: "Checkout SLO fast burn: budget exhausting in < 1h"

# Slow burn — 5 % budget in 6 h (6× rate), gated by 30 m → ticket
# Full PromQL in observability-as-code.md §6 and boundaries/slo.md §7.4
```

4. GitOps commit flow (`observability-as-code.md §7`):

```
git PR (checkout-slo.yaml + PrometheusRule)
  → CI: promtool check rules alerts/checkout-slo-burn-rate.yaml
  → CI: promtool test rules alerts/checkout_slo_test.yaml
  → peer review (SRE CODEOWNERS approval)
  → merge to main
  → Argo CD sync → Prometheus Operator applies PrometheusRule to cluster
```

5. Validate via synthetic error injection:

```bash
# Inject 20 % error rate for 3 minutes to trigger fast-burn alert
kubectl run error-injector --image=curlimages/curl --restart=Never -- \
  sh -c 'for i in $(seq 1 200); do curl -s -o /dev/null -w "%{http_code}" \
    http://checkout-api/error-inject; done'
```

Fast-burn alert fires in < 2 minutes.

6. Error budget policy (`boundaries/slo.md §8`):

| Budget remaining | Action |
|-----------------|--------|
| > 50 % | Normal feature velocity |
| 25–50 % | Review reliability vs feature ratio |
| < 25 % | Reliability sprint |
| 0 % exhausted | Deploy freeze until budget recovers |

When budget reaches 0 %, Flagger canary promotion gates use the same SLI error rate to block
new deploys (`boundaries/release.md`).

---

## Appendix: Vendor Query Cross-Reference

From `incident-forensics.md §4`; trace lookup by `trace_id` across backends:

| Vendor | Query syntax |
|--------|-------------|
| Grafana Tempo | `{ trace:id = "4bf92f3577b34da6a3ce929d0e0e4736" }` |
| Honeycomb | `trace:4bf92f3577b34da6a3ce929d0e0e4736` |
| Datadog APM | `@trace_id:4bf92f3577b34da6a3ce929d0e0e4736` |
| Jaeger HTTP API | `GET /api/traces/4bf92f3577b34da6a3ce929d0e0e4736` |
| Elastic APM | `trace.id: "4bf92f3577b34da6a3ce929d0e0e4736"` |
| Sentry | `trace_id:4bf92f3577b34da6a3ce929d0e0e4736` |

---

## Contribution Protocol

- Add new scenarios only if they demonstrate cross-file value spanning 3+ skill files.
- Each scenario must cite at least 3 other skill files by relative path from `resources/`.
- Scenarios are for human learning: keep them narrative, not exhaustive.
- Code and YAML snippets are illustrative excerpts; for authoritative full config see the
  referenced files.
- Do not add future skill references: if a domain is out of scope, cite the external tool
  from `SKILL.md §When NOT to use`.
