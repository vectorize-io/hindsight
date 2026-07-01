# Release Boundary

## 1. Scope

The release boundary is the **temporal boundary between versions of a service**; the window
where new code enters production and either stabilizes or is rolled back.

Covers: progressive delivery strategies, Flagger canary analysis, Argo Rollouts, OpenFeature
feature flags, GitOps engine reconcile observability, and Kubernetes operator reconcile metrics.

Out of scope: SLI/SLO definitions and error budget math (see `slo.md`); dashboard layout
and code (see `../observability-as-code.md`); post-incident timelines (see `../incident-forensics.md`).

This boundary **consumes** SLI/SLO metrics as promotion gates; it does not define SLOs.

---

## 2. Progressive Delivery Strategies

| Strategy | Traffic split | Blast radius | Metric feedback speed | When to use |
|----------|--------------|-------------|----------------------|-------------|
| **Blue/Green** | Full switch | High | Fast (all traffic) | Low-frequency, reversible deploys |
| **Canary** | Gradual % shift | Low to high | Progressive | Most stateless services |
| **A/B testing** | Cohort split | Low | Slow (UX signal) | Product/UX variant evaluation |

Choose based on: blast radius tolerance × metric feedback speed. Canary is the default
recommendation for services with Prometheus SLI coverage.

---

## 3. Flagger (CNCF Graduated, part of Flux)

Source: <https://flagger.app>

Flagger automates canary promotion and rollback decisions using Prometheus, Datadog, or
NewRelic metric queries. It integrates natively with Flux and supports any service mesh or
ingress (Istio, Linkerd, Nginx, Contour).

**Feedback loop:** Flagger samples metrics on a configurable interval (30s–60s). If the
metric breaches the threshold for N consecutive samples (`failureThreshold`), it rolls back.
Promotion requires all metric checks to pass for the full `iterations` count.

```yaml
# Flagger Canary CR — canary analysis referencing PromQL SLI
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: checkout
  namespace: prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: checkout
  progressDeadlineSeconds: 600
  service:
    port: 8080
  analysis:
    interval: 60s        # sample every 60 seconds
    threshold: 5         # max failed checks before rollback
    maxWeight: 50        # max canary traffic weight (%)
    stepWeight: 10       # increment per successful check
    metrics:
      - name: error-rate
        templateRef:
          name: error-rate
          namespace: flagger-system
        thresholdRange:
          max: 1         # fail if error rate > 1%
        interval: 60s
      - name: latency-p99
        templateRef:
          name: latency
          namespace: flagger-system
        thresholdRange:
          max: 500       # fail if p99 > 500ms
        interval: 60s
---
apiVersion: flagger.app/v1beta1
kind: MetricTemplate
metadata:
  name: error-rate
  namespace: flagger-system
spec:
  provider:
    type: prometheus
    address: http://prometheus.monitoring:9090
  query: |
    sum(rate(http_requests_total{
      namespace="{{ namespace }}",
      service="{{ target }}",
      status=~"5.."
    }[{{ interval }}]))
    /
    sum(rate(http_requests_total{
      namespace="{{ namespace }}",
      service="{{ target }}"
    }[{{ interval }}])) * 100
```

Cross-ref `slo.md §3` for the full SLI PromQL patterns used as the canary threshold source.

---

## 4. Argo Rollouts

Source: <https://argoproj.github.io/rollouts>

Argo Rollouts integrates with Argo CD for GitOps-native progressive delivery. Supports
canary, blue-green, and experiment steps. `AnalysisTemplate` CRD gates promotion with
metric-based pass/fail.

```yaml
# Argo Rollout — canary with AnalysisTemplate reference
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout
spec:
  replicas: 5
  strategy:
    canary:
      steps:
        - setWeight: 20
        - pause: { duration: 2m }
        - analysis:
            templates:
              - templateName: error-rate-check
        - setWeight: 50
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: error-rate-check
      canaryService: checkout-canary
      stableService: checkout-stable
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate-check
spec:
  metrics:
    - name: error-rate
      interval: 60s
      successCondition: result[0] < 1.0   # < 1% error rate
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{
              job="checkout-canary",
              status=~"5.."
            }[5m]))
            /
            sum(rate(http_requests_total{
              job="checkout-canary"
            }[5m])) * 100
```

---

## 5. Feature Flags via OpenFeature (CNCF Graduated, 2024-11)

Source: <https://openfeature.dev> | CNCF graduation: <https://www.cncf.io/projects/openfeature/>

OpenFeature defines a vendor-agnostic SDK specification for feature flag evaluation via the
**OFREP** (OpenFeature Remote Evaluation Protocol). The SDK core is provider-agnostic;
teams swap providers without changing application code.

**Supported providers:** LaunchDarkly, Flagsmith, GrowthBook, Unleash, ConfigCat, Harness FF.

**Evaluation context** carries targeting attributes:

| Attribute | Example | Use |
|-----------|---------|-----|
| `user.id` | `usr_abc123` | Per-user targeting |
| `tenant.id` | `org_xyz` | Tenant-gated rollout |
| `environment` | `production` | Environment guard |
| `app.version` | `2.4.1` | Version-gated flag |

Integration with Flagger/Rollouts: use feature flags to gate traffic routing at the
application layer while Flagger controls infrastructure-level traffic weight.

---

## 6. Observing Feature Flag Evaluations

OTel semantic conventions: `feature_flag.*` (Experimental as of semconv 1.27.0).

Key span attributes emitted on each flag evaluation:

```
feature_flag.key          = "checkout-v2-enabled"
feature_flag.variant      = "on"
feature_flag.provider_name = "flagsmith"
```

Example span attributes (JSON log-compatible):

```json
{
  "name": "feature_flag.evaluation",
  "attributes": {
    "feature_flag.key": "checkout-v2-enabled",
    "feature_flag.variant": "on",
    "feature_flag.provider_name": "flagsmith",
    "user.id": "usr_abc123",
    "tenant.id": "org_enterprise"
  }
}
```

**Metrics to track per flag:**

| Metric | Labels | Purpose |
|--------|--------|---------|
| `feature_flag_evaluation_total` | `flag`, `variant`, `tenant` | Variant distribution |
| Error rate delta per variant | `flag`, `variant` | A/B regression detection |
| Latency p99 delta per variant | `flag`, `variant` | Performance regression |

Dashboards cross-ref `../observability-as-code.md` for the flag evaluation panel template.

---

## 7. GitOps Engines

### 7.1 Argo CD

Source: <https://argo-cd.readthedocs.io>

Declarative Kubernetes sync using app-of-apps pattern. Reconcile metrics exposed via `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `argocd_app_info` | Gauge | App metadata; sync status label |
| `argocd_app_sync_total` | Counter | Sync operations by phase |
| `argocd_app_reconcile_bucket` | Histogram | Reconcile latency distribution |

Drift detection alert (cluster state ≠ git manifest):

```promql
argocd_app_info{sync_status="OutOfSync"} == 1
```

### 7.2 Flux (CNCF Graduated)

Source: <https://fluxcd.io>; Flagger ships as part of the Flux ecosystem.

| Metric | Type | Description |
|--------|------|-------------|
| `gotk_reconcile_duration_seconds` | Histogram | Per-controller reconcile latency |
| `gotk_reconcile_condition` | Gauge | `ready` or `stalled` per resource |

Stalled resource alert:

```promql
gotk_reconcile_condition{type="Ready",status="False"} == 1
```

Cross-ref `../signals/metrics.md` for Prometheus scrape config snippets for both engines.

---

## 8. Kubernetes Operator Reconcile Observability

Every operator built on `controller-runtime` exposes `/metrics` automatically.

**Built-in metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `controller_runtime_reconcile_total` | Counter | Reconcile attempts by result |
| `controller_runtime_reconcile_errors_total` | Counter | Failed reconcile loops |
| `controller_runtime_reconcile_time_seconds` | Histogram | Reconcile loop duration |

Monitor CRD readiness via `.status.conditions[]`:

```promql
# Alert on reconcile error spike
rate(controller_runtime_reconcile_errors_total[5m]) > 0.1
```

Custom CRDs must emit reconcile metrics. Pattern: wrap reconcile logic in `ObservedGeneration`
guard and expose a `status.conditions` entry per managed resource.

---

## 9. Release Markers in Telemetry

- `service.version` (OTel Stable): set on every span, metric data point, and log record
  via the OTel Resource at SDK initialization. Never patch per-signal.
- Emit a **deployment event** at release time:

```json
{
  "event.name": "deployment",
  "service.name": "checkout",
  "service.version": "2.4.1",
  "deployment.environment": "production",
  "timestamp": "2026-04-21T10:00:00Z"
}
```

- Pipeline this event to Grafana → vertical annotation line on all timeseries dashboards.
- **Correlation rule:** incidents within ≤30 minutes of a deployment event are flagged as
  release suspects. Cross-ref `../incident-forensics.md §Scenario C` for the triage playbook.

---

## 10. Canary Analysis Metric Suite

| Signal | Source | Gate |
|--------|--------|------|
| SLI compliance | `slo.md` PromQL | Pass/fail per iteration |
| Error rate delta (new vs prev) | `http_requests_total` | Max +0.5% allowed |
| Latency p99 delta | `http_request_duration_seconds_bucket` | Max +50ms allowed |
| Crash-Free Rate delta | Cross-ref `../layers/L7-application/crash-analytics.md` | Mobile/web only |
| Conversion / revenue delta | Product analytics | Product team domain; not in Flagger |

---

## 11. Rollback and Hotfix Pipeline

| Trigger | Mechanism | Action |
|---------|-----------|--------|
| Metric breach (N consecutive) | Flagger / Argo Rollouts automated | Instant rollback to stable |
| Manual override | `kubectl argo rollouts abort` or Flagger annotation | Operator-initiated |
| Hotfix deploy | Standard pipeline, skip canary analysis steps if authorized | Requires audit entry |

Manual override always remains possible regardless of automated analysis state.

Post-mortem audit trail required for every rollback event.
Cross-ref `../signals/audit.md` for the required audit log fields.

---

## 12. Matrix.md Cells (release row)

| Layer | Signal | Status | Detail |
|-------|--------|--------|--------|
| L7 × release | metrics | PASS | Canary SLI delta; error rate and latency per version |
| L7 × release | logs | PASS | Deployment events with `service.version` |
| L7 × release | traces | PASS | `service.version` tagging on every span |
| L7 × release | profiles | PARTIAL | Regression comparison v(new) vs v(prev); tooling-dependent |
| L7 × release | cost | PARTIAL | Cost delta per variant requires OpenCost label propagation |
| mesh × release | metrics | PARTIAL | Canary routing rules observable via Envoy stats if mesh present |

---

## 13. Anti-Patterns (candidates for `../anti-patterns.md §Section E Release & Deployment`)

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| No release markers in telemetry | Cannot correlate deploys to incidents | Set `service.version` on OTel Resource; emit deployment event |
| Canary analysis without SLI metric | Promotion is blind; regressions ship silently | Add MetricTemplate/AnalysisTemplate referencing PromQL SLI |
| Feature flag evaluation not observed | Variant effect invisible; A/B regression undetected | Emit `feature_flag.*` span attributes on every evaluation |
| GitOps drift unalerted | Ghost state; cluster diverges from git manifest silently | Alert on `argocd_app_info{sync_status="OutOfSync"}` or `gotk_reconcile_condition{status="False"}` |
| Rollback without post-mortem audit trail | No learning loop; same failure repeats | Record rollback event in audit log; cross-ref `../signals/audit.md` |

---

## References

- Flagger: <https://flagger.app>
- Argo Rollouts: <https://argoproj.github.io/rollouts>
- OpenFeature: <https://openfeature.dev>
- CNCF OpenFeature incubating announcement: <https://cncf.io/blog/2023/12/>
- Flux (includes Flagger): <https://fluxcd.io>
- Argo CD: <https://argo-cd.readthedocs.io>
- OTel feature_flag semconv: <https://opentelemetry.io/docs/specs/semconv/feature-flags/>
- controller-runtime metrics: <https://book.kubebuilder.io/reference/metrics>
