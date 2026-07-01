# SLO Boundary

## 1. Scope

SLO (Service Level Objective) is the contractual reliability target bounding acceptable error rates
and response times. Covers: SLI definition, SLO math, error budget, burn-rate alerts, tool options.

Out of scope: dashboard config (`observability-as-code.md`); progressive delivery gates (`release.md`);
pipeline SLOs (`../meta-observability.md §Section F`).

---

## 2. Terminology

| Term | Definition | Example |
|------|-----------|---------|
| **SLI** | Measurable property of a service | 99.2% of requests returned 2xx in past 30d |
| **SLO** | Internal reliability target | 99.9% availability, 28-day rolling window |
| **SLA** | Customer contract with consequences | SLO − buffer; breach triggers credits |
| **Error budget** | `100% − SLO`; headroom before breach | 0.1% = 43.2 min/month at 99.9% SLO |

Rule: SLA target < SLO target. Never set SLA = SLO (no operations buffer).

---

## 3. SLI Selection

| Method | Best for | Signals |
|--------|---------|---------|
| Golden Signals (Google SRE) | Any service | Latency, Traffic, Errors, Saturation |
| RED | Request-serving | Rate, Errors, Duration |
| USE | Resources | Utilization, Saturation, Errors |

Cross-ref `../signals/metrics.md §SLI` for PromQL sketches per method.

---

## 4. SLO Math

Availability SLI (28-day rolling window):

```promql
sum(rate(http_requests_total{status=~"2..|3.."}[28d]))
  / sum(rate(http_requests_total[28d]))
```

Latency SLI (fraction of requests under 300ms):

```promql
sum(rate(http_request_duration_seconds_bucket{le="0.3"}[28d]))
  / sum(rate(http_request_duration_seconds_count[28d]))
```

---

## 5. OpenSLO Spec

Source: <https://openslo.com>; community-driven, not CNCF. Vendor-neutral YAML adopted by Sloth, Pyrra, Nobl9.

```yaml
apiVersion: openslo.com/v1
kind: SLO
metadata:
  name: checkout-availability
spec:
  service: checkout
  sloType: Request-Based
  indicator:
    spec:
      ratioMetric:
        good:
          metricSource:
            type: Prometheus
            spec:
              query: sum(rate(http_requests_total{service="checkout",status=~"2..|3.."}[{{.Window}}]))
        total:
          metricSource:
            type: Prometheus
            spec:
              query: sum(rate(http_requests_total{service="checkout"}[{{.Window}}]))
  objectives:
    - target: 0.999
  timeWindow:
    - duration: 28d
      isRolling: true
```

---

## 6. Tool Options (as of 2026-Q2)

| Tool | Type | CNCF | Output |
|------|------|------|--------|
| Sloth | OSS CLI + k8s operator | No | PrometheusRule CRDs |
| Pyrra | OSS CLI + k8s operator | No | PrometheusRule CRDs |
| Grafana SLO | Grafana Cloud product | No | Cloud-native alerts |
| Nobl9 | Commercial SaaS | No | Multi-backend |
| Google Cloud Service Monitoring | GCP-native | No | GCP alerts |

Cross-ref `../vendor-categories.md §OSS Full-Stack`. Sloth and Pyrra are the recommended OSS path
for Kubernetes-native GitOps deployments.

---

## 7. Burn-Rate Alert Design

### 7.1 Problem

A simple error-budget-exhausted alert fires after damage is done. Multi-window burn-rate alerts
detect fast-burning incidents while they still have budget remaining.

Source: <https://sre.google/workbook/alerting-on-slos/>

### 7.2 Multi-Window Tiers (99.9% SLO)

| Tier | Budget consumed | Long window | Short window | Multiplier | Action |
|------|----------------|------------|-------------|-----------|--------|
| Fast burn | 2% in 1h | 1h | 5m | 14.4× | Page immediately |
| Slow burn | 5% in 6h | 6h | 30m | 6× | Create ticket |

**Multiplier derivation:** `budget_fraction / (window / 720h)`
- Fast: `2% / (1h/720h) = 14.4`
- Slow: `5% / (6h/720h) = 6`

### 7.3 Fast Burn PromQL

```promql
(
  (1 - sum(rate(http_requests_total{status=~"2..|3.."}[1h]))
         / sum(rate(http_requests_total[1h])))
  > (1 - 0.999) * 14.4
)
and
(
  (1 - sum(rate(http_requests_total{status=~"2..|3.."}[5m]))
         / sum(rate(http_requests_total[5m])))
  > (1 - 0.999) * 14.4
)
```

### 7.4 Slow Burn PromQL

```promql
(
  (1 - sum(rate(http_requests_total{status=~"2..|3.."}[6h]))
         / sum(rate(http_requests_total[6h])))
  > (1 - 0.999) * 6
)
and
(
  (1 - sum(rate(http_requests_total{status=~"2..|3.."}[30m]))
         / sum(rate(http_requests_total[30m])))
  > (1 - 0.999) * 6
)
```

The short window gates the long window: long window detects sustained burns; short window
suppresses false positives from transient spikes.

---

## 8. Error Budget Policy

| Budget remaining | Action |
|-----------------|--------|
| > 50% | Normal feature velocity |
| 25–50% | Review reliability vs feature ratio |
| < 25% | Reliability sprint |
| 0% (exhausted) | Freeze deploys until budget recovers |

Cross-ref `release.md`; Flagger and Argo Rollouts use the SLI error rate as canary promotion gate;
the same threshold enforces budget-aware deployment freeze.

---

## 9. Cross-Integration

| File | Integration |
|------|------------|
| `release.md` | Flagger/Argo use SLI metric for canary promotion; SLO failure = rollback |
| `observability-as-code.md` | OpenSLO YAML in Git; Sloth/Pyrra generate PrometheusRule CRDs in CI |
| `../meta-observability.md §Section F` | Pipeline has its own burn-rate alerts for computation lag |

---

## 10. Matrix Coverage (slo row)

| Layer | Signal | Status | Detail |
|-------|--------|--------|--------|
| L7-application | metrics | PASS | SLI calculation from HTTP/gRPC counters and histograms |
| L7-application | logs | PARTIAL | Burn-rate source when metrics unavailable; higher latency |
| L7-application | traces | PARTIAL | Critical path traces complement metrics-based SLI |
| mesh | metrics | PASS | Golden signals from Envoy; zero-code instrumentation |

---

## 11. Anti-Patterns (candidates for `../anti-patterns.md §Section D`)

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| SLO without burn-rate alert | Alert fires after budget exhausted | Add multi-window PrometheusRule |
| Burn-rate without multi-window | False alarms (short) or slow detection (long) | Gate long window with short window |
| SLO without error budget policy | No action framework for budget consumption | Define policy table (§8) |
| Customer SLA = SLO | No buffer; any breach triggers penalty | Set SLA target below SLO target |

---

## References

- Google SRE Workbook: Alerting on SLOs: <https://sre.google/workbook/alerting-on-slos/>
- OpenSLO specification: <https://openslo.com>
- Sloth: <https://github.com/slok/sloth>
- Pyrra: <https://github.com/pyrra-dev/pyrra>
- Grafana SLO plugin: <https://grafana.com/grafana/plugins/grafana-slo-app>
