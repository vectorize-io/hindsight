---
cross_ref: "meta-observability.md §Section F for pipeline alerts; boundaries/slo.md §5,7 for SLO math"
notes:
  - "Versions: Grafonnet 11.x; OTel Operator v1beta1; Terraform Grafana provider ~> 3.0"
---

# Observability-as-Code

## 1. Why Observability-as-Code

UI-edited dashboards and alerts accumulate silent debt:

- **No DR / no rollback.** A dashboard overwritten in the Grafana UI cannot be reverted without a backup.
- **No peer review.** Alert thresholds edited live bypass the same change-management gates required for code.
- **Scale problem.** 100+ dashboards across N environments managed by hand drift within weeks. Each environment develops its own undocumented fork.
- **Compliance.** SOC 2 change-management controls require an audit trail for every change to detection and response configuration. Git history provides that trail; Grafana's internal change log does not. Cross-ref `signals/audit.md` for WORM immutability requirements.

The non-negotiable principle: version every observability artifact; dashboards, alert rules, SLO definitions, collector config; in git, applied via CI/CD. This is design decision D9 in the design document.

---

## 2. Dashboards-as-Code

### 2.1 Grafana Jsonnet / Grafonnet

Source: <https://grafana.github.io/grafonnet/> (Grafonnet 11.x, replacing legacy grafonnet-lib)

Grafonnet is a Jsonnet library that generates Grafana's JSON dashboard model declaratively. Dashboards are composed programmatically; panels, data sources, variables, and annotations are all typed functions.

```jsonnet
// dashboards/collector-health.jsonnet
local grafonnet = import 'grafonnet/main.libsonnet';
local dashboard = grafonnet.dashboard;
local panel = grafonnet.panel;
local prometheus = grafonnet.query.prometheus;

dashboard.new('OTel Collector Health')
+ dashboard.withUid('otelcol-health-v1')
+ dashboard.withRefresh('30s')
+ dashboard.withPanels([
  panel.timeSeries.new('Delivery Ratio — Traces')
  + panel.timeSeries.withTargets([
    prometheus.new(
      '$datasource',
      |||
        sum(rate(otelcol_exporter_sent_spans[5m]))
          / (sum(rate(otelcol_receiver_accepted_spans[5m])) > 0)
      |||
    ),
  ]),
])
```

Render to JSON and apply:

```bash
jsonnet -J vendor dashboards/collector-health.jsonnet > dist/collector-health.json
# CI step: grafana-cli dashboards import dist/collector-health.json
```

Lint Jsonnet in CI:

```bash
jsonnetfmt --test dashboards/*.jsonnet
```

### 2.2 Perses (CNCF Sandbox)

Source: <https://perses.dev>; CNCF Sandbox (accepted 2023). YAML-first, vendor-neutral dashboard definition targeting a future CNCF standard for dashboards-as-code. Stricter schema than Grafana JSON; import from Grafana JSON is under development.

```yaml
# dashboards/collector-health.yaml (Perses)
kind: Dashboard
metadata:
  name: otelcol-health
  project: observability
spec:
  duration: 30m
  panels:
    delivery_ratio:
      kind: TimeSeriesChart
      spec:
        queries:
          - kind: TimeSeriesQuery
            spec:
              plugin:
                kind: PrometheusTimeSeriesQuery
                spec:
                  query: >
                    sum(rate(otelcol_exporter_sent_spans[5m]))
                    / (sum(rate(otelcol_receiver_accepted_spans[5m])) > 0)
```

Use Perses when vendor-neutral YAML is a hard requirement (e.g., multi-backend environments). See Section 12 for current maturity notes.

### 2.3 Datadog Terraform Provider

```hcl
resource "datadog_dashboard_json" "collector_health" {
  dashboard = file("${path.module}/dashboards/collector-health.json")
}
```

### 2.4 Honeycomb Terraform Provider

```hcl
resource "honeycombio_board" "slo_burn_rate" {
  name        = "SLO Burn Rate"
  description = "Multi-window burn-rate overview"
}
```

### 2.5 GitOps Workflow for Dashboards

```
git PR (dashboard Jsonnet / YAML)
  → CI: jsonnet lint + render to JSON
  → peer review
  → merge to main
  → CD (Argo CD / Flux) applies to Grafana via API or ConfigMap
```

---

## 3. Alerts-as-Code

### 3.1 PrometheusRule CRD (Prometheus Operator)

The `PrometheusRule` custom resource (group `monitoring.coreos.com/v1`) is the standard Kubernetes-native format for alert rules when running Prometheus Operator or kube-prometheus-stack.

Full burn-rate example in Section 6.

### 3.2 Alertmanager YAML

Notification routing is versioned alongside alert rules:

```yaml
# alertmanager-config.yaml
route:
  group_by: [alertname, cluster, service]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: slack-critical
  routes:
    - matchers:
        - severity="warning"
      receiver: slack-warning

receivers:
  - name: slack-critical
    slack_configs:
      - api_url: "${SLACK_WEBHOOK_URL}"   # injected from Secret; never hardcoded
        channel: "#incidents"
        title: "{{ .GroupLabels.alertname }}"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
```

### 3.3 Terraform Grafana Provider (Grafana Cloud Alerts)

Source: <https://github.com/grafana/terraform-provider-grafana>

```hcl
resource "grafana_rule_group" "slo_burn_rate" {
  name             = "slo-burn-rate"
  folder_uid       = grafana_folder.observability.uid
  interval_seconds = 60

  rule {
    name      = "SloBurnRateFast"
    condition = "A"

    data {
      ref_id         = "A"
      datasource_uid = var.prometheus_datasource_uid
      model = jsonencode({
        expr = "(sum(rate(http_requests_total{status=~\"5..\"}[1h])) / sum(rate(http_requests_total[1h]))) / (1 - 0.999) > 14.4"
      })
    }

    annotations = { summary = "Fast burn: SLO budget burning at 14.4x rate" }
    labels      = { severity = "critical" }
  }
}
```

### 3.4 GitHub Actions Alert Pipeline

```yaml
# .github/workflows/alerts.yaml
on:
  push:
    paths: ["alerts/**"]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check PrometheusRule YAML
        run: promtool check rules alerts/*.yaml
      - name: Lint Alertmanager config
        run: amtool check-config alertmanager-config.yaml
  apply:
    needs: validate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Apply via kubectl
        run: kubectl apply -f alerts/
```

---

## 4. SLO-as-Code

### 4.1 OpenSLO YAML

Source: <https://openslo.com>; community-driven (not CNCF). Adopted by Sloth, Pyrra, and Nobl9. Cross-ref `boundaries/slo.md §5` for the full OpenSLO spec example.

```yaml
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
              query: sum(rate(http_requests_total{service="checkout",status=~"2..|3.."}[{{.Window}}]))
        total:
          metricSource:
            type: Prometheus
            spec:
              query: sum(rate(http_requests_total{service="checkout"}[{{.Window}}]))
```

### 4.2 Sloth: YAML to PrometheusRule

Source: <https://sloth.dev>; generates multi-window burn-rate `PrometheusRule` CRDs from a concise YAML definition.

```yaml
# sloth/checkout-slo.yaml
version: prometheus/v1
service: checkout
slos:
  - name: availability
    objective: 99.9
    description: "Checkout HTTP availability"
    sli:
      events:
        error_query: sum(rate(http_requests_total{service="checkout",status=~"5.."}[{{.window}}]))
        total_query: sum(rate(http_requests_total{service="checkout"}[{{.window}}]))
    alerting:
      page_alert:
        labels: { severity: critical }
      ticket_alert:
        labels: { severity: warning }
```

```bash
sloth generate -i sloth/checkout-slo.yaml | kubectl apply -f -
```

### 4.3 Pyrra: Kubernetes CRD Operator

Source: <https://github.com/pyrra-dev/pyrra>; Kubernetes operator that reconciles `ServiceLevelObjective` CRDs directly into `PrometheusRule` + recording rules.

```yaml
apiVersion: pyrra.dev/v1alpha1
kind: ServiceLevelObjective
metadata:
  name: checkout-availability
  namespace: monitoring
spec:
  target: "99.9"
  window: 28d
  serviceMonitorSelector: {}
  indicator:
    http:
      selector:
        matchLabels: { job: checkout }
      errorsSelector:
        matchExpressions:
          - { key: code, operator: In, values: ["5xx"] }
```

Pyrra reconciles this into recording rules and PrometheusRule alerts automatically; no separate generation step.

---

## 5. OTel Collector-as-Code

### 5.1 OTel Operator: OpenTelemetryCollector CRD

Source: <https://github.com/open-telemetry/opentelemetry-operator>

The `OpenTelemetryCollector` CRD (`v1beta1`, beta) is the Kubernetes-native way to manage collector config as code. Four `spec.mode` values map to the four deployment strategies; cross-ref `transport/collector-topology.md §1`:

```yaml
apiVersion: opentelemetry.io/v1beta1
kind: OpenTelemetryCollector
metadata:
  name: otelcol-daemonset
  namespace: observability
spec:
  mode: daemonset          # deployment | daemonset | statefulset | sidecar
  image: otel/opentelemetry-collector-contrib:0.122.1   # pin to current contrib release; verify tag exists via docker pull before apply
  config:
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
      hostmetrics:
        collection_interval: 30s
        scrapers:
          cpu: {}
          memory: {}
    processors:
      memory_limiter:
        check_interval: 1s
        limit_percentage: 75
        spike_limit_percentage: 20
      batch: {}
    exporters:
      otlp:
        endpoint: otelcol-gateway:4317
        tls:
          insecure: false
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [otlp]
        metrics:
          receivers: [otlp, hostmetrics]
          processors: [memory_limiter, batch]
          exporters: [otlp]
```

The `Instrumentation` CR enables auto-injection into application pods. Cross-ref `layers/mesh.md §OTel Operator Instrumentation CR`.

### 5.2 Terraform: Helm Release or kubernetes_manifest

```hcl
resource "helm_release" "otel_operator" {
  name       = "opentelemetry-operator"
  repository = "https://open-telemetry.github.io/opentelemetry-helm-charts"
  chart      = "opentelemetry-operator"
  namespace  = "observability"
  version    = "0.78.0"   # pin to current stable; verify via `helm search repo opentelemetry/opentelemetry-operator --versions`

  set {
    name  = "manager.collectorImage.repository"
    value = "otel/opentelemetry-collector-contrib"
  }
}

resource "kubernetes_manifest" "otelcol_daemonset" {
  manifest = yamldecode(file("${path.module}/manifests/otelcol-daemonset.yaml"))
  depends_on = [helm_release.otel_operator]
}
```

### 5.3 CI Validation for Collector Config

```bash
# otelcol validate — verifies config syntax and component availability
otelcol validate --config otelcol-config.yaml
```

Add to CI before any collector config change merges.

---

## 6. Burn-Rate Multi-Window Alert (Concrete PromQL)

Multi-window burn-rate alerting from `boundaries/slo.md §7`. Two alert pairs protect against fast budget exhaustion and slow invisible drain.

- **Fast burn**: 2% budget consumed in 1h (14.4x rate), gated by a 5m short window to suppress transient spikes.
- **Slow burn**: 5% budget consumed in 6h (6x rate), gated by a 30m short window.

Error budget denominator for 99.9% SLO: `1 - 0.999 = 0.001`.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: checkout-slo-burn-rate
  namespace: monitoring
  labels:
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
    - name: checkout.slo.burn_rate
      interval: 30s
      rules:
        # Fast burn — 1h / 5m windows
        - alert: CheckoutSLOBurnRateFast
          expr: |
            (
              sum(rate(http_requests_total{service="checkout",status=~"5.."}[1h]))
              / sum(rate(http_requests_total{service="checkout"}[1h]))
            ) / (1 - 0.999) > 14.4
            and
            (
              sum(rate(http_requests_total{service="checkout",status=~"5.."}[5m]))
              / sum(rate(http_requests_total{service="checkout"}[5m]))
            ) / (1 - 0.999) > 14.4
          for: 2m
          labels:
            severity: critical
            slo: checkout-availability
          annotations:
            summary: "Checkout SLO fast burn: budget exhausting in < 1h"
            description: >
              Error rate is {{ $value | humanizePercentage }} of the error budget rate.
              At this pace the 28-day budget is exhausted in under 1 hour.
            runbook: "https://wiki.example.com/runbooks/checkout-slo-fast-burn"

        # Slow burn — 6h / 30m windows
        - alert: CheckoutSLOBurnRateSlow
          expr: |
            (
              sum(rate(http_requests_total{service="checkout",status=~"5.."}[6h]))
              / sum(rate(http_requests_total{service="checkout"}[6h]))
            ) / (1 - 0.999) > 6
            and
            (
              sum(rate(http_requests_total{service="checkout",status=~"5.."}[30m]))
              / sum(rate(http_requests_total{service="checkout"}[30m]))
            ) / (1 - 0.999) > 6
          for: 15m
          labels:
            severity: warning
            slo: checkout-availability
          annotations:
            summary: "Checkout SLO slow burn: budget exhausting in < 5 days"
            description: >
              Error rate is {{ $value | humanizePercentage }} of the error budget rate.
              At this pace the 28-day budget is exhausted in under 5 days.
            runbook: "https://wiki.example.com/runbooks/checkout-slo-slow-burn"
```

---

## 7. GitOps Integration

### 7.1 Argo CD

Source: <https://argoproj.github.io/cd/>

Use the app-of-apps pattern to manage the observability stack as a first-class application:

```
observability-root (App of Apps)
├── prometheus-stack (App)
├── otel-operator (App)
├── otelcol-daemonset (App)       ← OpenTelemetryCollector CRD
├── otelcol-gateway (App)
├── dashboards (App)              ← Grafonnet rendered JSON in ConfigMap
└── alert-rules (App)             ← PrometheusRule CRDs
```

Argo CD reconciles git state to the cluster on every commit. Drift is detected and surfaced in the Argo CD UI.

### 7.2 Flux (CNCF Graduated)

Source: <https://fluxcd.io>

Flux `Kustomization` resources watch git paths and apply them to the cluster. Flagger (a Flux ecosystem project) handles progressive delivery gates; cross-ref `boundaries/release.md` for canary/blue-green integration.

```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: observability-alerts
  namespace: flux-system
spec:
  interval: 5m
  path: ./alerts
  prune: true
  sourceRef:
    kind: GitRepository
    name: observability-repo
```

### 7.3 Self-Observability of the GitOps Stack

The observability stack is itself observed via meta-observability. Cross-ref `meta-observability.md §Section A` for the pipeline self-health metrics that must remain green to trust any downstream alert or dashboard. Argo CD and Flux both expose Prometheus metrics; scrape them with the same DaemonSet collector.

---

## 8. Immutable Review Workflow

All observability changes flow through the same review gate as application code:

| Step | Tool | What is checked |
|------|------|----------------|
| Open PR | GitHub / GitLab | Diff shows intent |
| CI: PromQL / rule lint | `promtool check rules` | PrometheusRule validity |
| CI: Collector config lint | `otelcol validate` | Collector YAML syntax and component graph |
| CI: Jsonnet lint | `jsonnetfmt --test` | Grafonnet formatting |
| CI: Dashboard lint | Grafana dashboard linter | Panel best practices (no orphan panels, datasource vars set) |
| Peer review | GitHub CODEOWNERS | At least one SRE approval |
| Merge → main | CI gate | All checks must pass |
| CD apply | Argo CD / Flux | Reconciles to cluster; Argo CD marks Synced |

Audit trail: git history is the authoritative change log. Cross-ref `signals/audit.md` for the WORM immutability requirement on audit logs generated from this pipeline.

---

## 9. Secrets Management for Observability Stack

Backend API keys, Grafana service account tokens, and OAuth client secrets MUST NOT be committed to git; even in encrypted form inside dashboard YAML.

| Pattern | Tool | Notes |
|---------|------|-------|
| Kubernetes secrets from Vault | External Secrets Operator | Syncs Vault path → Kubernetes Secret |
| Encrypted secrets in git | Sealed Secrets (Bitnami) | `SealedSecret` CRD; encrypted with cluster public key |
| Cloud-native secret store | AWS Secrets Manager, GCP Secret Manager | Pull via ESO provider |

Reference secrets in `OpenTelemetryCollector` CRD and Terraform via `secretKeyRef`, never inline:

```yaml
env:
  - name: GRAFANA_API_TOKEN
    valueFrom:
      secretKeyRef:
        name: grafana-credentials
        key: api-token
```

---

## 10. Environment Separation

| Concern | Approach |
|---------|----------|
| Separate namespaces | `observability-dev`, `observability-staging`, `observability-prod` |
| Dashboard parameterization | Grafana template variable `$env` filters all queries; built into Grafonnet at render time |
| Kustomize overlays | Base config + `overlays/dev`, `overlays/prod` for resource limits, replica counts, retention |
| Helm values | `values-dev.yaml`, `values-prod.yaml` per environment |

Never duplicate dashboard JSON per environment. Parameterize via variables; let the environment label filter do the work.

---

## 11. Testing Observability-as-Code

| Test type | Method | Tooling |
|-----------|--------|---------|
| Dashboard preview | Import rendered JSON into ephemeral Grafana via API; visual check | `grafana-cli`, Grafana HTTP API |
| Alert firing validation | Inject synthetic metric spike (set error counter to trigger threshold); confirm alert fires and routes | `promtool test rules` with `rule_test` fixtures |
| Collector config unit test | Feed sample OTLP telemetry fixtures through `otelcol` in test mode | `otelcol_test` package; `file` exporter to assert output |

```yaml
# promtool rule test — alerts/checkout_slo_test.yaml
rule_files:
  - checkout-slo-burn-rate.yaml

tests:
  - interval: 1m
    input_series:
      - series: 'http_requests_total{service="checkout",status="500"}'
        values: "0+100x10"    # 100 errors/min for 10 minutes
      - series: 'http_requests_total{service="checkout",status="200"}'
        values: "0+900x10"    # 900 success/min
    alert_rule_test:
      - eval_time: 10m
        alertname: CheckoutSLOBurnRateFast
        exp_alerts:
          - exp_labels:
              severity: critical
              slo: checkout-availability
```

---

## 12. Perses CNCF Sandbox Specifics

Source: <https://perses.dev> | CNCF Sandbox status as of 2026-Q2.

- YAML schema is stricter and more opinionated than Grafana JSON; dashboards are more portable but migration from Grafana JSON requires manual mapping (import tooling is under active development).
- Perses CLI (`percli`) provides `apply`, `get`, `delete`, and `lint` subcommands mirroring `kubectl`.
- Use Perses when: the organization requires a vendor-neutral CNCF-blessed format, or is evaluating a Grafana alternative.
- Do not use Perses as the sole dashboard-as-code solution in production yet unless the team accepts the current schema instability risk. Track promotion to CNCF Incubating as the stability signal.

---

## 13. Matrix Coverage Note

Observability-as-code is cross-cutting: it covers every row in `matrix.md` (all layers, all boundaries, all signals). It has no dedicated matrix row. Think of it as the delivery mechanism for everything else in the skill; the how, not the what.

---

## 14. Anti-Patterns

Append to `anti-patterns.md §Section I: As-Code and GitOps`:

| # | Anti-pattern | Consequence | Fix |
|---|-------------|-------------|-----|
| I-1 | Dashboards edited directly in Grafana UI | No rollback; peer review bypassed; drift from git within hours | All edits via PR; Grafana provisioning blocks UI edits (`allowUiUpdates: false`) |
| I-2 | Alert rules without automated tests | False alarms or silent failures reach production undetected | Add `promtool test rules` fixtures to CI for every new alert |
| I-3 | Secrets (API tokens, DSNs) committed in dashboard YAML or `values.yaml` | Credentials exposed in git history permanently | External Secrets Operator or Sealed Secrets; never inline |
| I-4 | No CI validation for PrometheusRule or Collector config | Invalid YAML / broken PromQL deployed to production; alerts silently stop firing | `promtool check rules` + `otelcol validate` in CI gate |
| I-5 | Environment-specific dashboards duplicated instead of parameterized | N copies of the same dashboard diverge; maintenance multiplied by N | Template variable `$env`; single parameterized source of truth |

---

## Cross-References

| Topic | File |
|-------|------|
| SLO math and error budget formula | `boundaries/slo.md §4` |
| OpenSLO YAML full example | `boundaries/slo.md §5` |
| Burn-rate multi-window alert math | `boundaries/slo.md §7` |
| OTel Operator four deployment modes | `transport/collector-topology.md §1` |
| Instrumentation CR for auto-injection | `layers/mesh.md §OTel Operator Instrumentation CR` |
| Pipeline self-health alerts (meta) | `meta-observability.md §Section F` |
| WORM immutability for audit trail | `signals/audit.md` |
| Progressive delivery (Flagger) | `boundaries/release.md` |
| Anti-patterns full list | `anti-patterns.md` |

---

## Primary Sources

- Grafonnet 11.x: <https://grafana.github.io/grafonnet/>
- Perses: <https://perses.dev>
- PrometheusRule CRD: <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>
- Terraform Grafana provider: <https://github.com/grafana/terraform-provider-grafana>
- OTel Operator: <https://github.com/open-telemetry/opentelemetry-operator>
- Argo CD: <https://argoproj.github.io/cd/>
- Flux: <https://fluxcd.io>
- OpenSLO: <https://openslo.com>
- Sloth: <https://sloth.dev>
- Pyrra: <https://github.com/pyrra-dev/pyrra>
