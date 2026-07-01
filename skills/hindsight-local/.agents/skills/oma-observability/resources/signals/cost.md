---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
specs:
  - "FOCUS Spec: 1.0 (FinOps Open Cost and Usage Specification)"
notes:
  - "OpenCost: CNCF Incubating (advanced 2024-10-31)"
---

# Cost Signal

## 1. Scope & Boundary

Cost is the 6th signal in the `oma-observability` skill (design decision D4: cost as 1st-class signal, standalone). It sits alongside metrics, logs, traces, profiles, audit, and privacy in the 4 × 4 × 7 coverage matrix.

**In scope:**
- Cost as a telemetry signal: metric surface (OpenCost Prometheus exposition), attribution rules, and unit economics
- Kubernetes-native cost allocation via OpenCost (CNCF Incubating)
- Per-tenant cost attribution (B2B SaaS) and per-feature cost deltas
- LLM/AI token-cost observability at the span attribute level
- Cross-cloud egress cost attribution
- Retention policy and access-control rules for cost data
- Cardinality rules specific to cost labels

**Out of scope:**
- Full FinOps strategy: procurement, Reserved Instances / Savings Plans, commitment analysis, contract negotiation; see [FinOps Foundation framework](https://www.finops.org/framework/)
- Full LLM observability (prompt versioning, evals, model routing economics); see Langfuse, Arize Phoenix, LangSmith
- Cluster cost optimization (rightsizing, spot orchestration): see Kubecost commercial tier or CloudZero

**Distinct boundary with `../boundaries/multi-tenant.md §Cost Attribution`:** that file uses cost as a _routing signal_ (tenant chargeback, showback, residency). This file defines cost as a _telemetry signal_; what to collect, how to attribute it, and how to store it. Cross-reference is mandatory.

---

## 2. Why Cost Is a 1st-Class Signal

Cost differs from metrics, logs, and traces in three structural ways:

| Dimension | Metrics / Logs / Traces | Cost |
|-----------|------------------------|------|
| Data source | Instrumented applications + infra agents | Cloud billing APIs + Kubernetes allocation engine |
| Primary consumer | Engineering, SRE | Finance, Platform Engineering |
| Retention requirement | 30–90 days typical | 2 years (financial audit compliance) |
| Granularity | Per-request, sub-second | Per-hour or per-day billing aggregates |

Unit economics is the motivation: understanding `$ per request`, `$ per tenant`, and `$ per feature` requires correlating cloud billing data with application-level telemetry. Neither data source alone is sufficient.

FinOps Foundation identifies cost visibility as a foundational practice across three phases: Inform → Optimize → Operate. The Inform phase is entirely an observability problem; cost data must be collected, attributed, and surfaced before optimization can occur.

---

## 3. OpenCost (CNCF Incubating)

Sources:
- <https://www.opencost.io/>
- <https://www.cncf.io/blog/2024/10/31/opencost-advances-to-cncf-incubator/>
- <https://github.com/opencost/opencost>

OpenCost is the open-source Kubernetes cost monitoring standard. It advanced to CNCF Incubating status on 2024-10-31. The OSS core is maintained by IBM Kubecost (post-acquisition) and the community. Kubecost's commercial tier adds UI, recommendations, and multi-cluster federation on top of the OpenCost core.

### 3.1 Architecture

```
Cloud Billing API (AWS CUR / GCP Billing Export / Azure Cost Management)
        |
        v
+----------------+        /metrics (Prometheus format)
| OpenCost Pod   | -----> Prometheus scrape -----> OTel Collector prometheusreceiver
| (namespace:    |                                          |
|  opencost)     |                                          v
+----------------+                               Your TSDB backend
        |                                        (Thanos / Mimir / VictoriaMetrics)
        v
Kubernetes API (node prices, pod labels, PVC claims)
```

The OpenCost pod reads node price data from a cloud-provider price configmap (or the cloud billing API directly), reads Kubernetes resource allocations from the kubelet and API server, and exposes Prometheus metrics at port 9003.

### 3.2 Key Metrics Exposed

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `opencost_namespace_cost_total` | Counter | USD | Cumulative cost per Kubernetes namespace |
| `opencost_workload_cost_total` | Counter | USD | Cumulative cost per Deployment / StatefulSet |
| `opencost_cpu_cost` | Gauge | USD/hr | CPU allocation cost, current window |
| `opencost_ram_cost` | Gauge | USD/hr | Memory allocation cost, current window |
| `opencost_network_cost` | Gauge | USD/hr | Network egress cost attribution |
| `opencost_storage_cost` | Gauge | USD/hr | PVC / persistent storage cost |
| `opencost_load_balancer_cost` | Gauge | USD/hr | Cloud load balancer cost |
| `opencost_cluster_management_cost` | Gauge | USD/hr | Managed control plane fee (GKE, EKS, AKS) |

### 3.3 Scrape Configuration (OTel Collector)

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

### 3.4 Allocation Model

Three allocation categories: compute (CPU + RAM), storage (PVC), and network (egress bytes). Each workload receives a share of node cost proportional to resource requests. Idle cost distributes across active workloads or a dedicated `__idle__` namespace via configurable rules. Cross-cloud pricing is supplied via a price configmap; on-prem environments set custom `$/CPU-hr` and `$/GB-RAM-hr` values.

---

## 4. Cost Attribution by Dimension

| Dimension | Kubernetes mechanism | Label / attribute |
|-----------|---------------------|-------------------|
| Namespace / team | Namespace label → team ownership mapping | `namespace`, `team` |
| Workload | Deployment / StatefulSet / DaemonSet name | `workload`, `controller_kind` |
| Tenant (B2B SaaS) | `tenant.id` label on pods + W3C Baggage propagation | `tenant_id` |
| Feature | Custom pod label (`feature.name`) | `feature_name` |
| Per-request (LLM) | `gen_ai.cost.total_usd` span attribute | trace attribute, not metric label |

**Tenant attribution** is the primary B2B SaaS use case: tag every pod with `tenant.id` at deploy time and OpenCost aggregates cost by that label automatically. Cross-ref `../boundaries/multi-tenant.md §Cost Attribution` for chargeback and showback patterns. **Feature attribution** uses a custom pod label (`feature.name`); cross-ref `../boundaries/release.md` for A/B cost comparison. **Per-request attribution** for LLM workloads uses span attributes only; writing cost to a metric label at request granularity causes a cardinality explosion (see Section 10).

---

## 5. Unit Economics

Three primary PromQL formulas for cost-to-business-metric conversion:

**Per-request cost (rolling 1-hour window):**

```promql
sum(rate(opencost_workload_cost_total{workload="checkout-api"}[1h])) * 3600
  /
sum(rate(http_requests_total{job="checkout-api"}[1h]))
```

**Per-tenant cost (daily, aggregated):**

```promql
sum by (tenant_id) (
  increase(opencost_workload_cost_total[24h])
  * on(namespace) group_left(tenant_id)
  kube_namespace_labels
)
```

> **Dependency**: the query joins against `kube_namespace_labels` from kube-state-metrics. `kube-state-metrics` by default does NOT expose custom namespace labels (only a small allowlist). To surface `tenant_id`, run kube-state-metrics with `--metric-labels-allowlist=namespaces=[tenant_id,tenant_tier]` (or via Helm `metricLabelsAllowlist`). Without this flag, the query silently returns empty; a common production pitfall.

**Per-namespace cost (current rate, USD/hr):**

```promql
sum by (namespace) (
  opencost_cpu_cost + opencost_ram_cost + opencost_storage_cost + opencost_network_cost
)
```

**SLO + cost trade-off (gold tier vs silver tier delta):**

```promql
# Gold tier: high-replica, low-latency p99
sum(opencost_workload_cost_total{workload=~".*-gold"}) by (tenant_id)
  /
sum(opencost_workload_cost_total{workload=~".*-silver"}) by (tenant_id)
```

This ratio surfaces the cost multiplier of tiered SLO guarantees and feeds tier-pricing decisions.

---

## 6. LLM / AI Cost Observability

Full LLM observability is out of scope for this skill. The tools for that domain are Langfuse, Arize Phoenix, LangSmith, and Braintrust (see `SKILL.md §When NOT to use`). This section covers only the intersection where LLM cost surfaces as a span attribute in the OTel pipeline.

**Token-based pricing attributes** (OTel `gen_ai.*` semconv, currently Experimental tier per `../standards.md §Semconv Stability`):

| Span attribute | Description |
|----------------|-------------|
| `gen_ai.usage.input_tokens` | Prompt tokens consumed |
| `gen_ai.usage.output_tokens` | Completion tokens generated |
| `gen_ai.cost.total_usd` | Computed cost in USD for the span (vendor-specific extension) |

**Threshold-based sampling rule:** if `gen_ai.cost.total_usd > 0.50`, always retain the span regardless of sampling rate. This prevents silent budget overruns from being dropped before they reach the backend. Cross-ref `../transport/sampling-recipes.md §Cost-aware sampling` for the tail-sampler rule configuration.

Note: `gen_ai.*` semconv is Experimental as of semconv 1.27.0. Do not use these attributes as inputs to production SLOs until the group reaches Stable or RC.

---

## 7. Cross-Cloud Cost

Each cloud exposes billing data through a distinct API and export format. FOCUS (FinOps Open Cost and Usage Specification) is the data-format unification effort.

| Cloud | Billing data source | Export format | Ingestion path |
|-------|--------------------|----|------|
| AWS | Cost & Usage Report (CUR) | Parquet / CSV → S3 | Athena query → OpenCost price configmap |
| GCP | Billing Export | BigQuery table | BigQuery export job → OpenCost |
| Azure | Cost Management Export | CSV → Blob Storage | Storage export → OpenCost |
| On-prem | Manual price configmap | Custom `$/CPU-hr`, `$/GB-hr` | OpenCost configmap |

**FOCUS Spec** (<https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS_Spec>) provides a vendor-neutral column schema (`BilledCost`, `ResourceId`, `ServiceName`) for cross-cloud cost joins. When FOCUS exports are available from a cloud provider, prefer them over native formats. For the OSS multi-cluster path, run one OpenCost instance per cluster and aggregate in Thanos or Mimir by namespace and workload labels.

---

## 8. Retention Policy

| Data type | Retention | Rationale |
|-----------|-----------|-----------|
| Raw cloud billing data (CUR / GCP export) | 2 years | Financial audit compliance (SOX, ISO 27001) |
| OpenCost Prometheus metrics (hourly resolution) | 2 years | Chargeback evidence per tenant |
| Aggregated per-tenant / per-feature cost | Long-term (cold storage) | Historical unit economics trending |
| Per-request span cost attributes (`gen_ai.cost.total_usd`) | 30–90 days | Operational debugging window |

Cross-ref `../meta-observability.md §Retention Matrix` for the full retention policy table covering all seven signals.

---

## 9. Privacy & Access Control

Cost per tenant reveals revenue tier, contract value, and resource consumption patterns. Restrict access by role:

**RBAC separation:**

| Role | Dashboard access | Data access |
|------|-----------------|-------------|
| Finance | Full cost by tenant, contract | Billing export (read-only) |
| Platform Engineering | Cost by namespace, workload, cluster | OpenCost metrics (read-only) |
| Application Engineering | Cost by their own service / namespace | Filtered by namespace label |
| Tenant admin (self-serve) | Their own tenant cost only | Filtered by `tenant_id` claim |

Cross-ref `privacy.md §Backend RBAC` for the Grafana RBAC configuration pattern and OPA policy rules for dashboard-level tenant isolation.

---

## 10. Anti-Cardinality Rules for Cost Labels

Follow the same principles as metric labels (cross-ref `../meta-observability.md §Cardinality Guardrails`). Cost workloads add two additional risk surfaces: per-request LLM spans and dynamic tenant growth.

| Rule | Applies to | Rationale |
|------|-----------|-----------|
| `tenant.id` allowed as metric label with top-N cap (≤ 1000) | OpenCost metrics | Bounded tenant count; use `"other"` bucket for overflow |
| `namespace` and `workload` always allowed | OpenCost metrics | Bounded by cluster size |
| `gen_ai.cost.total_usd` as span attribute only; never as metric label | LLM spans | One series per request = cardinality explosion |
| `feature.name` allowed as pod label; allowed as metric label with cap | OpenCost workload attribution | Feature set is bounded; gate new features through label allowlist |
| `request.id` never as metric label for cost | Any | Unbounded cardinality; use trace attribute only |

Per-request cost labels are safe only as trace span attributes, where cardinality is handled by the trace backend (not a TSDB). Aggregate by `tenant.id`, `namespace`, and `workload` for metric surfaces.

---

## 11. Vendors

As of 2026-Q2. Verify current status at <https://landscape.cncf.io>.

Cross-ref `../vendor-categories.md §FinOps / Cost` for full selection guidance.

| Vendor | Type | Notes |
|--------|------|-------|
| OpenCost | OSS, CNCF Incubating | Kubernetes-native; Prometheus exposition; community-maintained |
| Kubecost | Commercial (IBM) | OpenCost OSS core + commercial UI, multi-cluster, recommendations |
| CloudZero | Commercial SaaS | Engineering-focused cost attribution; no Kubernetes agent required |
| Vantage | Commercial SaaS | Cross-cloud cost reporting; FOCUS spec early adopter |

---

## 12. Matrix Cells: Cost Column

Quick navigation for cost-column cells in `../matrix.md`:

| Layer | Boundary | Status | Detail |
|-------|----------|--------|--------|
| L3-network | multi-tenant | PARTIAL | Egress byte attribution by tenant VPC; cost proxy, not unit economics |
| L3-network | cross-application | PARTIAL | Cross-VPC egress cost attribution; unreliable without flow tagging |
| L4-transport | multi-tenant | PARTIAL | L4 byte volume per tenant as cost proxy; rolls up into compute |
| L4-transport | cross-application | PARTIAL | Cross-application byte volume; correlate with L7 for accuracy |
| mesh | multi-tenant | PARTIAL | Request-level cost attribution by tenant via mesh telemetry; feeds OpenCost |
| mesh | cross-application | PASS | Per-service byte/request counts from Envoy; unit cost cross-service |
| mesh | release | PARTIAL | Canary cost delta via per-version request counts in mesh telemetry |
| L7-application | multi-tenant | PASS | Primary use case: per-tenant OpenCost unit economics with `tenant.id` pod label |
| L7-application | cross-application | PASS | Per-service unit cost model; `gen_ai.cost.total_usd` span attribute for LLM |
| L7-application | slo | PASS | Gold-tier vs silver-tier cost delta; cost trade-off for SLO tier selection |
| L7-application | release | PASS | Canary cost delta: compare per-request cost across canary vs stable version |

L3 and L4 cost cells are PARTIAL because they produce a cost proxy (egress bytes) that informs FinOps egress billing but is insufficient for full unit-economics modeling. See `../matrix.md §C6` for the detailed caveat.

---

## 13. Anti-Patterns

Candidates for `../anti-patterns.md §Section B Cardinality & Cost`:

| Anti-pattern | Impact | Correction |
|-------------|--------|------------|
| Cost label at per-request metric granularity | Cardinality explosion in TSDB; OOM on ingestor | Use `gen_ai.cost.total_usd` as span attribute; aggregate cost metrics by tenant/namespace/workload |
| No `tenant.id` pod label | Cannot attribute cost to tenants; chargeback impossible | Apply `tenant.id` at deploy time via Helm values or admission webhook |
| Cost dashboard with public access | Reveals tenant revenue tier and contract value | Apply Grafana RBAC; finance and engineering views are separate; no public embedding |
| Ignoring egress cost in FinOps | Often the largest surprise cost in multi-cloud or CDN-heavy architectures | Include `opencost_network_cost` in all cost dashboards; set egress budget alerts |
| LLM spans not tail-sampled on cost threshold | Silent budget blowup: expensive spans are dropped before alerting | Configure tail-sampler rule: if `gen_ai.cost.total_usd > 0.50`, always retain |
| One metric name per tenant for cost | Bypasses TSDB cardinality controls; cannot be aggregated | Use `opencost_workload_cost_total{tenant_id="acme"}` with top-N cap |

---

## Cross-References

| Topic | File |
|-------|------|
| Tenant cost chargeback and routing | `../boundaries/multi-tenant.md §Cost Attribution` |
| Canary cost delta and A/B comparison | `../boundaries/release.md` |
| Tail-sampler cost-aware rules | `../transport/sampling-recipes.md §Cost-aware sampling` |
| Cardinality guardrails and top-N cap | `../meta-observability.md §Cardinality Guardrails` |
| Retention matrix (all signals) | `../meta-observability.md §Retention Matrix` |
| Cost dashboard RBAC | `signals/privacy.md §Backend RBAC` |
| Vendor selection (FinOps category) | `../vendor-categories.md §FinOps / Cost` |
| Full 112-cell matrix with cost column | `../matrix.md` |
| OpenCost metric surface (metrics signal) | `signals/metrics.md §8 OpenCost Metric Surface` |
| LLM / gen_ai semconv stability | `../standards.md §3 OTel Semconv Stability Tiers` |
