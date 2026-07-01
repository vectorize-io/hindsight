---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
specs:
  - "W3C Baggage: Recommendation 2022-12-22"
  - "GDPR: Regulation (EU) 2016/679; KR PIPA: amended 2023"
notes:
  - "OpenCost: CNCF Incubating (advanced 2024-10-31)"
---

# Multi-Tenant Observability

## 1. Scope

Multi-tenant observability covers the collection, routing, isolation, attribution, and residency of telemetry signals in a B2B SaaS platform where multiple customer tenants share underlying infrastructure.

**In scope:**
- Tenant isolation strategy across four tiers (soft / routing / dedicated-collector / dedicated-backend)
- Tenant attribute propagation via W3C Baggage
- Per-tenant sampling policies
- Per-tenant retention schedules
- Cost attribution per tenant (chargeback / showback)
- Data residency routing for GDPR (EU) and PIPA (KR) tenants

**Out of scope (related but distinct):**
- Cross-service propagation mechanics: see `cross-application.md` (propagators, baggage rules)
- FinOps unit economics and OpenCost metric surface; see `../signals/cost.md`
- PII redaction and anonymization rules per tenant; see `../signals/privacy.md`

---

## 2. OTel Attribute Conventions for Tenant

OpenTelemetry Semantic Conventions do not include a stable `tenant.*` group as of semconv 1.27.0. The attributes below are **custom application-defined** attributes. Follow the naming convention from `../standards.md §3` and prefix with the domain noun to avoid collisions with future OTel semconv additions.

| Attribute | Type | Example | Status |
|-----------|------|---------|--------|
| `tenant.id` | string | `"acme-corp"` | Custom (not OTel Stable); use consistently across all signals |
| `tenant.tier` | string enum | `"free"` / `"pro"` / `"enterprise"` | Custom; drives sampling and routing decisions |
| `tenant.region` | string | `"eu-west-1"` / `"ap-northeast-2"` | Custom; drives data residency routing |

**Naming rationale:** dot-separated namespace (`tenant.*`) matches OTel semconv style and avoids the underscore ambiguity common in Prometheus label names. Do not use `customer_id`, `org_id`, or `account_id` for the same concept; pick one canonical key and propagate it everywhere.

---

## 3. Four-Tier Isolation Strategy

Most B2B SaaS organizations apply a mix of tiers: enterprise tenants get Tier 3 or 4, pro gets Tier 2, and free gets Tier 1. Select the highest tier required by the tenant's compliance obligations.

| Tier | Description | Relative Cost | Isolation Strength | Compliance Fit |
|------|-------------|---------------|--------------------|----------------|
| 1. Soft | Shared collector + shared backend; tenants are separated only by `tenant.id` label filtering in dashboards and queries | Low | Weak; no pipeline isolation; noisy-neighbor risk | Basic B2B without data separation requirements |
| 2. Routing | Shared collector pool; `routing_connector` or `tail_sampling` sub-policies split pipelines by tenant tier; still shared backend | Medium | Medium; pipeline isolation; shared storage | Regulated tiers with data processing agreements |
| 3. Dedicated collector | Per-tenant collector instance in a dedicated Kubernetes namespace; isolates ingestion and processing; shared or per-region backend | High | Strong; ingestion isolated; namespace-level blast radius | Enterprise tenants, HIPAA, ISO 27001 requirements |
| 4. Dedicated backend | Per-tenant observability backend project or account (e.g., separate Grafana org, separate Datadog account, separate GCP project) | Highest | Strongest; full stack isolation from ingestion to storage | Highest compliance obligations (FedRAMP, SOC 2 Type II per tenant, GDPR Art. 28 sub-processor separation) |

**Routing connector alpha caveat:** Tier 2 using `routing_connector` is subject to the alpha stability warning documented in `../transport/sampling-recipes.md §4`. For production Tier 2 deployments, prefer `tail_sampling` with `and` sub-policies (stable) over `routing_connector` (alpha as of 2025).

---

## 4. Tenant ID Propagation

Tenant context must be carried across service boundaries so every span, metric, and log record emitted by any service is attributable to its originating tenant.

**Mechanism:** W3C Baggage (`baggage` header, Recommendation 2022-12-22). Set `tenant.id` and `tenant.tier` as baggage entries at the ingress gateway. All downstream services read from the OTel Baggage API and apply these values as resource or span attributes.

```
Ingress Gateway
  → set baggage: tenant.id=acme-corp, tenant.tier=enterprise
      ↓
Service A (reads baggage → sets span attribute tenant.id)
      ↓
Service B (reads baggage → sets span attribute tenant.id)
```

**Trust-boundary warning:** W3C Baggage is visible to every service in the propagation chain, including third-party or external services. The W3C Baggage specification (https://www.w3.org/TR/baggage/) explicitly notes that baggage values cross trust boundaries. Carrying `tenant.id` to external egress endpoints leaks customer account existence information.

Rule: strip or validate `tenant.*` baggage entries at the egress gateway before forwarding to any external third-party endpoint. Internal propagation only. Cross-ref `../signals/privacy.md §Common PII in Telemetry` for baggage PII rules.

---

## 5. Per-Tenant Sampling

Different tenant tiers justify different sampling rates. Enterprise tenants have SLA obligations and full debugging requirements; free tenants justify only ambient visibility.

**Per-tier retention targets:**

| Tenant Tier | Trace Sampling Rate | Rationale |
|-------------|---------------------|-----------|
| `enterprise` | 100% | SLA obligations, full debugging capability, compliance audit trail |
| `pro` | 20% | Representative sample, cost-controlled |
| `free` | 2% | Ambient visibility only |

**Recommended configuration; `tail_sampling` with `and` sub-policies (stable, production-safe):**

```yaml
processors:
  tail_sampling:
    decision_wait: 30s
    num_traces: 100000
    expected_new_traces_per_sec: 1000
    policies:
      - name: enterprise
        type: and
        and:
          and_sub_policy:
            - name: tier-check
              type: string_attribute
              string_attribute:
                key: tenant.tier
                values: ["enterprise"]
            - name: probabilistic
              type: probabilistic
              probabilistic:
                sampling_percentage: 100

      - name: pro
        type: and
        and:
          and_sub_policy:
            - name: tier-check
              type: string_attribute
              string_attribute:
                key: tenant.tier
                values: ["pro"]
            - name: probabilistic
              type: probabilistic
              probabilistic:
                sampling_percentage: 20

      - name: free-baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 2   # catches free tier and unmatched traffic
```

Cross-ref `../transport/sampling-recipes.md §4` for the full tenant-aware sampling recipe including `routing_connector` Option A (alpha) and the combined error + cost + tenant four-policy example.

---

## 6. Per-Tenant Retention

Retention schedules must be enforced per tier. Hot storage is fast-query; warm is compressed but queryable; cold is archival with retrieval latency.

| Tier | Hot | Warm | Cold |
|------|-----|------|------|
| Enterprise | 90 days | 1 year | 3 years |
| Pro | 30 days | 90 days | None |
| Free | 7 days | 30 days | None |

**Implementation:** Apply Kubernetes-native or backend-native lifecycle policies keyed on the `tenant.id` and `tenant.tier` labels. For shared backends (Tier 1–2), use label-based TTL rules or index lifecycle management (e.g., OpenSearch ISM policies, Loki retention rules, Thanos compactor retention). For dedicated backends (Tier 3–4), set backend project-level retention per tenant.

Cross-ref `../meta-observability.md §Retention Matrix` for the full retention policy table covering all seven signals.

---

## 7. Cost Attribution

Tenant-level cost attribution enables chargeback (billing tenants for their resource consumption) and showback (internal reporting without billing).

**Kubernetes workload labeling:** Tag every pod at deploy time with `tenant.id` as a Kubernetes label. OpenCost reads workload labels and produces `opencost_workload_cost_total{tenant_id="acme-corp"}` automatically.

```yaml
# Kubernetes Pod template label (applied via Helm values or admission webhook)
metadata:
  labels:
    tenant.id: "acme-corp"
    tenant.tier: "enterprise"
```

**Telemetry cost attribution:** The observability bill itself (e.g., Datadog RUM sessions per tenant, Honeycomb events per tenant) must also be attributed. Instrument the collector pipeline with throughput counters per tenant to apportion the observability bill proportionally.

**Cardinality constraint:** `tenant.id` used as a metric label must be capped at the top-N active tenant count (100–1,000 tenants is safe for most TSDBs). Beyond this threshold, bucket overflow tenants under an `"other"` label to prevent metric series explosion. Cross-ref `../meta-observability.md §Cardinality Guardrails`.

Cross-ref `../signals/cost.md §4 Cost Attribution by Dimension` for FinOps unit economics (per-tenant PromQL formulas, OpenCost architecture, and FOCUS spec).

---

## 8. Data Residency

GDPR Chapter V (https://gdpr-info.eu/chapter-5/) restricts transfers of personal data outside the EU/EEA to countries or organizations that provide adequate protection. Korean PIPA (https://www.pipa.go.kr) applies equivalent restrictions for Korean resident data.

**Routing rules:**

| Tenant Region | Collector Placement | Backend Placement | Cross-Region Allowed? |
|---------------|--------------------|--------------------|----------------------|
| EU (`eu-*`) | EU-region edge collector | EU-region backend only | No; GDPR Chapter V |
| KR (`ap-northeast-2`) | KR-region edge collector | KR-region backend only | No; PIPA |
| US (`us-*`) | US-region collector | US-region or global backend | Yes (to non-EU/KR) |
| Other | Regional or global collector | Regional or global backend | Yes (check bilateral agreements) |

**Topology:** deploy per-region edge collectors that aggregate locally and export only to backends in the same region. No cross-region OTLP export for EU or KR tenants.

```
EU Tenants → EU Edge Collector → EU Backend (e.g., eu-west-1 Grafana Cloud)
KR Tenants → KR Edge Collector → KR Backend (e.g., ap-northeast-2 region)
US Tenants → US Edge Collector → US Backend or global aggregator
```

Route by `tenant.region` at the ingress gateway before data enters the collector pipeline. Do not allow EU or KR tenant telemetry to flow through a non-compliant region, even transiently.

**Source-of-truth rule (critical)**: `tenant.region` MUST be resolved from an internal, server-side authoritative source; tenant registry service, organization metadata table, or IdP claim stamped at session start. It MUST NOT be trusted from client-supplied input (HTTP header, baggage, query string, or JWT claim the client itself controls). A misconfigured or malicious tenant could otherwise self-declare a non-EU/KR region and bypass residency routing. Enforce at the ingress gateway: reject requests where a client-declared `tenant.region` disagrees with the registry lookup keyed on `tenant.id`. In practice, strip any inbound `tenant.region` attribute and re-attach the registry-sourced value before the Collector pipeline accepts the span/log.

Cross-ref `../transport/collector-topology.md §7 Federated / Multi-Cluster` for the multi-region edge topology diagram. Cross-ref `../signals/privacy.md §2 Regulatory Drivers` for GDPR and PIPA penalty context and `../signals/privacy.md §Cross-border transfer` for PII-specific cross-border rules.

---

## 9. Tenant Onboarding and Offboarding

**Onboarding checklist:**
1. Provision `tenant.id` and `tenant.tier` as Kubernetes labels on all tenant workloads (via Helm values or admission webhook).
2. Create per-tenant routing rule if Tier 2+.
3. Provision per-tenant collector namespace if Tier 3+.
4. Create per-tenant backend project or organization if Tier 4.
5. Create per-tenant dashboard folder with RBAC rules (see §10).
6. Register tenant in cardinality allowlist (top-N cap enforcement).

Automate steps 1–6 as code; cross-ref `../observability-as-code.md` for provisioning patterns.

**Offboarding; GDPR Art. 17 Right to Erasure:**

When a tenant terminates their contract, all telemetry data containing `tenant.id` must be deleted across every storage tier (hot, warm, cold) and every signal (metrics, logs, traces, profiles, cost records, audit records). This is a legal obligation under GDPR Art. 17, not an engineering convenience.

Offboarding procedure:
1. Trigger deletion job across all backends scoped to `tenant.id`.
2. Remove tenant from cardinality allowlist and routing rules.
3. Deprovision collector namespace (Tier 3) or backend account (Tier 4).
4. Emit an audit event recording the erasure action, timestamp, and operator identity.

Cross-ref `../signals/audit.md` for audit event schema for offboarding erasure events.

---

## 10. Dashboard Isolation

| Tier | Dashboard Isolation Mechanism |
|------|-------------------------------|
| Tier 1 | Grafana folder per tenant; dashboard variables filter by `tenant.id` label |
| Tier 2 | Grafana folder per tenant; Grafana RBAC restricts folder access by team |
| Tier 3 | Grafana organization per tenant; or Honeycomb environment per tenant |
| Tier 4 | Dedicated backend instance; tenant admin is org owner in their own account |

**RBAC rule:** a tenant admin identity claim must be mapped to a Grafana or backend role that scopes data access strictly to that tenant's `tenant.id`. Cross-org or cross-tenant data leakage via dashboard query is a compliance violation.

Cross-ref `../signals/privacy.md §Backend RBAC` for Grafana RBAC configuration patterns and OPA policy rules for query-level tenant isolation.

---

## 11. Noisy Neighbor Protection

Shared infrastructure (Tier 1 and Tier 2) is vulnerable to one high-volume tenant degrading the observability pipeline for all other tenants.

**Controls:**

| Control | Scope | Mechanism |
|---------|-------|-----------|
| Per-tenant ingress rate limit | Collector receiver level | `ratelimiter` extension (alpha) or `filter` processor tied to per-tenant token-bucket state; pair with `memory_limiter` as backpressure |
| Per-tenant cardinality quota | Metrics pipeline | Top-N series cap per `tenant.id`; overflow bucketed as `"other"` |
| Circuit breaker on ingress | Collector pipeline | `memory_limiter` processor per-tenant pipeline; shed load when memory exceeds threshold |
| Per-tenant queue depth limit | Exporter queue | `sending_queue` max size per tenant pipeline (Tier 2 routing) |

Apply rate limits at the first collector tier (agent or edge). A tenant exceeding its quota must receive a well-defined error (e.g., OTLP `ResourceExhausted` gRPC status) rather than silently dropping data.

---

## 12. Matrix Cells: Multi-Tenant Row

Quick navigation for multi-tenant boundary cells in `../matrix.md`:

| Layer | Signal | Status | Artifact |
|-------|--------|--------|----------|
| L3-network | metrics | PASS | VPC flow logs per tenant; egress bytes attributed by source CIDR mapped to `tenant.id` |
| L7-application | traces | PASS | W3C Baggage carries `tenant.id`; all spans tagged at ingress |
| L7-application | cost | PASS | OpenCost workload attribution via `tenant.id` pod label; per-tenant PromQL aggregation |
| L7-application | privacy | PASS | Per-tenant PII redaction rules; per-tenant residency routing |
| L7-application | audit | PASS | Per-tenant audit trail; erasure events on offboarding |

---

## 13. Anti-Patterns

The following are candidates for `../anti-patterns.md §Multi-Tenant`:

| Anti-Pattern | Impact | Correction |
|-------------|--------|------------|
| `tenant.id` as metric label without top-N cap | Cardinality explosion in TSDB; ingestor OOM; query timeouts for all tenants | Enforce top-N cap (100–1,000); bucket overflow as `"other"`; cross-ref `../meta-observability.md §Cardinality Guardrails` |
| `tenant.id` in W3C Baggage crossing trust boundaries | Tenant account existence leaks to third-party services; GDPR personal data transfer without legal basis | Strip `tenant.*` baggage at egress gateway before forwarding to any external endpoint |
| Shared backend for regulated tiers (Tier 1 for compliance tenants) | Co-mingled data violates data processing agreements; one breach affects all tenants | Upgrade regulated tenants to Tier 3 or 4; apply isolation tier based on contractual obligation, not cost convenience |
| Cross-region OTLP export for EU or KR tenants | GDPR Chapter V violation; personal data transfer to non-adequate third country; regulatory fine risk | Route EU/KR telemetry to region-local backends exclusively; enforce at ingress gateway by `tenant.region` |
| No tenant offboarding erasure process | GDPR Art. 17 violation; deleted tenant data persists in hot/warm/cold tiers and backup snapshots | Implement automated erasure job scoped by `tenant.id` across all storage tiers; emit audit event per erasure |

---

## Cross-References

| Topic | File |
|-------|------|
| Baggage propagation mechanics and trust-boundary rules | `cross-application.md` |
| Full FinOps cost attribution and OpenCost metric surface | `../signals/cost.md` |
| PII redaction, anonymization, and GDPR/PIPA regulatory detail | `../signals/privacy.md` |
| Tenant-aware sampling recipes (routing_connector, tail_sampling) | `../transport/sampling-recipes.md §4` |
| Multi-cluster and regional collector topology | `../transport/collector-topology.md §7` |
| Retention matrix for all seven signals | `../meta-observability.md §Retention Matrix` |
| Cardinality guardrails and top-N cap | `../meta-observability.md §Cardinality Guardrails` |
| Dashboard RBAC and query-level tenant isolation | `../signals/privacy.md §Backend RBAC` |
| Audit event schema for offboarding erasure | `../signals/audit.md` |
| Observability-as-code provisioning for tenant onboarding | `../observability-as-code.md` |
| Full 112-cell coverage matrix | `../matrix.md` |
