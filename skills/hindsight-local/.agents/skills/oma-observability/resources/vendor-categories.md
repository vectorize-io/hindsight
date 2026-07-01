---
notes:
  - "Example vendor lists are snapshots: verify via landscape.cncf.io before production selection"
---

# Observability Vendor Categories

> as of 2026-Q2; review quarterly. Verify live status at https://landscape.cncf.io

## Preamble: Why Categories, Not a Registry

This file is a **category taxonomy with timestamped example vendors**. It is not a vendor registry.

**Why this distinction matters:**

1. **Vendor names rot.** Examples from this codebase's own lifetime:
   - Keptn: archived by CNCF, 2025-09
   - Fluentd: deprecated by CNCF, 2025-10 (migration guide: Fluent Bit / OTel Collector)
   - Pyroscope: was CNCF Sandbox; acquired by Grafana 2023; CNCF Sandbox status is uncertain post-acquisition (verify at landscape.cncf.io before citing)

2. **Categories are stable.** "OSS full-stack", "SIEM", and "profiling specialist" have been coherent for years. The vendors filling them change.

3. **No duplication with vendor-owned skills.** The following published skills already describe themselves authoritatively:
   - `getsentry/sentry-sdk-setup`: Sentry SDK instrumentation
   - `honeycombio/agent-skill`: Honeycomb OTel setup (8+ published skills)
   - `Dash0 otel-instrumentation`: Dash0 OTel
   - `Microsoft Azure Monitor exporters`: Azure-specific exporters
   - `Datadog Labs dd-apm`: Datadog APM
   Duplicating their content here creates drift, not value.

4. **CNCF landscape is the authoritative live registry.** `landscape.cncf.io` tracks CNCF project status (Graduated / Incubating / Sandbox / Archived), vendor landscape, and category membership in real time. This file complements it with decision guidance; it does not copy or compete with it.

## Timestamp Discipline

All example vendor lists in this file are marked `as of 2026-Q2`.

**Review cadence:** quarterly (aligned with OTel spec release cadence).

When reviewing:
- Check CNCF project status changes at landscape.cncf.io
- Check acquisition or archival announcements
- Update the `as of YYYY-QX` markers on any changed sections

---

## Category Taxonomy

### (a) OSS Full-Stack

**Traits:** Self-hostable, covers metrics + logs + traces in one coherent stack, often includes UI, storage, and agents. No per-seat licensing. Operational burden is on the team.

**Example vendors** (as of 2026-Q2):

| Vendor | Metrics | Logs | Traces | Profiles | UI | Notes |
|--------|---------|------|--------|----------|----|-------|
| Grafana Labs LGTM+ | Mimir | Loki | Tempo | Pyroscope | Grafana | Alloy collector, Beyla eBPF auto-instr, Faro RUM, k6 load |
| Elastic Stack (ELK) | Yes | Elasticsearch | APM | No | Kibana | Mature; storage costs at scale |
| SigNoz | Yes | Yes | Yes | No | Yes | OTel-native from day 1; ClickHouse backend |
| OpenObserve | Yes | Yes | Yes | No | Yes | Rust-based; lower storage footprint claim |

**How to choose:**
- Already using Grafana dashboards → Grafana LGTM+
- Need full-text log search maturity → Elastic Stack
- Want OTel-native with no legacy shim → SigNoz
- Cost-sensitive storage + small team → OpenObserve or SigNoz

**Delegation target:** For Grafana stack setup, invoke `oma-search --docs "Grafana LGTM+ OTel Collector setup"`. For Elastic, use Elastic documentation directly. No single vendor-owned skill covers the full OSS stack; delegate to `oma-search` unless a specific component skill is installed.

---

### (b) Commercial SaaS Unified APM

**Traits:** Managed SaaS; covers metrics, logs, traces, and often RUM/synthetics/profiling in one product. Per-host or per-ingestion pricing. Reduced operational burden; vendor lock-in risk.

**Example vendors** (as of 2026-Q2):

| Vendor | Metrics | Logs | Traces | Profiles | RUM | Notes |
|--------|---------|------|--------|----------|-----|-------|
| Datadog | Yes | Yes | Yes | Yes | Yes | Broadest feature surface; highest cost at scale |
| New Relic | Yes | Yes | Yes | Yes | Yes | Usage-based pricing; OTel-native ingest |
| Dynatrace | Yes | Yes | Yes | Yes | Yes | AI-driven auto-discovery; OneAgent proprietary |
| Sentry | Partial | Yes | Yes | No | Yes | Error-first; strong release tracking |
| Grafana Cloud | Yes | Yes | Yes | Yes | Faro | Managed LGTM+; per-signal pricing |

**How to choose:**
- Maximum feature coverage, budget flexible → Datadog
- OTel-native ingest preferred, usage pricing → New Relic
- Auto-discovery in complex microservice env → Dynatrace
- Error tracking + release correlation primary need → Sentry
- Prefer OSS tools managed → Grafana Cloud

**Delegation target:** `Datadog Labs dd-apm`, `getsentry/sentry-sdk-setup`. For New Relic / Dynatrace: `oma-search --docs "{vendor} OTel integration setup"`.

---

### (c) High-Cardinality Specialist

**Traits:** Purpose-built for high-cardinality event-based observability. Avoids pre-aggregation. Enables arbitrary dimension slicing at query time. Dynamic sampling to control cost.

**Example vendors** (as of 2026-Q2):

| Vendor | Approach | Key features | Notes |
|--------|----------|-------------|-------|
| Honeycomb | Event-first, columnar | BubbleUp root-cause, dynamic sampling, Query Builder | OTel-native; no metrics tier (use separate tool) |

**How to choose:**
- Cardinality explosion on Prometheus/Datadog metrics → Honeycomb for traces/events
- Need to slice by any arbitrary field post-hoc → Honeycomb
- Metrics + high-cardinality in one product → pair Honeycomb with a TSDB (category i)

**Delegation target:** `honeycombio/agent-skill` (8+ published skills for OTel setup, sampling config, and BubbleUp usage).

---

### (d) Profiling Specialist

**Traits:** Continuous profiling (CPU, memory, goroutine, heap). Always-on, low-overhead. pprof or OTel profiling (OTEP 0239, currently alpha) as wire format. MELT+P fifth pillar.

**Note on Pyroscope:** Originally CNCF Sandbox. Acquired by Grafana Labs in 2023. CNCF Sandbox status is uncertain post-acquisition; verify at landscape.cncf.io before citing CNCF affiliation.

**Example vendors** (as of 2026-Q2):

| Vendor | CNCF status | Backend | Notes |
|--------|------------|---------|-------|
| Parca | CNCF Sandbox | ClickHouse / S3 | OTEP 0239-aligned; open-source |
| Grafana Pyroscope | See note above | Object storage | Integrated into Grafana LGTM+; formerly CNCF Sandbox |
| Polar Signals Cloud | Commercial | Polar Signals | Enterprise managed Parca |

**How to choose:**
- Full OSS control → Parca
- Already on Grafana LGTM+ → Grafana Pyroscope (co-located storage)
- Managed + enterprise support → Polar Signals Cloud

**Delegation target:** `oma-search --docs "Parca OTel profiling setup"` or `oma-search --docs "Grafana Pyroscope integration"`.

---

### (e) SIEM / Enterprise Logs

**Traits:** Security information and event management. Compliance-grade log retention, correlation rules, threat detection, alerting. Often WORM / immutable storage. Targets SOC2 / ISO 27001 audit requirements.

**Example vendors** (as of 2026-Q2):

| Vendor | Log ingest | Threat detection | Compliance | Notes |
|--------|-----------|-----------------|-----------|-------|
| Splunk | Yes | Yes | Yes | Market leader; high licensing cost |
| Elastic Security | Yes | Yes | Yes | Elastic Stack + SIEM rules; lower cost than Splunk |
| Sumo Logic | Yes | Yes | Yes | Cloud-native; per-GB pricing |
| Datadog Cloud SIEM | Yes | Yes | Partial | Add-on to Datadog unified APM |

**How to choose:**
- Enterprise compliance, existing Splunk investment → Splunk
- Cost-sensitive, already on Elastic → Elastic Security
- Cloud-native, no on-prem requirement → Sumo Logic
- Already on Datadog, want unified → Datadog Cloud SIEM

**Delegation target:** `oma-search --docs "{vendor} SIEM log pipeline setup"`. For audit-specific requirements see `signals/audit.md` in this skill.

---

### (f) FinOps / Cost Observability

**Traits:** Kubernetes / cloud cost visibility, unit economics, per-tenant / per-namespace attribution. Cost is a first-class observability signal (D4 in design), not a boundary attribute. Distinct from APM cost-of-ownership analysis.

**Example vendors** (as of 2026-Q2):

| Vendor | CNCF status | K8s cost | Cloud cost | Notes |
|--------|------------|---------|-----------|-------|
| OpenCost | CNCF Incubating | Yes | AWS/GCP/Azure | Open spec + OSS implementation; Prometheus-native |
| Kubecost | Commercial (OpenCost-based) | Yes | Yes | Adds multi-cluster, RBAC, Slack alerts |
| CloudZero | Commercial SaaS | Partial | Yes | Business-unit cost allocation focus |

**How to choose:**
- OSS, Prometheus-integrated, per-namespace attribution → OpenCost
- Multi-cluster + enterprise reporting → Kubecost
- Cloud bill decomposed by product team → CloudZero

**Delegation target:** `oma-search --docs "OpenCost Prometheus integration"`. For cost signal context see `signals/cost.md` in this skill.

---

### (g) Feature Flags / Progressive Delivery

**Traits:** Controlled rollout of features to subsets of users or traffic. Release observability; correlate flag states with error rates and latency. GitOps-integrated progressive delivery.

**Example vendors** (as of 2026-Q2):

| Vendor | CNCF status | Flag eval | Progressive delivery | Notes |
|--------|------------|---------|---------------------|-------|
| OpenFeature | CNCF Graduated (2024-11) | Yes (SDK standard) | No | Standardizes flag SDK; vendor-agnostic |
| Flagger | CNCF Graduated | No | Yes (canary/A-B/blue-green) | Prometheus/Datadog metric gating |
| Argo Rollouts | N/A (Argo project) | No | Yes (canary/blue-green) | Kubernetes CRD; integrates with analysis templates |
| LaunchDarkly | Commercial | Yes | Partial | Mature commercial flag platform |
| Unleash | OSS + Commercial | Yes | No | Self-hostable flag platform |

**How to choose:**
- Standardize flag SDK across vendors → OpenFeature (pair with LaunchDarkly or Unleash as backend)
- Automated canary promotion with metric gates → Flagger or Argo Rollouts
- Per-tenant progressive rollout (feature flags) → LaunchDarkly or Unleash
- GitOps-native canary on Kubernetes → Flagger (CNCF Graduated, Flux / Helm native)

**Delegation target:** `oma-search --docs "OpenFeature SDK {runtime} setup"` or `oma-search --docs "Flagger canary analysis Prometheus"`. For release boundary observability see `boundaries/release.md` in this skill.

---

### (h) Log Pipeline / Collection

**Traits:** High-throughput log and event collection, transformation, routing, and forwarding. Runs as DaemonSet agents (preferred) or standalone pipeline. OTel Collector is the 2026 standard for new deployments.

**Fluentd note:** Deprecated by CNCF 2025-10. Official migration guide: CNCF Blog 2025-10-01 "Fluentd to Fluent Bit: A Migration Guide". For migration assistance, use intent `migrate` in this skill, target category (h), with CNCF 2025-10 guide as reference.

**Example vendors** (as of 2026-Q2):

| Vendor | CNCF status | Protocol support | Transform | Notes |
|--------|------------|----------------|---------|-------|
| Fluent Bit | CNCF Graduated | Multiple | Lua / built-in | Preferred Fluentd replacement; C, low memory |
| OpenTelemetry Collector | CNCF Incubating | OTLP + many | processors | OTel-native; recommended for new deployments |
| Vector (Datadog OSS) | Non-CNCF | Multiple | VRL | Rust; high throughput; Datadog-backed |
| Cribl Stream | Commercial | Multiple | Yes | Enterprise routing + data shaping |

**How to choose:**
- Migrating from Fluentd → Fluent Bit (drop-in config compatibility)
- New OTel-native deployment → OpenTelemetry Collector
- High-throughput with Rust reliability → Vector
- Enterprise data routing with UI + compliance → Cribl Stream

**Delegation target:** `oma-search --docs "Fluent Bit Kubernetes DaemonSet setup"` or `oma-search --docs "OTel Collector configuration {backend}"`. For collector topology guidance see `transport/collector-topology.md` in this skill.

---

### (i) Time Series Storage / Long-term Metrics

**Traits:** Purpose-built for metrics storage at scale. Prometheus-compatible query (PromQL). Long-term retention beyond Prometheus 15-day default. HA + multi-tenancy for production.

**Example vendors** (as of 2026-Q2):

| Vendor | CNCF status | Multi-tenant | Object storage | Notes |
|--------|------------|------------|---------------|-------|
| Prometheus | CNCF Graduated | No (single-tenant) | No | Standard scrape source; 15d retention default |
| Thanos | CNCF Graduated | Yes | Yes (S3/GCS/Azure) | Sidecar or receive mode; global query view |
| Cortex | CNCF Incubating | Yes | Yes | Horizontally scalable; more complex ops |
| Grafana Mimir | Non-CNCF (Grafana OSS) | Yes | Yes | Evolved from Cortex; simpler ops |
| VictoriaMetrics | Non-CNCF (OSS + Commercial) | Cluster edition | Yes | High compression; low resource usage |
| InfluxDB | Non-CNCF (Commercial OSS) | Yes (Cloud) | Yes | Line protocol; IOx rewrite in Rust |

**How to choose:**
- Starting fresh, Kubernetes, team familiar with Grafana → Grafana Mimir
- Need CNCF Graduated status for compliance → Thanos
- Multi-tenant with complex federation needs → Cortex
- Resource-constrained or high compression priority → VictoriaMetrics
- Time-series with SQL query surface needed → InfluxDB IOx

**Delegation target:** `oma-search --docs "Thanos sidecar Prometheus setup"` or `oma-search --docs "Grafana Mimir distributed mode"`.

---

### (j) Crash Analytics (Mobile-Heavy)

**Traits:** Symbolication of native crash stacks. Crash-free rate (CFR) tracking. Release-correlated crash trends. ANR (Application Not Responding) detection. Mobile session replay context.

**Example vendors** (as of 2026-Q2):

| Vendor | iOS | Android | Web | Notes |
|--------|-----|---------|-----|-------|
| Firebase Crashlytics | Yes | Yes | No | Free; Google ecosystem; dSYM auto-upload |
| Sentry | Yes | Yes | Yes | Cross-platform; OTel trace correlation |
| Bugsnag | Yes | Yes | Yes | Strong breadcrumb API; Smartisan noise reduction |
| Embrace | Yes | Yes | No | Mobile-first; session replay; network body capture |
| Datadog Error Tracking | Yes | Yes | Yes | Unified with Datadog APM |

**How to choose:**
- Mobile-only, budget zero → Firebase Crashlytics
- Cross-platform (mobile + web + backend) correlation → Sentry
- Mobile session replay + user journey → Embrace
- Already on Datadog APM → Datadog Error Tracking
- Need noise reduction + grouping quality → Bugsnag

**Delegation target:** `getsentry/sentry-sdk-setup` for Sentry. For others: `oma-search --docs "{vendor} iOS/Android crash reporting setup"`. For crash analytics context see `layers/L7-application/crash-analytics.md` in this skill.

---

## Intent to Category Routing

| User intent | Primary categories | Notes |
|------------|-------------------|-------|
| Setup full-stack OSS monitoring | (a) OSS Full-Stack, (h) Log Pipeline, (i) TSDB, (d) Profiling | Start with (a); (h)+(i)+(d) fill gaps |
| Setup commercial managed APM | (b) Commercial SaaS Unified APM | Select vendor by criteria in (b) |
| Migrate off Fluentd | (h) Log Pipeline | Fluent Bit preferred (CNCF 2025-10 guide); OTel Collector for OTel-native |
| High-cardinality trace investigation | (c) High-Cardinality Specialist | Honeycomb + dynamic sampling |
| FinOps cost attribution program | (f) FinOps / Cost Observability | OpenCost + `signals/cost.md` |
| Per-tenant progressive rollout | (g) Feature Flags / Progressive Delivery | OpenFeature SDK + Flagger or Argo Rollouts |
| Mobile crash rate SLO | (j) Crash Analytics | Pair with (b) or (a) for backend traces |
| Long-term metric retention | (i) TSDB | Thanos or Mimir on top of Prometheus |
| Continuous profiling (MELT+P) | (d) Profiling Specialist | Parca (OSS) or Pyroscope (Grafana) |
| SIEM / SOC2 audit log compliance | (e) SIEM / Enterprise Logs | See `signals/audit.md` for WORM requirements |

---

## How to Delegate

### When a vendor-specific skill is installed

Invoke the skill directly:

| Skill | Coverage |
|-------|---------|
| `getsentry/sentry-sdk-setup` | Sentry SDK instrumentation, error tracking, release tracking |
| `honeycombio/agent-skill` | Honeycomb OTel setup, BubbleUp, dynamic sampling (8+ published skills) |
| `Dash0 otel-instrumentation` | Dash0 OTel generic instrumentation |
| `Microsoft Azure Monitor exporters` | Azure-specific OTel exporters and Monitor integration |
| `Datadog Labs dd-apm` | Datadog APM, log correlation, distributed tracing |

### When no vendor-specific skill is installed

Route to search with vendor + observability context:

```
oma-search --docs "{vendor} observability setup"
oma-search --docs "{vendor} OTel integration {runtime}"
oma-search --docs "{vendor} {signal} configuration"
```

Examples:
- `oma-search --docs "Grafana Mimir Prometheus remote_write setup"`
- `oma-search --docs "OpenCost Kubernetes namespace cost attribution"`
- `oma-search --docs "Fluent Bit Kubernetes DaemonSet migration from Fluentd"`
- `oma-search --docs "Parca continuous profiling Go gRPC"`

### CNCF landscape as canonical vendor source

For questions about current CNCF project status, vendor landscape membership, or whether a project is still maintained:

```
oma-search --docs "site:landscape.cncf.io {category or vendor}"
```

Or navigate directly to https://landscape.cncf.io; it is the authoritative live registry. This file does not attempt to replicate it.

---

## Footer

**Timestamp:** as of 2026-Q2

**Review cadence:** Quarterly. Triggered by: CNCF project status changes, acquisitions, deprecations, or new CNCF Graduated/Incubating entrants in covered categories.

**Authoritative live source:** https://landscape.cncf.io

**Known pending verifications (as of 2026-Q2):**
- Pyroscope CNCF status post-Grafana acquisition (2023): check landscape.cncf.io before citing CNCF affiliation
- Thanos Graduated status (confirmed 2024 per design verification log; verify remains current)
- OpenFeature CNCF status timestamped as of 2024-11 Graduated; verify next tier/changes via landscape.cncf.io
