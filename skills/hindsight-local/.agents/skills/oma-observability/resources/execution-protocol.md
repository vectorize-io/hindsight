---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# Observability Agent - Execution Protocol

## Step 0: Parse Query

1. **Extract flags** from the query string:
   - `--setup`, `--migrate`, `--investigate`, `--alert`, `--trace`, `--tune`, `--route`: force specific intent
   - `--strict`: require Stable semconv only; reject Development/Experimental attributes
   - `--multi-tenant`, `--multi-cloud`: force boundary dimension to `multi-tenant`
   - `--layer=L3|L4|mesh|L7`: force specific layer focus
   - `--signal=metrics|logs|traces|profiles|cost|audit|privacy`: force signal focus
2. **Detect keywords** (Korean and English) against `resources/intent-rules.md` trigger list
3. **Extract ambient context** from user message or project files:
   - Target platform: k8s, serverless (Lambda/Cloud Run), VM, bare metal
   - Scale tier: single-service, multi-service, multi-region
   - Compliance requirements: SOC2, ISO 27001, GDPR, PIPA
4. **Log** resolved flags, detected language, and ambient context for transparency

## Step 1: Classify Intent

1. If a flag is present → use that intent directly; skip keyword matching
2. If no flag → apply keyword pattern matching from `resources/intent-rules.md`
3. If ambiguous or no match → default to `investigate + tune` in parallel
4. **Sparse-context gate**: even when an intent is matched (score ≥ 1), before proceeding to Step 2 check that ambient context is sufficient for the chosen intent. Minimum required context per intent:
   - `investigate`: service name OR symptom category (error code, metric name) OR time window
   - `setup` / `migrate`: target platform (k8s/serverless/VM) OR language/framework
   - `alert`: SLI target or service name
   - `trace`: at least one boundary hint (service hop, mesh, cloud)
   - `tune`: signal type (metrics/logs/traces) OR problem (cost/cardinality/MTU)
   - `route`: tenant OR region OR cloud axis

   If the minimum is not present, request clarification from the user before consuming reference material. A 1-2 keyword query that hits one intent keyword but carries no context (e.g., "metrics broken", "tracing broken") should prompt for service/symptom/scope rather than executing a playbook that cannot succeed.
5. Log selected intent and whether selection was `flag` or `auto`, plus any clarification requested

Intent vocabulary:

| Intent | Meaning |
|--------|---------|
| `setup` | First-time instrumentation or pipeline deployment |
| `migrate` | Moving from a deprecated or legacy tool |
| `investigate` | Incident analysis or anomaly root-cause |
| `alert` | SLO burn-rate or threshold alert authoring |
| `trace` | Distributed trace propagation and context design |
| `tune` | Transport, cardinality, sampling, or collector optimization |
| `route` | Signal routing, tenant isolation, or topology design |

## Step 2: Matrix Navigation

1. Based on (intent × layer × boundary × signal), identify relevant cells in `resources/matrix.md`
2. Collect file references for each cell marked as covered (checkmark) or conditional (warning marker)
3. Flag any N/A cells the user is asking about; redirect to an alternative dimension rather than producing a stub answer
4. Record the active (layer, boundary, signal) triple for use in Step 6 output header

## Step 3: Route Dispatch

Dispatch based on intent. Use the table below as the primary routing map, then apply intent-specific detail.

| Intent | Primary resource | Fallback |
|--------|-----------------|---------|
| `setup` | `resources/vendor-categories.md` → vendor-owned skill | `resources/standards.md` (OTel semconv) |
| `migrate` | CNCF 2025-10 guide + `resources/vendor-categories.md §(h) Log Pipeline` | OTel Collector bridge config |
| `investigate` | `resources/incident-forensics.md` (MRA + 6-dim localization) | `resources/signals/traces.md` + `resources/signals/logs.md` |
| `alert` | `resources/boundaries/slo.md` (burn-rate rules) | `resources/observability-as-code.md` |
| `trace` | `resources/boundaries/cross-application.md` (propagator matrix) | `resources/layers/mesh.md` (zero-code auto-instr) |
| `tune` | `resources/transport/` (4 files; see below) | `resources/meta-observability.md` (cardinality guardrails) |
| `route` | `resources/boundaries/multi-tenant.md` + `resources/transport/collector-topology.md` | `resources/boundaries/cross-application.md` |

### setup intent
- Consult `resources/vendor-categories.md`: select category based on constraints (OSS vs commercial, high-cardinality, FinOps, profiling)
- Delegate to vendor-owned skill when one is installed (e.g., getsentry/sentry-sdk-setup, honeycombio/agent-skill, Dash0 otel-instrumentation, Datadog Labs dd-apm)
- If no matching vendor skill is installed → guide user to `/oma-search --docs` for vendor documentation

### migrate intent
- Fluentd as source → apply CNCF 2025-10 deprecation guide; recommend Fluent Bit or OTel Collector
- Legacy APM as source → provide OTel bridge config patterns; reference `resources/vendor-categories.md §(h)`

### investigate intent
- Invoke `resources/incident-forensics.md` full playbook (MRA + 6-dimension narrowing: code / service / layer / host / region / infra)
- Cross-reference signal files based on symptom category (latency, error rate, saturation, data loss)

### alert intent
- Use `resources/boundaries/slo.md` for burn-rate calculation and multi-window alert rules
- Use `resources/observability-as-code.md` for PrometheusRule CRD and Alertmanager routing tree

### trace intent
- Use `resources/boundaries/cross-application.md` for W3C Trace Context propagator matrix across cloud providers
- Use `resources/layers/mesh.md` for zero-code auto-instrumentation via service mesh

### tune intent
- `resources/transport/udp-statsd-mtu.md`: UDP payload size thresholds and fragmentation risk
- `resources/transport/otlp-grpc-vs-http.md`: protocol selection by environment and firewall constraints
- `resources/transport/collector-topology.md`: DaemonSet vs sidecar vs gateway deployment patterns
- `resources/transport/sampling-recipes.md`: head-based vs tail-based sampling policy selection

### route intent
- `resources/boundaries/multi-tenant.md`: tenant isolation strategies (attribute-based, pipeline-based, storage-based)
- `resources/transport/collector-topology.md`: routing topology for signal fan-out and load balancing

## Step 4: Collect Reference Material

1. Pull referenced file sections into working context based on Step 3 routing results
2. Check `resources/vendor-categories.md` timestamp: if older than one quarter, advise the user to verify against the CNCF landscape at https://landscape.cncf.io
3. For commercial vendor references, check whether a vendor-owned skill is installed locally before suggesting manual setup

## Step 5: Validate Against Constraints

1. Consult `resources/anti-patterns.md`: does the proposed approach violate any of the 18 items?
2. Consult `resources/checklist.md`: will this pass Pre-prod and Prod gates?
3. Run `resources/meta-observability.md` cardinality guardrail preview: flag any label dimension that risks unbounded growth
4. If `--strict` flag is set → reject any semconv attribute in Development or Experimental stability tier; cite stable alternative
5. If PII is involved → apply `resources/signals/privacy.md` redaction and sampling-aware baggage rules at collection, not only at storage
6. If `--multi-tenant` or `--multi-cloud` → apply `resources/boundaries/multi-tenant.md` isolation rules; verify data residency is explicit

## Step 6: Present

Format output as:

```
Intent: {intent}  Mode: {auto|flag}
Layer:  {L3|L4|mesh|L7}    Boundary: {multi-tenant|cross-application|slo|release}    Signal: {metrics|logs|traces|profiles|cost|audit|privacy}

Primary recommendation:
  - {file:section} — {1-line rationale}

Secondary considerations:
  - {file:section} — {caveat or anti-pattern hit}

Delegation target:
  - {vendor-owned skill name, or /oma-search command}

Checklist items to verify:
  - [ ] {item from checklist.md}
  - [ ] {item}
```

Example output for "setup OTel stack on k8s":

```
Intent: setup  Mode: auto
Layer: L7 + mesh    Boundary: cross-application    Signal: metrics + logs + traces

Primary recommendation:
  - resources/transport/collector-topology.md §Two-tier hybrid — DaemonSet + Deployment gateway
  - resources/vendor-categories.md §OSS Full-Stack — Grafana LGTM+ (2026-Q2)

Secondary considerations:
  - resources/transport/sampling-recipes.md §Tail-based — consistent routing via loadbalancing exporter
  - resources/anti-patterns.md §Section C — avoid sidecar as default on k8s; use DaemonSet

Delegation target:
  - No single vendor skill (OSS self-host); use oma-tf-infra for Terraform + /oma-search for component docs

Checklist items to verify:
  - [ ] memory_limiter processor placed before batch processor in pipeline
  - [ ] NTP synced on all nodes (< 100 ms drift)
  - [ ] cardinality budget set per service before enabling high-cardinality labels
  - [ ] Fluentd replaced or bridged — CNCF 2025-10 deprecation in effect
```

## On Error

See `resources/checklist.md §7 Recovery` for recovery steps.
