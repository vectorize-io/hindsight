---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# Intent Classification Rules

## Purpose

- Classify a user query into one of 7 intents to route to the right resources
- Used by Step 1 of `execution-protocol.md`
- This file is an **English-only intent classification reference**. Multi-language keyword detection (Korean, Japanese, Chinese, etc.) is owned by `.agents/hooks/core/triggers.json §oma-observability.keywords`; do not duplicate per-locale tokens here.

## Classification Priority

1. **Override flags**: always win, skip classification entirely (e.g., `--investigate`, `--tune`)
2. **Keyword pattern matching**: scan query for intent-specific keywords
3. **Signal detection**: contextual clues (tool names, error messages, metric names, cloud regions)
4. **Fallback**: `investigate + tune` parallel when no clear signal

## Override Flags

| Flag | Forced Intent |
|------|--------------|
| `--setup` | `setup` |
| `--migrate` | `migrate` |
| `--investigate` | `investigate` |
| `--alert` | `alert` |
| `--trace` | `trace` |
| `--tune` | `tune` |
| `--route` | `route` |

---

## The 7 Intents

### Intent 1: `setup`

Bootstrap a new observability pipeline or instrument a new service from scratch.
Use when the user is starting fresh: no existing pipeline, first-time SDK integration, or onboarding a new environment.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | setup, install, bootstrap, instrument, configure, onboard, initialize, start, enable, add, integrate, begin, getting started, first time |
| Synonyms | init, SDK setup, OTel setup, collector setup, agent install |

**Signals:**
- "How do I add OTel to ..." queries
- Service name + "for the first time"
- New environment or cluster name mentioned with no existing config

#### Example Queries

- "Set up OTel stack on our k8s cluster"
- "Instrument our Node.js backend with OTel"
- "How do I enable tracing on a new Spring Boot service?"

#### Primary Route

`resources/vendor-categories.md` → category selection; `resources/transport/collector-topology.md` for Kubernetes

#### Secondary Considerations

`resources/standards.md` for semconv requirements before first instrumentation commit

---

### Intent 2: `migrate`

Move from a legacy tool to a modern equivalent. Applies to logging agents, APM platforms, and deprecated CNCF projects.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | migrate, migration, transition, move, replace, upgrade, modernize, switch, deprecate, port, convert, off-board, phase out |
| Synonyms | Fluentd → Fluent Bit, legacy APM → OTel, lift-and-shift, rip-and-replace |

**Signals:**
- Source tool name (Fluentd, New Relic, Datadog, old APM) + destination intent
- "Replace", "away from", "moving off of"
- CNCF deprecation context (Fluentd 2025-10)

#### Example Queries

- "Migrating Fluentd to Fluent Bit per CNCF guidance"
- "Move from New Relic to OpenTelemetry"
- "How do I port StatsD metrics to OTLP?"

#### Primary Route

`resources/vendor-categories.md §(h) Log Pipeline` (deprecation notes); CNCF 2025-10 Fluentd migration guide

#### Secondary Considerations

`resources/transport/otlp-grpc-vs-http.md` for destination protocol selection during migration

---

### Intent 3: `investigate`

Production incident or bug root-cause analysis. Use when something is broken, degraded, or unexpectedly slow right now.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | investigate, debug, diagnose, incident, outage, postmortem, forensics, root-cause, why, broken, failing, spike, degraded, 5xx, 403 storm, error rate, latency, slow, down, regression, N+1, slow query, connection pool, pool exhaustion, db span, memory leak, OOM, flame graph, stuck, WAF block, WAF false positive, firewall block, ruleset rollback, cascading failure, dependency outage, third-party outage, fail-closed, status page |
| Synonyms | RCA, postmortem, blameless review, p99 spike, tail latency, DB timeout, FP storm, edge 403, ruleset bump, third-party down |

**Signals:**
- HTTP status codes (5xx, 4xx) or error rate numbers
- Cloud region name mentioned with a problem symptom
- Service name + "suddenly", "again", "just started"
- Time window phrase ("since 14:00", "after deploy")

#### Example Queries

- "5xx spike in ap-northeast-2, need to find root cause"
- "Why is checkout service p99 high?"
- "Auth service started timing out after today's release; help"
- "Customers report 403s but our backend looks fine; check the WAF"
- "Stripe webhooks failing in bursts — is it us or a third-party outage?"

#### Primary Route

`resources/incident-forensics.md` (MRA + 6-dimension localization flow)

#### Secondary Considerations

`resources/signals/traces.md`, `resources/signals/logs.md`, `resources/boundaries/multi-tenant.md` (if single-tenant degradation pattern), `resources/layers/L7-application/waf.md` (if edge 403 storm / WAF FP suspected), `resources/boundaries/cross-application.md` (if third-party dependency outage suspected)

---

### Intent 4: `alert`

Define alerts, SLO burn-rate rules, or monitor configuration. Use when the user wants to be notified proactively before or during an incident.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | alert, alarm, notification, burn-rate, SLO, SLI, SLA, threshold, page, warn, monitor, PrometheusRule, alerting rule, error budget, firing |
| Synonyms | burn rate alert, fast burn, slow burn, PD (PagerDuty), firing rule, recording rule |

**Signals:**
- SLO/SLI/SLA acronyms
- "Alert when ...", "notify me if ..."
- PrometheusRule, Alertmanager, or burn-rate framing
- Error budget percentage mentioned

#### Example Queries

- "Set up SLO burn-rate alert for payment service"
- "Need alerts when error budget is burning fast"
- "How do I configure a slow-burn SLO alert in Grafana?"

#### Primary Route

`resources/boundaries/slo.md`; `resources/observability-as-code.md` (PrometheusRule CRD, OpenSLO YAML)

#### Secondary Considerations

`resources/meta-observability.md §Section F` for meta-observability pipeline health alerts

---

### Intent 5: `trace`

Design or debug distributed tracing: propagators, baggage, cross-service context, or mesh trace continuity.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | trace, tracing, propagator, traceparent, tracestate, baggage, span, distributed, context propagation, correlation, W3C, X-Amzn-Trace-Id, context break, missing spans |
| Synonyms | W3C Trace Context, B3, Zipkin, X-Ray trace, OpenTelemetry tracing, end-to-end trace |

**Signals:**
- "Trace breaks", "missing parent span", "trace not showing"
- Propagator format names (W3C, B3, X-Ray, Jaeger)
- Mesh or gateway name + trace context question
- Baggage field name mentioned

#### Example Queries

- "Our traces break at the Istio ingress; how to propagate context?"
- "Design baggage for multi-tenant trace correlation"
- "How do I bridge AWS X-Ray trace headers into W3C Trace Context?"

#### Primary Route

`resources/boundaries/cross-application.md` (propagator matrix); `resources/layers/mesh.md` (zero-code auto-instrumentation)

#### Secondary Considerations

`resources/signals/privacy.md` (baggage PII rules; no user identifiers in traceparent/baggage without redaction)

---

### Intent 6: `tune`

Optimize performance, reduce cost, tame cardinality, configure sampling, or fix throughput bottlenecks in the telemetry pipeline.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | tune, optimize, performance, throughput, cost, cardinality, sampling, budget, reduce, bandwidth, MTU, rate-limit, too much, expensive, overhead, head sampling, tail sampling, drop |
| Synonyms | high cardinality, metric explosion, bill shock, DPM (data points per minute), ingest cost, backpressure |

**Signals:**
- Cost number with "jumped", "tripled", "too high"
- Cardinality, label explosion, or DPM framing
- UDP MTU or OTLP throughput question
- Sampling ratio or rate-limit configuration

#### Example Queries

- "Datadog bill jumped 3x: need to reduce cardinality"
- "UDP statsd throughput is low at peak"
- "How do I set a cost-aware tail sampling policy for the checkout service?"

#### Primary Route

`resources/transport/` (all 4 files: `udp-statsd-mtu.md`, `otlp-grpc-vs-http.md`, `collector-topology.md`, `sampling-recipes.md`); `resources/meta-observability.md §Cardinality`

#### Secondary Considerations

`resources/vendor-categories.md` for alternative tool selection when current vendor is causing the cost spike

---

### Intent 7: `route`

Multi-tenant, multi-cloud, or multi-region telemetry routing, isolation, data residency, or federation.

#### Keywords

| Category | Tokens |
|----------|--------|
| Primary | route, routing, multi-tenant, multi-cloud, region, residency, isolation, segregation, gateway, fan-out, federation, data locality, GDPR residency, pipeline split, tenant routing |
| Synonyms | data sovereignty, geo-fencing, per-tenant collector, routing_connector, OTel Collector routing |

**Signals:**
- Specific cloud region name + compliance or residency requirement
- "Tenant A vs Tenant B" pipeline separation
- Multiple collectors or clusters mentioned with routing context
- GDPR, PIPA, or data sovereignty regulation cited

#### Example Queries

- "Route tenant A telemetry to ap-northeast-2 only (GDPR residency)"
- "Federated collectors across 3 k8s clusters"
- "How do I fan-out logs to two different Loki stacks by tenant tier?"

#### Primary Route

`resources/boundaries/multi-tenant.md`; `resources/transport/collector-topology.md`

#### Secondary Considerations

`resources/boundaries/cross-application.md` for trust boundary enforcement when routing crosses application domains

---

## Ambiguous / Fallback

When no intent is detected with confidence:

- Default to `investigate + tune` in parallel (observability work is often both diagnosing a problem and reducing noise)
- Present both result sets; let the user pick
- If automation is critical (e.g., CI pipeline), require the user to pass an explicit flag

**Ambiguity resolution examples:**

| Query | Detected Intent | Reason |
|-------|----------------|--------|
| "OTel Collector configuration" | `setup` (or `tune` if already deployed) | "configure" matches setup; check for existing deployment context |
| "p99 high, why?" | `investigate` | Problem symptom + causal question |
| "How do I sample?" | `tune` | Sampling = throughput/cost optimization |
| "Tenant data isolation" | `route` | Isolation + tenant framing |
| "SLO alert setup" | `alert` | SLO + alarm keywords |
| "Want to drop Fluentd" | `migrate` | Deprecation + replacement intent |
| "Istio trace propagation" | `trace` | Mesh + trace context propagation |
| "Something off with OTel" (no detail) | `investigate + tune` fallback | No specific signal; dispatch parallel |

---

## Matching Algorithm

1. Lower-case both query and keyword list
2. For each intent, count keyword hits (whole-word boundary match)
3. Pick the intent with the highest score
4. On a tie, apply tiebreak priority: `investigate` > `setup` > `tune` > `alert` > `trace` > `route` > `migrate` (ordered by business-incident impact)
5. If all scores are zero: fall back to `investigate + tune` parallel with a clarification prompt

---

## Integration with Hooks

- **Multi-language keyword detection** at `UserPromptSubmit` is owned by `.agents/hooks/core/triggers.json §oma-observability.keywords` (entries for `*`, `en`, `ko`, `ja`, `zh`). That file is the single source of truth for activation tokens.
- This file documents the **English intent classification rules** consumed by `resources/execution-protocol.md §Step 1` once the skill has been activated.
- Override flags (`--investigate`, etc.) bypass keyword scoring entirely and force the named intent.
- When adding tokens for a new domain (e.g., WAF, cascading failure), update both: (a) English tokens in the relevant intent's Keywords table here, and (b) per-locale activation tokens in `triggers.json`.
