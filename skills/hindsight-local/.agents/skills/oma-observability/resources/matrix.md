---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# Observability Coverage Matrix

## 1. Purpose

This file is the coverage map for the `oma-observability` skill. It answers one question per cell:

> "What observability artifact belongs at the intersection of this layer, this boundary, and this signal?"

The matrix has three goals:

1. **Prevent hand-wave instrumentation.** "Just add metrics" is not a plan. Every cell forces an explicit decision: what artifact, from which source, tagged how, stored where.
2. **Surface N/A early.** Some combinations are physically or semantically meaningless (e.g., L3-level CPU profiling). Declaring them N/A is a correct engineering answer, not a gap.
3. **Provide navigation.** Each covered cell links to the authoritative layer/boundary/signal file within this skill so implementers can jump directly to the detailed guidance.

**Normative base:** all taxonomy, semconv stability tiers, and OSI boundary decisions are declared in `resources/standards.md`. Read that file before using this matrix.

---

## 2. Taxonomy

### Layers (4)

| Label | Meaning |
|-------|---------|
| L3-network | IP routing, VPC flow logs, BGP/BMP, ICMP, PMTUD |
| L4-transport | TCP retransmits/RTT, QUIC/HTTP3 transport, eBPF (Beyla/Pixie) |
| mesh | Istio/Linkerd/Envoy: zero-code instrumentation, mTLS metadata as security context |
| L7-application | HTTP/gRPC/WebSocket, RUM (web + mobile), crash analytics, messaging |

### Boundaries (4)

| Label | Meaning |
|-------|---------|
| multi-tenant | Signals must be isolated, attributed, and possibly residency-constrained per tenant |
| cross-application | Propagation across service, domain, or vendor boundaries |
| slo | Signals that feed SLI computation, burn-rate alerting, or error-budget accounting |
| release | Signals that are correlated to a specific deployment, canary, or feature flag state |

### Signals (7)

| Label | Meaning |
|-------|---------|
| metrics | Numeric time-series (counters, gauges, histograms) |
| logs | Structured log records and events |
| traces | Distributed spans and context propagation |
| profiles | Continuous CPU/memory/off-CPU flame graphs (OTEP 0239 alpha) |
| cost | FinOps unit-economics and resource attribution |
| audit | Immutable, tamper-evident event records for compliance |
| privacy | PII detection, redaction, anonymization, pseudonymization controls |

---

## 3. How to Read the Matrix

- **Row** = one boundary within a given layer.
- **Column** = one signal.
- **Cell content** = 1-2 word artifact description + file reference.

### Cell legend

| Token | Meaning |
|-------|---------|
| `PASS` | Covered: artifact is well-defined; see referenced file and section |
| `PARTIAL` | Partially covered or requires caveat; see referenced file and "Caveats" section below |
| `N/A` | Not meaningful at this layer/boundary or produces no actionable artifact |

Every `PASS` and `PARTIAL` cell includes at least one file reference. File references point to files within this skill tree. Files not yet written (Phase 1b/1c) are forward references and are explicitly allowed.

---

## 4. The Matrix

### Layer: L3-network

L3 is the IP routing layer: VPC flow logs, BGP/BMP, ICMP unreachables, and PMTUD probing. Observability at this layer is network-flow-centric. Traces are not native to L3 packets; profiles are not applicable. SLO and release boundaries are defined at the application layer and cannot be computed from IP flow data alone.

| Boundary \ Signal | metrics | logs | traces | profiles | cost | audit | privacy |
|---|---|---|---|---|---|---|---|
| multi-tenant | PASS per-tenant VPC flow byte/packet counters → `layers/L3-network.md`, `boundaries/multi-tenant.md §Metric attribution` | PASS VPC flow log stream tagged by tenant network CIDR → `layers/L3-network.md`, `signals/logs.md` | PARTIAL trace-ID egress tagging at L3 boundary only; trace context not carried in IP headers → `layers/L3-network.md §BGP advanced`, `signals/traces.md` | N/A; CPU/memory profiling has no L3 artifact | PARTIAL egress byte attribution by tenant VPC → `signals/cost.md §egress`, `boundaries/multi-tenant.md` | PASS VPC flow audit trail tagged by tenant → `signals/audit.md`, `layers/L3-network.md` | PARTIAL IP addresses are PII in GDPR/PIPA; mask or hash before retention → `signals/privacy.md §IP addresses` |
| cross-application | PASS inter-VPC/peering flow metrics → `layers/L3-network.md`, `signals/metrics.md` | PASS VPC flow logs across peering or transit gateway → `layers/L3-network.md`, `signals/logs.md` | PARTIAL L3 packets carry no trace context natively; use trace-ID in DNS or SNI side-channel only → `layers/L3-network.md`, `boundaries/cross-application.md` | N/A | PARTIAL cross-VPC egress cost attribution; unreliable without flow tagging → `signals/cost.md` | PASS inter-VPC flow audit for SOC2 network controls → `signals/audit.md`, `layers/L3-network.md` | PARTIAL source/destination IPs crossing application boundary are PII candidates → `signals/privacy.md §IP addresses` |
| slo | N/A; SLO error budgets are defined at L7; L3 availability feeds infra health at most | N/A; VPC flow logs are operational, not SLO inputs | N/A | N/A | N/A | N/A | N/A |
| release | N/A; release events carry no L3 signal | N/A | N/A | N/A | N/A | N/A | N/A |

---

### Layer: L4-transport

L4 is the TCP/UDP/QUIC transport layer. eBPF (Beyla, Pixie) is the primary observability mechanism, producing socket-level metrics and off-CPU flame graphs without code changes. TCP retransmits, RTT, and connection states are rich signals. Trace context is not native to TCP; it lives at the application framing layer. SLO and release boundaries are application concerns.

| Boundary \ Signal | metrics | logs | traces | profiles | cost | audit | privacy |
|---|---|---|---|---|---|---|---|
| multi-tenant | PASS per-tenant TCP retransmit rate, RTT histograms via eBPF socket filter → `layers/L4-transport.md §eBPF`, `boundaries/multi-tenant.md` | PASS TCP connection lifecycle events per tenant socket namespace → `layers/L4-transport.md`, `signals/logs.md` | PARTIAL TCP is not trace-native; mesh or L7 must carry trace context; L4 can log socket tuples for correlation → `layers/L4-transport.md`, `signals/traces.md` | PASS eBPF CPU/off-CPU profiling at socket-level covers L4 overhead; pprof-compatible output → `signals/profiles.md`, `layers/L4-transport.md §eBPF` | PARTIAL L4 byte volume per tenant as cost proxy; not a direct cost signal → `signals/cost.md` | PARTIAL connection-level audit (who connected to what port, when); useful for SOC2 network access controls → `signals/audit.md`, `layers/L4-transport.md` | PARTIAL source IPs in TCP metadata are PII; apply same IP-masking rules as L3 → `signals/privacy.md §IP addresses` |
| cross-application | PASS cross-service TCP RTT and retransmit metrics via eBPF → `layers/L4-transport.md §eBPF`, `signals/metrics.md` | PASS TCP connection events across service socket pairs → `layers/L4-transport.md`, `signals/logs.md` | PARTIAL trace context does not exist at L4; identify cross-app flows by socket 5-tuple and correlate to L7 spans → `layers/L4-transport.md`, `boundaries/cross-application.md` | PASS eBPF off-CPU profiles covering network-wait time across application boundaries → `signals/profiles.md`, `layers/L4-transport.md §eBPF` | PARTIAL cross-application byte volume as FinOps signal; correlate with L7 cost attribution for accuracy → `signals/cost.md` | PASS TCP connection audit across trust boundaries → `signals/audit.md`, `layers/L4-transport.md` | PARTIAL connection metadata contains IPs; apply masking at pipeline ingress → `signals/privacy.md` |
| slo | N/A; SLO windows are application-defined; L4 connection success rate may inform infra SLI but is not a canonical SLO boundary | N/A | N/A | N/A | N/A | N/A | N/A |
| release | N/A; release events are application-layer; L4 has no deployment semantic | N/A | N/A | N/A | N/A | N/A | N/A |

---

### Layer: mesh

The service mesh layer (Istio, Linkerd, Envoy) operates as a transparent L4-L7 proxy sidecar or ambient mode agent. It provides zero-code instrumentation, mTLS metadata as security context, and W3C Trace Context propagation without application code changes. This layer has the strongest native multi-signal coverage of any layer in this skill.

| Boundary \ Signal | metrics | logs | traces | profiles | cost | audit | privacy |
|---|---|---|---|---|---|---|---|
| multi-tenant | PASS per-tenant RED (Rate/Error/Duration) metrics from Envoy telemetry; tagged via `tenant.id` baggage → `layers/mesh.md`, `boundaries/multi-tenant.md §Metric attribution`, `signals/metrics.md` | PASS Envoy access logs per tenant with baggage-derived tenant tag → `layers/mesh.md`, `signals/logs.md` | PASS Envoy zero-code span injection + W3C Baggage `tenant.id` propagation → `layers/mesh.md`, `boundaries/multi-tenant.md`, `signals/traces.md` | PARTIAL mesh proxies add latency overhead visible in eBPF profiles; not a mesh-native profiling source → `signals/profiles.md`, `layers/mesh.md` | PARTIAL request-level cost attribution by tenant via mesh telemetry; feed into OpenCost → `signals/cost.md`, `layers/mesh.md` | PASS mTLS identity + access log provide SOC2 accountability trail per tenant → `signals/audit.md`, `layers/mesh.md` | PASS mTLS config enforces transport encryption; baggage scrubbing at ingress gateway removes PII before propagation → `signals/privacy.md`, `layers/mesh.md`, `resources/standards.md §W3C Baggage` |
| cross-application | PASS cross-service RED metrics at mesh proxy; primary use case for service topology mapping → `layers/mesh.md`, `boundaries/cross-application.md`, `signals/metrics.md` | PASS Envoy access logs across service-to-service calls; correlation by `trace_id` → `layers/mesh.md`, `signals/logs.md` | PASS primary trace origin for cross-application spans; Envoy injects spans without code changes; W3C Trace Context propagation → `layers/mesh.md`, `boundaries/cross-application.md`, `signals/traces.md` | PARTIAL mesh overhead visible via eBPF off-CPU profiles on sidecar process; not mesh-native → `signals/profiles.md`, `layers/L4-transport.md §eBPF` | PASS request cost attribution across services; mesh provides per-service byte/request counts → `signals/cost.md`, `layers/mesh.md` | PASS mTLS peer identity in access log provides cross-application accountability → `signals/audit.md`, `layers/mesh.md` | PASS baggage trust boundary enforced at mesh ingress gateway; strip or validate external baggage → `signals/privacy.md`, `boundaries/cross-application.md`, `resources/standards.md §W3C Baggage` |
| slo | PASS mesh request rate and error rate are primary SLI sources for latency and availability SLOs → `layers/mesh.md`, `boundaries/slo.md`, `signals/metrics.md §SLI` | PARTIAL Envoy access logs as burn-rate event source; log-based SLI possible but metric-based preferred → `signals/logs.md`, `boundaries/slo.md` | PARTIAL trace sampling must be configured to retain error traces for SLO error-budget forensics; not a direct SLO input → `signals/traces.md`, `boundaries/slo.md` | N/A; profiles do not feed SLO computation | N/A; cost is a separate FinOps concern, not an SLO boundary input | PARTIAL SLO audit trail: record burn-rate threshold breach events as audit log entries → `signals/audit.md`, `boundaries/slo.md` | N/A |
| release | PASS mesh request split metrics for canary traffic (Flagger/Argo Rollouts proxy rules) → `layers/mesh.md`, `boundaries/release.md`, `signals/metrics.md` | PASS deployment event injected as Envoy log entry with `service.version` tag → `layers/mesh.md`, `signals/logs.md`, `boundaries/release.md` | PARTIAL canary proxy rules at mesh layer route a percentage of traces to new version; trace header carries `service.version` → `layers/mesh.md`, `boundaries/release.md`, `signals/traces.md` | N/A | PARTIAL canary cost delta visible via per-version request counts in mesh telemetry → `signals/cost.md`, `boundaries/release.md` | PASS release event audit: mesh records version-tagged traffic split events → `signals/audit.md`, `boundaries/release.md` | N/A |

---

### Layer: L7-application

L7 is the application layer: HTTP/gRPC, RUM (web + mobile), crash analytics, and messaging. This is the primary signal domain for this skill. All seven signals are meaningful here. Multi-tenant instrumentation is richest at L7 because application code has full access to user context, tenant identifiers, and business semantics.

| Boundary \ Signal | metrics | logs | traces | profiles | cost | audit | privacy |
|---|---|---|---|---|---|---|---|
| multi-tenant | PASS per-tenant RED + custom business metrics tagged with `tenant.id`; histogram by tenant tier → `signals/metrics.md §tenant`, `boundaries/multi-tenant.md`, `layers/L7-application/web-rum.md` | PASS structured log stream with `tenant.id` on every record; tenant log routing via OTel Collector `routing_connector` → `signals/logs.md`, `boundaries/multi-tenant.md` | PASS W3C Baggage `tenant.id` propagated on every span; trace exported per-tenant collector pipeline if residency required → `signals/traces.md`, `boundaries/multi-tenant.md`, `resources/standards.md §W3C Baggage` | PASS per-tenant continuous profiling with Parca/Pyroscope; label by `tenant.id` for flame graph isolation → `signals/profiles.md`, `boundaries/multi-tenant.md` | PASS request-level cost attribution by tenant; feed OpenCost unit economics model → `signals/cost.md §unit economics`, `boundaries/multi-tenant.md` | PASS per-tenant audit event stream; WORM storage per tenant for SOC2 evidence → `signals/audit.md`, `boundaries/multi-tenant.md` | PASS tenant-scoped PII redaction; per-tenant privacy config for GDPR right-to-erasure → `signals/privacy.md`, `boundaries/multi-tenant.md` |
| cross-application | PASS inter-service request rate and latency histograms; use `service.name` + `peer.service` labels → `signals/metrics.md`, `boundaries/cross-application.md` | PASS correlation log: `trace_id` + `span_id` on every log record enables log-trace join across applications → `signals/logs.md`, `boundaries/cross-application.md` | PASS primary trace origin; W3C Trace Context `traceparent` on all outbound calls; DDD namespace baggage for bounded-context attribution → `signals/traces.md`, `boundaries/cross-application.md`, `resources/standards.md §W3C Trace Context` | PASS application-level profiling showing cross-service call overhead; correlate with traces via `trace_id` label → `signals/profiles.md`, `boundaries/cross-application.md` | PASS request cost attribution across services; per-service unit cost model → `signals/cost.md`, `boundaries/cross-application.md` | PASS cross-application audit events carry caller identity and `trace_id` for accountability chain → `signals/audit.md`, `boundaries/cross-application.md` | PASS baggage trust boundary at API gateway; validate or strip PII-bearing baggage from external callers → `signals/privacy.md`, `boundaries/cross-application.md`, `resources/standards.md §W3C Baggage` |
| slo | PASS SLI metric computation (availability, latency p99, error rate); SLO targets defined in OpenSLO YAML → `signals/metrics.md §SLI`, `boundaries/slo.md`, `resources/observability-as-code.md` | PARTIAL log-based SLI possible (error log count / total); valid for non-metrics-instrumented services but metric-based SLI preferred → `signals/logs.md`, `boundaries/slo.md` | PARTIAL critical path trace sampling for SLO forensics; tail-sampler keeps error traces within error budget window → `signals/traces.md`, `boundaries/slo.md`, `transport/sampling-recipes.md` | N/A; profiling does not feed SLO computation directly | N/A; cost SLO is a FinOps budget concern, not an error-budget SLO | PARTIAL SLO breach audit record: persist burn-rate threshold crossing as immutable audit event → `signals/audit.md`, `boundaries/slo.md` | N/A |
| release | PASS release marker metric event; `service.version` label on all metrics for before/after comparison → `signals/metrics.md`, `boundaries/release.md`, `resources/observability-as-code.md` | PASS deployment event as structured log record; deployment SHA, version, and rollout strategy logged → `signals/logs.md`, `boundaries/release.md` | PASS `service.version` attribute on all spans; canary trace routing by version tag for A/B error comparison → `signals/traces.md`, `boundaries/release.md` | PARTIAL pre/post-release profile comparison for performance regression detection; Parca/Pyroscope diff view → `signals/profiles.md`, `boundaries/release.md` | PARTIAL release cost delta: compare per-request cost across canary vs stable version → `signals/cost.md`, `boundaries/release.md` | PASS release audit event: immutable record of who deployed what version when → `signals/audit.md`, `boundaries/release.md` | N/A |

---

## 5. Caveats

Rationale for the rarer N/A / PARTIAL cells. Each entry justifies the marker and points to the authoritative file.

- **C1. L3 × {SLO, release}**: N/A. SLO and release are L7 constructs (OpenSLO YAML, canary markers); L3 IP flow feeds infra health, not application error budgets. An L3 event (BGP leak, PMTUD black hole) that causes an SLO burn surfaces as an L7 error spike first; investigate via `resources/incident-forensics.md` 6-dim localization, not via L3 SLO policy. See `boundaries/slo.md`, `layers/L3-network.md`.
- **C2. L4 × {SLO, release}**: same rationale as C1. TCP connection success rate can serve as a fallback SLI when L7 instrumentation is absent, but it is not recommended. See `boundaries/slo.md §fallback SLI sources`.
- **C3. {L3, L4} × profiles**: profiling (Parca, Pyroscope, OTEP 0239) is process-level; L3 has no equivalent artifact. L4 × profiles PASS reflects eBPF kernel-socket and off-CPU wait measurements; the closest meaningful artifact; attributed to `layers/L4-transport.md §eBPF`.
- **C4. {L3, L4} × traces (PARTIAL)**: W3C Trace Context lives in HTTP/gRPC headers, not IP/TCP. The PARTIAL reflects a correlation technique (log socket 5-tuple alongside trace ID) rather than native propagation. Native propagation begins at mesh or L7. See `resources/standards.md §W3C Trace Context`.
- **C5. mesh × profiles (PARTIAL)**: sidecar proxies (Envoy, Linkerd-proxy) are separate processes; their CPU overhead is visible via eBPF on the sidecar, but that is an L4 artifact, not a mesh-native profiling signal. Mesh exposes no profiling API. See `signals/profiles.md`, `layers/L4-transport.md §eBPF`.
- **C6. {L3, L4} × cost (PARTIAL)**: byte volume is a cost proxy sufficient for cloud egress billing but insufficient for unit economics, which requires L7 per-request attribution. See `signals/cost.md §egress attribution`.
- **C7. {L3, L4} × privacy (PARTIAL)**: IP addresses are personal data (GDPR Art. 4(1), PIPA equivalent). VPC flow and TCP connection logs are operationally required but must be masked, hashed, or pseudonymized before long-term retention. Masking technique (prefix truncation, HMAC+salt, pseudonymization) is in `signals/privacy.md §IP addresses`.
- **C8. {L3, L4} × {SLO, release} × audit**; audit requires an identity-bearing, time-bound subject; IP/TCP-layer events carry no such subject tied to an SLO policy or deployment action. Audit at SLO/release boundaries is always L7-originated. See `signals/audit.md`.

---

## 6. Cell Count Verification

| Layer | Boundaries | Signals | Cells |
|-------|-----------|---------|-------|
| L3-network | 4 | 7 | 28 |
| L4-transport | 4 | 7 | 28 |
| mesh | 4 | 7 | 28 |
| L7-application | 4 | 7 | 28 |
| **Total** | | | **112** |

All 112 cells are populated. No cell is blank.

---

## 7. Cross-references

All files referenced in this matrix belong to the `oma-observability` skill tree. Files not yet written (Phase 1b/1c per the design document rollout plan) are forward references.

| Category | Files |
|----------|-------|
| Layers | `layers/L3-network.md`, `layers/L4-transport.md`, `layers/mesh.md`, `layers/L7-application/web-rum.md`, `layers/L7-application/mobile-rum.md`, `layers/L7-application/crash-analytics.md` |
| Boundaries | `boundaries/multi-tenant.md`, `boundaries/cross-application.md`, `boundaries/slo.md`, `boundaries/release.md` |
| Signals | `signals/metrics.md`, `signals/logs.md`, `signals/traces.md`, `signals/profiles.md`, `signals/cost.md`, `signals/audit.md`, `signals/privacy.md` |
| Transport | `transport/sampling-recipes.md`, `transport/collector-topology.md` |
| Resources | `resources/standards.md`, `resources/incident-forensics.md`, `resources/observability-as-code.md`, `resources/meta-observability.md` |

---

## 8. Review and Maintenance

- **Review cadence**: quarterly, aligned with `resources/standards.md` version update cadence.
- **On semconv promotion** (Development → RC → Stable): re-evaluate PARTIAL cells that cite stability as a caveat; promote to PASS if the semconv group is now stable.
- **On new layer or boundary addition**: this file requires a new table section and all cross-references updated. Signal columns do not change without a taxonomy revision in the design document.
- **On N/A re-evaluation**: add a Caveat entry (Section 5) explaining why a previously N/A combination is now meaningful before changing the cell marker.
- **Owner**: CTO direct review required for any change to this file (see design document Ownership & Quality Gates table).
