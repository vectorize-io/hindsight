---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
specs:
  - "W3C Trace Context: Level 1 Recommendation 2020-02-06; Level 2 Candidate Recommendation"
  - "W3C Baggage: Recommendation 2022-12-22"
  - "ISO/IEC 25010: 2023; ISO/IEC 27001:2022; ISO/IEC 42010:2011"
notes:
  - "Pinned versions (update quarterly or on spec promotion)"
---

# Observability Standards Reference

## 1. Purpose

This file defines the normative standards baseline for the `oma-observability` skill. It answers:

- Which specifications govern trace propagation, attribute naming, and signal transport?
- How do ISO quality characteristics map onto observability concerns?
- Which OSI layers are in scope and why?
- What clock discipline is required for trustworthy timestamp ordering?

All other files in this skill reference this document as the authoritative source. Implementers MUST read this before writing instrumentation, configuring collectors, or authoring SLO policies.

---

## 2. Primary De Facto Standards

### 2.1 OpenTelemetry (CNCF Incubating)

Source: <https://opentelemetry.io/docs/specs/otel/> | CNCF: <https://www.cncf.io/projects/opentelemetry/>

OpenTelemetry is the vendor-neutral, CNCF-incubating specification for telemetry APIs, SDKs, semantic conventions, and the wire protocol OTLP. It is the single de facto standard this skill operates on.

Four pillars in scope for this skill:

| Pillar | Specification | Wire protocol |
|--------|---------------|---------------|
| API / SDK | opentelemetry.io/docs/specs/otel/ | N/A |
| Semantic Conventions (semconv) | opentelemetry.io/docs/specs/semconv/ | N/A |
| Protocol (OTLP) | opentelemetry.io/docs/specs/otlp/ | gRPC :4317 / HTTP :4318 |
| Collector | opentelemetry.io/docs/collector/ | N/A |

Key constraints:
- Use the stable semconv groups listed in Section 3 for production instrumentation.
- OTLP is the mandatory export format; vendor-native formats are acceptable only as secondary sinks.
- The OTel Operator (`v1beta1`) manages Collector and auto-instrumentation CRs in Kubernetes.

### 2.2 W3C Trace Context

- Level 1 Recommendation: <https://www.w3.org/TR/trace-context/> (published 2020-02-06)
- Level 2 Candidate Recommendation: <https://www.w3.org/TR/trace-context-2/>

W3C Trace Context defines the `traceparent` and `tracestate` HTTP headers that establish a distributed trace across service boundaries. Level 1 is the production-stable baseline. Level 2 adds the `traceflags` precision extension and is safe to implement against CR status.

Mandatory propagation rule: every outbound HTTP/gRPC call MUST forward `traceparent`. Stripping it silently is an anti-pattern.

Vendor header compatibility reference (informative):
- AWS X-Ray: `X-Amzn-Trace-Id`; compatible bridge via OTel X-Ray propagator
- GCP Cloud Trace: prefers W3C; legacy `X-Cloud-Trace-Context` is supported for backward compat
- Datadog: `X-Datadog-Trace-Id`; use OTel Datadog exporter for bridging

### 2.3 W3C Baggage

Recommendation: <https://www.w3.org/TR/baggage/> (2022-12-22)

W3C Baggage defines the `baggage` HTTP header for key-value pairs propagated alongside a trace. The W3C spec itself contains trust-boundary and PII guidance: baggage is visible to all downstream services and MUST NOT carry secrets, tokens, or personally identifiable information without explicit downstream trust agreement.

OTel Baggage API reference: <https://opentelemetry.io/docs/specs/otel/baggage/api/>

Usage rules enforced by this skill:
- Allowed: tenant ID, feature flag state, deployment SHA, region hint.
- Not allowed: user email, session tokens, authentication credentials.
- Trust boundary: strip or validate baggage at ingress gateway before forwarding to external services.

---

## 3. OpenTelemetry Semconv Stability Tiers

Semantic convention stability determines which attributes can be used in production without risk of breaking changes. Pin the semconv version in the file header above and update on quarterly review.

Source: <https://opentelemetry.io/docs/specs/semconv/general/attribute-requirement-level/>

| Tier | Groups | Production use |
|------|--------|----------------|
| Stable | `service.*`, `host.*`, `cloud.*`, `k8s.*`, `http.*`, `db.*` (core), `network.*` (core), `error.*` | Yes, without caveat |
| Release Candidate (RC) | `rpc.*`, gRPC semconv | Yes, expect minor changes |
| Development | `tls.*`, `network.connection.*` | Test environments; production use requires change-tolerance |
| Experimental | `gen_ai.*`, profiles (OTEP 0239 alpha) | Not for production SLOs |

Notes:
- `network.*` core attributes (e.g., `network.protocol.name`, `network.transport`) are Stable. `network.connection.*` (e.g., `network.connection.type`, `network.connection.subtype`) are Development.
- `tls.*` (all) are Development as of semconv 1.27.0. For TLS deep inspection, use Wireshark or vendor-specific TLS tooling rather than OTel attributes.
- OTEP 0239 (profiling signal) is alpha. Parca and Pyroscope are in production, but the OTel profiling spec is not yet stable. Mark any profiling-related SLOs as experimental.

Verified sources:
- TLS attrs: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/tls/>
- Network attrs: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>
- RPC/gRPC: <https://opentelemetry.io/docs/specs/semconv/rpc/grpc/>

---

## 4. ISO/IEC Indirect Mapping

ISO/IEC 25010:2023 (<https://www.iso.org/standard/78176.html>) defines the Systems and Software Quality Model. There is no dedicated "Observability" quality characteristic in the 2023 edition. Observability concerns map indirectly through three characteristics:

| ISO/IEC 25010:2023 characteristic | Sub-characteristic | Observability concern |
|---|---|---|
| Maintainability | Analysability | Can operators diagnose system state from telemetry? |
| Security | Accountability | Are actions traceable to an authenticated identity (audit logs)? |
| Reliability | Faultlessness | Does telemetry surface defects before users encounter them? |

This mapping is informative, not normative. It is useful when presenting observability investment in terms that quality assurance or audit stakeholders recognize.

Related standards:
- **ISO/IEC 27001:2022 / 27002:2022** (<https://www.iso.org/standard/27001>); controls for information security management systems; governs log integrity, access control on observability backends, and audit trail requirements.
- **ISO/IEC 42010:2011** (<https://www.iso.org/standard/50508.html>); architecture description standard. An "observability viewpoint" is a valid architecture viewpoint under 42010 for documenting how stakeholders inspect system internals.

---

## 5. OSI Boundary Decision

This skill operates on OSI layers L3, L4, mesh (L4-L7 hybrid), and L7. The following table makes the in-scope and out-of-scope decisions explicit.

### In scope

| Layer | Label | Coverage |
|-------|-------|----------|
| L3 | Network | IP routing, VPC flow logs, BGP/BMP, ICMP, PMTUD |
| L4 | Transport | TCP retransmits/RTT, eBPF (Beyla/Pixie), QUIC/HTTP3 transport |
| L4-L7 | Service mesh | Istio/Linkerd/Envoy: zero-code instrumentation, mTLS metadata as security context |
| L7 | Application | HTTP/gRPC/WebSocket, RUM (web + mobile), crash analytics, messaging |

### Out of scope

| Layer | Label | Reason | Use instead |
|-------|-------|--------|-------------|
| L1 | Physical | SaaS hypervisor hides; no OTel semconv coverage | Vendor DCIM tooling (Nlyte, Sunbird, Device42) |
| L2 | Data Link | Same as L1; SNMP/IPMI are hardware domains | Vendor DCIM tooling; SNMP exporters for Prometheus if needed |
| L5 | Session (full) | gRPC session semantics moved to L7 instrumentation; WebSocket deferred | WebSocket-specific vendor tooling |
| L6 | Presentation (full) | TLS kept as security context attributes only (`tls.*` Development tier); full TLS inspection is not OTel's domain | Wireshark for packet-level TLS; Cloudflare Radar for TLS ecosystem data; vendor-specific TLS inspection tooling |

Rationale: approximately 90% of production debugging occurs at L3, L4, mesh, and L7. L1/L2 are opaque to SaaS workloads. L5/L6 full coverage requires OTel semconv maturity that does not yet exist (Development tier). This decision is design decision D2 in the design document.

---

## 6. Clock Discipline

Distributed traces depend on monotonic, synchronized clocks across all nodes. Clock drift corrupts waterfall charts and makes parent-before-child ordering unreliable.

### Requirements

- **NTP or chrony** MUST be running and synchronized on all host VMs and container hosts.
- Acceptable drift tolerance for trace timestamp correlation: **< 100 ms** (typical NTP accuracy on well-connected hosts).
- For sub-millisecond precision (financial, telco, or high-frequency workloads): use **PTP (IEEE 1588)** hardware timestamping.

### Cloud hypervisor time sync notes

| Cloud | Mechanism | Notes |
|-------|-----------|-------|
| AWS | Chrony + Amazon Time Sync Service (169.254.169.123) | PTP-backed time source; verify with `chronyc tracking` and `chronyc sources -v` |
| GCP | Internal hypervisor clock sync (`metadata.google.internal` via `time.google.com`) | GCP VMs inherit host clock. `timedatectl show` only confirms a sync daemon is active; to verify the actual offset, use `chronyc sources -v` (chrony) or `timedatectl timesync-status` (systemd-timesyncd) |
| Azure | Hyper-V IC timesync primary; external NTP fallback only if Hyper-V IC tools absent | Azure Linux VMs with the Hyper-V Integration Services use host time as the authoritative source. External NTP (for example `time.windows.com`) as a peer source can conflict with IC timesync; treat it as a fallback, not a parallel source. Docs: <https://learn.microsoft.com/azure/virtual-machines/linux/time-sync> |

### Span timestamp validation rule

A valid trace satisfies:

```
parent_span.start_time <= child_span.start_time
child_span.end_time <= parent_span.end_time
```

A violation where `child_span.end_time > parent_span.end_time` is a clock-drift indicator, not necessarily a code bug. Flag these in meta-observability pipeline checks (see `resources/meta-observability.md`).

### Anti-pattern

NTP drift left unmonitored is anti-pattern #18 in this skill: waterfall charts appear causally wrong, leading engineers to chase phantom race conditions. Monitor `otelcol_receiver_accepted_spans` and node-level NTP offset metrics together.

---

## 7. Versioning and Review Cadence

- The spec version block in this file's header MUST be updated when:
  - Any listed semconv group promotes from Development → RC or RC → Stable.
  - A listed W3C document advances maturity (CR → PR → Recommendation).
  - OTel releases a new minor version with breaking semconv changes.
- **Review cadence**: quarterly, aligned with OTel spec releases and CNCF landscape updates (<https://landscape.cncf.io/>).
- When updating, also check and update affected signal/layer files that reference the changed attribute group.
- Skill minor version bump required on any normative change to this file.

---

## 8. References

Primary sources, in order of precedence:

1. OpenTelemetry specification: <https://opentelemetry.io/docs/specs/otel/>
2. OTel Semantic Conventions: <https://opentelemetry.io/docs/specs/semconv/>
3. OTLP specification: <https://opentelemetry.io/docs/specs/otlp/>
4. W3C Trace Context L1 Recommendation: <https://www.w3.org/TR/trace-context/>
5. W3C Trace Context L2 Candidate Recommendation: <https://www.w3.org/TR/trace-context-2/>
6. W3C Baggage Recommendation: <https://www.w3.org/TR/baggage/>
7. ISO/IEC 25010:2023; <https://www.iso.org/standard/78176.html>
8. ISO/IEC 27001:2022; <https://www.iso.org/standard/27001>
9. ISO/IEC 42010:2011; <https://www.iso.org/standard/50508.html>
10. CNCF Projects: <https://www.cncf.io/projects/>
11. CNCF Landscape (authoritative vendor registry): <https://landscape.cncf.io/>
12. OTel TLS attributes registry: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/tls/>
13. OTel network attributes registry: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>
14. OTel RPC/gRPC semconv: <https://opentelemetry.io/docs/specs/semconv/rpc/grpc/>
15. OTel Baggage API: <https://opentelemetry.io/docs/specs/otel/baggage/api/>
16. AWS Time Sync Service: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/set-time.html>
17. Azure Linux time sync: <https://learn.microsoft.com/azure/virtual-machines/linux/time-sync>
18. OTEP 0239 (profiling signal): <https://github.com/open-telemetry/oteps/blob/main/text/profiles/0239-profiles-data-model.md>
