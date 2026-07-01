---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
tools:
  - "Istio: 1.22+; Envoy: 1.29+; OTel Operator: v0.100+"
---

# Service Mesh Observability

## 1. Why the Service Mesh Has Its Own Observability File

The service mesh is classified as a **L4-L7 hybrid layer** in the taxonomy defined in `resources/standards.md §OSI Boundary Decision`. It differs from the adjacent L4-transport layer in a fundamental way: the mesh proxy (sidecar or ambient agent) operates at the HTTP/gRPC framing level, not at raw TCP socket level.

Three properties make mesh observability distinct enough to warrant a dedicated file:

1. **Zero-code auto-instrumentation.** Sidecar injection (or ambient mode) intercepts all inbound and outbound traffic without requiring any change to application code. Metrics, access logs, and distributed trace spans are produced automatically from day one of mesh deployment. This contrasts with `layers/L4-transport.md`, where eBPF probes attach to kernel socket events and produce transport-centric (TCP RTT, retransmit) artifacts.

2. **Proxy-centric, not transport-centric.** Mesh observability is Envoy-proxy-centric or Linkerd-proxy-centric. The primary artifacts are Envoy listeners/clusters, access log records, and HTTP/gRPC span metadata; not TCP segments or IP packets. `layers/L4-transport.md` covers kernel-level transport; this file covers proxy-level L7 framing.

3. **mTLS identity as a first-class signal.** The mesh terminates mTLS at the sidecar, producing certificate identity (SPIFFE SVID), cipher suite, and expiry as observable attributes alongside each request. No other layer produces this security context natively.

---

## 2. Service Mesh Options (as of 2026-Q2)

| Mesh | CNCF Status | Proxy engine | Deployment modes | Primary differentiator |
|------|------------|-------------|-----------------|----------------------|
| **Istio** | Graduated | Envoy | Sidecar, Ambient (ztunnel + waypoint) | Richest Envoy telemetry; Telemetry API v2; OTLP direct export since 1.22 |
| **Linkerd** | Graduated | linkerd2-proxy (Rust) | Sidecar only | Lightweight; minimal CPU overhead; built-in mTLS; own propagator headers |
| **Envoy** | Graduated | Self | Standalone gateway or sidecar | Proxy engine underlying Istio; also deployed as standalone API gateway |
| **Consul Connect** | Not CNCF | Envoy (via xDS) | Sidecar | HashiCorp ecosystem; strong multi-datacenter support |
| **Kuma** | CNCF Sandbox | Envoy (via xDS) | Sidecar, Universal (VM) | Multi-cloud and multi-zone; Kong-backed |

CNCF status source: <https://landscape.cncf.io>; verify quarterly.

---

## 3. Native Telemetry from the Mesh

### 3.1 Metrics: Envoy Golden Signals

Envoy exposes golden signals per **listener** (inbound) and **cluster** (outbound). Istio enables a controlled subset by default to avoid cardinality explosion caused by per-endpoint label permutations.

| Signal | Metric names (Istio/Envoy) | Label cardinality note |
|--------|---------------------------|----------------------|
| Request rate (throughput) | `istio_requests_total` | High: source × destination × method × status code |
| Error rate | `istio_requests_total{response_code=~"5.."}` | Subset of throughput labels |
| Latency p50/p95/p99 | `istio_request_duration_milliseconds` | Histogram; disable per-endpoint if >1000 pods |
| Saturation (pending) | `envoy_cluster_upstream_rq_pending_total` | Per cluster |
| mTLS handshake errors | `istio_tcp_connections_closed_total` | Proxy-level |

Istio 1.22+ Telemetry API disables `destination_service_name` by default at high pod counts to cap cardinality. Override only after confirming cardinality budget in `resources/meta-observability.md`.

### 3.2 Distributed Traces

Envoy automatically creates **ingress and egress spans** for every HTTP/gRPC call that traverses the proxy. Each span captures:

- `http.method`, `http.status_code`, `http.url` (Stable semconv)
- `peer.service` derived from Envoy cluster name
- mTLS peer identity as `net.peer.name` (when available)
- `x-request-id` correlation ID (Istio-specific; carries across internal calls)

Spans are created without SDK changes. The caveat is that without application-level header forwarding, each service starts a **new root span** rather than contributing to the existing trace. This is the context propagation problem addressed in Section 5.

### 3.3 Access Logs

Envoy access logs provide per-request metadata: source workload identity, destination service, HTTP method and status code, request duration, bytes sent and received, and upstream cluster name. Access logs complement traces; they are always available (100% sampling) even when traces are sampled at 1%.

Structured access log format (JSON) is required for trace correlation:

```json
{
  "trace_id": "%TRACE_ID%",
  "span_id": "%SPAN_ID%",
  "upstream_cluster": "%UPSTREAM_CLUSTER%",
  "response_code": "%RESPONSE_CODE%",
  "duration": "%DURATION%",
  "source_principal": "%DOWNSTREAM_PEER_SUBJECT%"
}
```

> Envoy command operators `%TRACE_ID%` and `%SPAN_ID%` (available since Envoy 1.25) extract the W3C `traceparent` context Envoy itself emitted. Using `%REQ(X-B3-TRACEID)%` is only correct if the upstream explicitly sent the B3 header. Since this skill standardizes on W3C (see §6 Propagator Headers), prefer the command operators.

### 3.4 mTLS Certificate Observability

Istio exposes certificate expiry and rotation events as Prometheus metrics:

- `citadel_server_csr_count`: certificate signing request volume
- `pilot_xds_pushes{type="sds"}`: SDS secret delivery (certificate rotation)
- Cert expiry: scraped from `istio-proxy` via `/pki` endpoint, alertable as a PrometheusRule

Cross-reference: `signals/privacy.md §Security Context (TLS attrs Development)` for attribute stability notes.

---

## 4. OpenTelemetry Direct Export (Envoy 1.29+ / Istio 1.22+)

### 4.1 Envoy OTLP Tracer

Since Envoy 1.29, the built-in OTLP tracer exports spans directly over OTLP/HTTP to any OTel-compatible backend or OTel Collector, replacing the older Zipkin and Jaeger tracers.

Istio 1.22 surfaces this via the **Telemetry API** `opentelemetry` provider type, which avoids editing raw Envoy bootstrap config.

### 4.2 Telemetry CR (Istio 1.22+)

```yaml
# Configure Istio tracing to export OTLP/HTTP to an OTel Collector
apiVersion: telemetry.istio.io/v1
kind: Telemetry
metadata:
  name: otel-tracing
  namespace: istio-system
spec:
  tracing:
    - providers:
        - name: otel-tracing-provider
      randomSamplingPercentage: 10.0
---
# Register the OTel tracing provider in MeshConfig
# (set via IstioOperator or helm values: meshConfig.extensionProviders)
# meshConfig:
#   extensionProviders:
#     - name: otel-tracing-provider
#       opentelemetry:
#         service: otel-collector.observability.svc.cluster.local
#         port: 4318
#         resourceDetectors:
#           environment: {}
```

The `resourceDetectors.environment` block activates the **Environment Resource Detector**, which reads `OTEL_RESOURCE_ATTRIBUTES` from the proxy environment and enriches every span with `service.name`, `service.namespace`, `k8s.pod.name`, and `k8s.node.name` (Stable semconv) automatically.

### 4.3 Custom Samplers (Envoy 1.29+)

Envoy 1.29 introduced the OTel Sampler interface, allowing parent-based or trace-ID-ratio samplers to be configured without patching application code. Tail-based sampling decisions are still made at the Collector tier; see `resources/transport/sampling-recipes.md` for tail-sampler configuration.

---

## 5. Zero-Code Auto-Instrumentation Pitfall: Broken Context Propagation

### 5.1 The Problem

The mesh sidecar creates spans but cannot inject `traceparent` into the application's outbound HTTP calls by itself. If the application does not read the incoming `traceparent` header and forward it on outbound calls, Envoy on the egress side starts a **new trace**, breaking the trace chain. The result is a trace forest of disconnected single-hop spans rather than a unified distributed trace.

This is anti-pattern #1 for mesh observability: mesh-only tracing without application header forwarding.

### 5.2 Solution A: Application-Level Propagator Library

Add the W3C Trace Context propagator to the application SDK. The SDK reads `traceparent` from incoming requests and automatically injects it into all outbound HTTP/gRPC calls.

```python
# Python example (opentelemetry-sdk)
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry import propagate

set_global_textmap(CompositePropagator([
    propagate.get_global_textmap(),  # W3C TraceContext + Baggage
    B3MultiFormat(),                  # B3 for Zipkin-legacy services
]))
```

### 5.3 Solution B: OTel Operator Instrumentation CR

The **OpenTelemetry Operator** (`github.com/open-telemetry/opentelemetry-operator`) injects the OTel SDK init container and environment variables into Pods automatically via a mutating webhook, without any application code change.

Supported language runtimes: Java, NodeJS, Python, .NET, Go, Apache HTTPD, Nginx.

**Step 1; Deploy the Instrumentation CR:**

```yaml
apiVersion: opentelemetry.io/v1alpha1
kind: Instrumentation
metadata:
  name: default-instrumentation
  namespace: my-app
spec:
  exporter:
    endpoint: http://otel-collector.observability.svc.cluster.local:4318
  propagators:
    - tracecontext   # W3C traceparent / tracestate
    - baggage        # W3C baggage
    - b3             # B3 single-header (Zipkin legacy compatibility)
  sampler:
    type: parentbased_traceidratio
    argument: "0.1"
  java:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-java:latest
  nodejs:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-nodejs:latest
  python:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-python:latest
  dotnet:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-dotnet:latest
  go:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-go:latest
```

**Step 2; Activate per Pod via annotation:**

```yaml
metadata:
  annotations:
    instrumentation.opentelemetry.io/inject-java: "true"
    # or: inject-nodejs, inject-python, inject-dotnet, inject-go
```

The Operator injects the SDK agent, sets `OTEL_EXPORTER_OTLP_ENDPOINT`, and configures the propagator list. The sidecar then sees correctly formed `traceparent` headers on all outbound calls, connecting mesh spans to application spans in a single trace.

---

## 6. Propagator Headers per Mesh

Cross-reference: `resources/boundaries/cross-application.md` for full propagator compatibility matrix.

| Header | Standard / Origin | Carried by | Notes |
|--------|------------------|-----------|-------|
| `traceparent` | W3C Trace Context L1 | W3C-compliant SDKs, Envoy OTLP, Istio | Recommended default for all new deployments |
| `tracestate` | W3C Trace Context L1 | Same as above | Vendor-specific trace flags; opaque to intermediaries |
| `baggage` | W3C Baggage | OTel SDK, Istio | Key-value propagation; strip PII at ingress gateway |
| `x-b3-traceid` | B3 (Zipkin) | Envoy B3 propagator, Linkerd (legacy compat) | 128-bit; use B3 multi-header format |
| `x-b3-spanid` | B3 (Zipkin) | Same | 64-bit |
| `x-b3-parentspanid` | B3 (Zipkin) | Same | 64-bit; absent on root spans |
| `x-b3-sampled` | B3 (Zipkin) | Same | `1` = keep, `0` = drop; overrides downstream sampler |
| `x-request-id` | Envoy / Istio internal | Istio sidecar injected | Stable correlation ID across retries; not a trace ID |
| `x-ot-span-context` | OpenTracing (legacy) | Envoy OpenTracing tracer | Deprecated; replaced by OTLP tracer in Envoy 1.29+ |
| `l5d-ctx-trace` | Linkerd | linkerd2-proxy | Linkerd trace context; not W3C-compatible |
| `l5d-ctx-span` | Linkerd | linkerd2-proxy | Current span identifier in Linkerd format |
| `l5d-ctx-parent` | Linkerd | linkerd2-proxy | Parent span reference |
| `l5d-ctx-deadline` | Linkerd | linkerd2-proxy | Deadline propagation (timeout budget) |

**Cross-mesh compatibility rule:** when Istio and Linkerd coexist (or when a service exits the mesh boundary to an external system), standardize on W3C `traceparent` / `tracestate` as the translation surface. Configure each mesh's propagator list to include W3C as the first entry. Translate Linkerd `l5d-ctx-*` headers to W3C at the mesh boundary gateway to avoid split-brain traces.

---

## 7. Combining Mesh and Application Telemetry

The mesh alone covers **network boundary spans**: the ingress proxy creates a server-side span, the egress proxy creates a client-side span. What it cannot see is what the application does between receiving a request and making the next outbound call; database queries, cache reads, external API calls, or business-logic operations.

| Source | Spans covered | Business context |
|--------|--------------|-----------------|
| Mesh (Envoy/Linkerd) | Ingress and egress HTTP/gRPC per service hop | None; proxy has no access to app state |
| Application SDK | Internal operations: DB, cache, queue, business logic | Full; SDK operates inside the application process |
| Combined via OTel Collector | All of the above, correlated by `trace_id` | Complete; best result |

Recommended pipeline architecture:

```
Envoy sidecar
  └─ OTLP/HTTP → OTel Collector (DaemonSet or sidecar)
                    └─ traces → observability backend

Application SDK (via OTel Operator Instrumentation CR)
  └─ OTLP/gRPC → OTel Collector (same or separate instance)
                    └─ traces → observability backend (same trace_id)
```

Both pipelines feed the same backend. The backend joins spans by `trace_id`. The result is a waterfall chart with proxy-level boundary spans and application-level internal spans in the same view.

For Collector topology options see `resources/transport/collector-topology.md`.

---

## 8. mTLS Observability (Security Context)

When Istio PeerAuthentication is set to `STRICT` mode, all pod-to-pod traffic is encrypted with mTLS. The mesh then exposes security context as observable attributes:

- **TLS version**: `tls.protocol_version` (Development tier per semconv 1.27.0; see `resources/standards.md §Semconv Stability Tiers`)
- **Cipher suite**: `tls.cipher` (Development tier)
- **Certificate expiry**: alertable via PrometheusRule on `citadel_server_root_cert_expiry_timestamp`
- **SPIFFE peer identity**: available in Envoy access log via `%DOWNSTREAM_PEER_SUBJECT%`; identifies source workload for zero-trust audit

```yaml
# PrometheusRule: alert when any Istio cert expires within 7 days
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: istio-cert-expiry
  namespace: istio-system
spec:
  groups:
    - name: istio.certificates
      rules:
        - alert: IstioCertExpiryWarning
          expr: |
            (citadel_server_root_cert_expiry_timestamp - time()) / 86400 < 7
          for: 1h
          labels:
            severity: warning
          annotations:
            summary: "Istio root certificate expires in fewer than 7 days"
```

Cross-reference: `resources/signals/privacy.md §Security Context` for TLS attribute stability notes.

---

## 9. Sampling Considerations

Istio's default sampling rate is **1%** (`randomSamplingPercentage: 1.0`). At 1%, you need 100 requests before the first trace is visible in the backend; this is insufficient for debugging low-traffic endpoints or canary deployments.

| Scenario | Recommended sampling rate | Configuration location |
|----------|--------------------------|----------------------|
| Development / staging | 100% | Telemetry CR `randomSamplingPercentage: 100` |
| Production (baseline) | 10% | Telemetry CR `randomSamplingPercentage: 10` |
| High-traffic SLO critical path | Tail-based at Collector | Collector `tail_sampling` processor |
| Canary release tracing | 100% on canary subset | Telemetry CR scoped to canary namespace or label |

Tune via the Telemetry API `samplingPercentage` field (Istio 1.22+) or by configuring a custom OTel Sampler via Envoy 1.29+. For tail-based sampling (retain error traces, drop successful traces after SLO window), configure at the gateway Collector tier as documented in `resources/transport/sampling-recipes.md`.

---

## 10. Matrix Coverage Reference (mesh row)

These cells from `resources/matrix.md` are the primary coverage drivers for this file:

| matrix cell | symbol | artifact |
|------------|--------|---------|
| mesh × cross-application × traces | PASS | Zero-code L7 trace continuity; primary use case of the mesh layer |
| mesh × multi-tenant × traces | PASS | `tenant.id` via W3C Baggage; proxy can enforce baggage scrubbing at gateway |
| mesh × release × traces | PARTIAL | Canary routing rules are observable; `service.version` on spans; trace continuity requires OTel Operator CR |
| mesh × privacy × * | PASS | mTLS config observability; baggage scrubbing; SPIFFE identity in access log |
| mesh × slo × metrics | PASS | Envoy request rate + error rate are primary SLI sources |

---

## 11. Anti-Patterns

The following are anti-patterns for this layer. They are candidates for inclusion in `resources/anti-patterns.md`.

| # | Anti-pattern | Impact | Remedy |
|---|-------------|--------|--------|
| AP-M1 | Mesh-only tracing without application SDK | Spans cover network hops only; DB queries, business logic, and external API calls invisible | Deploy OTel Operator Instrumentation CR to inject SDK into all application pods |
| AP-M2 | W3C Trace Context not standardized across meshes | Linkerd and Istio produce disconnected trace forests when a request crosses mesh boundaries | Configure W3C as first propagator in both meshes; translate at boundary gateway |
| AP-M3 | Default 1% Istio sampling rate in production without awareness | First trace visible only after 100 requests; low-traffic endpoints never appear in trace backend | Raise to 10% baseline; use tail-based sampling at Collector for cost control |
| AP-M4 | OTel Operator Instrumentation CR not deployed | `traceparent` not forwarded by application code; mesh spans are disconnected root spans | Deploy `Instrumentation` CR and annotate all application Deployments |
| AP-M5 | Cardinality explosion from per-endpoint Envoy labels | `istio_requests_total` with high-cardinality `destination_service_name` × `source_workload` × URL path exceeds Prometheus cardinality budget | Use Telemetry API to suppress high-cardinality labels; aggregate at Collector metric transform processor |

---

## 12. References


Internal cross-references:
- `resources/standards.md`: normative semconv stability tiers and W3C Trace Context requirements
- `resources/matrix.md`: full 112-cell coverage map (mesh row)
- `resources/transport/sampling-recipes.md`: tail-based sampling at Collector tier
- `resources/transport/collector-topology.md`: DaemonSet vs sidecar Collector topology
- `resources/meta-observability.md`: cardinality guardrails and pipeline self-health
- `resources/layers/L4-transport.md`: eBPF socket-level profiles for mesh sidecar overhead
- `resources/boundaries/cross-application.md`: full propagator compatibility matrix
- `resources/boundaries/multi-tenant.md`: baggage-based tenant attribution
- `resources/boundaries/release.md`: canary trace routing with `service.version`
- `resources/signals/privacy.md`: TLS attribute stability and mTLS security context
- `resources/signals/traces.md`: OTel SDK trace patterns for application layer

## References

- Istio observability concepts: <https://istio.io/latest/docs/concepts/observability/>
- Istio OpenTelemetry tracing task: <https://istio.io/latest/docs/tasks/observability/distributed-tracing/opentelemetry/>
- Envoy + Istio OTel features (2024): <https://opentelemetry.io/blog/2024/new-otel-features-envoy-istio/>
- OTel Operator GitHub: <https://github.com/open-telemetry/opentelemetry-operator>
- Linkerd observability and propagator docs: <https://linkerd.io/2.15/features/distributed-tracing/>
- W3C Trace Context L1: <https://www.w3.org/TR/trace-context/>
- OTel semconv `tls.*`: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/tls/>
