---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
notes:
  - "Profiles signal (OTEP 0239) is alpha; semconv in development"
---

# Profiles Signal

> **EXPERIMENTAL; OTEP 0239 alpha (as of 2026-Q2)**
> Continuous profiling tooling (Parca, Pyroscope) is production-ready.
> The OpenTelemetry profiling data model and OTLP signal are in alpha / active development; semantic conventions and wire format may change without backward compatibility.
> Do NOT define SLOs against OTel-native profiling attributes. Treat any `profile.*` OTel semconv as experimental tier.
> Cross-ref: `../standards.md §3 OTel semconv stability` for tier definitions.

---

## 1. Scope

Continuous profiling; whole-cluster or whole-service CPU, memory, and allocation sampling running at low, steady overhead in production. Flamegraphs are the primary visualization artifact. The output answers the question "in which function is time or memory being spent?"

This file does NOT cover ad-hoc developer profiling sessions (e.g., `go tool pprof http://localhost:6060/debug/pprof/profile` run once during a development investigation). Those are single-use diagnostics. Continuous profiling generates persistent, comparable data over time and is as much a production signal as metrics or logs.

---

## 2. The 5th Pillar: MELT to MELT+P

Observability was originally described as three pillars (Metrics, Logs, Traces). Events as LogRecords and cost extended that. Profiles add the fifth pillar.

| Signal | Answers |
|--------|---------|
| Metrics | What is slow / broken? (rate, error rate, latency quantile) |
| Traces | Where is it slow? (which service, which span, which dependency) |
| Profiles | In which function is CPU time or memory spent? (line-level attribution) |
| Logs | Why, in context? (structured event with variables and error detail) |
| Cost | What is the financial impact of the above? |

Correlation is the mechanism that makes MELT+P actionable rather than five isolated views. Profiles share resource labels (`service.name`, `service.version`, `k8s.pod.name`) with metrics and traces, and can be linked to a specific trace by attaching a `trace_id` label to profile samples; a pattern pioneered by Grafana Tempo + Pyroscope.

---

## 3. Tools Landscape (as of 2026-Q2)

Verify live status at <https://landscape.cncf.io>.

| Tool | Category | Mechanism | CNCF Status | Notes |
|------|----------|-----------|-------------|-------|
| Parca | OSS, continuous | eBPF kernel-level, no agent code | Sandbox | Written in Go; Polar Signals led; whole-cluster, no language SDK required for CPU profiles |
| Grafana Pyroscope | OSS / SaaS, continuous | Pull + push (eBPF + language SDKs) | Sandbox (verify; CNCF status uncertain post-Grafana Labs acquisition, 2023) | Formerly independent; Grafana acquired 2023; integrated into Grafana stack |
| Polar Signals Cloud | Commercial SaaS | Parca-based, hosted | N/A | Commercial offering backed by Parca OSS; Polar Signals is the primary Parca maintainer |
| Go `net/http/pprof` | Language-specific, on-demand | HTTP handler; sample on request | N/A | Built-in; not continuous; heavier per request; MUST NOT be left exposed on production endpoints |
| Fgprof | Language-specific | Wall-clock profiler for Go | N/A | Supplements `pprof` by measuring off-CPU time as well as on-CPU |
| py-spy | Language-specific, continuous | Native code (Rust); attaches without code change | N/A | Low overhead; works on CPython; no SDK injection |
| pyroscope-python | Language-specific, continuous | Python SDK; sends to Pyroscope server | N/A | Requires SDK import; supports async runtimes |
| async-profiler | JVM, continuous | JVMTI + `perf_events`; async-safe | N/A | Primary choice for JVM profiling; supports CPU, alloc, lock, wall modes |
| JFR (Java Flight Recorder) | JVM, continuous | JVM-native; available from JDK 11+ | N/A | Enable with JVM flags; low overhead; integrates with async-profiler |
| `0x` / V8 Inspector | Node.js, on-demand | V8 CPU profiler | N/A | `0x` wraps `node --prof` for flamegraph generation |
| pyroscope-node | Node.js, continuous | Node SDK; sends to Pyroscope | N/A | Continuous alternative to `0x` |
| stackprof | Ruby, continuous | Sampling profiler | N/A | Low overhead; supports wall, object, custom modes |
| php-spx / phpspy | PHP, continuous | Extension (spx) or attach (spy) | N/A | `php-spx` requires extension install; `phpspy` attaches without code change |
| Beyla / Pixie | eBPF, language-agnostic | eBPF user-space probes | Beyla CNCF Incubating (2024); Pixie CNCF Sandbox | No language integration needed; covers Go, Python, JVM, Node.js for latency profiling |

**eBPF unified approach note:** For CPU and off-CPU profiling without language-level SDK integration, Parca, Beyla, and Pixie are the primary choices. They instrument at the kernel/eBPF level, making them language-agnostic for CPU time attribution.

---

## 4. Sampling Overhead

Continuous profiling is viable in production only when overhead is provably low.

| Tool / Mode | Typical CPU overhead | Sampling rate | Notes |
|------------|---------------------|---------------|-------|
| Parca (eBPF) | < 1% CPU | ~19 Hz per CPU core | Kernel-level; process namespaces handled correctly |
| Pyroscope (eBPF mode) | < 1% CPU | Configurable, default ~100 Hz | Overhead scales with sampling rate; start conservatively |
| async-profiler (CPU) | < 1% CPU | Configurable; default 100 Hz | AsyncGetCallTrace avoids safe-point bias |
| JFR (CPU + alloc) | < 1% CPU typical | JVM-managed | Monitor GC pause impact separately |
| py-spy | < 1% CPU | Configurable; default 100 Hz | External; no Python overhead |
| `net/http/pprof` | ~5-10% during active profile | On-demand only | Never enable `/debug/pprof` permanently on production endpoints; see anti-patterns |

Always measure baseline and profiled CPU utilization in staging before enabling continuous profiling in production. Set a budget threshold (e.g., < 1% CPU) and alert if actual overhead exceeds it.

---

## 5. OTEP 0239: OTel Profiling Signal Data Model

Source: <https://github.com/open-telemetry/oteps/blob/main/text/profiles/0239-profiles-data-model.md>

Status: **alpha / in active development (2026-Q2)**. The data model, wire format, and semantic conventions are all subject to breaking change.

Key design points:
- Profiles are transmitted as a fourth OTLP signal alongside metrics, traces, and logs. A new `ExportProfilesServiceRequest` protobuf message parallels `ExportMetricsServiceRequest`.
- The data model derives from the pprof format (Google); a profile is a directed call graph with sample counts per stack frame.
- Proposed profile types: CPU, Heap, Allocation, Mutex, Lock. Stored in `profile.type` attribute.
- `profile.name` identifies the profile within a service and collection window.
- Semantic conventions for profile attributes (e.g., mapping to `service.*`, `k8s.*`, `thread.*`) are in draft. Do not treat them as stable.

```
# Conceptual OTLP profile payload (pseudocode — not final spec)
ProfilesData {
  resource_profiles: [
    ResourceProfiles {
      resource: { attributes: [service.name, service.version, k8s.pod.name] }
      scope_profiles: [
        ScopeProfiles {
          profiles: [
            Profile {
              profile_id: <16-byte random>
              start_time_unix_nano: ...
              end_time_unix_nano: ...
              sample_type: [{ type: "cpu", unit: "nanoseconds" }]
              sample: [{ location_id: [...], value: [...], label: [...] }]
            }
          ]
        }
      ]
    }
  ]
}
```

Most production deployments (2026-Q2) use Parca's own gRPC API or Pyroscope's push API, not OTLP profiles. OTLP profile support in collectors and backends is experimental. Plan OTLP profile adoption only after the spec reaches RC status.

---

## 6. Use Cases

| Use case | Profile type | How to investigate |
|----------|-------------|-------------------|
| CPU bottleneck; which function consumes CPU? | CPU | Flamegraph: wide frame = high CPU; sort by self-time |
| Memory leak; heap growing over time? | Heap | Heap profile over hours; compare allocation counts |
| Allocation rate; GC pressure? | Allocation | Allocation profile; identify hot allocation sites |
| Lock contention; threads blocked? | Mutex / Lock | Mutex profile; look for frames holding locks across many samples |
| Off-CPU; threads blocked on I/O or sleep? | Off-CPU (eBPF) | eBPF off-CPU flamegraph; I/O, syscall, and scheduler frames visible |

---

## 7. Correlation with Traces

Profiles are the diagnostic tool you reach for after traces identify which service and span is slow. The linkage mechanism is attaching the current `trace_id` as a label on profile samples.

```
# Pyroscope Go SDK — attach trace_id to profile (conceptual)
pyroscope.TagWrapper(ctx, pyroscope.Labels("trace_id", span.SpanContext().TraceID().String()), func() {
    // code under profiling
})
```

Grafana Tempo + Pyroscope integration implements this as a first-class UI feature: clicking a slow trace span offers a "View Profile" link that loads the flamegraph for that time window and service.

**Limitation:** Profiles are statistical (sampled across all requests). Traces are per-request. The correlation is probabilistic; a trace_id label on a profile sample means that request was running during that sample, not that the sample is causally attributed to that exact request. This is a fundamental constraint of sampling-based profiling; document it clearly when presenting correlation results to stakeholders.

---

## 8. Mobile and Client-Side Profiling

Client-side continuous profiling in production is not yet established practice. Platform tools provide opt-in session profiling:

| Platform | Tool | Mode | Notes |
|----------|------|------|-------|
| iOS | Instruments (Time Profiler, Allocations) | Developer / TestFlight | Requires Xcode; not continuous in App Store builds |
| Android | Android Studio Profiler (CPU, Memory) | Developer / debug build | Production CPU profiling requires `<profileable>` manifest flag |

For production mobile observability, rely on crash analytics and RUM metrics to surface performance regressions. Cross-ref `../layers/L7-application/mobile-rum.md` for mobile RUM patterns and `../layers/L7-application/crash-analytics.md` for crash rate metrics.

---

## 9. Storage Backends

| Backend | Primary use | Symbol resolution |
|---------|------------|-------------------|
| Parca server | OSS; local or k8s deployment; uses Parquet-backed storage | Upload ELF debug info / DWARF symbols; agent resolves at collection time |
| Pyroscope OSS | Self-hosted; integrates with Grafana stack | Upload symbol files per build |
| Grafana Cloud Profiles | Managed Pyroscope; SaaS | Upload symbols via Pyroscope API |
| Polar Signals Cloud | Managed Parca; SaaS; automatic symbol upload | Symbols uploaded via CI step; no manual resolution needed |

Symbol resolution is a critical operational concern. Without debug symbols (or source maps for Node.js), flamegraph frames appear as raw memory addresses or mangled names. Treat symbol upload as a required CI step alongside binary deployment; it has the same relationship to profiles as sourcemaps have to crash analytics.

---

## 10. Privacy and Security

Stack traces expose function names, which may reveal proprietary algorithm structure or internal service architecture. This is especially sensitive for multi-tenant SaaS operators.

- Apply access control to the profiling backend (Parca server, Pyroscope) equivalent to trace backend access control. Cross-ref `../signals/privacy.md §Backend RBAC`.
- For multi-tenant deployments, isolate flamegraph access by tenant label. A tenant MUST NOT be able to query another tenant's flamegraph.
- Function name redaction (stripping internal frames from exported profiles) is technically possible but rare and operationally complex. Consider it only if profile data is shared externally (e.g., vendor support handoff).

---

## 11. Matrix Cells: Profiles Column Summary

Cross-ref: `../matrix.md` for full cell detail. Summary for the profiles signal column:

| Layer × Boundary | profiles cell | Rationale |
|-----------------|---------------|-----------|
| L3 × all | N/A | IP routing has no process-level profiling artifact |
| L4 × multi-tenant | PASS | eBPF CPU/off-CPU profiling covers socket-level overhead per process |
| L4 × cross-application | PASS | eBPF off-CPU shows network-wait time between services |
| L4 × slo | N/A | SLO is application-defined; L4 profiles are diagnostic, not SLO inputs |
| L4 × release | N/A | Release events are L7 constructs |
| mesh × all | PARTIAL | Proxy CPU visible via eBPF on sidecar process; not mesh-native |
| L7 × multi-tenant | PASS | Per-tenant flamegraph via label-based isolation in Parca/Pyroscope |
| L7 × cross-application | PASS | Application-level profiling with trace_id correlation |
| L7 × slo | N/A | SLO is computed from metrics; profiles are diagnostic |
| L7 × release | PARTIAL | Pre/post-release profile diff for performance regression detection |

---

## 12. Anti-Patterns

These candidates belong in `../anti-patterns.md` and are flagged here for consolidation.

| Anti-pattern | Risk | Mitigation |
|-------------|------|-----------|
| Continuous profiling enabled in production without overhead measurement in staging | Unexpected CPU spike > 1%; service degradation | Benchmark overhead in staging first; set alert on profiler CPU budget |
| Symbol upload missing at deploy time | Flamegraphs show raw addresses or mangled names; engineers cannot act on data | Add symbol upload as a required CI step immediately after binary upload; gate deployment on symbol upload success |
| `/debug/pprof` endpoint left enabled on production HTTP servers | Exposes CPU/heap profiles and internal function names to any caller who discovers the path; also a denial-of-service vector (profile requests are expensive) | Disable `net/http/pprof` import in production builds, or restrict to internal network behind authentication middleware |
| Profile labels include high-cardinality dimensions (e.g., `request_id`, `user_id`) | Storage blowup; query performance degradation in Parca/Pyroscope backend | Label with low-cardinality dimensions only: `service.name`, `service.version`, `tenant.id` (for multi-tenant), `environment` |

---

## 13. References


Internal cross-references:

- `../standards.md §3`: OTel semconv stability tiers (experimental badge rationale)
- `../matrix.md`: full 112-cell coverage matrix
- `../layers/L4-transport.md`: eBPF profiling at socket level
- `../layers/L7-application/mobile-rum.md`: mobile performance signals
- `../layers/L7-application/crash-analytics.md`: crash rate and symbolication
- `../boundaries/multi-tenant.md`: per-tenant label isolation
- `../boundaries/release.md`: pre/post-release profile comparison
- `../signals/privacy.md`: backend RBAC and access control
- `../anti-patterns.md`: consolidated anti-pattern list

## References

- OTEP 0239 (profiling data model): <https://github.com/open-telemetry/oteps/blob/main/text/profiles/0239-profiles-data-model.md>
- Parca: <https://parca.dev>
- Grafana Pyroscope: <https://grafana.com/oss/pyroscope/>
- Polar Signals Cloud: <https://polarsignals.com>
- async-profiler: <https://github.com/jvm-profiling-tools/async-profiler>
- py-spy: <https://github.com/benfred/py-spy>
- CNCF Landscape (verify current CNCF status): <https://landscape.cncf.io>
