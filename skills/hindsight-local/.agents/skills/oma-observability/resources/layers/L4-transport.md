---
otel_semconv: "1.27.0 (2024-11)"
---

# L4 Transport Layer Observability

> **DISAMBIGUATION; read before proceeding.**
>
> This file covers **observing the OSI Layer 4 (TCP/UDP/QUIC) transport in your application systems**; 
> retransmits, RTT, connection lifecycle, eBPF-based socket instrumentation, and QUIC/HTTP3 semantics.
>
> This file is **NOT** about the observability pipeline transport:
> - `../transport/udp-statsd-mtu.md`; StatsD payload sizing over UDP
> - `../transport/otlp-grpc-vs-http.md`; OTLP exporter protocol choice
> - `../transport/collector-topology.md`; OTel Collector deployment topologies
> - `../transport/sampling-recipes.md`; tail/head sampling strategies
>
> The word "transport" is overloaded. When a cross-reference says `transport/`, it refers to the
> observability pipeline. When it says "L4", "TCP/UDP/QUIC", or this filename, it refers to this file.

---

## 1. Scope

OSI Layer 4 observability for production systems: TCP, UDP, QUIC/HTTP3, eBPF auto-instrumentation,
and connection lifecycle. Out of scope: L5 session (gRPC/WebSocket → L7), L6 full TLS inspection
(use Wireshark/vendor tooling), security-focused eBPF (Cilium Tetragon/Falco → security skill).
See `../standards.md §5` for the authoritative OSI boundary decision.

---

## 2. L4 Semconv (OTel)

Source: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>

| Attribute | Values | Stability | Notes |
|-----------|--------|-----------|-------|
| `network.transport` | `tcp`, `udp`, `unix`, `pipe`, `quic` | **Stable** | Set on spans and metrics describing a socket |
| `network.protocol.name` | `http`, `grpc`, `amqp`, … | **Stable** | Application protocol over the transport |
| `network.protocol.version` | `1.1`, `2`, `3` | **Stable** | `3` = HTTP/3 over QUIC |
| `network.peer.address` | IP or hostname | **Stable** | Apply IP-masking before retention; `../signals/privacy.md §IP addresses` |
| `network.connection.state` | `established`, `close_wait`, … | **Development** | Do not build SLOs against this; schema may change |
| `network.connection.type` | `wifi`, `cell`, … | **Development** | Mobile/RUM only; flag in production use |

`network.connection.*` attributes are Development tier (`../standards.md §3`). Use them in test
environments and tolerate breaking changes before promoting to production dashboards.

---

## 3. TCP Observability

### 3.1 Key Metrics

| Metric | Type | Measures | Source |
|--------|------|----------|--------|
| `tcp_retransmits_total` | Counter | Retransmit segments; rate spike = congestion or loss | hostmetrics, eBPF |
| `tcp_rtt_ms` (p50/p99) | Histogram | Round-trip time per connection | eBPF socket stats |
| `tcp_active_connections` | Gauge | In-flight established connections; growth = pool leak | `/proc/net/sockstat` |
| `tcp_time_wait_connections` | Gauge | TIME_WAIT count; high under short-lived connection churn | `/proc/net/sockstat` |
| `tcp_syn_queue_drops_total` | Counter | SYN queue overflow; SYN flood or low `tcp_max_syn_backlog` | netlink / eBPF |
| `tcp_close_wait_connections` | Gauge | CLOSE_WAIT growth = server not closing half-open connections | hostmetrics |

```promql
# Retransmit rate per node
rate(tcp_retransmits_total[5m])

# p99 RTT per service
histogram_quantile(0.99, sum by (le, service_name) (rate(tcp_rtt_ms_bucket[5m])))

# TIME_WAIT pressure alert
tcp_time_wait_connections > 10000
```

### 3.2 Collection

| Method | Mechanism | Privilege |
|--------|-----------|-----------|
| OTel hostmetrics receiver | `/proc/net/tcp`, `/proc/net/snmp`, `/proc/net/sockstat` | Root or CAP_SYS_PTRACE |
| eBPF socket filter | kprobes on `tcp_retransmit_skb`, `tcp_sendmsg` | CAP_BPF (kernel ≥ 5.8) or root |
| Linux netlink SOCK_DIAG | TCP socket state enumeration | Unprivileged in most distros |

### 3.3 Common Pitfalls

**Connection pool exhaustion.** High `tcp_active_connections` with high application latency but no TCP errors
points to pool queue saturation, not network failure. Cross-ref: `../signals/traces.md §DB connection pool patterns`.

**SYN cookie activation.** `tcp_syn_queue_drops_total` > 0 in normal traffic means `tcp_max_syn_backlog`
is too low. SYN cookies mask drop counters in `/proc/net/snmp`; use eBPF for reliable counts.

**TIME_WAIT accumulation.** Short-lived HTTP/1.1 connections exhaust ephemeral ports. Remediation:
`SO_REUSEADDR`, wider `ip_local_port_range`, or HTTP/2 multiplexing.

---

## 4. UDP Observability

| Metric | Source | Measures |
|--------|--------|----------|
| `udp_datagrams_sent_total` | `/proc/net/snmp` | Baseline send throughput |
| `udp_receive_buffer_errors_total` | `/proc/net/snmp` `RcvbufErrors` | Socket buffer overflow; increase `net.core.rmem_max` |
| `udp_no_ports_total` | `/proc/net/snmp` `NoPorts` | Drop because no listener on destination port |
| `udp_checksum_errors_total` | `/proc/net/snmp` `InCsumErrors` | Corrupted datagrams; NIC or path corruption |

OTel hostmetrics `network` scraper covers these counters automatically.
Cross-ref: `../transport/udp-statsd-mtu.md` for StatsD payload sizing constraints that affect
`udp_receive_buffer_errors_total` in high-throughput StatsD pipelines.

---

## 5. eBPF-Based L4 Observability

eBPF instruments the Linux kernel without application code changes. Three tools are in scope.

### 5.1 Tool Comparison

| Tool | CNCF Status | Primary signal | OTel output | Key use case |
|------|-------------|----------------|-------------|--------------|
| **Grafana Beyla** | CNCF Incubating (2024) | HTTP/gRPC traces + Go/C++ runtime | OTLP natively | Zero-code HTTP/gRPC auto-instrumentation via uprobes; also emits `tcp_rtt_us` histograms. Requires kernel ≥ 5.2 with BTF enabled |
| **Pixie** | CNCF Sandbox | Multi-protocol traces + L4 metrics via PxL | OTLP via `px-export` | Scriptable cluster-wide eBPF; auto-discovers pods |
| **Parca** | CNCF Sandbox | Continuous CPU + off-CPU flame graphs | pprof-compatible | Whole-cluster profiling; cross-ref `../signals/profiles.md` |

Sources: <https://grafana.com/oss/beyla/>, <https://px.dev/>, <https://parca.dev/>

**Out of scope: Cilium Tetragon.** Security-focused runtime policy enforcement (syscall filtering, process
exec tracing) belongs to a security skill. Falco is the comparable alternative for kernel security events.

### 5.2 Kernel and Privilege Requirements

| Requirement | Detail |
|-------------|--------|
| Minimum kernel | ≥ 4.14 for most eBPF features; ≥ 5.8 for CAP_BPF without root |
| Capabilities | CAP_BPF + CAP_PERFMON (kernel ≥ 5.8) or root on older kernels |
| GKE Autopilot | eBPF DaemonSets disallowed; use Standard node pools |
| EKS Fargate | No DaemonSet support; eBPF unavailable |
| AKS | Supported on standard node pools; verify with `uname -r` |

```bash
# Preflight kernel check (run in initContainer or node admission)
uname -r           # must be >= 4.14, ideally >= 5.8
capsh --print | grep cap_bpf

# Beyla deploy via Helm
helm repo add grafana https://grafana.github.io/helm-charts
helm install beyla grafana/beyla --namespace beyla --create-namespace
```

### 5.3 Pixie PxL (Conceptual)

```python
# Conceptual PxL — TCP retransmit counts per pod
import px
df = px.DataFrame(table='tcp_retransmits', start_time='-5m')
df = df.groupby(['pod', 'namespace']).agg(retransmits=('retransmits', px.sum))
px.display(df)
```

---

## 6. Connection Lifecycle Observability

```
LISTEN → SYN_RECEIVED → ESTABLISHED → (data transfer)
  → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED   (active close)
  → CLOSE_WAIT → LAST_ACK → CLOSED                  (passive close)
```

| State | Observable signal | Tooling |
|-------|------------------|---------|
| SYN → ESTABLISHED | Setup latency (SYN-ACK RTT) | eBPF `tcp_v4_connect` kprobe |
| ESTABLISHED idle | Keepalive probe success/failure | `/proc/net/tcp` + hostmetrics |
| FIN_WAIT / TIME_WAIT | TIME_WAIT gauge; ephemeral port pressure | hostmetrics `/proc/net/sockstat` |
| CLOSE_WAIT growth | Server not releasing half-open connections | hostmetrics + eBPF alert |

**TLS handshake boundary.** TLS sits at L4/L5. `tls.*` semconv is Development tier; do not build
production SLOs on it. Use eBPF uprobes on OpenSSL/BoringSSL or vendor tooling for handshake latency.
Cross-ref: `../signals/privacy.md §TLS context`.

**Keepalive tuning.** TCP keepalive interval must be shorter than the upstream LB idle timeout
(AWS ALB: 60 s; AWS NLB: 350 s; GCP CLB: 600 s default). Mismatches cause silent connection resets.

**WebSocket and gRPC streams**; full lifecycle coverage is deferred to L7 (`../standards.md §5`).
Long-lived streams mitigate TIME_WAIT accumulation but introduce reconnection-storm risk on restart.

---

## 7. QUIC / HTTP/3 Transport Semantics

QUIC (RFC 9000) is a UDP-based transport with built-in TLS 1.3, stream multiplexing, and connection
migration. HTTP/3 (RFC 9114) runs exclusively over QUIC.

### 7.1 Observability Differences from TCP

| TCP concept | QUIC equivalent | Observability impact |
|-------------|----------------|----------------------|
| 3-way SYN handshake | 1-RTT or 0-RTT | No SYN/FIN to observe; connection state is QUIC-internal |
| 5-tuple for correlation | Connection ID (opaque bytes) | Connection ID survives NAT rebinding; log it on establishment |
| `/proc/net/tcp` retransmit | QUIC loss recovery (packet number gaps) | Not visible in proc; requires QUIC-aware tooling |
| TCP FIN / RST | QUIC GOAWAY / CONNECTION_CLOSE frames | Surfaced in Envoy access logs |
| HTTP/2 HOL blocking | Per-stream flow control; no HOL blocking | Stream errors do not block other streams |

### 7.2 OTel Semconv for QUIC

```
network.transport: quic
network.protocol.name: http
network.protocol.version: "3"
```

### 7.3 0-RTT Observability

0-RTT sends data before the handshake completes. Emit a span attribute or metric label
distinguishing 0-RTT from 1-RTT connections to detect replay exposure on non-idempotent endpoints.

### 7.4 Firewall and Enterprise Deployment Considerations

Enterprise networks frequently block outbound UDP 443 and UDP 8443 because legacy firewall rules assume HTTP(S) is TCP-only. QUIC (and therefore HTTP/3) rides on UDP and silently falls back to HTTP/1.1 or HTTP/2 when UDP is blocked; the fallback is invisible to application telemetry unless you instrument it.

Detect QUIC blocking:
- Emit `network.transport` attribute on client spans; compare ratio of `quic` vs `tcp` across deployment zones
- CDN access logs (Cloudflare, Fastly) expose QUIC negotiation outcome per edge
- Alert when `quic` ratio drops below baseline (for example, office network where UDP 443 is firewalled)

Action: coordinate with network security to allowlist UDP 443/8443 for validated destinations; document the fallback behavior so regressions do not silently halve transport performance.

### 7.5 Tool Support (2026)

| Tool | QUIC support |
|------|-------------|
| Envoy 1.29+ | QUIC upstream and downstream; stats in admin API and access logs |
| Istio 1.22+ | QUIC upstream via Envoy; mTLS over QUIC experimental; see `layers/mesh.md` |
| Grafana Beyla | HTTP/3 uprobes in roadmap; verify release notes before QUIC-heavy deployments |
| Cloudflare QUIC telemetry | <https://blog.cloudflare.com/tag/quic/>; public connection migration + 0-RTT metrics |
| Google QUIC metrics | <https://research.google>; packet loss recovery timing |

---

## 8. OS-Level Integration

| Source | Signal | Cross-ref |
|--------|--------|-----------|
| systemd-journal / syslog | TCP resets, netfilter drops, NIC errors | `../signals/logs.md §OS-level log sources` |
| cgroup v2 net accounting | Per-container byte/packet counters | `../transport/collector-topology.md §kubernetes` |
| nf_conntrack | NAT and stateful firewall connection table size | Requires `nf_conntrack` kernel module |

---

## 9. Matrix Cells: L4 Row

From `../matrix.md §L4-transport`:

| Cell | Symbol | Rationale |
|------|--------|-----------|
| L4 × cross-application × traces | PARTIAL | TCP is not trace-native; log socket 5-tuple for post-hoc correlation to L7 spans only |
| L4 × * × profiles | PASS | eBPF off-CPU profiling via Parca captures kernel socket overhead and network-wait; pprof output |
| L4 × privacy (all boundaries) | PARTIAL | Source/destination IPs in TCP logs are PII (GDPR Art. 4(1)); apply masking at pipeline ingress |
| L4 × SLO and L4 × release | N/A | SLO budgets and release markers are application-layer; TCP health is infra-SLI fallback only |

---

## 10. Anti-Patterns: Candidates for `../anti-patterns.md §Section H`

**H-L4-1: Connection pool observability absent.** Pool queue saturation is invisible in TCP metrics
alone; neither retransmit rate nor error rate spikes until the pool times out. Instrument pool size,
wait time, and timeout counters at the application layer. Severity: HIGH.

**H-L4-2: QUIC adoption without HTTP/3 trace tooling validation.** Enabling QUIC without verifying
that OTel SDKs emit `network.transport: quic` and `network.protocol.version: "3"` creates a
transport-layer blind spot. Add a canary assertion in staging before production rollout. Severity: HIGH.

**H-L4-3: eBPF agent deployed without kernel/capability preflight.** Beyla or Pixie DaemonSets on
incompatible kernels fail silently with no operator-visible error. Add an `initContainer` asserting
kernel version and CAP_BPF presence; fail non-zero if requirements are unmet. Severity: HIGH.

**H-L4-4: WebSocket reconnection storm undetected.** Server restart without graceful WebSocket drain
causes simultaneous mass-reconnect. TCP SYN counts look like normal new connections. Instrument client
reconnection counters and alert on rate spikes; implement exponential backoff + jitter. Severity: MEDIUM.

---

## 11. References

- OTel network attributes registry: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>
- Grafana Beyla: <https://grafana.com/oss/beyla/>
- Pixie (CNCF Sandbox): <https://px.dev/>
- Parca (CNCF Sandbox): <https://parca.dev/>
- QUIC RFC 9000: <https://www.rfc-editor.org/rfc/rfc9000>
- HTTP/3 RFC 9114: <https://www.rfc-editor.org/rfc/rfc9114>
- Cloudflare QUIC telemetry: <https://blog.cloudflare.com/tag/quic/>
- OTel hostmetrics receiver: <https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/hostmetricsreceiver>
- Linux `/proc/net/tcp` format: <https://www.kernel.org/doc/html/latest/networking/proc_net_tcp.html>
- Semconv stability tiers: `../standards.md §3`
- OSI boundary decisions: `../standards.md §5`
- Coverage matrix (L4 row): `../matrix.md §L4-transport`
- IP masking rules: `../signals/privacy.md §IP addresses`
- Profiling (eBPF/Parca): `../signals/profiles.md`
- OS-level logs: `../signals/logs.md §OS-level log sources`
- Collector Kubernetes topology: `../transport/collector-topology.md`
- StatsD UDP sizing: `../transport/udp-statsd-mtu.md`
- Service mesh + QUIC: `layers/mesh.md`
