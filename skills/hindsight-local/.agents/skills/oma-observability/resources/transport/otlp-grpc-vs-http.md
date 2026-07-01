---
otel_spec: "1.x (stable API/SDK)"
---

# OTLP Transport: gRPC vs HTTP Decision Guide

## 1. OTLP Basics

The OpenTelemetry Protocol (OTLP) uses a single Protobuf schema for traces, metrics, and logs.
It defines three wire formats over two transports:

| Transport      | Content-Type                       | Port (default) |
|----------------|------------------------------------|----------------|
| gRPC           | `application/grpc`                 | **4317**       |
| HTTP/Protobuf  | `application/x-protobuf`           | **4318**       |
| HTTP/JSON      | `application/json`                 | **4318**       |

All three formats carry identical semantic data. Transport choice is an operational decision.

Sources:
- OTLP specification: opentelemetry.io/docs/specs/otlp/
- Collector configuration: opentelemetry.io/docs/collector/configuration/
- OTLP receiver README: github.com/open-telemetry/opentelemetry-collector (receiver/otlpreceiver)

---

## 2. Decision Tree

```
Is the sender a browser or a runtime without gRPC support?
  YES → HTTP/protobuf (port 4318) — smallest payload, wide compatibility
        HTTP/JSON if debugging or if protobuf encoding unavailable

  NO → Does a corporate proxy or layer-7 firewall sit between sender and collector?
         YES → HTTP (port 4318); proxies handle HTTP/1.1 and HTTP/2 cleartext reliably
               Confirm with: curl -v http://<collector>:4318/v1/traces

         NO → Is traffic internal to a Kubernetes cluster (pod-to-pod)?
                YES → gRPC (port 4317): multiplexed streams, lower per-RPC overhead,
                      native health checking via grpc_health_probe
                      Use headless Service or client-side LB (see section 6)

                NO → Is volume high (>50 K spans/sec) or latency critical?
                       YES → gRPC + gzip compression
                       NO  → Either transport works; prefer gRPC for consistency
```

| Axis                      | Recommended transport | Reason                                          |
|---------------------------|-----------------------|-------------------------------------------------|
| Browser / Electron        | HTTP/JSON or HTTP/protobuf | No gRPC-Web support in standard fetch API  |
| k8s pod-to-pod            | gRPC 4317             | Multiplexing, streaming, native health check    |
| Corporate proxy / firewall| HTTP 4318             | Proxies often terminate gRPC streams early      |
| VPN with deep inspection  | HTTP 4318             | gRPC ALPN negotiation blocked by some DPI       |
| High-volume batch         | gRPC + gzip           | Header compression (HPACK), stream reuse        |
| IoT / constrained device  | HTTP/JSON             | Simpler implementation, no Protobuf required    |

---

## 3. Port Conventions

| Port | Protocol | Notes                                                  |
|------|----------|--------------------------------------------------------|
| 4317 | gRPC     | IANA-registered; TLS optional (recommended in prod)   |
| 4318 | HTTP     | IANA-registered; paths `/v1/traces`, `/v1/metrics`, `/v1/logs` |

Do not expose either port to the public internet without TLS and authentication.

---

## 4. Server-Side Tuning (Collector Receiver)

### 4.1 gRPC Receiver

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
        max_recv_msg_size_mib: 16          # default 4 MiB; raise for large trace batches
        max_concurrent_streams: 100        # per-connection stream limit
        read_buffer_size: 524288           # 512 KiB per stream
        keepalive:
          server_parameters:
            time: 30s                      # ping interval when idle
            timeout: 5s                    # wait before closing unresponsive connection
          enforcement_policy:
            min_time: 10s                  # reject client pings faster than this
            permit_without_stream: true
```

### 4.2 HTTP Receiver

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: "0.0.0.0:4318"
        max_request_body_size: 16777216    # 16 MiB; matches gRPC max_recv_msg_size_mib
        include_metadata: true             # forward request headers as resource attributes
        cors:
          allowed_origins: ["https://app.example.com"]
          allowed_headers: ["*"]
```

### 4.3 Memory Limiter (Required)

Place `memory_limiter` first in every pipeline; it back-pressures receivers before OOM.

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128       # refuse new data when usage > limit_mib - spike_limit_mib

  batch:
    send_batch_size: 8192
    send_batch_max_size: 16384
    timeout: 5s

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/tier2]
```

---

## 5. Client-Side Tuning (Exporter)

### 5.1 Compression

gzip is the default and should remain enabled (typically 70–85% payload reduction).
Do not disable it unless your TLS offload device double-compresses.

```yaml
exporters:
  otlp:
    endpoint: "collector:4317"
    compression: gzip          # gzip | zstd | none
    tls:
      insecure: false
      ca_file: /etc/ssl/certs/ca.crt
```

### 5.2 Sending Queue and Retry

```yaml
exporters:
  otlp:
    endpoint: "collector:4317"
    compression: gzip
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 5000
      storage: file_storage/otlp_queue   # persist queue to disk across restarts
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s             # give up after 5 min total
```

**OTLP retry semantics** (opentelemetry.io/docs/specs/otlp/): `RESOURCE_EXHAUSTED` returns
`RetryInfo.retry_delay`; honor it. `UNAVAILABLE` → exponential backoff. `INVALID_ARGUMENT` →
permanent failure, do not retry.

### 5.3 Persistent Queue Extension

```yaml
extensions:
  file_storage/otlp_queue:
    directory: /var/otel/queue
    timeout: 10s
    compaction:
      on_start: true
      rebound_needed_threshold_mib: 100
```

---

## 6. Load Balancing

gRPC uses long-lived HTTP/2 connections, defeating L4 round-robin. Use client-side LB.

**Two-tier pattern** (recommended for tail sampling):

```
[SDK exporters]
      |  gRPC 4317
      v
[Tier-1: loadbalancingexporter]   ← routes by traceID for consistent routing
      |  gRPC 4317
      v
[Tier-2: tailsamplingprocessor]   ← sees full trace before sampling decision
      |  OTLP
      v
[Backend / storage]
```

```yaml
exporters:
  loadbalancing:
    protocol:
      otlp:
        tls:
          insecure: false
        timeout: 5s
    resolver:
      dns:
        hostname: otel-collector-tier2-headless.monitoring.svc.cluster.local
        port: 4317
        interval: 5s
        timeout: 1s
```

For Kubernetes: use a headless Service (`clusterIP: None`) for Tier-2. The `loadbalancingexporter`
resolves DNS and maintains one gRPC connection per pod.

---

## 7. Security: TLS and mTLS

### Collector Receiver (server-side)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
        tls:
          cert_file: /etc/ssl/certs/collector.crt
          key_file: /etc/ssl/private/collector.key
          client_ca_file: /etc/ssl/certs/ca.crt   # enforce mTLS
```

### Exporter (client-side)

```yaml
exporters:
  otlp:
    endpoint: "collector:4317"
    tls:
      ca_file: /etc/ssl/certs/ca.crt
      cert_file: /etc/ssl/certs/client.crt        # mTLS client cert
      key_file: /etc/ssl/private/client.key
```

For Kubernetes workloads, use cert-manager to auto-rotate TLS credentials (90-day TTL).
Use SPIFFE/SPIRE for zero-touch mTLS in multi-cluster scenarios.
