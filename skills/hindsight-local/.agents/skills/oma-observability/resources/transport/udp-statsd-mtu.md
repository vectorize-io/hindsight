---
otel_spec: "1.x (stable API/SDK)"
---

# UDP StatsD MTU and Fragmentation Guide

## 1. Scope

This document covers StatsD metric payloads transported over UDP or Unix Domain Socket (UDS)
into an OpenTelemetry Collector `statsdreceiver` or a DogStatsD agent. It does not address
TCP or HTTP transports, or metrics emitted directly via OTLP.

Sources:
- OTel statsdreceiver: github.com/open-telemetry/opentelemetry-collector-contrib/receiver/statsdreceiver
- DogStatsD high-throughput: docs.datadoghq.com/developers/dogstatsd/high_throughput/
- Prometheus statsd_exporter issue #35 (buffer sizing)

---

## 2. Why Fragmentation Matters for UDP

UDP is connectionless and provides no retransmission. When an IP datagram exceeds the path MTU:

- **No retransmission**: if any IP fragment is lost in transit, the kernel discards the entire
  reassembled datagram silently. StatsD has no acknowledgement mechanism, so the loss is invisible.
- **Middlebox drops**: NAT gateways, stateful firewalls, and VPN concentrators frequently drop
  IP fragments entirely, or track only the first fragment for NAT state and discard the rest.
- **Reassembly CPU overhead**: fragment reassembly at the receiver incurs kernel memory allocation
  and timer management. Under high packet rate this degrades throughput measurably.

**Rule**: keep each datagram within the path MTU so it is transmitted as a single unfragmented packet.

---

## 3. Optimal Datagram Size Table

| Network path                     | Effective MTU | Max UDP payload | Notes                                      |
|----------------------------------|---------------|-----------------|---------------------------------------------|
| External network, IPv4 Ethernet  | 1500 B        | **1472 B**      | 1500 − 20 (IP) − 8 (UDP)                   |
| External network, IPv6 Ethernet  | 1500 B        | **1452 B**      | 1500 − 40 (IPv6) − 8 (UDP)                 |
| VPN / PPPoE encapsulation        | ~1460 B       | **1432 B**      | Conservative; accounts for tunnel overhead  |
| Same-host loopback (Linux/macOS) | ~65535 B      | **~16384 B**    | Kernel loopback; practical limit ~16 K      |
| Unix Domain Socket (`unixgram`)  | N/A           | **8192 B**      | Recommended; avoids kernel socket buffer pressure |

> IPv6 minimum MTU is 1280 B (RFC 8200); on links with lower MTU, fragmentation is performed
> by the source host only; not routers; making drops more likely. Always test your actual path.

---

## 4. Verification

### 4.1 PMTUD Probe (Linux / macOS)

Find the largest datagram that reaches a destination without fragmentation:

```bash
# Linux — DF bit set, vary -s until you find the boundary
ping -M do -s 1472 -c 3 <collector-host>

# macOS equivalent
ping -D -s 1472 -c 3 <collector-host>
```

Decrease `-s` in steps of 10 until you stop seeing "Frag needed" / "Message too long" errors.
The largest passing value is your PMTU; set `max_packet_size` to that value minus 28 (IPv4)
or 48 (IPv6) to leave room for IP and UDP headers.

### 4.2 Wireshark Fragment Inspection

Capture on the collector interface and filter for IP fragments:

```
ip.flags.mf == 1
```

The **More Fragments (MF)** bit set on any packet indicates fragmentation is occurring.
A fragment offset > 0 identifies continuation fragments. If you see these in production
traffic, your client is sending oversized datagrams.

For UDS traffic, capture on a loopback or use `socat` to proxy and inspect:

```bash
socat -v UNIX-RECV:/tmp/statsd.sock UDP:127.0.0.1:8126
```

---

## 5. Client-Side Batching

Most StatsD client libraries default to sending **one metric per datagram**, which is wasteful:

- A single counter line (`my.counter:1|c`) is ~20 bytes; one UDP send per metric at 1 M metrics/min
  generates 16 K packets/sec of syscall overhead.
- Enable multi-metric batching in your client library.

| Client library     | Batching config                          | Recommended buffer   |
|--------------------|------------------------------------------|----------------------|
| dogstatsd-go       | `WithMaxMessagesPerPayload(N)`           | 1472 (external UDP)  |
| statsd (npm)       | `maxBufferSize: 1432`                    | 1432 (VPN/default)   |
| statsd-client (py) | `maxudpsize=1432`                        | 1432                 |
| Any UDS client     | Set buffer ≤ 8192                        | 8192                 |

Rules:
- For external UDP: keep buffered payload at or below the PMTU value found in section 4.1.
- For UDS (same-host): buffer up to 8192 bytes per send; larger values risk ENOBUFS under load.
- Never exceed 65507 bytes (IPv4 UDP maximum); the kernel will return EMSGSIZE.

---

## 6. OTel StatsD Receiver Configuration

Default UDP port is 8125. Switch to UDS for same-host agents to eliminate network stack overhead.

```yaml
receivers:
  statsd:
    endpoint: "0.0.0.0:8125"      # UDP; change to unixgram path for UDS
    # endpoint: "/var/run/statsd.sock"
    transport: udp                  # udp | unixgram
    aggregation_interval: 60s       # flush interval to next processor
    enable_metric_type: true        # attach metric type as attribute
    is_monotonic_counter: false     # set true for always-increasing counters
    timer_histogram_mapping:
      - statsd_type: "timing"
        observer_type: "histogram"
      - statsd_type: "histogram"
        observer_type: "histogram"

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128
  batch:
    send_batch_size: 8192
    timeout: 10s

exporters:
  otlp:
    endpoint: "otel-collector-tier2:4317"
    compression: gzip

service:
  pipelines:
    metrics:
      receivers: [statsd]
      processors: [memory_limiter, batch]
      exporters: [otlp]
```

**Bridge pattern**: `StatsD clients` → UDP/UDS → `OTel Collector statsdreceiver` → OTLP gRPC →
downstream Collector or backend. This decouples metric ingestion from processing and allows the
statsdreceiver to be scaled or replaced without changing client configuration.

For UDS on Linux, set socket permissions to allow the application user:

```bash
chmod 0660 /var/run/statsd.sock
chown root:app /var/run/statsd.sock
```

Set `SO_RCVBUF` on the collector socket to handle burst traffic (Linux default is typically 208 KB;
increase to 8–25 MB for high-throughput agents via `net.core.rmem_max`):

```bash
sysctl -w net.core.rmem_max=26214400
```
