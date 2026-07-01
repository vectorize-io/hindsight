---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
---

# L3-Network Layer Observability

## 1. Scope

**In scope**: IP routing, VPC flow logs, ICMP (reachability and PMTUD), Path MTU Discovery, and; for organizations with their own Autonomous System Number (ASN); BGP/BMP inter-AS routing monitoring.

**Out of scope**: L1 physical (NIC, cable, IPMI) and L2 data-link (MAC, VLAN, STP). Those domains belong to vendor DCIM tooling (Nlyte, Sunbird, Device42). See `../SKILL.md §Out of Scope` for the full boundary declaration.

**SaaS / cloud-native teams**: the primary artifact is cloud-provider VPC flow logs (Section 3). BGP/BMP (Section 6) applies only to ISPs, CDNs, and organizations that announce their own prefixes. SaaS-only teams using only VPC routing can skip Section 6.

---

## 2. OTel Semantic Conventions Relevant to L3

The following attributes from the `network.*` group are **Stable** (semconv 1.27.0) and safe for production instrumentation. Reference: `../standards.md §3` for the full stability tier table.

| Attribute | Type | Example values | Notes |
|-----------|------|----------------|-------|
| `network.type` | string enum | `ipv4`, `ipv6` | Stable; always set on socket-level spans |
| `network.transport` | string enum | `tcp`, `udp`, `quic` | Stable; distinguish protocol at L4 framing |
| `network.local.address` | string | `10.0.1.5` | Stable; local IP of the socket |
| `network.local.port` | int | `443` | Stable; local port |
| `network.peer.address` | string | `192.0.2.10` | Stable; remote IP; treat as PII, see Section 7 |
| `network.peer.port` | int | `8080` | Stable; remote port |
| `network.protocol.name` | string | `http`, `grpc` | Stable; application protocol atop L3/L4 |

Note: `network.connection.*` (e.g., `network.connection.type`, `network.connection.subtype`) are **Development** tier and should not be used in production SLOs. Reference: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>

OTel does not define BGP-specific semconv. BGP monitoring uses a separate pipeline (Section 6) that is explicitly outside the OTLP ecosystem.

---

## 3. Cloud VPC Flow Logs

VPC flow logs capture IP-layer traffic metadata: source address, destination address, bytes, packets, protocol, and firewall action. They are the primary L3 observability artifact for cloud-native teams.

### 3.1 Per-Cloud Collection Pipeline

| Cloud | Flow log source | Recommended OTel receiver | Key config |
|-------|----------------|--------------------------|------------|
| AWS | VPC Flow Logs → CloudWatch Logs or S3 | `cloudwatchlogsreceiver` or `filelogreceiver` (S3 path) | Enable `pkt-srcaddr`, `pkt-dstaddr`, `bytes`, `packets`, `action`, `protocol` fields in flow log format v3+ |
| GCP | VPC Flow Logs → Cloud Logging | `googlecloudpubsubreceiver` (via Pub/Sub export sink) | Configure aggregation interval (5s–15min); lower interval increases cost |
| Azure | NSG Flow Logs v2 → Storage Account → Event Hubs | `kafkareceiver` (Event Hubs uses Kafka protocol) | Enable Traffic Analytics for enriched flow data |

References:
- AWS: <https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html>
- GCP: <https://cloud.google.com/vpc/docs/flow-logs>
- Azure: <https://learn.microsoft.com/azure/network-watcher/nsg-flow-logs-overview>

### 3.2 Key Flow Log Fields

| Field | AWS name | GCP name | Azure name | Notes |
|-------|----------|----------|------------|-------|
| Source IP | `srcaddr` | `src_ip` | `sourceAddress` | PII; see Section 7 |
| Destination IP | `dstaddr` | `dest_ip` | `destinationAddress` | PII; see Section 7 |
| Source port | `srcport` | `src_port` | `sourcePort` | |
| Destination port | `dstport` | `dest_port` | `destinationPort` | |
| Protocol | `protocol` | `protocol` | `protocol` | IANA protocol number |
| Bytes | `bytes` | `bytes_sent` | `bytesForwardedDenied` + `bytesForwardedAllowed` | Used for egress cost attribution |
| Packets | `packets` | `packets_sent` | `packetsForwardedAllowed` | |
| Firewall action | `action` | N/A (use firewall logs) | `trafficType` | `ACCEPT` / `REJECT`; audit signal |
| Start/end time | `start`, `end` | `start_time`, `end_time` | `startTime`, `endTime` | |

### 3.3 Privacy Note

`srcaddr` and `dstaddr` (and their equivalents) are IP addresses. Under GDPR Article 4(1) and PIPA, IP addresses linked to natural persons are personal data. Before long-term retention:

- Truncate the last octet for IPv4 (e.g., `203.0.113.0/24` prefix preservation).
- Hash with a rotating salt (SHA-256 minimum) for per-flow pseudonymization.
- Do not store raw IP addresses in shared observability backends unless the tenant has explicit lawful basis.

Cross-reference: `../signals/privacy.md §IP addresses` for the full masking decision tree.

---

## 4. PMTUD: Path MTU Discovery

### 4.1 Mechanism

Path MTU Discovery (PMTUD, RFC 1191 for IPv4; RFC 8201 for IPv6) allows a sender to discover the maximum transmission unit supported along an end-to-end path without fragmentation. RFC 8899 defines Datagram PLPMTUD (DPLPMTUD), a transport-level variant used by QUIC and SCTP that does not rely on ICMP.

IPv4 PMTUD relies on ICMP Type 3 Code 4 ("Fragmentation Needed, DF set") messages returned by intermediate routers when a packet is too large and the DF (Don't Fragment) bit is set. The sender reduces its segment size upon receiving this ICMP message.

### 4.2 PMTUD Black Hole

**Root cause**: firewall rules that block ICMP Type 3 Code 4 prevent the sender from receiving the path MTU signal. The sender continues transmitting oversized segments that are silently dropped.

**Symptom pattern**:
- Small payloads (< 576 bytes) succeed; large payloads (> path MTU) stall or fail.
- TCP connection establishes (SYN/SYN-ACK are small) but bulk data transfer hangs.
- HTTP responses for large objects time out; health checks (small packets) pass.

**Detection in flow logs**: look for sessions with normal packet counts but abnormally low byte counts per session; correlate with application-layer timeout logs.

**Remediation options**:
1. **MSS clamping** (TCP Maximum Segment Size): configure at the firewall or router to clamp MSS to `interface_MTU - 40` (IPv4) or `interface_MTU - 60` (IPv6). This is the preferred fix for VPN and tunnel endpoints.
2. **Explicit MTU pinning**: set the outgoing interface MTU on the host to a conservative value (e.g., 1452 for PPPoE, 1422 for GRE over IPSec).
3. **ICMP passthrough**: allow ICMP Type 3 Code 4 through security groups and firewalls. Verify with: `ping -M do -s 1472 <destination>`.

Cross-reference: `../transport/udp-statsd-mtu.md` for related UDP MTU thresholds (1472 IPv4, 1452 IPv6, 8192 UDS) that interact with the same path MTU constraints.

---

## 5. ICMP Observability

### 5.1 Active Monitoring Tools

| Tool | Signal type | Use case |
|------|------------|---------|
| Blackbox Exporter (Prometheus) | ICMP echo probe | Synthetic availability from a vantage point |
| SmokePing | ICMP RTT time series | Long-term latency trend and packet-loss distribution |
| `ping` (scripted) | ICMP echo RTT | Ad hoc reachability during incident investigation |
| `traceroute` / `mtr` | ICMP TTL-exceeded path | Hop-by-hop latency and packet-loss localization |

### 5.2 Cloud Provider ICMP Constraints

| Cloud | ICMP behavior | Notes |
|-------|--------------|-------|
| AWS | ICMP echo blocked by default on security groups | Explicitly allow inbound ICMP Type 8 (echo request) for Blackbox Exporter probes to work |
| GCP | ICMP echo allowed by default in default VPC firewall rules | Review custom VPC firewall rules |
| Azure | ICMP echo allowed; NSG allows by default for same-VNet | Blocked at NSG if explicit deny rule added |

ICMP Rate limits: all major clouds rate-limit ICMP responses from managed infrastructure (e.g., NAT gateways, load balancers). Do not interpret ICMP loss at these endpoints as a network fault; correlate with TCP-level application metrics.

### 5.3 ICMP in OTel

OTel does not define ICMP-specific semconv. Use the Blackbox Exporter's native Prometheus metrics (`probe_success`, `probe_duration_seconds`, `probe_icmp_reply_hop_limit`) and scrape them via the OTel Collector `prometheusreceiver`. Tag probes with `network.peer.address` for correlation.

---

## 6. BGP / Inter-AS Routing (Advanced Subsection)

**Scope boundary**: this subsection applies to **ISPs, CDNs, and organizations that own and operate an Autonomous System (ASN)** and announce their own IP prefixes via BGP. SaaS-only teams that rely entirely on cloud-provider VPC routing can skip this subsection.

OTel does not define BGP semconv. BGP monitoring is a parallel pipeline ecosystem documented here for completeness. It does not use OTLP as the transport.

### 6.1 BGP Monitoring Protocol (BMP)

BMP (RFC 7854, <https://www.rfc-editor.org/rfc/rfc7854>) is the standard protocol for exporting real-time BGP session state, RIB (Routing Information Base) snapshots, and route change events from a BGP speaker to a monitoring collector.

**BMP exporters** (BGP speakers that support BMP):

| Platform | BMP support |
|----------|------------|
| FRRouting (OSS) | Native BMP export (`bmp` config stanza) |
| Cisco IOS-XR | Native BMP (`router bgp ... bmp server`) |
| Juniper Junos | Native BMP (`protocols bgp monitoring-protocol bmp`) |
| BIRD 2.x | BMP via `bmp` protocol block |

### 6.2 BMP Collectors and Aggregators

| Tool | Role | Notes |
|------|------|-------|
| OpenBMP | Open-source BMP collector + PostgreSQL/ClickHouse backend | Reference implementation; supports MOAS detection |
| pmacct | Multi-purpose network accounting; BMP + BGP support | Used at large ISPs; outputs to Kafka, InfluxDB, Elasticsearch |
| SNAS (Streaming Network Analytics System) | BMP → Kafka pipeline; CAIDA-affiliated | Suitable for research and large-scale ISP deployments |
| GoBMP | Lightweight Go BMP collector | Good for parsing and forwarding to custom pipelines |

### 6.3 Recommended Pipeline

```
BGP Speaker (FRR / IOS-XR / Junos)
    |  BMP (RFC 7854, TCP port 11019)
    v
BMP Collector (OpenBMP / pmacct / GoBMP)
    |  Kafka topic: bgp.updates
    v
Stream Processor (Kafka Streams / Flink)
    |  enrichment: RPKI ROA lookup, MOAS detection
    v
ClickHouse (columnar storage for BGP RIB history)
    |
    v
Grafana (BGP topology dashboards, hijack alerts)
```

This pipeline is **not OTel**. It operates independently of OTLP. The two pipelines (OTel metrics/logs/traces and BGP BMP) share only the storage and visualization layer (ClickHouse + Grafana).

### 6.4 Security Observability: Hijack and Route Leak Detection

#### BGP Hijack: MOAS Detection

A BGP hijack occurs when a rogue AS announces a prefix it does not legitimately own, diverting traffic. Multiple Origin AS (MOAS) detection identifies prefixes announced by more than one origin AS simultaneously.

Detection: compare real-time BMP route updates against a known-good origin AS database (your own RPKI ROAs + IRR records). Alert on any observed origin AS that does not match the expected set.

Tools: OpenBMP includes MOAS detection; BGPalerter (<https://github.com/nttgin/BGPalerter>) provides real-time alerting on MOAS, route leaks, and more-specific prefix announcements.

#### Route Leaks

A route leak occurs when a route received from one BGP neighbor is re-announced to another neighbor in violation of routing policy (e.g., customer routes leaked to a peer). Detection requires policy-aware analysis of AS_PATH attributes.

Tool: ARTEMIS (<https://github.com/FORTH-ICS-INSPIRE/artemis>) provides automated detection and mitigation orchestration for both hijacks and route leaks.

#### RPKI-ROV (Route Origin Validation)

RPKI (Resource Public Key Infrastructure) and ROA (Route Origin Authorization) records bind a prefix to an authorized origin AS with a cryptographic signature. ROV (Route Origin Validation) is the process of checking incoming BGP announcements against the RPKI trust anchor.

**Verify your own prefixes**: ensure all prefixes you announce have valid ROA records in your RIR (ARIN, RIPE NCC, APNIC, LACNIC, AFRINIC). An announced prefix without an ROA is marked "Not Found" by downstream validators; not "Invalid", but not cryptographically anchored either.

**Enforce ROV on your router**: configure BGP to drop or de-prefer routes with RPKI Invalid status. Reference NIST SP 800-189 (<https://csrc.nist.gov/publications/detail/sp/800-189/final>) for deployment guidance.

**Tools**: Cloudflare Radar RPKI dashboard (<https://radar.cloudflare.com/routing/rpki>) provides global visibility into ROA coverage and RPKI Invalid route counts by ASN.

### 6.5 Public BGP Feeds for Baselining

| Feed | Provider | Use |
|------|----------|-----|
| RIPE RIS (Routing Information Service) | RIPE NCC | Global BGP route collector data; MRT format |
| RouteViews | University of Oregon / CAIDA | Historical BGP table snapshots; MRT format |
| BGPStream | CAIDA | Streaming API over RIPE RIS + RouteViews; Python/C library |
| Cloudflare Radar | Cloudflare | Public BGP anomaly detection; hijack and outage notifications |

Use these feeds to baseline what your prefixes look like to the global routing table and to cross-check your own RPKI ROA coverage.

---

## 7. Matrix Cross-Reference (L3 Row)

The L3 row of `../matrix.md` covers 28 cells (4 boundaries × 7 signals). The following summarizes the coverage status and primary artifacts:

| Boundary | metrics | logs | traces | profiles | cost | audit | privacy |
|----------|---------|------|--------|----------|------|-------|---------|
| multi-tenant | PASS per-tenant egress byte/packet counters from VPC flow logs | PASS VPC flow stream tagged by tenant CIDR | PARTIAL trace-ID egress tagging only; no native L3 trace context | N/A | PARTIAL egress byte attribution as cost proxy | PASS VPC flow audit trail tagged by tenant | PARTIAL IP addresses are PII (GDPR/PIPA); mask before retention |
| cross-application | PASS inter-VPC peering flow metrics | PASS VPC flow logs across peering / transit gateway | PARTIAL socket 5-tuple correlation to L7 spans only | N/A | PARTIAL cross-VPC egress cost proxy | PASS inter-VPC flow audit for SOC2 network controls | PARTIAL source/destination IPs crossing app boundary are PII candidates |
| slo | N/A; SLO belongs at L7 | N/A | N/A | N/A | N/A | N/A | N/A |
| release | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Notes:
- **SLO cells**: all N/A. SLO error budgets are computed from L7 application metrics. If an L3 event (BGP hijack, PMTUD black hole) causes an SLO burn, the causal chain surfaces as an L7 error-rate spike first. Investigate via `../incident-forensics.md` 6-dimension localization.
- **Audit cells**: VPC flow logs with `ACCEPT`/`REJECT` actions satisfy SOC2 CC6.6 (network access control) audit evidence requirements. Store in WORM-compatible storage (S3 Object Lock, GCS retention policy, Azure immutable blob) per `../signals/audit.md`.
- **Privacy cells**: cross-reference `../signals/privacy.md §IP addresses` for the masking decision tree. The threshold for masking versus pseudonymization depends on the data's correlation potential with other identifiers.

---

## 8. Anti-Patterns

The following are candidates for `../anti-patterns.md §Section H` (L3-specific):

| Anti-pattern | Risk | Remediation |
|-------------|------|------------|
| Own-ASN BGP hijack left unmonitored | Prefix hijack undetected for hours or days; traffic silently diverted | Deploy BGPalerter or ARTEMIS with MOAS detection; subscribe to Cloudflare Radar alerts for your prefixes |
| Raw IP addresses retained in logs without redaction | GDPR Article 5(1)(c) data minimization violation; 4% global turnover fine risk | Truncate last octet on ingest; use rotating-salt SHA-256 for pseudonymization; see `../signals/privacy.md` |
| RPKI-ROV not configured on advertised prefixes | Rogue origin AS can announce your prefixes without RPKI Invalid signal to downstream validators | Create ROA records at your RIR; enable ROV enforcement on border routers |
| PMTUD black hole left uncorrected | Large TCP transfers stall; health checks pass (small packets), masking the problem | Enable MSS clamping at VPN/tunnel endpoints; allow ICMP Type 3 Code 4 through firewalls |

---

## 9. References


Internal cross-references:

- `../standards.md §3`: OTel semconv stability tiers
- `../matrix.md §Layer: L3-network`; full 28-cell coverage map
- `../signals/privacy.md §IP addresses`: IP address masking decision tree
- `../signals/audit.md`: WORM storage and SOC2 audit trail requirements
- `../signals/cost.md §egress attribution`: L3 egress byte cost proxy
- `../transport/udp-statsd-mtu.md`: UDP MTU thresholds (1472/1452/1432/8192/16K)
- `../incident-forensics.md`: 6-dimension localization when L3 event causes L7 SLO burn

## References

- RFC 7854: BGP Monitoring Protocol (BMP): <https://www.rfc-editor.org/rfc/rfc7854>
- RFC 1191: Path MTU Discovery (IPv4): <https://www.rfc-editor.org/rfc/rfc1191>
- RFC 8201: Path MTU Discovery for IPv6: <https://www.rfc-editor.org/rfc/rfc8201>
- RFC 8899: Datagram PLPMTUD (used by QUIC/SCTP): <https://www.rfc-editor.org/rfc/rfc8899>
- RFC 6811: BGP Prefix Origin Validation (RPKI-ROV): <https://www.rfc-editor.org/rfc/rfc6811>
- IANA ICMP Type registry: <https://www.iana.org/assignments/icmp-parameters/icmp-parameters.xhtml>
- IANA protocol numbers: <https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml>
- NIST SP 800-189 (RPKI deployment): <https://csrc.nist.gov/publications/detail/sp/800-189/final>
- Cloudflare Radar (public BGP + RPKI): <https://radar.cloudflare.com/routing>
- OTel network semconv registry: <https://opentelemetry.io/docs/specs/semconv/attributes-registry/network/>
- AWS VPC Flow Logs: <https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html>
- GCP VPC Flow Logs: <https://cloud.google.com/vpc/docs/flow-logs>
- Azure NSG Flow Logs: <https://learn.microsoft.com/azure/network-watcher/nsg-flow-logs-overview>
