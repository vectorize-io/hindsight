---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
notes:
  - "Fluent Bit: v3.x (CNCF Graduated); OTel Collector: v0.100+"
---

# Logs Signal Reference

## 1. Scope

Logs are the "L" in MELT+P. Covers: OTel LogRecord data model, structured logging, events-as-logs, systemd journal, and pipeline options.

**Normative base:** `../standards.md`. **Deprecated:** Fluentd (CNCF 2025-10); use Fluent Bit or OTel Collector.

---

## 2. OTel LogRecord Data Model

Source: <https://opentelemetry.io/docs/specs/otel/logs/data-model/>

| Field | Type | Description |
|---|---|---|
| `Timestamp` | uint64 (ns) | When the event occurred; 0 if unknown |
| `ObservedTimestamp` | uint64 (ns) | When the collector observed the record; always set |
| `SeverityText` | string | e.g. `"ERROR"`; original string from source |
| `SeverityNumber` | int (1–24) | Normalized OTel severity |
| `Body` | any | String, map, or primitive |
| `Attributes` | map[string]any | Semconv-stable keys preferred |
| `TraceId` | 16-byte hex | W3C trace ID; empty if not in a trace |
| `SpanId` | 8-byte hex | W3C span ID; empty if not in a span |
| `Resource` | map[string]any | `service.name`, `host.name`, etc. |

**Severity scale:** TRACE 1–4 · DEBUG 5–8 · INFO 9–12 · WARN 13–16 · ERROR 17–20 · FATAL 21–24

**Export:** OTLP gRPC `:4317` or HTTP `:4318/v1/logs`. Configure the SDK's `LogRecordExporter` to point at the OTel Collector.

---

## 3. Structured Logging

JSON over text: queryable, typed, machine-readable. No grok patterns.

```json
{
  "timestamp": "2026-04-21T09:15:32.847Z",
  "level": "ERROR",
  "message": "payment gateway timeout after 3 retries",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "service.name": "checkout-api",
  "service.version": "1.4.2",
  "deployment.environment": "production",
  "error.type": "GatewayTimeoutError",
  "retry.count": 3
}
```

**Required fields on every log record:**

| Field | Format | Why |
|---|---|---|
| `timestamp` | ISO 8601 UTC (`Z`) | Clock-ordered queries; cross-ref `../standards.md §Clock Discipline` |
| `level` | OTel SeverityText | Filtering and sampling |
| `message` | human-readable string | On-call readability |
| `trace_id` + `span_id` | 32/16-char hex | Join with traces; critical for incident forensics |
| `service.name`, `service.version` | semconv stable | Release comparison |
| `deployment.environment` | `production` / `staging` | Noise isolation |

**Metric correlation:** use `service.name`, `cloud.region`, `k8s.namespace.name` as shared labels across logs and metrics. Cross-ref `../incident-forensics.md §MRA` for the full mandatory resource attribute list.

---

## 4. Events as LogRecords

OTel spec v1+ folded events into LogRecord: set `event.name` + `event.*` attributes. Source: <https://opentelemetry.io/docs/specs/otel/logs/event-api/>

```json
{
  "timestamp": "2026-04-21T09:15:33.001Z",
  "level": "INFO",
  "message": "order placed",
  "event.name": "order.placed",
  "order.id": "ord_8f3k2",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "service.name": "order-service",
  "deployment.environment": "production"
}
```

Use cases: `feature_flag.evaluated`, `deployment.completed`, `order.placed`, `user.signup`.

**Distinction from metrics:** events are discrete occurrences; metrics are aggregated rates.  
**Distinction from logs:** events have a declared `event.name` and attribute contract; logs are often free-form.

---

## 5. Systemd Journal Integration

On most Linux cloud VMs, systemd-journald is the primary log sink. Map via `journaldreceiver` (stable).

```yaml
# otelcol-config.yaml
receivers:
  journald:
    directory: /var/log/journal
    units: [kubelet.service, containerd.service]
    priority: info
processors:
  resource:
    attributes:
      - { key: host.name, from_attribute: _HOSTNAME, action: upsert }
      - { key: service.name, from_attribute: _SYSTEMD_UNIT, action: upsert }
exporters:
  otlp:
    endpoint: "otel-gateway.observability.svc:4317"
service:
  pipelines:
    logs:
      receivers: [journald]
      processors: [resource]
      exporters: [otlp]
```

Source: <https://github.com/open-telemetry/opentelemetry-collector-contrib/receiver/journaldreceiver>

**journald field mapping:** `MESSAGE` → `Body` · `PRIORITY` → `SeverityNumber` · `_HOSTNAME` → `Resource["host.name"]` · `_SYSTEMD_UNIT` → `Resource["service.name"]` · `_PID` → `Attributes["process.pid"]`

**Kubernetes log sources:**
- kubelet → container stdout/stderr → runtime log files → `filelogreceiver` (`/var/log/pods/*/*/*.log`)
- Kubernetes events → `k8seventsreceiver` (OTel Collector contrib)

---

## 6. Collector and Agent Options

Cross-ref `../vendor-categories.md §Log Pipeline Collection` (category h).

| Tool | Runtime | RAM | CNCF | Notes |
|---|---|---|---|---|
| **Fluent Bit** | C/Rust | 5–15 MB | Graduated | Edge DaemonSet; native OTLP output |
| **OTel Collector** | Go | 30–100 MB | Incubating | Unified MELT; gateway aggregation |
| **Vector** | Rust | 20–60 MB | Datadog OSS | Log + metric pipeline with transforms |
| **Cribl Stream** | Proprietary | 100+ MB | Commercial | Advanced routing |
| ~~Fluentd~~ | Ruby | 100+ MB | **Deprecated** | CNCF 2025-10 |

**2026 best practice:** Fluent Bit DaemonSet (edge) → OTLP → OTel Collector gateway (enrichment, PII redaction, routing) → backends.

---

## 7. Trace ID Injection Rules

**EVERY log line from an instrumented service MUST carry `trace_id` + `span_id`.**

Without them, log-trace join fails during incident forensics, adding 15–30 min to MTTR.

**Injection mechanisms:**

| Language | Library | Method |
|---|---|---|
| Python | `structlog` | Context processor extracting OTel span context |
| Java | Logback / Log4j2 | `OpenTelemetryAppender` MDC integration |
| Node.js | `pino` | `pino-opentelemetry-transport` |
| Go | `zap` / `slog` | Manual `trace.SpanFromContext` extraction |

**Verification:** assert in integration tests that the `trace_id` in a log record matches bytes 1–16 of the inbound `traceparent` header. Cross-ref `../standards.md §W3C Trace Context`.

---

## 8. Log Severity and Sampling

| Severity | Production policy |
|---|---|
| TRACE / DEBUG | 1% sample |
| INFO | 10–20% for high-volume services |
| WARN / ERROR / FATAL | 100% retained |

Apply sampling at the collection tier (Fluent Bit throttle filter or OTel Collector `probabilistic_sampler`), not at the application tier.

**Tail sampling for logs:** retain 100% of records whose `trace_id` is associated with an error trace. Cross-ref `../transport/sampling-recipes.md` for the OTel Collector tail-sampling processor config.

---

## 9. Log Retention and Compliance

| Category | Retention | Storage |
|---|---|---|
| Operational (INFO/DEBUG) | 7–30 days | Hot |
| WARN / ERROR | 90 days | Hot or warm |
| Audit logs | 7 years | WORM object storage |

**PII redaction** MUST occur at ingestion (Fluent Bit or OTel Collector `redaction` processor), not at storage time. Cross-ref `privacy.md §PII redaction pipeline`.

Cross-ref `../meta-observability.md §Retention Matrix` for unified policy across all signals.

---

## 10. OS-Level Log Sources

| Source | Collection | Notes |
|---|---|---|
| systemd-journald | `journaldreceiver` | Primary on Linux; all systemd units |
| syslog (legacy) | `syslogreceiver` | Facilities: kern, daemon, auth |
| kernel dmesg | journald (forwarded automatically) | OOM kills, NIC errors |
| Linux auditd | `filelogreceiver` on `/var/log/audit/audit.log` | Cross-ref `audit.md §auditd` |

---

## 11. Backends

Cross-ref `../vendor-categories.md` for category (e) SIEM / Enterprise Logs and (a) OSS Full-Stack.

| Backend | Query model | Notes |
|---|---|---|
| **Loki** (Grafana Labs) | LogQL; label-indexed | Chunks in object storage; cloud-native OSS |
| **Elasticsearch** | Lucene / EQL | Full-text search; large existing footprints |
| **OpenSearch** (AWS fork) | Lucene / SQL | No Elastic licensing |
| **ClickHouse** | SQL; columnar | Rising default for high-volume structured log analytics; used by SigNoz |

---

## 12. Matrix Coverage: Logs Column

Cells from `../matrix.md` owned by this file:

| Layer | Boundary | Status | Artifact |
|---|---|---|---|
| L3-network | multi-tenant | PASS | VPC flow log stream tagged by tenant CIDR |
| L4-transport | multi-tenant | PASS | TCP connection events per tenant socket namespace |
| mesh | multi-tenant | PASS | Envoy access logs with baggage-derived tenant tag |
| mesh | cross-application | PASS | Envoy access logs; `trace_id` correlation |
| mesh | slo | PARTIAL | Burn-rate event source; metric-based SLI preferred |
| mesh | release | PASS | Deployment event with `service.version` tag |
| L7-application | multi-tenant | PASS | `tenant.id` on every record; OTel Collector `routing_connector` |
| L7-application | cross-application | PASS | `trace_id` + `span_id` enables log-trace join across services |
| L7-application | slo | PARTIAL | Log-based SLI valid fallback; metric-based preferred |
| L7-application | release | PASS | Deployment SHA, version, rollout strategy as structured event |

---

## 13. Anti-patterns

Append candidates for `../anti-patterns.md §Logs`:

| ID | Anti-pattern | Fix |
|---|---|---|
| A-L1 | Free-form log text without JSON | Adopt structured JSON; no grok |
| A-L2 | `trace_id` missing in logs | Inject via SDK hook (Section 7); enforce in CI |
| A-L3 | `user.email` in log body | PII violation; redact at collection tier; cross-ref `privacy.md` |
| A-L4 | Unique message IDs as log labels | Cardinality explosion in Loki; use attributes, not labels |
| A-L5 | Fluentd as new deployment in 2026+ | Deprecated; use Fluent Bit or OTel Collector |

---

## 14. References

1. OTel log data model: <https://opentelemetry.io/docs/specs/otel/logs/data-model/>
2. OTel event API: <https://opentelemetry.io/docs/specs/otel/logs/event-api/>
3. journaldreceiver: <https://github.com/open-telemetry/opentelemetry-collector-contrib/receiver/journaldreceiver>
4. CNCF Fluentd migration guide: <https://cncf.io/blog/2025/10/01/fluentd-to-fluent-bit-migration-guide>
5. Fluent Bit docs: <https://docs.fluentbit.io>
6. `../standards.md` · `../matrix.md` · `../incident-forensics.md §MRA` · `../meta-observability.md §Retention Matrix` · `../transport/sampling-recipes.md` · `../vendor-categories.md`
