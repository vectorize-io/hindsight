# oma-observability Checklist

> Consolidated verification checklist. Run before shipping observability changes to production.
> Each item: action; criterion; priority.
> Source files live under `resources/` from the skill root.

## Priority legend

- **P0**: block ship; do not merge or deploy without this
- **P1**: must-before-prod; complete before promoting to production
- **P2**: this-sprint; schedule and close within the current sprint

---

## Section 1: Setup validation (pre-commit / pre-merge)

- [ ] Pin OTel spec version and semconv version in `standards.md` header; must match deployed SDK version (P0)
- [ ] Define `service.name`, `service.namespace`, `service.version` on every instrumented service OTel Resource; all three attributes present and non-empty (P0)
- [ ] Define `deployment.environment`, `cloud.provider`, `cloud.region`, `cloud.availability_zone` on every OTel Resource; required for region-dimension pivot in incident forensics (P0)
- [ ] Define `host.id`, `k8s.pod.name`, `k8s.node.name`, `k8s.cluster.name`, `container.id` on every Kubernetes workload Resource; required for server-dimension pivot (P0)
- [ ] Configure W3C Trace Context (`traceparent`) as default propagator; every outbound HTTP/gRPC call must forward the header; confirm by checking `trace_id` consistency across service log streams (P0)
- [ ] Verify W3C Baggage carries only allowed keys; `tenant.id`, feature flag state, deployment SHA, region hint are allowed; user email, session tokens, and credentials are prohibited per `standards.md §W3C Baggage` (P0)
- [ ] Strip or validate inbound baggage at ingress gateway; external callers must not inject arbitrary baggage into internal services (P0)
- [ ] Use only stable semconv groups (`service.*`, `host.*`, `cloud.*`, `k8s.*`, `http.*`, `db.*`, `network.*` core, `error.*`) for production SLO inputs; Development-tier attributes (`tls.*`, `network.connection.*`) must not feed SLOs (P0)
- [ ] Configure `memory_limiter` processor as the first processor in every Collector pipeline; `limit_percentage: 75`, `spike_limit_percentage: 20` (P0)
- [ ] Place `memory_limiter` before `batch` in pipeline order; prevents OOM crash under burst (P0)
- [ ] Validate that OTLP transport choice matches topology; gRPC port 4317 for pod-to-pod; HTTP port 4318 for browser SDKs and proxy-traversal paths (P1)
- [ ] Verify UDP StatsD datagrams fit within path MTU; max 1472 B on standard Ethernet IPv4; use Unix Domain Socket (`unixgram`) for same-host paths (P1)
- [ ] Version all observability artifacts in git; dashboards, alert rules, SLO definitions, and Collector configs must be applied via CI/CD, not edited in the UI (P1)

---

## Section 2: Pre-production readiness

- [ ] Confirm matrix coverage gaps are identified and acknowledged; every uncovered cell in `matrix.md` has an explicit N/A rationale or a remediation plan (P1)
- [ ] Set cardinality budget per service; alert at 80% of budget (default 5 000 series/service); configure `count({job="<service>"}) > 4000` alert (P1)
- [ ] Allow-list metric attributes explicitly in the OTel SDK View; use `attribute_keys` to prevent unbounded label sets; never let raw `http.url`, `user.id`, `request.id`, `trace.id`, or `error.message` appear as metric labels (P0)
- [ ] Normalize high-cardinality route labels; apply `transform` processor to replace numeric path segments with `/_` (e.g., `/users/42` → `/users/_`) (P1)
- [ ] Define per-tenant sampling policy when multi-tenant; enterprise tenants may require 100% error retention + dedicated pipeline; free-tier tenants use probabilistic baseline (P1)
- [ ] Configure four-tier tenant isolation strategy; select Tier (soft label / routing / dedicated-collector / dedicated-backend) per tenant compliance obligation and document the decision (P1)
- [ ] Propagate `tenant.id` via W3C Baggage on all spans, logs, and metrics in multi-tenant services; use the canonical key `tenant.id`; do not use `customer_id`, `org_id`, or `account_id` as synonyms (P1)
- [ ] Verify `tenant.region` drives data residency routing; EU and KR tenants must have telemetry routed to compliant regional Collector pipelines before reaching a cross-region backend (P0)
- [ ] Configure two-tier Collector topology for Kubernetes; DaemonSet agent (hostmetrics, filelog, kubeletstats, k8sattributes) forwarding to Deployment gateway (batch, tail_sampling, exporters) (P1)
- [ ] Deploy loadbalancing exporter upstream of tail_sampling processor; consistent hash by `trace_id` ensures complete traces arrive at the same gateway replica (P0)
- [ ] Confirm `exception.type`, `exception.message`, `exception.stacktrace`, `code.function`, `code.filepath`, `code.lineno` are populated on every ERROR span; use `span.recordException(e)` for atomicity (P0)
- [ ] Confirm IP address logging is masked or hashed before long-term retention; IP addresses are personal data under GDPR Art. 4(1) and PIPA; apply prefix truncation or HMAC+salt at pipeline ingress (P0)

---

## Section 3: Production operations

- [ ] Scrape OTel Collector self-metrics (`otelcol_*`) from a separate Prometheus instance or second Collector; Collector failures must not destroy their own observability (P0)
- [ ] Alert on pipeline delivery ratio dropping below 99%; `(rate(otelcol_exporter_sent_spans[5m]) / rate(otelcol_receiver_accepted_spans[5m])) < 0.99` sustained for 5 minutes (P0)
- [ ] Alert on exporter send failures above 1%; `rate(otelcol_exporter_send_failed_spans[5m]) / rate(otelcol_exporter_sent_spans[5m]) > 0.01` for 5 minutes (P0)
- [ ] Alert on `otelcol_receiver_refused_spans > 0` sustained for 2 minutes; indicates queue full or parse failures (P0)
- [ ] Alert on Collector heap usage exceeding 75% of container memory limit; `otelcol_process_runtime_heap_alloc_bytes > 0.75 * container_limit` (P0)
- [ ] Configure NTP or chrony on all host VMs and Kubernetes nodes; clock drift must stay below 100 ms for reliable trace waterfall ordering (P0)
- [ ] Alert on node clock drift exceeding 100 ms; `node_clock_drift_ms > 100` for 5 minutes triggers waterfall inversion risk (P0)
- [ ] Emit `node_clock_drift_ms` from each host via chrony textfile collector or node exporter; required for the clock drift alert to fire (P1)
- [ ] Configure tail_sampling policy with 100% error + 100% high-latency + 5-10% probabilistic baseline; gateway Collector only; `decision_wait: 30s` to buffer all spans (P1)
- [ ] Apply exporter retry configuration; `initial_interval: 5s`, `max_interval: 30s`, `max_elapsed_time: 300s`, `queue_size: 1000` to handle transient backend unavailability without infinite queue growth (P1)
- [ ] Scrape Fluent Bit self-metrics at `:2020/api/v1/metrics/prometheus`; monitor `fluentbit_output_errors_total` alongside `otelcol_*` for unified pipeline health view (P1)
- [ ] Publish a meta-observability Grafana dashboard; must include pipeline delivery ratio panels, receiver accepted vs refused, exporter sent vs failed, queue depth, heap usage, clock drift heatmap, and top cardinality metrics table (P1)
- [ ] Enforce retention policy per signal; metrics: 15d full-res / 90d 5m / 2y 1h; operational logs: 7d / 30d / 90d; sampled traces: 30d; profiles: 14d (P1)

---

## Section 4: Incident forensics readiness

- [ ] Verify all MRA resource attributes are present on every signal before production; `service.name`, `service.namespace`, `service.version`, `deployment.environment`, `cloud.*`, `host.id`, `k8s.*`, `container.id` (P0)
- [ ] Confirm `trace_id` appears on every structured log record; log-trace join is impossible without it; set via OTel SDK current span context (P0)
- [ ] Confirm `span_id` appears on every structured log record alongside `trace_id`; required for sub-trace log correlation (P0)
- [ ] Inject `request_id` at the API gateway; propagate as `x-request-id` header; log on every record; expose to end user for support tickets (P1)
- [ ] Verify metric exemplars are enabled and linked to `trace_id`; required for metric-to-trace pivot in Step 2 of the 6-dimension narrowing flow (P1)
- [ ] Confirm the 6-dimension narrowing flow can execute in under 15 minutes; run a tabletop drill: symptom capture → trace_id acquisition → region → server → service → layer → code → cross-signal validation (P1)
- [ ] Validate vendor query patterns are documented for all backends in use; Honeycomb, Datadog, Grafana Tempo, Jaeger, Sentry, Elastic query syntax must be accessible during an incident (P1)
- [ ] Test log-trace join by `trace_id` across at least two services in the production environment; the join must return matching records within the same request's span (P1)
- [ ] Confirm release marker log events include `service.name`, `service.version`, `deployment.environment`, and deployment SHA; without markers, Step 5 release correlation in the forensics playbook is blind (P0)

---

## Section 5: Compliance & Audit

- [ ] Store audit logs in WORM immutable storage; S3 Object Lock, GCS Object Hold, or Azure Immutable Blob Storage; minimum retention 7 years (P0)
- [ ] Configure audit log retention alert; `audit_log_retention_days < 2555` (7 years) triggers a critical alert via scheduled compliance check (P0)
- [ ] Include all six mandatory audit event categories; authentication, authorization, data access, administrative, security events, system events; before claiming SOC 2 or ISO 27001 compliance (P0)
- [ ] Carry required attributes on every audit event; `user.id` (pseudonymized), `actor.type`, `action`, `resource.type`, `resource.id`, `event.outcome` (P0)
- [ ] Implement PII redaction at pipeline ingestion; apply Collector `transform` processor to scrub or hash PII fields before they reach any storage backend (P0)
- [ ] Enforce GDPR storage limitation; raw personal data must not exceed the retention period justified by the processing purpose; default operational logs ≤ 7 days raw (P0)
- [ ] Separate audit and privacy pipelines; same-pipeline storage would create contradictory retention requirements (WORM vs. erasable); keep distinct pipelines per `signals/audit.md` Design Decision D5 (P0)
- [ ] Sign a Data Processing Agreement (DPA) with every observability vendor that receives personal data; required before routing production telemetry to SaaS backends (P0)
- [ ] Enforce cross-region data residency for EU and KR tenants; EU tenant telemetry must not transit or rest outside EEA without adequacy decision; KR tenants subject to PIPA § 29 (P0)
- [ ] Store pseudonymization keys separately from pseudonymized telemetry data; GDPR Art. 32 requires independent access control and audit trail for the key store (P1)
- [ ] Verify `user.id` and `user.email` are never used as metric labels; cardinality explosion and PII in TSDB are a double violation; use `filter` processor to drop any datapoint carrying these attributes (P0)

---

## Section 6: SLO & Release gate

- [ ] Define SLO with error budget policy for every user-facing service; SLO target, SLA target (SLA < SLO), 28-day rolling window, and burn-rate alert thresholds (P1)
- [ ] Implement multi-window burn-rate alerts; fast burn: 2% budget in 1 hour; slow burn: 5% budget in 6 hours; both required to avoid alert fatigue and missed slow leaks (P1)
- [ ] Store SLO definitions in OpenSLO YAML in git; treated as code; PR review required for any SLO target change (P1)
- [ ] Select SLI source from stable signals; mesh RED metrics or L7 application metrics preferred; log-based SLI is acceptable only as a fallback for un-instrumented services (P1)
- [ ] Configure canary analysis (Flagger or Argo Rollouts) to use SLI metrics as promotion gates; success rate threshold and latency p99 threshold both required (P1)
- [ ] Add memory saturation metric as a custom Flagger analysis gate; OOMKilled pods may not fail the success rate threshold before crashing; `container_memory_working_set_bytes` gate prevents silent OOM canary progression (P1)
- [ ] Emit `service.version` on all metrics, logs, and spans at deploy time; required for before/after comparison in canary analysis and in Step 3c of incident forensics (P0)
- [ ] Emit a structured release marker event at every deployment; must carry `service.name`, `service.version`, `deployment.environment`, deployment strategy, and deployment SHA (P0)
- [ ] Record SLO burn-rate threshold crossings as immutable audit events; provides compliance evidence for availability breach; cross-ref `signals/audit.md` (P1)
- [ ] Define rollback criteria and document the rollback procedure; include metric recovery verification steps in the runbook (P1)

---

## Section 7: Recovery

- [ ] Document Collector queue backpressure remediation runbook; steps: identify via `otelcol_processor_queued_retry_send_queue_length`, scale gateway Collector replicas, check vendor rate limits (P1)
- [ ] Document clock drift remediation procedure; `chronyc makestep` for immediate re-sync; escalation to PTP for sub-millisecond requirements; check NTP source reachability if drift persists (P1)
- [ ] Document cardinality bomb mitigation procedure; identify offending metric with `topk(10, count({__name__!=""}) by (__name__))`, add `filter` processor to drop offending label, or `transform` processor to remove the label value (P1)
- [ ] Document Collector OOM recovery path; verify `memory_limiter` is configured; if OOM still reached, increase container memory limit or reduce `limit_percentage`; never disable `memory_limiter` (P1)
- [ ] Document vendor fallback plan for observability backend outage; identify secondary backend or local Collector buffering strategy; `sending_queue` with `queue_size` provides short-term buffering (P2)
- [ ] Document PII incident response procedure; GDPR Art. 33 breach notification to supervisory authority within 72 hours; identify affected telemetry stores, scope the affected data subjects, and engage DPO (P0)
- [ ] Document audit log retention violation response; verify S3/GCS Object Lock policy; confirm `retention_period` in log backend matches 7y; restore from backup if records were prematurely deleted (P0)
- [ ] Test Flagger auto-rollback by injecting a synthetic error rate above the failure threshold in a staging environment; verify rollback completes before the `progressDeadlineSeconds` expires (P2)
- [ ] Validate that `memory_limiter` refused spans alert fires before Collector OOM; run a controlled load test; confirm `otelcol_processor_memory_limiter_refused_spans > 0` alert fires and data loss stops short of crash (P2)

---

## Contribution Protocol

- Append items under the section that best fits the intent (setup / pre-prod / ops / forensics / compliance / SLO / recovery)
- Format: `- [ ] {action verb} {specific target} — {acceptance criterion} (priority: P0/P1/P2)`
- Include a `cross-ref` comment pointing to the source file using a path relative to the `resources/` root when the item derives from a specific section
- Mark priority: P0 (block ship) | P1 (must before prod) | P2 (this sprint)
- Deduplicate before adding: search existing items by keyword before inserting
- CTO review required for any change to P0 items in Sections 1, 5, or 6
