# oma-observability Anti-Patterns

> Consolidated catalog of patterns that break observability/traceability.
> Each entry: pattern name, why-it-fails, remediation, severity.
> Sources: all resource files in this skill tree. See individual "See also" references.

## Severity legend

- **CRITICAL**: security breach / compliance violation / production data loss
- **HIGH**: blocks on-call ability to resolve incidents
- **MEDIUM**: degrades observability quality / cost / maintainability
- **LOW**: style or minor optimization

---

## A: Privacy & Sensitive Data

### A.1 Claiming "anonymization" for pseudonymized data

**Severity**: CRITICAL
**Why it fails**: Pseudonymized data (reversible with a key) is still personal data under GDPR Article 4(1). Misclassifying it as anonymous data leads to non-compliant retention periods, missing erasure obligations, and exposes the organization to 4% global-turnover fines.
**Remediation**: Apply the reversibility test from `signals/privacy.md §3`: "Could we recover the original value if compelled?" If yes, it is pseudonymization. Update your ROPA and re-classify accordingly.
**See also**: `signals/privacy.md §3 Anonymization vs Pseudonymization`

### A.2 OTel Collector `hash` action on low-entropy user IDs

**Severity**: CRITICAL
**Why it fails**: The Collector's built-in `hash` action applies SHA-256 without a salt. Numeric user IDs (e.g., 6-digit codes) are reversible via rainbow tables, making the pseudonymization ineffective and constituting a GDPR breach.
**Remediation**: Perform HMAC-SHA256 with a vault-managed key at the SDK layer before emitting spans. Store the salt in a separate region with independent IAM (GDPR Art. 32). Rotate quarterly.
**See also**: `signals/privacy.md §8 Salted Hashing Caveats`

### A.3 Baggage carrying PII across trust boundaries

**Severity**: CRITICAL
**Why it fails**: W3C Baggage propagates to every downstream service, including external partners. Placing `user.email`, session tokens, or credentials in baggage violates W3C Baggage §Security and GDPR Article 5(1)(c) minimization, leaking PII to untrusted collectors.
**Remediation**: Apply a baggage allowlist at every egress trust boundary (API gateway, external webhook). Allowed values: `tenant.id`, `feature.variant`, `deployment.sha`, `region.hint`. Strip all other keys.
**See also**: `boundaries/cross-application.md §4 Baggage Rules`, `signals/privacy.md §4 Common PII`

### A.4 PII in crash stack traces without redaction filter

**Severity**: CRITICAL
**Why it fails**: Exception messages frequently capture SQL queries, HTTP headers, and URL query strings verbatim, exposing `user.email`, `Authorization` tokens, card numbers, and passwords in vendor SaaS storage; a GDPR and PIPA breach.
**Remediation**: Implement a `beforeSend` / `before_send` allowlist hook in your crash SDK. Strip `Authorization`, `Cookie`, and any regex-matching email/card patterns before the crash report is serialized. Do not rely on server-side redaction alone.
**See also**: `layers/L7-application/crash-analytics.md §8 Privacy and PII`, `signals/privacy.md §7 SDK-Layer Redaction`

### A.5 Raw IP addresses retained without redaction

**Severity**: CRITICAL
**Why it fails**: IP addresses linked to natural persons are personal data under GDPR Article 4(1) and PIPA. Storing raw `srcaddr` / `client.address` in long-retention backends violates the data minimization principle (GDPR Art. 5(1)(c)) and is grounds for regulatory action.
**Remediation**: Truncate the last IPv4 octet (e.g., `203.0.113.0`) or apply rotating-salt SHA-256 pseudonymization at pipeline ingestion. Never store raw IPs beyond the 7-day raw retention tier.
**See also**: `signals/privacy.md §5 PII Handling Rules`, `layers/L3-network.md §3.3 Privacy Note`

### A.6 `db.query.text` with untrimmed PII

**Severity**: CRITICAL
**Why it fails**: SQL query text in `db.query.text` span attributes frequently contains `WHERE email = 'user@example.com'` or `WHERE ssn = '...'`. These surface verbatim in trace backends, violating GDPR minimization and leaking credentials.
**Remediation**: Use the OTel Collector `redaction` processor with an allowlist; block free-text query content unless it matches safe patterns. Apply SQL parameterization at the SDK layer so user data never enters query text.
**See also**: `signals/privacy.md §6 OTel Collector Processors`, `signals/traces.md`

### A.7 Observability backend open to all engineers (no RBAC)

**Severity**: HIGH
**Why it fails**: Production traces and logs containing even pseudonymized user data are sensitive. Unrestricted access violates the principle of least privilege, increases PII exposure risk, and fails SOC 2 CC7.2 and ISO/IEC 27001 A.8.15 access controls.
**Remediation**: Implement role-scoped access: on-call engineers see own-service traces (24h); security analysts see all services (30d); auditors get read-only audit index. Use Grafana folder permissions or Datadog Teams scopes.
**See also**: `signals/privacy.md §12 Backend RBAC`

### A.8 Routing telemetry to 3rd-party vendor without DPA

**Severity**: HIGH
**Why it fails**: Observability vendors receiving personal data are data processors under GDPR Article 28. Routing data before signing a Data Processing Agreement is a direct GDPR violation.
**Remediation**: Block data flows to Datadog, Sentry, Grafana Cloud, Honeycomb, New Relic, or Elastic Cloud until a DPA is signed. Confirm storage region matches your compliance obligations. Reference the legal team's approved vendor list.
**See also**: `signals/privacy.md §13 Third-Party Processor Obligations`

### A.9 Session replay without client-side PII masking

**Severity**: HIGH
**Why it fails**: Session replay captures DOM mutations including `<input>` fields. Without SDK-level masking, email addresses, credit card numbers, and passwords are captured in replay payloads before reaching the vendor; a GDPR Article 6 consent violation.
**Remediation**: Enable input masking in the SDK config (Sentry, Datadog both support this). Wire replay consent to the cookie consent flow. Test via automated replay review that sensitive fields are masked.
**See also**: `layers/L7-application/web-rum.md §8 Session Replay`, `signals/privacy.md §5`

### A.10 Unencrypted telemetry queue on mobile device

**Severity**: HIGH
**Why it fails**: Mobile telemetry queued to disk is PII at rest on user devices. If the device is lost, compromised, or forensically examined, unencrypted queue files expose personal data; violating GDPR Art. 32 and PIPA § 29 safety measures.
**Remediation**: Encrypt the queue using platform-native key storage: iOS Keychain, Android Keystore. Apply field-level redaction before write, not before send.
**See also**: `layers/L7-application/mobile-rum.md §3 Offline-First Queuing`

### A.11 Cross-region telemetry routing without GDPR mechanism

**Severity**: HIGH
**Why it fails**: GDPR Chapter V prohibits transfer of personal data to non-adequate countries without Standard Contractual Clauses or an adequacy decision. Routing EU telemetry through a US-hosted collector without SCC in place is a direct violation.
**Remediation**: Implement routing connector per `signals/privacy.md §10`: route EU traffic to EU-region backend, KR traffic to KR-region backend. Confirm SCC or adequacy decision with legal before routing.
**See also**: `signals/privacy.md §10 Cross-Border Transfer`, `transport/collector-topology.md §7`

---

## B: Cardinality & Cost

### B.1 `user.id` as metric label

**Severity**: CRITICAL
**Why it fails**: Creates one TSDB time series per user. For a service with 1M users, this is 1M series; instant storage explosion, query latency degradation, and SaaS vendor bill spike. Additionally, user IDs are PII under GDPR Art. 4(1).
**Remediation**: Replace with `user.tier`, `user.cohort`, or aggregated bucket labels. Never use any unbounded identifier as a metric label. Enforce via OTel SDK View attribute allow-list.
**See also**: `signals/metrics.md §9 Cardinality Budget`, `meta-observability.md §Section C`

### B.2 New metric name per tenant

**Severity**: HIGH
**Why it fails**: Creating `http_requests_total_tenant_acme` for each tenant bypasses TSDB cardinality controls entirely and cannot be aggregated across tenants. It also disables cardinality budget alerting.
**Remediation**: Use `http_requests_total{tenant_id="acme"}` with a top-N cap (≤ 1000 explicit tenants). Map overflow to label value `"other"`. Apply Collector `transform` processor for normalization.
**See also**: `signals/metrics.md §9.3 Tenant Cap`, `signals/cost.md §10`

### B.3 Raw `http.url` as metric label

**Severity**: HIGH
**Why it fails**: URL query strings are unbounded and may contain tokens or email addresses (`?token=...`, `?email=...`). Using raw `url.full` as a label causes cardinality explosion and PII leakage in the TSDB.
**Remediation**: Use `http.route` (normalized route) instead. Apply `replace_pattern` in OTel Collector `transform` processor to strip numeric IDs and UUIDs from route segments.
**See also**: `meta-observability.md §Section C`, `signals/metrics.md §3.3 Label Rules`

### B.4 Cost label at per-request metric granularity

**Severity**: HIGH
**Why it fails**: Writing `gen_ai.cost.total_usd` as a metric label at request granularity creates one series per request; causing OOM on the TSDB ingestor and making cost attribution unusable.
**Remediation**: Use `gen_ai.cost.total_usd` as a span attribute only. Aggregate cost metrics by `tenant_id`, `namespace`, and `workload` at the metric surface. Configure tail-sampler to always retain spans where cost exceeds $0.50.
**See also**: `signals/cost.md §10`, `transport/sampling-recipes.md §3`

### B.5 Summary instrument for cross-service aggregation

**Severity**: MEDIUM
**Why it fails**: Prometheus Summary computes quantiles client-side per process. p99 from three replicas cannot be merged into a fleet-level p99; the values are mathematically incompatible.
**Remediation**: Replace Summary with Histogram + `histogram_quantile()` at query time. Set explicit bucket boundaries matching the expected value range (e.g., seconds-scale for request duration, not the default millisecond buckets).
**See also**: `signals/metrics.md §2.5 Summary`

### B.6 Histogram with default bucket boundaries

**Severity**: MEDIUM
**Why it fails**: The OTel SDK default buckets are millisecond-scale `[0, 5, 10, 25, 50, ...]`. For second-scale operations (database queries, LLM inference, file uploads), all measurements land in the last bucket, making `histogram_quantile()` meaningless.
**Remediation**: Set explicit bucket boundaries per instrument view using `View(instrument_name=..., aggregation=ExplicitBucketHistogramAggregation([0.01, 0.05, 0.1, 0.5, 1, 2, 5]))`.
**See also**: `signals/metrics.md §2.4 Histogram`

### B.7 LLM spans not tail-sampled on cost threshold

**Severity**: MEDIUM
**Why it fails**: High-cost LLM spans (e.g., $2+ per trace) may be dropped by probabilistic sampling before they reach the backend. Silent budget blowup is invisible until the cloud billing invoice arrives.
**Remediation**: Use a `transform` processor to set `sampling.keep_reason = "high_cost"` on spans where `gen_ai.cost.total_usd > 0.50`, then add a `string_attribute` policy in `tail_sampling` that always retains these spans.
**See also**: `signals/cost.md §6`, `transport/sampling-recipes.md §3`

---

## C: Pipeline & Collector

### C.1 Missing `memory_limiter` processor

**Severity**: CRITICAL
**Why it fails**: Without `memory_limiter`, a traffic spike or backend backpressure causes the Collector heap to grow unbounded until the process OOM-kills. All in-flight signals are lost and the pipeline is silent until the pod restarts.
**Remediation**: Add `memory_limiter` as the **first** processor in every pipeline (traces, metrics, logs). Set `limit_percentage: 75` and `spike_limit_percentage: 20`. Apply to both DaemonSet agents and gateway tiers.
**See also**: `meta-observability.md §Section A3`, `transport/collector-topology.md §2`, `transport/otlp-grpc-vs-http.md §4.3`

### C.2 Sidecar collectors on standard Kubernetes nodes

**Severity**: HIGH
**Why it fails**: Sidecar mode injects one Collector per application pod. A 100-pod deployment runs 100 Collectors. CPU and memory cost scales linearly with pod count, overwhelming cluster resources. Additionally, sidecar collectors see only spans from their own pod, breaking tail sampling.
**Remediation**: Use DaemonSet mode (one Collector per node) for standard Kubernetes. Reserve sidecar mode for AWS Fargate or GCP Cloud Run where DaemonSets are unavailable.
**See also**: `transport/collector-topology.md §3 When to Use Sidecar`

### C.3 Tail sampling in sidecar

**Severity**: HIGH
**Why it fails**: A sidecar Collector only sees spans from its own pod. A trace spanning multiple services has spans on different pods, each with a different sidecar. No single sidecar has the complete trace; sampling decisions are based on incomplete data, producing systematically wrong retention.
**Remediation**: Run `tail_sampling` processor in the gateway tier (Deployment mode) only, combined with a `loadbalancing` exporter using consistent hash by `trace_id` to ensure trace completeness.
**See also**: `transport/collector-topology.md §3`, `transport/sampling-recipes.md §2`

### C.4 Single gateway replica

**Severity**: HIGH
**Why it fails**: A single gateway Collector is a single point of failure. Under backend backpressure or a rolling restart, all telemetry is lost for the duration. This eliminates observability exactly when it is most needed.
**Remediation**: Deploy a minimum of 2–3 gateway replicas. Use a PodDisruptionBudget to prevent simultaneous eviction. Add a `loadbalancing` exporter upstream for trace-complete routing.
**See also**: `transport/collector-topology.md §6 High-Throughput Gateway Scaling`

### C.5 Fluentd as new log pipeline deployment in 2026+

**Severity**: HIGH
**Why it fails**: CNCF announced Fluentd deprecation on 2025-10. Choosing Fluentd for a new deployment in 2026+ means adopting a deprecated tool with no future community investment, missing Fluent Bit's OTLP native output, and incurring Ruby runtime overhead (100+ MB vs Fluent Bit's 5–15 MB).
**Remediation**: Use Fluent Bit (CNCF Graduated, C/Rust, native OTLP output) as the edge DaemonSet log agent. Use OTel Collector for gateway aggregation, PII redaction, and routing.
**See also**: `signals/logs.md §6 Collector and Agent Options`

### C.6 Prometheus receiver on multiple replicas without target allocator

**Severity**: MEDIUM
**Why it fails**: When multiple gateway Collector replicas each run a `prometheus` receiver, every replica scrapes every target. This produces duplicate metrics in the TSDB backend and inflates series counts.
**Remediation**: Enable the OTel Operator's `spec.targetAllocator` to distribute scrape targets across replicas. Each replica scrapes a non-overlapping subset of targets.
**See also**: `transport/collector-topology.md §4 Component Placement Reference`

### C.7 Collector self-metrics not scraped by independent instance

**Severity**: MEDIUM
**Why it fails**: If the Collector scrapes its own `:8888/metrics` endpoint and the Collector fails, the metrics that would reveal the failure are also lost. The pipeline's health is invisible at the moment of failure.
**Remediation**: Scrape Collector self-metrics from a separate Prometheus instance or second Collector instance. Expose `otelcol_*` metrics at `service.telemetry.metrics.address: 0.0.0.0:8888` with `level: detailed`.
**See also**: `meta-observability.md §Section A1`

---

## D: Sampling & Retention

### D.1 Head-based sampling on multi-service call paths

**Severity**: HIGH
**Why it fails**: Head-based sampling makes a sampling decision at the trace root. If a downstream service independently samples out, its spans are dropped; the reconstructed trace has gaps. Tracing backends show incomplete traces that mislead incident investigation.
**Remediation**: Use tail-based sampling in the gateway tier for multi-service systems. Propagate `traceparent` on every hop regardless of local sampling decision. The gateway buffers all spans and decides after the trace is complete.
**See also**: `transport/sampling-recipes.md §1`, `transport/sampling-recipes.md §6 Pitfalls`

### D.2 Missing `loadbalancing` exporter before tail sampler

**Severity**: HIGH
**Why it fails**: Without consistent-hash routing, spans for the same trace arrive at different gateway replicas. Each replica's `tail_sampling` processor sees an incomplete trace and makes wrong retention decisions; high-value traces are dropped, low-value traces are retained.
**Remediation**: Deploy `loadbalancing` exporter (with `routing_key: traceID`) in the tier upstream of `tail_sampling`. Use a headless Kubernetes Service so the exporter resolves per-pod DNS.
**See also**: `transport/sampling-recipes.md §2`, `transport/collector-topology.md §6`

### D.3 Audit logs not stored in WORM (mutable audit storage)

**Severity**: CRITICAL
**Why it fails**: SOC 2 CC7.2, PCI DSS Requirement 10, and HIPAA §164.312(b) require tamper-evident, immutable audit storage. Mutable audit logs can be deleted or altered, nullifying compliance evidence and enabling concealment of unauthorized actions.
**Remediation**: Apply S3 Object Lock in **Compliance** mode (not Governance), GCS retention policy with locked bucket, or Azure Immutable Blob Storage at bucket creation time. Set 7-year retention as a baseline. Do not use Governance mode; it allows privileged override.
**See also**: `signals/audit.md §5 Immutable WORM Storage`

### D.4 Audit log retention below regulatory minimum

**Severity**: HIGH
**Why it fails**: HIPAA requires 6-year retention; PCI DSS requires 1 year online + offline; SOC 2 audit periods are typically 12 months. Flushing audit logs before the minimum period is a direct compliance violation discoverable during any audit.
**Remediation**: Use a 7-year baseline with automated lifecycle policy (S3 → Glacier Deep Archive after 90 days). Set WORM Object Lock at write time; it cannot be applied retroactively. Monitor compliance via an `audit_log_retention_days` metric alert.
**See also**: `signals/audit.md §7 7-Year Retention Policy`, `meta-observability.md §Section F Alert 5`

### D.5 No tamper evidence on audit trail

**Severity**: HIGH
**Why it fails**: WORM prevents deletion but does not prove that records were not silently modified or that records are not missing. An audit trail without a hash chain or Merkle root anchoring cannot satisfy forensic integrity requirements for SOC 2 or ISO/IEC 27001.
**Remediation**: Implement per-event hash chain: `event_N.hash = SHA256(event_N.payload + event_{N-1}.hash)`. Anchor daily Merkle root to an external transparency log (rekor/sigstore). Run automated chain verification weekly.
**See also**: `signals/audit.md §6 Tamper Evidence`

### D.6 Kubernetes audit logs routed to operational log store

**Severity**: HIGH
**Why it fails**: Operational log stores (Loki, Elasticsearch) are mutable and have short retention policies. Kubernetes API audit logs; which record secret access, RBAC mutations, and cluster-admin actions; require WORM storage with multi-year retention to satisfy PCI DSS Requirement 10.
**Remediation**: Route K8s audit logs via a separate pipeline to the WORM cold tier (S3 Glacier Deep Archive with Object Lock). Tag with `source: k8s_apiserver`. Do not co-mingle with operational logs.
**See also**: `signals/audit.md §9 Kubernetes Audit Logs`

### D.7 Decision wait too short in tail sampler

**Severity**: MEDIUM
**Why it fails**: If `decision_wait` is shorter than the p99 inter-service latency, spans from slow downstream services arrive after the sampling decision is finalized. Those spans are dropped unconditionally, making tail sampling silently incomplete.
**Remediation**: Set `decision_wait` to exceed your p99 cross-service latency (typically 30–60 s for synchronous calls). For cross-region deployments, account for additional propagation delay. Monitor `otelcol_processor_queued_retry_send_queue_length` for buffer pressure.
**See also**: `transport/sampling-recipes.md §6 Pitfalls`

---

## E: Release & Deployment

### E.1 No release markers in telemetry

**Severity**: HIGH
**Why it fails**: Without `service.version` on spans, metrics, and logs; and without a deployment event at release time; there is no way to correlate a metric anomaly or error spike to a specific deploy. Incident investigation degrades to git bisect guesswork.
**Remediation**: Set `service.version` on the OTel Resource at SDK init (injected via CI as `OTEL_RESOURCE_ATTRIBUTES`). Emit a structured deployment event at release time and pipeline it to Grafana as a vertical annotation line.
**See also**: `boundaries/release.md §9 Release Markers`

### E.2 Canary analysis without SLI metric

**Severity**: HIGH
**Why it fails**: Promoting a canary without a Flagger `MetricTemplate` or Argo Rollouts `AnalysisTemplate` means the promotion decision is manual and intuition-based. Regressions in error rate or latency ship silently to 100% of traffic.
**Remediation**: Gate every canary on at minimum: error rate (max +0.5% allowed) and p99 latency (max +50ms allowed). Reference PromQL SLI queries from `boundaries/slo.md` as the metric source.
**See also**: `boundaries/release.md §10 Canary Analysis Metric Suite`

### E.3 Symbol upload not automated in CI (mobile / web)

**Severity**: HIGH
**Why it fails**: Without automated dSYM / ProGuard / source map upload on every release, production crashes produce unreadable minified stack traces. Incident investigation time for mobile/web crashes increases by 30–60 minutes per incident.
**Remediation**: Add symbol upload as a mandatory CI step, gated on the same pipeline step as container image push or app store submission. Use Sentry CLI, `fastlane-plugin-sentry`, or Gradle Sentry plugin. Never store symbols in git LFS.
**See also**: `layers/L7-application/crash-analytics.md §9 CI Integration`

### E.4 GitOps drift unalerted

**Severity**: MEDIUM
**Why it fails**: When the cluster state diverges from the git manifest (OutOfSync in Argo CD, stalled Flux reconcile), the cluster is running unintended configuration. This silently breaks observability contracts; dashboards may reference metrics from a previous config version.
**Remediation**: Alert on `argocd_app_info{sync_status="OutOfSync"} == 1` and `gotk_reconcile_condition{type="Ready",status="False"} == 1`. Treat drift as an incident, not a warning.
**See also**: `boundaries/release.md §7 GitOps Engines`

### E.5 Feature flag evaluation not observed

**Severity**: MEDIUM
**Why it fails**: Without `feature_flag.*` span attributes on every flag evaluation, A/B experiment effects are invisible. A latency regression introduced by a new feature variant cannot be attributed to the flag without telemetry.
**Remediation**: Emit `feature_flag.key`, `feature_flag.variant`, and `feature_flag.provider_name` as span attributes on every evaluation (OTel `feature_flag.*` semconv, Experimental tier). Track `feature_flag_evaluation_total` and error rate delta per variant.
**See also**: `boundaries/release.md §6 Observing Feature Flag Evaluations`

### E.6 Rollback without post-mortem audit trail

**Severity**: MEDIUM
**Why it fails**: A rollback event without an audit record severs the learning loop. The same deployment failure repeats because there is no evidence of what was deployed, who authorized it, and what metric triggered rollback.
**Remediation**: Record every rollback in the audit log with: `actor.type`, `action: rollback`, `service.version` (both from and to), `event.outcome`, and `trace_id`. Cross-reference to the triggering alert.
**See also**: `signals/audit.md §3 Audit Event Categories`, `boundaries/release.md §11`

---

## F: Security & Compliance

### F.0 Audit pipeline supply-chain integrity unverified

**Severity**: CRITICAL
**Why it fails**: If the Collector binary, Falco rules, or sigstore/rekor clients shipped into the audit pipeline are tampered with, every downstream "immutable" audit record inherits the compromise. WORM storage guarantees nothing about the ingestor's trustworthiness.
**Remediation**: Sign all audit-pipeline container images with cosign (sigstore). Pin Falco rule SHA digests in Helm values. Verify rekor transparency-log entries for rule provenance. Deploy an admission controller (e.g., Kyverno + `verifyImages`) that rejects unsigned images from the observability namespace. Cross-ref `signals/audit.md §8 Falco` and sigstore documentation.
**See also**: `signals/audit.md`, sigstore.dev

### F.1 `traceparent` stripped at service boundary

**Severity**: HIGH
**Why it fails**: Stripping `traceparent` on outbound calls silently breaks distributed trace continuity. The downstream service starts a new root trace, making it impossible to correlate a customer-reported request across services. MTTR increases by 15–30 minutes per incident.
**Remediation**: Every outbound HTTP/gRPC call MUST forward `traceparent`. Configure the OTel SDK auto-instrumentation to inject the header automatically. Test with an integration test that asserts the `traceparent` header on every outbound call.
**See also**: `standards.md §2.2 W3C Trace Context`

### F.2 Mixed propagators without normalization at ingress

**Severity**: HIGH
**Why it fails**: When B3 (Zipkin), AWS X-Ray, and W3C headers coexist without a composite propagator at the ingress gateway, spans from different origins appear as disconnected root spans in the trace backend. Waterfall correlation is impossible across cloud or vendor boundaries.
**Remediation**: Configure the ingress gateway with a composite propagator that extracts all known formats (tracecontext, b3multi, awsxray, datadog) and emits only W3C downstream.
**See also**: `boundaries/cross-application.md §3 Propagator Matrix`

### F.3 `request_id` not exposed to frontend users

**Severity**: MEDIUM
**Why it fails**: Without a user-visible `request_id` in error banners, customer support cannot correlate a user-reported error to backend traces. Support agents must rely on user-provided timestamps and symptoms; escalations that should take 2 minutes take 20.
**Remediation**: Return `x-request-id` in every HTTP response header. Display it in frontend error banners. Customer support uses this ID; engineers use it to pivot to `trace_id` in the log system.
**See also**: `boundaries/cross-application.md §7 request_id to trace_id Integration`

### F.4 `service.namespace` not set on services

**Severity**: MEDIUM
**Why it fails**: Without `service.namespace`, a system with 80+ microservices presents a flat, unordered list in trace backends and dashboards. Domain-level KPIs (e.g., payment-domain error rate vs. inventory-domain error rate) are impossible to compute.
**Remediation**: Assign `service.namespace` per DDD bounded context at deployment time via `OTEL_RESOURCE_ATTRIBUTES` or OTel Collector `resource` processor. Align namespaces with domain boundaries, not team names.
**See also**: `boundaries/cross-application.md §5 DDD Bounded Context`

### F.5 New `trace_id` generated on DLQ replay

**Severity**: MEDIUM
**Why it fails**: Generating a new `trace_id` at dead letter queue replay severs the forensic chain from the original failure span to the replay event. Root cause analysis of repeated failures requires tracing back to the originating request.
**Remediation**: Re-inject the original `traceparent` and `causation_id` from the failed message headers when replaying. Use a span link; not parent-child; to connect replay span to the original trace.
**See also**: `boundaries/cross-application.md §8 Idempotency and Event-Driven Trace Lineage`

### F.6 WAF rule hit / block rate unmonitored

**Severity**: HIGH
**Why it fails**: When a WAF rule update over-blocks legitimate traffic, the edge returns 403 to users while backend services see request volume drop. Service-level error rate stays flat; the on-call investigates "missing traffic" rather than "WAF blocking my users", and the FP storm is discovered only via customer complaints. MTTR runs into hours.
**Remediation**: Emit `waf_action_total{action, rule_id, rule_set_version}` from the edge tier. Alert when the block rate per `rule.id` step-changes more than 3σ above its rolling 24h baseline. Dashboard the action distribution alongside the application 5xx rate so block-vs-error is visible side-by-side.
**See also**: `layers/L7-application/waf.md §5 False-Positive Surge`

### F.7 WAF rule-set deploy without release marker

**Severity**: HIGH
**Why it fails**: Without a release marker stream for ruleset rollouts, an edge 403 surge cannot be aligned with the rule change that caused it. The rollback decision is guesswork; engineers may roll back the application instead of the ruleset.
**Remediation**: Emit a structured deployment event on every ruleset rollout carrying `waf.rule.set.version`, `waf.vendor`, and the rollout strategy (full vs canary). Treat WAF ruleset versions as a peer to `service.version` for release-correlation purposes per `boundaries/release.md`.
**See also**: `layers/L7-application/waf.md §3 Required Attributes`, `boundaries/release.md`

### F.8 WAF fail-closed without dependency-health signal

**Severity**: HIGH
**Why it fails**: WAFs that consult external services (IP reputation, bot scoring, GeoIP, threat-intel API) inherit those dependencies' failure modes. A fail-closed policy without dependency telemetry means a third-party API outage silently blocks 100% of traffic; on-call has no signal that the security layer is the cause.
**Remediation**: Emit `waf_dependency_failure_total{dependency, policy}` per external dependency. Alert when non-zero for more than 60 s. Document the fail-open vs fail-closed policy per rule and surface it on the WAF dashboard.
**See also**: `layers/L7-application/waf.md §6 Fail-Open vs Fail-Closed`

### F.9 WAF logs detached from application trace context

**Severity**: MEDIUM
**Why it fails**: When the WAF blocks a request and emits a log without trace correlation, the application has no record of the attempt. On-call cannot answer "did this user even reach our service?" without manually joining vendor-specific identifiers (e.g., `cf-ray`, `X-Amzn-Trace-Id`) to the application trace.
**Remediation**: Configure the WAF to propagate `traceparent` even on block actions. Map vendor headers (`cf-ray`, `X-Amzn-Trace-Id`, Akamai `X-Akamai-*`) to W3C baggage at the edge so application traces and WAF logs join cleanly on `trace_id`.
**See also**: `layers/L7-application/waf.md §7 Vendor Telemetry Surfaces`, `boundaries/cross-application.md §Propagators`

### F.10 WAF rules promoted to `block` without `log`-mode soak

**Severity**: MEDIUM
**Why it fails**: A new rule promoted directly to `block` action carries unknown FP rate; first contact with production traffic doubles as the rule's correctness test. The failure mode is a mass-block incident affecting real customers.
**Remediation**: Mandatory minimum 24 h `log`-mode (detect-only) soak per new rule with FP rate verified below a target ratio (commonly < 0.1% of matched traffic) before promotion to `block`. Encode the soak requirement as a CI gate on the ruleset repository so promotion bypass is auditable.
**See also**: `layers/L7-application/waf.md §2 WAF Action Model §5 False-Positive Surge`

---

## G: Frontend / Mobile

### G.1 3rd-party script loaded without CSP monitoring

**Severity**: HIGH
**Why it fails**: Third-party scripts loaded without a Content Security Policy are an unmonitored XSS and supply-chain attack vector. A compromised CDN script executes in the user's browser with full page access. LCP regressions from script load delays are also invisible without attribution.
**Remediation**: Add `Content-Security-Policy` header with `report-to` endpoint. Pipe CSP violation reports to the log backend and alert on new `blocked-uri` origins. Pin script hashes with Subresource Integrity (`integrity="sha384-..."`).
**See also**: `layers/L7-application/web-rum.md §6 Third-Party Scripts and CSP`

### G.2 Source maps not uploaded to error vendor

**Severity**: HIGH
**Why it fails**: Minified production bundles produce unreadable stack traces: `at t.<anonymous> (bundle.min.js:1:74821)`. Without source maps, engineers cannot identify the failing line of code. Incident investigation for frontend crashes is impossible.
**Remediation**: Upload source maps to the error tracking vendor on every CI release pipeline step, before the release is considered complete. Use Sentry CLI or equivalent. Never store source maps in git LFS; use vendor symbol storage.
**See also**: `layers/L7-application/web-rum.md §9 Error Tracking`, `layers/L7-application/crash-analytics.md §3`

### G.3 FID still reported in dashboards after March 2024

**Severity**: MEDIUM
**Why it fails**: First Input Delay (FID) was removed from Core Web Vitals in March 2024 and replaced by Interaction to Next Paint (INP). Dashboards still reporting FID mislead SLO reviews; FID scores passing does not mean the INP SLO is met.
**Remediation**: Replace FID with INP in all dashboards and OpenSLO definitions. SLI target: INP p75 ≤ 200 ms. Use `web-vitals` JS library v4.x which provides `onINP`.
**See also**: `layers/L7-application/web-rum.md §2 Core Web Vitals`

### G.4 `propagateTraceHeaderCorsUrls` / `allowedTracingUrls` not configured

**Severity**: MEDIUM
**Why it fails**: Browser CORS preflight rejects injection of the `traceparent` header to origins not listed in the SDK allowlist. Client-to-server trace correlation silently breaks; frontend traces appear disconnected from backend traces.
**Remediation**: Add all API origin patterns to `FetchInstrumentation`'s `propagateTraceHeaderCorsUrls` (OTel JS) or Datadog RUM's `allowedTracingUrls`. Test in a browser network inspector to confirm the header is present.
**See also**: `layers/L7-application/web-rum.md §5 Client-to-Server Error Correlation`

### G.5 Missing `traceparent` injection on mobile outbound HTTP

**Severity**: MEDIUM
**Why it fails**: Without `traceparent` on outbound requests from the mobile app, the mobile user session is invisible in backend distributed traces. Customer-reported mobile errors cannot be linked to backend spans.
**Remediation**: Configure the mobile SDK's HTTP interceptor at initialization time (not at individual call sites). Verify with a network proxy tool (Charles, mitmproxy) that `traceparent` appears on all outbound API calls.
**See also**: `layers/L7-application/mobile-rum.md §9 W3C Trace Context Propagation`

### G.6 No event TTL on mobile offline queue

**Severity**: MEDIUM
**Why it fails**: Stale events queued on-device for hours or days eventually upload when the network reconnects. Events with device timestamps that are 24+ hours old mislead dashboards and SLO calculations; a crash from two days ago appears as a current incident.
**Remediation**: Set an event TTL of 24–72 hours on the mobile offline queue. Drop events that exceed the TTL before upload, not after. Log dropped event counts as a metric for monitoring queue health.
**See also**: `layers/L7-application/mobile-rum.md §3 Offline-First Queuing`

### G.7 No release marker for crash correlation on mobile

**Severity**: MEDIUM
**Why it fails**: Without `service.version` set on every crash event and without a release marker event at deploy time, a crash rate spike cannot be attributed to a specific app version. Investigation requires manual version comparison across crash groups.
**Remediation**: Set `service.version` as a custom key on every crash report. Automate release marker events at submission time. Use the release marker to draw vertical lines on CFR trend charts for before/after comparison.
**See also**: `layers/L7-application/crash-analytics.md §4 Release Tracking Integration`

---

## H: Network / BGP / Clock

### H.1 NTP drift left unmonitored

**Severity**: HIGH
**Why it fails**: Distributed traces depend on synchronized clocks. When two nodes diverge by more than 100 ms, waterfall charts show child spans starting before their parent; engineers chase phantom race conditions instead of real bugs. MTTR increases by hours.
**Remediation**: Emit `node_clock_drift_ms` from every host (chrony textfile collector or node exporter). Alert when drift exceeds 100 ms for 5 minutes. Run `chronyc makestep` to force resync. For financial / telco workloads requiring sub-ms precision, use PTP (IEEE 1588).
**See also**: `standards.md §6 Clock Discipline`, `meta-observability.md §Section B`

### H.2 Own-ASN BGP hijack left unmonitored

**Severity**: HIGH
**Why it fails**: A BGP prefix hijack diverts traffic to a rogue AS, which may cause a complete outage or a silent man-in-the-middle attack. Without MOAS detection, hijacks go undetected for hours or days while traffic is silently stolen.
**Remediation**: Deploy BGPalerter or ARTEMIS with MOAS detection for your ASN prefixes. Subscribe to Cloudflare Radar alerts for your prefixes. Validate RPKI ROA records at your RIR and enable ROV enforcement on border routers.
**See also**: `layers/L3-network.md §6.4 Security Observability`

### H.3 PMTUD black hole left uncorrected

**Severity**: HIGH
**Why it fails**: When firewalls block ICMP Type 3 Code 4 ("Fragmentation Needed"), PMTUD fails silently. Large TCP transfers stall while health checks (small packets) pass; masking the problem. Services appear healthy in monitors while bulk data transfers time out.
**Remediation**: Enable MSS clamping at VPN/tunnel endpoints. Allow ICMP Type 3 Code 4 through security groups. Verify the fix with `ping -M do -s 1472 <destination>`. Cross-reference UDP MTU constraints for StatsD pipelines.
**See also**: `layers/L3-network.md §4 PMTUD`, `transport/udp-statsd-mtu.md §2`

### H.4 RPKI-ROV not configured on advertised prefixes

**Severity**: HIGH
**Why it fails**: An IP prefix announced without a valid RPKI ROA record is marked "Not Found" by downstream validators; not "Invalid", but also not cryptographically anchored. Rogue AS announcements for your prefix are undetectable by validators, increasing hijack risk.
**Remediation**: Create ROA records at your RIR (ARIN, RIPE NCC, APNIC) for all advertised prefixes. Enable ROV enforcement on border routers to drop or de-prefer RPKI Invalid routes.
**See also**: `layers/L3-network.md §6.4 RPKI-ROV`

### H.5 Connection pool observability absent

**Severity**: HIGH
**Why it fails**: Pool queue saturation causes application latency spikes that are invisible in TCP metrics alone. Neither retransmit rate nor error rate spikes until connection timeouts fire; engineers investigate network issues while the actual problem is a saturated database pool.
**Remediation**: Instrument connection pool size, wait time, and timeout counters at the application layer (not only at the TCP layer). Alert on pool utilization > 80% before timeouts occur.
**See also**: `layers/L4-transport.md §3.3 Common Pitfalls`

### H.6 QUIC adoption without HTTP/3 trace tooling validation

**Severity**: HIGH
**Why it fails**: Enabling QUIC without verifying that OTel SDKs emit `network.transport: quic` and `network.protocol.version: "3"` creates a transport-layer blind spot. UDP-based QUIC flows are invisible in TCP metrics (`/proc/net/tcp`), leaving L4 observability dark.
**Remediation**: Add a canary assertion in staging: verify `network.transport: quic` appears in spans and that Envoy access logs show QUIC connection IDs. Confirm tooling (Beyla QUIC uprobe) before production rollout.
**See also**: `layers/L4-transport.md §7 QUIC / HTTP3 Transport Semantics`

### H.7 eBPF agent deployed without kernel/capability preflight

**Severity**: HIGH
**Why it fails**: Beyla or Pixie DaemonSets on incompatible kernels (< 4.14) or without CAP_BPF fail silently, providing no operator-visible error or metric. The observability gap is discovered only during an incident.
**Remediation**: Add an `initContainer` that asserts kernel version (`uname -r >= 4.14`) and CAP_BPF presence. Exit non-zero if requirements are unmet. This surfaces the incompatibility during deployment, not during an incident.
**See also**: `layers/L4-transport.md §5.2 Kernel and Privilege Requirements`

---

## I: As-Code & GitOps

### I.1 Production dashboards edited directly in UI

**Severity**: HIGH
**Why it fails**: UI-edited dashboards have no version history, no rollback path, no peer review, and no audit trail. SOC 2 change-management controls require an audit-traceable change process for detection and response configuration. A dashboard overwritten by mistake cannot be recovered without a backup.
**Remediation**: Version all dashboards in git using Grafonnet (Jsonnet) or Terraform Grafana provider. Apply via CI/CD only. Gate on PR review and linting (`jsonnetfmt --test`). Treat the git history as the audit trail.
**See also**: `observability-as-code.md §1 Why Observability-as-Code`

### I.2 Alert thresholds hardcoded per-environment without parameterization

**Severity**: MEDIUM
**Why it fails**: Hardcoded thresholds in per-environment alert YAML files diverge silently over time. Production runs a different error-rate threshold than staging, masking regressions that staging was supposed to catch.
**Remediation**: Parameterize alert thresholds as Jsonnet or Terraform variables. Derive environment-specific values from a shared defaults map. Apply the same alert code to all environments with environment-scoped variable overrides.
**See also**: `observability-as-code.md §2 Dashboards-as-Code`

### I.3 SLO definitions stored only in the vendor UI

**Severity**: MEDIUM
**Why it fails**: SLO definitions stored only in Datadog, Grafana Cloud, or Honeycomb UIs are not version-controlled, not peer-reviewed, and cannot be reconstructed after a vendor migration. SLO drift goes undetected.
**Remediation**: Define SLOs in OpenSLO YAML, committed to git. Apply via sloth or the vendor's Terraform provider. The git commit history is the audit trail for SLO changes.
**See also**: `observability-as-code.md`, `boundaries/slo.md`

---

## Z: Cross-cutting

### Z.1 `trace_id` missing from log records

**Severity**: HIGH
**Why it fails**: Without `trace_id` and `span_id` on every log record, log-trace join during incident forensics fails. Engineers cannot pivot from a log error to the distributed trace waterfall, adding 15–30 minutes to MTTR per incident.
**Remediation**: Inject `trace_id` and `span_id` into every log record via the OTel SDK context hook (Python: `structlog`; Java: Logback `OpenTelemetryAppender`; Node.js: `pino-opentelemetry-transport`). Assert in integration tests.
**See also**: `signals/logs.md §7 Trace ID Injection Rules`, `incident-forensics.md §2.3`

### Z.2 `service.version` missing from resource attributes

**Severity**: HIGH
**Why it fails**: Without `service.version` on every signal, before/after comparison across a release is impossible. Canary analysis, SLO delta calculation, and post-incident release attribution all fail.
**Remediation**: Set `service.version` on the OTel Resource at SDK initialization. Inject via CI as `OTEL_RESOURCE_ATTRIBUTES=service.version=${GIT_SHA}`. Never set it per-signal; the Resource is the single source.
**See also**: `incident-forensics.md §2.1 Resource Attributes`, `boundaries/release.md §9`

### Z.3 Pipeline delivery ratio unmonitored

**Severity**: HIGH
**Why it fails**: If the OTel Collector is silently dropping 10% of traces, every SLO dashboard and alert is built on incomplete data. The pipeline degradation is invisible until SLO violations appear; at which point on-call cannot distinguish real incidents from telemetry gaps.
**Remediation**: Alert when `sum(rate(otelcol_exporter_sent_spans[5m])) / sum(rate(otelcol_receiver_accepted_spans[5m])) < 0.99` for 5 minutes. This is the single most important meta-observability alert.
**See also**: `meta-observability.md §Section A6`, `meta-observability.md §Section F Alert 1`

### Z.4 Tenant ID absent from multi-tenant telemetry

**Severity**: HIGH
**Why it fails**: In a multi-tenant system, telemetry without `tenant.id` makes per-tenant SLO computation, chargeback, and incident isolation impossible. All tenants are indistinguishable in dashboards; a single noisy tenant can mask SLO violations for the entire fleet.
**Remediation**: Propagate `tenant.id` via W3C Baggage from the API gateway through all downstream services. Emit it on every span, log record, and metric data point. Apply top-N cap (≤ 1000) when used as a metric label.
**See also**: `incident-forensics.md §2.3`, `boundaries/multi-tenant.md`

### Z.5 Incident forensics without 6-dimension MRA attributes

**Severity**: HIGH
**Why it fails**: Missing Minimum Required Attributes (`service.name`, `service.namespace`, `service.version`, `deployment.environment`, `cloud.region`, `k8s.pod.name`) break the 6-dimension narrowing flow; Code / Service / Layer / Host / Region / Infra pivots fail silently.
**Remediation**: Enforce MRA completeness at CI via an OTel attribute coverage gate. Set all resource attributes via `OTEL_RESOURCE_ATTRIBUTES` or the OTel Collector `resource` processor. Validate with `otelcol` debug exporter in staging before production rollout.
**See also**: `incident-forensics.md §2 Minimum Required Attributes`

### Z.6 Cost dashboard accessible to all engineers without RBAC

**Severity**: MEDIUM
**Why it fails**: Per-tenant cost data reveals revenue tier, contract value, and resource consumption patterns. Exposing this to all engineers violates least-privilege access principles and may constitute a data breach under GDPR if cost data is linked to identifiable customers.
**Remediation**: Separate cost dashboards by role: Finance sees full cost by tenant; Platform Engineering sees cost by namespace/workload; Application Engineering sees only their own service. Apply Grafana folder permissions or OPA policies.
**See also**: `signals/cost.md §9 Privacy & Access Control`

---

## Contribution Protocol

When adding an entry from another doc:

1. Place it in the correct section (A–Z) by primary concern.
2. If the same anti-pattern appears in multiple source files, one canonical entry only; list all sources in "See also".
3. Use the format: `### {Section}.{n} {Pattern name}` / `**Severity**` / `**Why it fails**` / `**Remediation**` / `**See also**`.
4. Order within each section: CRITICAL → HIGH → MEDIUM → LOW.
5. All cross-references use relative paths from the `resources/` root.
6. No forward references to planned but unwritten files.
