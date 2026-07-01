---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11); no stable WAF-specific semconv group; custom `waf.*` attributes used below"
tools:
  - "ModSecurity v3 (libmodsecurity3): mature; OWASP CRS 4.x"
  - "Coraza v3.x: OWASP-led, Go-native; Envoy/Caddy/HAProxy integrations"
  - "AWS WAF v2; Cloudflare WAF; Akamai App & API Protector; Imperva; F5 NGINX App Protect"
---

# WAF Rule Observability

## 1. Scope

This file covers Web Application Firewall (WAF) rule observability: rule hit rate, block/false-positive trends, edge latency overhead, rule-change correlation, and how WAF actions propagate into downstream service health.

**In scope:** rule hit and block telemetry, OWASP CRS paranoia level effects, edge-tier latency attribution, false-positive (FP) and false-negative (FN) tracking, WAF rule release markers, cascading 4xx storms caused by FP surges, WAF-to-application trace correlation, bypass-attempt audit, multi-tenant rule attribution.

**Out of scope:** WAF rule authoring strategy (use OWASP CRS docs and vendor playbooks); DDoS mitigation orchestration (use vendor anti-DDoS such as Cloudflare Magic Transit, AWS Shield Advanced); TLS-inspection content analysis (see SKILL.md OOS table for Wireshark / Cloudflare Radar); bot management product evaluation (vendor-specific).

Cross-references:
- L3 firewall and security-group rule visibility: `../L3-network.md §3 VPC flow logs §6.4 Security Observability`
- L4 UDP/QUIC firewall fallback: `../L4-transport.md §7.4 Firewall and Enterprise Deployment`
- Mesh ingress + Envoy + Coraza WAF integration: `../mesh.md`
- SIEM/enterprise log routing of WAF events: `../../vendor-categories.md §(e) SIEM / Enterprise Logs`
- WAF block as audit signal: `../../signals/audit.md`
- IP in WAF logs as PII: `../../signals/privacy.md §IP addresses`

---

## 2. WAF Action Model

A WAF rule evaluation produces one of these actions on every request:

| Action | Meaning | Downstream effect | OTel attribute |
|--------|---------|------------------|----------------|
| `allow` | Rule did not match; request forwarded | Normal | `waf.action = "allow"` |
| `block` | Rule matched and blocked | 403 to client; no backend call | `waf.action = "block"` |
| `challenge` | Rule matched; CAPTCHA/JS challenge issued | Backend call deferred until pass | `waf.action = "challenge"` |
| `log` (detect-only) | Rule matched; request still forwarded | Backend serves normally; signal only | `waf.action = "log"` |
| `count` (vendor) | Rule matched; counter incremented; pass-through | Backend serves normally | `waf.action = "count"` |

`log` / `count` modes are the safe deploy path for new rules; never promote a rule to `block` until its FP rate is verified at a target ratio (commonly < 0.1% of traffic) over a representative window.

---

## 3. Required Attributes per WAF Event

Every WAF rule evaluation emitted as a span event or structured log record MUST carry these attributes. They make rule effectiveness, FP investigation, and cascading failure attribution tractable.

| Attribute | Format | Purpose |
|-----------|--------|---------|
| `waf.vendor` | string (`modsecurity`, `coraza`, `aws-waf`, `cloudflare`, `akamai`, `imperva`, `nginx-app-protect`) | Vendor pivot in unified dashboards |
| `waf.rule.id` | string (e.g., `942100`, `crs-942100-sqli`, `AWSManagedRulesCommonRuleSet`) | Per-rule effectiveness analysis |
| `waf.rule.set.version` | string (CRS version, AWS managed-rules version, Cloudflare ruleset SHA) | **Release correlation; FP surges align with this bump** |
| `waf.action` | enum from Section 2 | Action distribution |
| `waf.match.category` | string (`sqli`, `xss`, `lfi`, `rfi`, `rce`, `protocol`, `scanner`, `bot`, `dos`, `geo`) | Trend analysis by attack class |
| `waf.score` | int (OWASP CRS anomaly score) | Threshold investigation |
| `waf.paranoia.level` | int (1–4 for CRS) | FP rate scales with PL; track per release |
| `http.request.method` | semconv (stable) | Baseline pivot |
| `http.route` | semconv (stable) | Per-route FP isolation |
| `http.response.status_code` | semconv (stable) | `403` from WAF distinguishable from app 403 by combining with `waf.action = "block"` |
| `client.address` | semconv (stable; IP); **PII** | Apply masking per `../../signals/privacy.md` |
| `trace_id`, `span_id` | OTel context | Join WAF event → application trace |
| `service.version` | semconv (stable) | Application-side release pivot |
| `tenant.id` | W3C Baggage | Per-tenant WAF rule scoping in multi-tenant SaaS |

Note on stability: there is no OTel `waf.*` semconv group as of 2026-Q2 (semconv 1.27.0). The attribute names above are this skill's convention; they share the dot-namespace style with stable groups. If `waf.*` is later standardized, treat the above as a migration source.

---

## 4. Edge Latency Attribution

WAF runs at the edge or in-mesh and adds latency to every request. Without per-rule timing, a slow rule (regex backtracking, GeoIP lookup, IP reputation API) is invisible until it shows as a service-wide p99 spike.

| Metric | Type | Purpose |
|--------|------|---------|
| `waf_request_duration_seconds` | Histogram | Total WAF processing time per request (label by `waf.rule.set.version`) |
| `waf_rule_duration_seconds` | Histogram | Per-rule processing time (`waf.rule.id` label, capped to top-N rules) |
| `waf_regex_backtrack_total` | Counter | ReDoS-prone rules; alerts on positive trend |
| `waf_origin_call_total` | Counter | Reputation/threat-intel API calls; rate-limit-prone dependency |
| `waf_origin_call_failure_total` | Counter | When reputation API fails, fail-open vs fail-closed policy must be explicit |

**Anti-pattern:** treating WAF as a transparent zero-cost layer. ModSecurity with CRS paranoia level 4 routinely adds 5–15 ms p50; at the 99th percentile a single bad regex can add hundreds of ms. Always carry `waf.processing.duration_ms` as a span attribute on the ingress span.

---

## 5. False-Positive Surge: Cascading 4xx Storm

When a WAF rule update over-blocks legitimate traffic, every affected request returns 403 from the edge. Downstream services see request volume drop (good), but customers see errors (bad). Without WAF-aware dashboards, on-call investigates "missing traffic" rather than "WAF blocking my users".

### Signature

```
1. waf_action_total{action="block"} step-changes upward at T0.
2. http_requests_total{status="403"} at edge spikes; service-level error rate does NOT spike.
3. service_requests_total at backend drops 5–30%.
4. waf.rule.set.version recorded a change at T0 (release marker).
```

### Required correlated artifacts

| Artifact | Source | Purpose |
|----------|--------|---------|
| Edge 403 rate by `waf.rule.id` | WAF telemetry | Identify the offending rule |
| Per-customer block rate by `tenant.id` | WAF telemetry with baggage | Is this single-customer or fleet-wide? |
| `waf.rule.set.version` release marker | `../../boundaries/release.md` | Time alignment with rollout |
| Application-side `http.server.request.duration` segmented by edge-vs-backend 403 | App OTel SDK | Confirm the 403 originated at WAF, not the app |

### Forensics path

This is a Step-5-first scenario (release correlation before service pivot). See `../../incident-forensics.md §3` for the standard 6-dimension flow; for WAF FP storms invert it:

1. Check `waf.rule.set.version` change in the incident window.
2. Group `waf_action_total{action="block"}` by `waf.rule.id`; identify the rule whose block count grew.
3. Sample a blocked `trace_id` (most WAFs inject `traceparent` or `X-Amzn-Trace-Id` even on blocks); pull the original request URL and headers from the WAF log (with PII masking applied).
4. Decide: rollback the ruleset version, or shift the rule to `log` mode while a tuning patch is prepared.

`log` mode rollback is preferred over full ruleset rollback when only one rule is the offender; it preserves protection from the other rules in the same release.

### Recovery time budget

Per the incident-forensics playbook's 15-minute SLA, WAF FP rollback should complete inside the same window. Maintain a pre-approved "rule-to-detect-mode" runbook so the on-call does not need a security review to flip a rule's action during an active incident.

---

## 6. WAF as a Dependency: Fail-Open vs Fail-Closed

WAFs that consult external services (IP reputation, bot scoring, threat-intel API, GeoIP lookup) inherit those dependencies' failure modes. When the external service degrades, the WAF policy choice is consequential.

| Policy | Behavior on dependency timeout | Risk |
|--------|--------------------------------|------|
| **Fail-open** | Allow the request; emit `waf.action = "allow"` + `waf.dependency.failed = true` | Brief security gap; legitimate users unaffected |
| **Fail-closed** | Block the request; emit `waf.action = "block"` + `waf.dependency.failed = true` | Mass legitimate-user blocking when the dependency is down; full outage to your users |

Both policies are valid choices, but the choice must be observable. Required signal:

```
waf_dependency_failure_total{dependency="ip-reputation-api", policy="fail-open"} = N
```

Alert when this counter is non-zero for more than 60 s. The dependency outage may not affect users today, but it changes your security posture; on-call must know.

**Anti-pattern:** fail-closed without monitoring the dependency. A third-party reputation API can take down 100% of your traffic with no advance warning. Always emit `waf.dependency.failed` and treat its rate as an SLO input.

---

## 7. Vendor Telemetry Surfaces

Routing index only. Vendor-specific configuration (YAML, plan tiers, query syntax, pricing) is owned by each vendor's authoritative docs; this table delegates to them via `../../vendor-categories.md §(e) SIEM / Enterprise Logs`.

| Vendor | Log surface | Trace header to map | Docs |
|--------|------------|---------------------|------|
| Coraza (Envoy WASM / Caddy / HAProxy SPOE) | Mesh ingress access logs | `traceparent` (native) | <https://www.coraza.io/> |
| ModSecurity v3 | NGINX / Apache logs | `traceparent` via NGINX OTel module | <https://github.com/owasp-modsecurity/ModSecurity> |
| AWS WAF | CloudWatch Logs / Kinesis Firehose | `X-Amzn-Trace-Id` | <https://docs.aws.amazon.com/waf/latest/developerguide/logging.html> |
| Cloudflare WAF | Logpush / Logpull | `cf-ray` → W3C baggage at edge | <https://developers.cloudflare.com/logs/logpush/> |
| Akamai App & API Protector | DataStream 2 | `X-Akamai-*` → origin OTel map | <https://techdocs.akamai.com/datastream2/docs> |
| Imperva (Incapsula) | SIEM connector; CEF / LEEF | Custom session ID → origin map | <https://docs.imperva.com/bundle/cloud-application-security/page/integrations-overview.htm> |
| F5 NGINX App Protect | Splunk / CEF / syslog | NGINX OTel module passthrough | <https://docs.nginx.com/nginx-app-protect-waf/> |

Normalize all surfaces into the common `waf.*` attribute set (§3) via the OTel Collector `transform` processor before reaching the backend. WAF events are simultaneously operational telemetry and audit records.

---

## 8. Audit and Compliance

WAF events carry security significance and must be retained per the audit signal rules in `../../signals/audit.md`.

| Event class | Retention | Storage |
|-------------|-----------|---------|
| All `block` actions | 1 year | Hot or warm; queryable for FP investigation |
| All `log` (detect-only) actions | 90 days | Hot |
| `allow` actions | Sampled at 1% (tail-keep on flagged categories) | Hot |
| Authenticated bypass attempts (rule match + valid session) | 7 years | WORM |
| Rule-set version change events | 7 years | WORM; tied to release-marker stream |

Cross-ref `../../signals/audit.md §retention matrix` for the unified retention policy.

`client.address` (source IP) appears in every WAF record and is PII under GDPR Article 4(1) and PIPA. Apply pipeline-stage redaction per `../../signals/privacy.md §IP addresses` before the records reach long-term storage. Block records retain enough fidelity for forensics (e.g., `/24` prefix preserved, last octet hashed) without retaining identifiable IP after the operational window expires.

---

## 9. Multi-Tenant WAF Rule Scoping

In multi-tenant SaaS, different tenants may have different threat profiles or contractual WAF policies. Per-tenant rule attribution requires:

1. `tenant.id` propagated to the edge via baggage (`../../boundaries/cross-application.md §Baggage at trust boundaries`).
2. WAF rules scoped by tenant (vendor support varies: Cloudflare Custom Rules + tenant header match; AWS WAF rule groups per CloudFront distribution per tenant; Coraza per-route rule lists).
3. Telemetry tagged with `tenant.id` so a tenant-specific FP surge does not contaminate the global rate.

Anti-pattern: a global rule set with no tenant scoping forces every customer onto the strictest profile, which inflates global FP rate and obscures tenant-specific tuning needs. See `../../boundaries/multi-tenant.md §isolation tiers`.

---

## 10. Matrix Coverage Contribution

This file contributes to existing L7-application matrix cells. It does not add new cells (the matrix taxonomy is fixed at 112 cells per `../../matrix.md §6`).

| Matrix cell | Contribution |
|-------------|--------------|
| L7 × cross-application × logs | WAF block log records carry `trace_id` for log-trace join into the application call chain |
| L7 × cross-application × audit | WAF block stream as compliance audit feed; cross-application security boundary |
| L7 × multi-tenant × audit | Per-tenant WAF rule effectiveness and bypass-attempt evidence |
| L7 × release × metrics | `waf.rule.set.version` as release marker; ruleset rollouts visible in pre/post comparison |
| L7 × release × logs | Ruleset version change events as structured deployment records |
| L7 × {all boundaries} × privacy | `client.address` masking and selective retention per audit class |

---

## 11. Anti-Patterns (Section F: Security & Compliance Candidates)

Candidates added to `../../anti-patterns.md §F`:

| ID | Anti-pattern | Impact | Remedy |
|----|--------------|--------|--------|
| F.6 | WAF rule hit/block rate unmonitored | Cascading 4xx storms from rule updates discovered by customer complaints, not by on-call | Emit `waf_action_total{action,rule_id,rule_set_version}`; alert on block rate step change > 3σ vs 24h baseline |
| F.7 | WAF rule-set deploy without release marker | Cannot align block-rate surge with rule update; rollback decision is guesswork | Emit a structured deployment event with `waf.rule.set.version` on every ruleset rollout; treat same as `service.version` |
| F.8 | WAF fail-closed without dependency-health signal | Third-party reputation API outage silently blocks all traffic; no on-call signal | Emit `waf_dependency_failure_total` per external dependency; alert when non-zero for > 60s; document fail-open/closed policy per rule |
| F.9 | WAF logs detached from application trace context | Block at edge invisible inside application trace; on-call cannot determine "did this user even reach us?" | Configure WAF to propagate `traceparent` even on blocks; map vendor headers (`cf-ray`, `X-Amzn-Trace-Id`) to baggage at the edge |
| F.10 | WAF rules in `block` mode without prior `log`-mode soak | New rule promoted to block on day 0; FP rate unknown; high risk of mass-block incident | Mandatory minimum 24h `log`-mode soak per new rule with FP rate verified < target ratio before promotion to `block` |

These follow the existing `### F.{n}` format in `../../anti-patterns.md`.

---

## 12. References

Internal cross-references:
- `../../standards.md`: stable semconv tiers (`http.*`, `client.*`, `service.*`) used as anchors for the `waf.*` extension
- `../../matrix.md`: 112-cell coverage map; this file contributes to L7-application cells without adding new ones
- `../../incident-forensics.md`: 6-dim narrowing flow; WAF FP storms invert the order (release marker first)
- `../../anti-patterns.md`: Section F candidates F.6–F.10
- `../../vendor-categories.md §(e) SIEM / Enterprise Logs`: WAF event routing as audit records
- `../../signals/audit.md`: WAF block retention and WORM rules
- `../../signals/privacy.md`: IP-address masking pipeline for `client.address`
- `../../signals/logs.md`: structured logging contract; WAF records inherit it
- `../../boundaries/cross-application.md`: header mapping (`cf-ray`, `X-Amzn-Trace-Id`) to W3C baggage
- `../../boundaries/release.md`: WAF ruleset version as a release marker stream
- `../../boundaries/multi-tenant.md`: per-tenant rule scoping isolation tiers
- `../mesh.md`: mesh ingress integration point for in-mesh WAF
- `../L3-network.md`: firewall ACCEPT/REJECT visibility; complementary to WAF L7 visibility
- `../L4-transport.md`: UDP/QUIC firewall fallback behavior

External references (vendor URLs in §7 table are authoritative for their products):
- OWASP Core Rule Set (CRS) v4: <https://coreruleset.org/>
- W3C Trace Context: <https://www.w3.org/TR/trace-context/>
