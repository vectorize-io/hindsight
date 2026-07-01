---
otel_semconv: "1.27.0 (2024-11); GDPR: 2016/679; PIPA: 2023 amendment"
---

# Privacy in Telemetry

## 1. Scope and Distinction from Audit

**Privacy** answers: *what personal data flows through telemetry, and how do we minimize, redact, or anonymize it?*

**Audit** answers: *what actions occurred, who performed them, and can we prove it?*

These are architecturally opposite concerns. See Design Decision D5 for the split rationale.

| Dimension | Privacy | Audit |
|-----------|---------|-------|
| Goal | Collect less, protect subjects | Prove what happened |
| Mutability | Supports deletion (right to erasure) | Immutable, append-only |
| Retention | Short raw tier; long anonymized tier | Long-term (7–10 years typical) |
| Consumer | DPO, privacy engineers, legal | Security ops, compliance, auditors |
| Regulation driver | GDPR Art. 5, Art. 17; PIPA § 21 | SOX, PCI-DSS, ISO/IEC 27001 |

**D5 Split rationale.** Merging privacy and audit into a single signal pipeline produces conflicting retention policies (erasure vs. immutability), access control models (broad deletion rights vs. tamper-evident storage), and regulatory obligations. Separating them lets each pipeline satisfy its own obligations without compromise. Cross-references between the two are explicit rather than implicit (see §9 and `audit.md`).

Primary consumers of this file: Data Protection Officers (DPO), privacy engineers, legal counsel.

---

## 2. Regulatory Drivers

| Regulation | Jurisdiction | Key Articles / Principles | Max Penalty |
|------------|-------------|--------------------------|-------------|
| GDPR | EU / EEA | Art. 5(1)(c) minimization; Art. 5(1)(e) storage limitation; Art. 17 erasure; Art. 32 security | 4% global annual turnover |
| PIPA | South Korea | § 3 minimization; § 21 destruction; § 29 safety measures; separate sensitive data consent | KRW 300M or 3% revenue |
| CCPA / CPRA | California, US | Opt-out of sale; right to delete; sensitive PI category | USD 7,500 per intentional violation |
| HIPAA | United States | PHI definition; minimum necessary standard; breach notification | USD 100–50,000 per violation |
| PIPEDA | Canada | Fair information principles; breach of security safeguards | CAD 100,000 per violation |
| LGPD | Brazil | Art. 6 finality + necessity; Art. 18 erasure rights | 2% Brazil revenue, max BRL 50M |

**Critical misclassification risk.** Claiming "anonymization" for data that is in fact pseudonymized (reversible with a key) constitutes a GDPR violation. Regulators have levied 4% penalties on this misclassification. See §3 for the definitive distinction.

---

## 3. Anonymization vs Pseudonymization vs Tokenization

The reversibility question is the only legally meaningful test: *Can you recover the original value with information you hold?*

| Technique | Reversible? | GDPR Applies? | Typical Tooling |
|-----------|-------------|---------------|-----------------|
| Anonymization | No; irreversible | No | k-anonymity, differential privacy, aggregation, generalization |
| Pseudonymization | Yes; with key/salt | Yes | HMAC+salt, format-preserving encryption (FPE), AES-FF1 |
| Tokenization | Yes; with vault lookup | Yes | Payment token vaults, HSM-backed services |
| Hashing (no salt, low entropy) | Yes; brute-force feasible | Yes (treated as pseudonymization) | SHA-256 without salt on numeric IDs |

**Decision rule.** Ask "Could we reverse this if compelled?" If no, it is anonymization and GDPR does not apply to the result. If yes (even theoretically), GDPR applies and you have pseudonymization obligations.

**Key separation.** GDPR Art. 32 requires that pseudonymization keys be stored separately from the pseudonymized data, with independent access control and audit trail.

---

## 4. Common PII in Telemetry

| PII Category | Where It Appears in Telemetry | OTel Semconv Attribute |
|-------------|------------------------------|------------------------|
| IP address | HTTP server spans, logs, network metrics | `client.address`, `network.peer.address` |
| User ID | Span attributes, baggage, log body | `enduser.id` (deprecated → `user.id`) |
| Email, name, phone | Log body, error traces, baggage | None; must not be added |
| Session token / cookie | HTTP headers, span attributes | `http.request.header.*` |
| Browser user-agent | HTTP server spans | `user_agent.original` |
| Geolocation | IP-derived enrichment, custom attrs | `geo.city`, `geo.country_iso_code` |
| Query string parameters | URL full string | `url.full`, `url.query` |
| Stack trace with user data | Exception events | `exception.stacktrace` |
| Request body on error | Trace event attributes | Custom; avoid entirely |
| Baggage values | W3C Baggage header, propagated downstream | See `../boundaries/cross-application.md §Baggage rules` |

Baggage is especially dangerous: values propagate across service boundaries and can be logged or traced by any downstream collector. Apply trust-boundary filters before injecting user-derived values into baggage.

---

## 5. PII Handling Rules by Field

| Field | Required Action | Technique |
|-------|----------------|-----------|
| IPv4 address | Mask last octet (`192.0.2.0`) or drop | Truncation / drop |
| IPv6 address | Mask last 80 bits or drop | Truncation / drop |
| Email address | Drop local part (`***@example.com`) or drop entirely | Redaction |
| User ID (internal) | Salted HMAC with vault-managed key | Pseudonymization |
| Card number, SSN | Drop entirely; never collect | Drop |
| Phone number | Drop entirely | Drop |
| Session token, password, `Authorization` header | Drop entirely | Drop |
| Timestamp (high-resolution) | Truncate to minute (metrics) or hour (logs) depending on aggregation tier | Generalization |
| Geolocation (precise) | Truncate to city or region | Generalization |
| User-agent string | Keep major browser family only, or drop | Generalization / drop |
| Query string | Drop or allowlist known safe params | Drop / allowlist |
| Request body | Drop on error traces; never capture by default | Drop |

---

## 6. OTel Collector Processors for Redaction

Use a layered pipeline: **attributes** (known PII keys) → **transform/OTTL** (pattern-based) → **redaction** (safety-net allowlist) → storage.

### 6.1 `attributes` Processor: Named Key Actions

```yaml
processors:
  attributes/minimize:
    actions:
      - key: user.email
        action: delete
      - key: user.id
        action: hash
      - key: http.request.header.authorization
        action: delete
      - key: http.request.header.cookie
        action: delete
      - key: url.query
        action: delete
```

> **Warning on `action: hash`**: the Collector's built-in `hash` action does not apply a salt. For low-entropy inputs such as numeric user IDs, an unsalted hash is reversible by brute force and is treated as pseudonymization (not anonymization) under GDPR. See §8 for salted-hash alternatives at the SDK layer.

### 6.2 `transform` Processor (OTTL): Pattern-Based Transformation

```yaml
processors:
  transform/ip_mask:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          # IPv4: replace last octet with .0
          - replace_pattern(attributes["client.address"], "\\.\\d+$", ".0")
          # Drop precise geolocation
          - delete_key(attributes, "geo.coordinates")
          # Truncate email to domain only
          - replace_pattern(attributes["user.email"], "^[^@]+", "***")
```

### 6.3 `redaction` Processor: Allowlist Safety Net

```yaml
processors:
  redaction/allowlist:
    allow_all_keys: false
    allowed_keys:
      - http.method
      - http.status_code
      - http.route
      - service.name
      - trace.id
      - span.id
      - db.system
      - rpc.system
    blocked_values:
      - "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"   # email pattern
      - "\\b(?:\\d[ -]?){13,16}\\b"                                    # card number pattern
    summary: debug
```

### 6.4 Pipeline Assembly

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors:
        - attributes/minimize
        - transform/ip_mask
        - redaction/allowlist
        - batch
      exporters: [otlp/backend]
```

---

## 7. SDK-Layer Redaction (Prefer over Pipeline)

Redacting at the instrumentation layer is always preferable: PII never enters the pipeline or wire.

| SDK / Tool | Mechanism |
|------------|-----------|
| OTel SDK (all languages) | Attribute processor on `TracerProvider` at construction time |
| Sentry | `before_send` / `before_send_transaction` callbacks |
| Datadog | `before_send` callback; `obfuscation_config` in agent |
| OpenTelemetry JS/Python | `SpanProcessor` with custom `onStart` / `onEnd` |

**Rule.** SDK-layer redaction is the first line of defense. Collector processors are the second line. Treat collector processors as a safety net, not the primary mechanism.

---

## 8. Salted Hashing Caveats

OTel Collector's built-in `hash` action uses SHA-256 **without salt**. On low-entropy inputs (numeric user IDs, 6-digit codes), this is reversible via brute-force rainbow tables.

**Requirements for safe pseudonymization:**

1. Apply salt using a vault-managed key (e.g., HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager).
2. Store the salt/key in a separate region and access-controlled vault; never co-located with the pseudonymized data (GDPR Art. 32).
3. Rotate the salt on a defined schedule (quarterly recommended); document rotation in the key management policy.
4. Use HMAC-SHA256 with the salt, not bare SHA-256.
5. Because the Collector `hash` action cannot inject vault-managed salts, perform HMAC at the **SDK layer** before emitting spans.

```python
# SDK-layer HMAC pseudonymization (Python example)
import hmac, hashlib, os

def pseudonymize_user_id(user_id: str, vault_key: bytes) -> str:
    return hmac.new(vault_key, user_id.encode(), hashlib.sha256).hexdigest()
```

---

## 9. Retention and Right to Erasure (GDPR Art. 17)

| Tier | Content | Max Retention | Erasure Support |
|------|---------|---------------|-----------------|
| Raw (with PII) | Full spans, logs, unredacted | 7 days | Automated on data-subject request |
| Redacted short | PII removed, full attributes | 30 days | N/A (no PII present) |
| Anonymized long | Aggregated metrics, anonymized traces | 90+ days | N/A (irreversible) |
| Audit events | Immutable access/action log | 7–10 years | Not erasable (legal obligation) |

Erasure pipeline: data-subject request → identity verification → lookup raw-tier records by pseudonymized ID → delete → log erasure action in audit trail.

Cross-reference: `../meta-observability.md §Retention Matrix` for full retention policy. Cross-reference: `audit.md` for erasure audit trail requirements.

---

## 10. Cross-Border Transfer

GDPR Chapter V prohibits transfer of personal data to non-adequate countries without a legal mechanism.

| Mechanism | When to Use |
|-----------|-------------|
| Standard Contractual Clauses (SCC) | EU → US, EU → most non-adequate countries |
| Binding Corporate Rules (BCR) | Intra-group transfers |
| Adequacy decision | EU → UK, Japan, Canada (PIPEDA), South Korea (partial) |
| PIPA § 17 | KR → non-KR: contractual safeguards + PIPC notification |

**Collector routing for regional compliance.**

```yaml
# Route EU telemetry to EU backend, KR to KR backend
connectors:
  routing:
    default_pipelines: [traces/global]
    table:
      - statement: attributes["deployment.region"] == "eu-west-1"
        pipelines: [traces/eu]
      - statement: attributes["deployment.region"] == "ap-northeast-2"
        pipelines: [traces/kr]
```

Cross-reference: `../transport/collector-topology.md §Multi-cluster + regional`.

---

## 11. TLS Context Attributes (L6 Development Status)

OTel `tls.*` semantic conventions are **Development** stability tier as of semconv 1.27.0. See `../standards.md` for stability tier definitions.

| Attribute | Type | Use Case |
|-----------|------|----------|
| `tls.protocol.version` | string | Detect TLS 1.0/1.1 downgrade attacks |
| `tls.cipher` | string | Flag weak ciphers (RC4, DES, 3DES) |
| `tls.established` | boolean | Alert on handshake failures |
| `tls.resumed` | boolean | Session resumption analysis |
| `tls.server.certificate.expiry` | int (epoch) | Certificate expiry alerting |

**Scope boundary.** These attributes provide *security context* for observability purposes (downgrade detection, cert expiry alerting). Full TLS inspection (decrypting payload for content analysis) is out of scope; use dedicated network inspection tooling.

**Privacy note.** TLS cipher and protocol version are not PII. Certificate subject fields (CN, SAN) may contain hostnames; do not log end-user certificate subjects from mTLS without legal basis.

---

## 12. Backend Access Control (RBAC on Observability Data)

Production traces and logs containing even pseudonymized data are sensitive. Access must be role-gated.

| Role | Scope | Time Range | Example Tools |
|------|-------|-----------|---------------|
| On-call engineer | Own service traces + logs | Last 24h | Grafana folder permission |
| Incident responder | Cross-service traces | Incident window | Datadog Teams, Honeycomb environments |
| Security analyst | All services, security events | 30 days | Separate security-tier datasource |
| Auditor | Audit events only (see `audit.md`) | Full retention | Read-only audit index |
| Finance | Cost metrics only | 90 days | See `cost.md` |
| DPO / Privacy engineer | Redaction pipeline config, raw tier | Short window only | Separate admin role |

**Environment isolation.** Production backends must be strictly separated from non-production. Never allow dev/staging pipelines to receive or store production telemetry; this is a direct PII leak path.

Vendor-specific RBAC: Grafana folder permissions + team sync; Datadog Teams with scoped monitors; Honeycomb environments with team-scoped API keys.

---

## 13. Third-Party Processor Obligations (GDPR Art. 28)

Observability vendors receiving telemetry containing personal data are **data processors** under GDPR. A Data Processing Agreement (DPA) is mandatory before routing data to them.

| Obligation | Action Required |
|------------|----------------|
| DPA in place | Signed before any data flows to vendor |
| Sub-processor transparency | Vendor must disclose and notify of sub-processor changes |
| Data residency | Confirm vendor's storage region matches your compliance obligations |
| Deletion / return | Vendor must support data deletion on contract termination |

Common vendors requiring DPA review: Datadog, Honeycomb, Grafana Cloud, Sentry, New Relic, Elastic Cloud. Each vendor publishes a DPA at their legal/privacy page. Reference your legal team's approved vendor list before routing production data.

---

## 14. Matrix Coverage Notes

The following cells in `../matrix.md` have privacy-specific guidance from this file:

| Layer × Boundary | Privacy Status | Notes |
|-----------------|---------------|-------|
| L3 (Network) × any | Warning | IP addresses are GDPR Art. 4(1) personal data identifiers; mask or drop |
| L4 (Transport) × any | Warning | Connection 5-tuple (src IP, dst IP, ports) can identify individuals |
| L7 (Application) × multi-tenant | Primary coverage | Per-tenant data minimization; tenant ID must not leak across boundaries |
| L7 × cross-application | Warning | Baggage crosses trust boundaries; apply filter at ingress (§4) |
| Release / feature flags × any | Overlap | Feature flag cohort membership may constitute profiling under GDPR Art. 4(4) |

---

## 15. Anti-Patterns

> These entries feed into `../anti-patterns.md §Section A Privacy`.

| Anti-Pattern | Risk | Remediation |
|-------------|------|-------------|
| Claiming "anonymization" for pseudonymized data | 4% GDPR penalty; regulatory misrepresentation | Apply reversibility test (§3); re-classify and update ROPA |
| Hash without salt on low-entropy IDs | Rainbow-table reversal; GDPR breach | HMAC-SHA256 with vault-managed salt at SDK layer (§8) |
| Salt/key stored in same region as pseudonymized data | Key compromise = full de-anonymization | Separate vault in independent region with independent IAM (GDPR Art. 32) |
| Baggage crossing trust boundary without filter | PII propagated to untrusted downstream collectors | Apply baggage allowlist at service ingress (cross-ref `../boundaries/cross-application.md`) |
| Observability backend open to all engineers (no RBAC) | PII exposure; compliance failure | Implement role-scoped access per §12 |
| Routing to 3rd-party vendor without DPA | GDPR Art. 28 violation | Block data flow until DPA is signed (§13) |
| Capturing raw request body on error trace | Request bodies frequently contain credentials, PII | Drop request body by default; allowlist specific non-sensitive fields at SDK layer (§7) |
| Cross-region telemetry routing without GDPR mechanism | GDPR Chapter V violation | Implement routing connector per §10; confirm SCC or adequacy decision |
| OTel Collector `hash` action on production user IDs | Unsalted; brute-force reversible | Migrate to SDK-layer HMAC with vault key (§8) |
| Logging `user_agent.original` without assessment | User-agent can be a unique identifier for profiling | Generalize to browser family only or drop (§5) |

---

## References

- GDPR text: <https://gdpr-info.eu>
- PIPA (Korean): <https://www.pipc.go.kr>
- OTel sensitive data handling: <https://opentelemetry.io/docs/security/handling-sensitive-data/>
- OTel Collector processors: <https://opentelemetry.io/docs/collector/configuration/>
- W3C Baggage PII warning: <https://www.w3.org/TR/baggage/>
- Cross-references: `../standards.md`, `../matrix.md`, `../meta-observability.md`, `../boundaries/cross-application.md`, `../transport/collector-topology.md`, `audit.md`, `cost.md`
