---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
tools:
  - "@opentelemetry/sdk-trace-web: 1.x (stable); browser instrumentations: some experimental"
notes:
  - "web-vitals JS library: 4.x (2024); INP replaced FID: March 2024"
---

# Web RUM: Real User Monitoring

## 1. Scope

This file covers browser-side Real User Monitoring (RUM) for web applications.

**In scope:** performance signals (Core Web Vitals, Navigation Timing, custom marks), error tracking (`window.onerror`, unhandled promise rejections, source map correlation), user interaction (click, navigation, long tasks), 3rd-party script impact and CSP violation reporting, synthetic monitoring (scheduled browser/API probes), client-to-server trace correlation via W3C `traceparent`, session replay (vendor overview).

**Out of scope:** mobile RUM (see `mobile-rum.md`); native app crash analytics (see `crash-analytics.md`).

---

## 2. Core Web Vitals (2024 Update)

Google updated the Core Web Vitals (CWV) set in March 2024. **INP (Interaction to Next Paint) replaced FID (First Input Delay)** as the official responsiveness metric. FID is deprecated; do not report it in new dashboards or SLOs.

Reference implementation: `web-vitals` JS library (`google/web-vitals`, v4.x).

### Official CWV Targets

| Metric | Full name | Dimension | Good | Needs improvement | Poor |
|--------|-----------|-----------|------|-------------------|------|
| **LCP** | Largest Contentful Paint | Loading performance | ≤ 2.5 s | 2.5 s – 4.0 s | > 4.0 s |
| **INP** | Interaction to Next Paint | Responsiveness (replaces FID) | ≤ 200 ms | 200 ms – 500 ms | > 500 ms |
| **CLS** | Cumulative Layout Shift | Visual stability | ≤ 0.1 | 0.1 – 0.25 | > 0.25 |

### Additional Operational Signals (not official CWV)

| Metric | Meaning | Typical target |
|--------|---------|---------------|
| TTFB | Time to First Byte; server + network latency | ≤ 800 ms |
| FCP | First Contentful Paint; when first content is painted | ≤ 1.8 s |
| TTI | Time to Interactive; main thread unblocked | ≤ 3.8 s (3G) |

Use `web-vitals` to collect all signals with a uniform API:

```js
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics({ name, value, rating, id }) {
  // forward to your RUM backend or OTel OTLP endpoint
  navigator.sendBeacon('/rum', JSON.stringify({ name, value, rating, id }));
}

onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);
```

SLI recommendation: use **p75** for each CWV metric as the SLI value (aligns with Google Search Console scoring). Cross-reference: `../../boundaries/slo.md` for SLO burn-rate alert configuration.

---

## 3. Browser OpenTelemetry SDK

The OTel JavaScript SDK provides browser instrumentation for traces and metrics. Logs from the browser are currently in progress (no stable browser LogRecord exporter as of semconv 1.27.0).

**Stability note:** `@opentelemetry/sdk-trace-web` and `@opentelemetry/sdk-metrics` are stable. Some browser instrumentation packages (e.g., Core Web Vitals plugin for OTel) remain experimental. Do not build production SLOs on experimental instrumentations without a fallback to `web-vitals` directly.

### Typical Instrumentations

| Package | What it instruments | Status |
|---------|-------------------|--------|
| `@opentelemetry/instrumentation-document-load` | Navigation timing, resource timing | Stable |
| `@opentelemetry/instrumentation-user-interaction` | Click events → spans | Stable |
| `@opentelemetry/instrumentation-fetch` | `fetch()` calls + `traceparent` injection | Stable |
| `@opentelemetry/instrumentation-xml-http-request` | `XMLHttpRequest` + header injection | Stable |

### Minimal SDK Initialization

```js
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load';
import { UserInteractionInstrumentation } from '@opentelemetry/instrumentation-user-interaction';
import { FetchInstrumentation } from '@opentelemetry/instrumentation-fetch';
import { W3CTraceContextPropagator } from '@opentelemetry/core';
import { CompositePropagator } from '@opentelemetry/core';
import { Resource } from '@opentelemetry/resources';

const provider = new WebTracerProvider({
  resource: new Resource({
    'service.name': 'my-web-app',
    'service.version': import.meta.env.VITE_RELEASE_VERSION, // CI-injected
    'deployment.environment': import.meta.env.MODE,
  }),
});

provider.addSpanProcessor(
  new BatchSpanProcessor(
    new OTLPTraceExporter({ url: 'https://otel-collector.example.com/v1/traces' })
  )
);

provider.register({
  propagator: new CompositePropagator({
    propagators: [new W3CTraceContextPropagator()],
  }),
});

registerInstrumentations({
  instrumentations: [
    new DocumentLoadInstrumentation(),
    new UserInteractionInstrumentation(),
    new FetchInstrumentation({
      // Allow traceparent injection only to trusted origins (see §5)
      propagateTraceHeaderCorsUrls: [/https:\/\/api\.example\.com/],
    }),
  ],
});
```

**Session and view identifier strategy:** Generate a `session.id` UUID on tab open (stored in `sessionStorage`) and a `view.id` UUID on each route change. Attach both as span attributes. This enables session-level RUM aggregation without a vendor SDK.

---

## 4. Vendor Options

Vendors below are examples as of 2026-Q2. This is not a registry; verify currency at `https://landscape.cncf.io` and `../../vendor-categories.md §RUM category`.

| Vendor | Bundle size | Session replay | OTel-based | Full-stack correlation | Notes |
|--------|------------|---------------|-----------|----------------------|-------|
| **Sentry Browser** | Medium | Yes (opt-in) | Partial | Strong | Error-first; release tracking; delegation: `getsentry/sentry-sdk-setup` |
| **Grafana Faro** | Lightweight | No | Yes | Grafana stack | OSS; emits OTel-compatible signals; pairs with LGTM+ |
| **Datadog RUM** | Heavy | Yes | Partial | APM link | Strongest APM-to-RUM pivot; `allowedTracingUrls` CORS config |
| **Bugsnag** | Light | No | No | Partial | Stability-focused; good grouping; SmartBear owned |

For vendor-specific setup, delegate to the vendor skill listed or use `oma-search --docs "{vendor} browser RUM setup"`.

---

## 5. Client-to-Server Error Correlation

**Problem:** A server-side 5xx error causes the browser to retry. Without shared trace context, each retry appears as an independent request in backend traces. The root cause (one bad deploy) generates a cascade that looks like independent failures.

**Solution:** emit the same `trace_id` on both sides using W3C `traceparent`.

The `FetchInstrumentation` (OTel JS) and Datadog RUM inject `traceparent` automatically into outbound requests; but only for origins listed in the CORS allowlist configuration.

```js
// OTel JS SDK — propagateTraceHeaderCorsUrls
new FetchInstrumentation({
  propagateTraceHeaderCorsUrls: [
    /https:\/\/api\.example\.com/,
    /https:\/\/gateway\.example\.com/,
  ],
});

// Datadog RUM — allowedTracingUrls
datadogRum.init({
  allowedTracingUrls: [
    { match: 'https://api.example.com', propagatorTypes: ['tracecontext'] },
  ],
});
```

The server must emit the same `trace_id` in its own spans. When the browser console shows an error, the `trace_id` links directly to the backend trace in Sentry Performance, Datadog APM, or any OTel-compatible backend.

**Anti-pattern to avoid:** client retry loop on 5xx without a circuit breaker. Retrying immediately amplifies server load. Implement exponential backoff + circuit breaker in the fetch layer before enabling distributed tracing; otherwise the correlation data documents a cascading failure, not a single event. Cross-reference: anti-patterns §Section G below and `../../signals/traces.md` for retry trace patterns.

---

## 6. Third-Party Scripts and CSP

3rd-party scripts are the dominant cause of LCP regression and a primary XSS / supply-chain attack vector. Observability of 3rd-party scripts requires both performance attribution and security violation reporting.

### Content Security Policy Violation Reporting

```http
Content-Security-Policy: default-src 'self';
  script-src 'self' https://cdn.trusted-vendor.com;
  report-to csp-violations;
  report-uri https://csp-reports.example.com/collect

Reporting-Endpoints: csp-violations="https://csp-reports.example.com/collect"
```

> `report-uri` is the legacy fallback directive for browsers that do not yet support the Reporting API Level 1 (`report-to`). Firefox gained full `Reporting-Endpoints` support in 2023; keeping both ensures older stable channels still deliver violation reports.

CSP violations are reported to your server endpoint as JSON. Pipe them to your log backend and alert on new `blocked-uri` origins; these indicate either a new 3rd-party load attempt or a supply-chain injection.

### Subresource Integrity (SRI)

Pin the exact hash of any 3rd-party CDN script to detect tampered deliveries:

```html
<script
  src="https://cdn.trusted-vendor.com/lib.min.js"
  integrity="sha384-<base64-hash>"
  crossorigin="anonymous"
></script>
```

### Performance Attribution

Use the Resource Timing API (`PerformanceObserver` type `resource`, `buffered: true`) to attribute LCP and INP delays to specific 3rd-party origins by comparing `new URL(entry.name).origin` against `window.location.origin`. Report per-origin `entry.duration` to your RUM backend.

Cross-reference: `../../signals/privacy.md` for 3rd-party cookie and tracking signal rules; `../../anti-patterns.md §Section G` for the "unmonitored 3rd-party script" anti-pattern.

---

## 7. Synthetic Monitoring

Synthetic monitoring provides an "outside-in" view that complements RUM. RUM shows what real users experience; synthetic shows what a deterministic probe experiences from a specific region, at any time; including before any real user visits.

| Tool | Type | Best for |
|------|------|---------|
| **Checkly** | Playwright-based browser checks | Full user-journey verification; integrates with CI/CD |
| **Grafana k6** | Script-based load + synthetic | Load testing + baseline synthetic probes in one tool |
| **Prometheus Blackbox Exporter** | HTTP/TCP/ICMP probe | Lightweight availability checks; PromQL-native alerting |
| **Datadog Synthetics** | Browser + API | Unified with Datadog RUM; managed SaaS |
| **Pingdom** | Managed HTTP probe | Simple uptime monitoring; low operational overhead |

### When to Use RUM vs Synthetic

| Scenario | Use |
|----------|-----|
| Understand real user performance distribution | RUM |
| Detect issues before users notice | Synthetic |
| Low-traffic page that may have issues | Synthetic (RUM has too few samples) |
| Geo-distribution of performance | Both (synthetic per region + RUM aggregated) |
| SLO availability measurement | Synthetic (deterministic; not subject to sampling) |
| Investigating a real user complaint | RUM session + trace correlation |

Recommendation: run synthetic probes from at least 3 regions. Alert on synthetic probe failures with a tighter SLA than RUM-based alerts (synthetic = canary; RUM = ground truth).

---

## 8. Session Replay

Session replay records DOM mutations to reconstruct a visual playback. Vendors include Sentry Session Replay, Datadog Session Replay (both mask all inputs by default), FullStory, Hotjar, and LogRocket.

**Privacy requirements:**
- Email addresses, credit card fields, and `input[type=password]` MUST be masked client-side before the payload is sent. Do not rely on server-side masking alone.
- Session replay consent must be tied to the cookie consent flow (GDPR Article 6(1)(a)).
- Cross-reference: `../../signals/privacy.md` for replay sanitization rules and PII classification.

---

## 9. Error Tracking

```js
// Global error handler
window.addEventListener('error', (event) => {
  myRumSdk.captureException(event.error, {
    'span.id': currentSpan?.spanContext().spanId,
    'trace.id': currentSpan?.spanContext().traceId,
  });
});

// Unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  myRumSdk.captureException(event.reason);
});
```

**Source map uploads:** Minified production bundles produce unreadable stack traces. Upload source maps to your error tracking vendor on every release.

```bash
# Example: Sentry CLI source map upload (run in CI after build)
sentry-cli sourcemaps inject ./dist
sentry-cli sourcemaps upload --org my-org --project my-web ./dist
```

Cross-reference: `../../boundaries/release.md` for CI integration patterns; source map upload should be gated on the same pipeline step as container image push.

---

## 10. Browser-Specific Observability APIs

| API | Purpose |
|-----|---------|
| `performance.mark()` / `performance.measure()` | Custom timing for business-critical flows (e.g., checkout duration as CWV supplement) |
| Long Task API (`PerformanceObserver` type `longtask`) | Detect main-thread blocking events > 50 ms; correlate with INP failures |
| Navigation Timing API (`performance.getEntriesByType('navigation')`) | Full page load breakdown: DNS, TCP, TLS, TTFB, DOMContentLoaded, load |

Use `performance.mark('flow-start')` / `performance.measure('flow-duration', 'flow-start', 'flow-end')` to emit named timing spans that complement automated CWV collection.

---

## 11. Matrix Coverage (L7 Row, Web Slice)

These cells from `../../matrix.md` are the primary drivers for this file:

| Matrix cell | Symbol | Artifact |
|------------|--------|---------|
| L7 × multi-tenant × metrics | PASS | Per-tenant CWV distribution (LCP p75, INP p75, CLS p75 segmented by `tenant.id`) |
| L7 × cross-application × traces | PASS | Primary: `traceparent` propagated via `propagateTraceHeaderCorsUrls` / `allowedTracingUrls` |
| L7 × SLO × metrics | PASS | CWV as SLI; LCP p75 ≤ 2.5s, INP p75 ≤ 200ms, CLS p75 ≤ 0.1 (see `../../boundaries/slo.md`) |
| L7 × release × traces | PASS | `service.version` on spans carries frontend build SHA + backend version for correlated release analysis |
| L7 × privacy × logs | PASS | PII masking in error stack traces and session replay; no `user.email` in metric labels |

---

## 12. Anti-Patterns (Section G: Frontend/Mobile Candidates)

These are candidates for `../../anti-patterns.md §Section G Frontend/Mobile`:

| # | Anti-pattern | Impact | Remedy |
|---|-------------|--------|--------|
| G1 | 3rd-party script loaded without CSP monitoring | Silent XSS or supply-chain injection; no alert | Add `Content-Security-Policy` with `report-to`; pipe violations to log backend |
| G2 | Source maps not uploaded to error vendor | Stack traces are unreadable minified symbols in production | Upload source maps on every CI release pipeline step |
| G3 | Client retry loop on 5xx without circuit breaker | Backend saturation cascade from amplified retry storm | Implement exponential backoff + client-side circuit breaker before enabling trace correlation |
| G4 | `user.email` as metric label | Cardinality explosion + PII violation (GDPR Article 5(1)(c)) | Use opaque `user.id` (stable hash) or remove user dimension from metrics |
| G5 | FID still reported in dashboards after March 2024 | Stale metric; no longer part of CWV; misleads SLO reviews | Replace FID with INP in all dashboards and SLO definitions |
| G6 | `propagateTraceHeaderCorsUrls` / `allowedTracingUrls` not configured | Browser CORS preflight rejects `traceparent` injection; client-server correlation silently broken | Add API origins to the SDK CORS allowlist configuration |
| G7 | Session replay without client-side PII masking | PII (email, card numbers) captured in replay payload before masking | Enable input masking in SDK config; do not rely on server-side redaction alone |

---

## 13. References


Internal cross-references:
- `../../standards.md`: normative semconv stability tiers and W3C Trace Context requirements
- `../../matrix.md`: full 112-cell coverage map (L7 row)
- `../../vendor-categories.md`: RUM vendor category taxonomy and delegation targets
- `../../anti-patterns.md`: full anti-pattern registry (Section G: Frontend/Mobile)
- `../../boundaries/slo.md`: OpenSLO definitions for CWV SLIs and burn-rate alerts
- `../../boundaries/release.md`: source map upload CI integration and `service.version` tagging
- `../../boundaries/cross-application.md`: full propagator compatibility matrix (for `traceparent` CORS allowlist guidance)
- `../../signals/privacy.md`: PII masking rules, 3rd-party tracking signals, replay sanitization
- `../../signals/traces.md`: server-side trace patterns for backend correlation
- `../mesh.md`: mesh propagator headers (cross-reference for `traceparent` propagation chain)
- `mobile-rum.md`: mobile RUM (offline-first queuing, battery, app lifecycle)
- `crash-analytics.md`: native crash symbolication and CFR tracking

## References

- Core Web Vitals (2024): <https://web.dev/articles/vitals>
- INP replaces FID: <https://web.dev/blog/inp-cwv>
- `web-vitals` JS library: <https://github.com/GoogleChrome/web-vitals>
- OTel JS browser instrumentation: <https://opentelemetry.io/docs/languages/js/getting-started/browser/>
- OTel frontend instrumentation guide: <https://elastic.co/observability-labs/blog/web-frontend-instrumentation-with-opentelemetry>
- Datadog RUM browser SDK: <https://docs.datadoghq.com/real_user_monitoring/browser/>
- Sentry Browser SDK: <https://docs.sentry.io/platforms/javascript/>
- Grafana Faro: <https://grafana.com/oss/faro/>
