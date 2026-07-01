---
otel_spec: "1.x (stable API/SDK)"
otel_semconv: "1.27.0 (2024-11)"
notes:
  - "OTel mobile semconv still experimental; vendor-native SDKs dominate in 2026"
---

# Mobile RUM: L7 Application Layer

---

## 1. Scope

**In scope:** Native mobile app Real User Monitoring for iOS (Swift/Obj-C), Android (Kotlin/Java), React Native, and Flutter. Covers performance, errors, app lifecycle events, network egress observability, offline telemetry queuing, and battery impact.

**Out of scope (see sibling files):**
- Mobile crash analytics specifics (symbolication pipeline, Crash-Free Rate computation, dSYM/ProGuard upload) → `crash-analytics.md`. This file references symbolication patterns but does not own the full pipeline.
- Mobile web RUM running inside an in-app WebView → `web-rum.md`. WebView telemetry is browser-runtime based and follows Web RUM patterns, not native SDK patterns.

---

## 2. OTel Mobile SDK Status (as of 2026-Q2)

OpenTelemetry mobile SDKs are less mature than backend or web SDKs. Both are experimental status:

| Platform | OTel SDK | Status |
|----------|----------|--------|
| Android | `opentelemetry-android` + `opentelemetry-java-instrumentation` | Experimental |
| iOS | `opentelemetry-swift` | Experimental |
| React Native | No official OTel SDK | Vendor SDK only |
| Flutter | No official OTel SDK | Vendor SDK only |

In practice, mobile teams use **vendor-native SDKs** and map signals to OTel concepts where possible. This file uses vendor SDKs as the primary path and notes where OTel semconv applies.


---

## 3. Offline-First Telemetry Queuing

Offline queuing is the defining mobile-specific concern. Mobile networks are intermittent, metered, roaming-constrained, and subject to airplane mode. A telemetry SDK that drops events on network failure is unsuitable for mobile.

### Queue design requirements

| Constraint | Recommended value | Rationale |
|------------|------------------|-----------|
| Disk storage cap | 1–5 MB | Balance completeness vs. user storage impact |
| Event TTL | 24–72 hours | Drop stale events before they mislead dashboards |
| Duplicate suppression | Idempotency key per event | Retry on reconnect must not produce duplicates |
| Backoff strategy | Exponential + jitter | Avoid thundering herd on reconnect |
| Send trigger | Network reachability change + foreground transition | Align with OS connectivity APIs |

### Quality considerations

- **Event ordering:** Queue flush order may differ from emit order. Attach a local device timestamp (`observed_time_unix_nano`) to every event. Server must tolerate out-of-order arrivals and sort by device timestamp for waterfall reconstruction.
- **Clock skew:** Device clocks may drift relative to server. Cross-reference `../../meta-observability.md §Clock skew` for handling. Do not rely on server receipt time as the canonical event timestamp.
- **Server tolerance:** OTLP backends (and vendor equivalents) are designed for out-of-order ingestion. Verify TTL settings on the collector pipeline so stale events are dropped before indexing.

### Privacy: queued data on disk

Events queued to disk are **PII at rest on user devices**. Requirements:
- Encrypt the queue using platform-native key storage: iOS Keychain, Android Keystore.
- Apply the same field-level redaction to queued events as to events sent immediately (mask email, card number, password fields before write, not before send).
- Anti-pattern: unencrypted telemetry queue on device; see Section 12.

---

## 4. App Lifecycle Events

App lifecycle events anchor session boundaries and correlate performance observations to foreground/background transitions. Each platform exposes different lifecycle hooks; standardize on a common semconv attribute set.

### Lifecycle hook mapping

| Event | iOS | Android | React Native | Flutter |
|-------|-----|---------|--------------|---------|
| App becomes active / foreground | `applicationDidBecomeActive` | `Activity.onResume` / `ProcessLifecycleOwner.ON_START` | `AppState → 'active'` | `AppLifecycleState.resumed` |
| App enters background | `applicationDidEnterBackground` | `Activity.onPause` / `ProcessLifecycleOwner.ON_STOP` | `AppState → 'background'` | `AppLifecycleState.paused` |
| App will terminate | `applicationWillTerminate` | `Activity.onDestroy` | (no guaranteed callback) | `AppLifecycleState.detached` |
| Memory pressure | `didReceiveMemoryWarning` | `onTrimMemory(level)` | N/A (JS heap managed) | N/A |

### Semconv (experimental)

OTel mobile semconv is in progress as of 2026-Q2. Use custom attributes pending stabilization:

```
app.lifecycle.event   = "foreground" | "background" | "terminate" | "memory_warning"
app.session.id        = <UUID generated at foreground entry, reset on cold start>
device.model.name     = <product model string>
os.name               = "ios" | "android"
os.version            = <platform version string>
```

Swift one-liner (OTel span on foreground):
```swift
tracer.spanBuilder(spanName: "app.lifecycle").setSpanKind(.internal)
    .setAttribute("app.lifecycle.event", "foreground").startSpan().end()
```

Kotlin one-liner (OTel span on resume):
```kotlin
tracer.spanBuilder("app.lifecycle").setSpanKind(SpanKind.INTERNAL)
    .setAttribute("app.lifecycle.event", "foreground").startSpan().end()
```

---

## 5. Battery Impact of RUM SDK

RUM SDKs are frequently among the largest third-party battery consumers on a mobile device, due to background network pings, location access, and sensor polling.

### Mitigation strategies

| Technique | iOS mechanism | Android mechanism |
|-----------|--------------|-------------------|
| Batch sends | `BGAppRefreshTask` (15-min minimum OS interval) | `WorkManager` with network constraint |
| Defer on low battery | `ProcessInfo.isLowPowerModeEnabled` | `BatteryManager.BATTERY_STATUS_DISCHARGING` + `PowerManager.isPowerSaveMode` |
| Skip cellular sends (optional) | `NWPathMonitor` check for `.cellular` interface | `ConnectivityManager.getNetworkCapabilities` → `NET_CAPABILITY_NOT_METERED` |
| Respect background execution limits | `BGAppRefreshTask` registration | Doze mode + App Standby bucket awareness |
| Sampling reduction on low signal | Custom rule in SDK config | Custom rule in SDK config |

### Measurement tools

- iOS: Xcode Energy Organizer + Instruments Energy Log
- Android: Battery Historian (<https://developer.android.com/topic/performance/power/battery-historian>)

### Carbon operations consideration

Mobile device CPU and network activity contribute to device-level energy consumption. For green observability practices, minimize telemetry flush frequency and batch size. Server-side carbon impact is tracked via Kepler (CNCF Sandbox) using CPU and network metrics on the collector nodes; mobile-side impact is the device CPU time and radio-on duration during flush.

---

## 6. Network Egress Observability

| Concern | Guidance |
|---------|---------|
| Metered data awareness | Detect cellular vs. Wi-Fi; warn or throttle SDK sends on cellular by default (opt-in to always-on) |
| Request coalescing | Combine multiple queued spans into a single OTLP export request per flush cycle |
| Certificate pinning | Implement mTLS or certificate pinning for telemetry endpoints beyond mesh-level TLS; use platform SecTrustEvaluate / TrustManager accordingly |
| PII leakage via URL | Strip query parameters containing tokens, emails, or IDs before recording `http.url` in spans |

Network request spans follow OTel HTTP client semconv (`http.request.method`, `server.address`, `http.response.status_code`, `network.protocol.name`). These are Stable semconv as of 1.27.0.

---

## 7. iOS vs Android SDK Comparison (as of 2026-Q2)

### iOS SDKs

| SDK | Coverage | Platforms | Notes |
|-----|----------|-----------|-------|
| Sentry Cocoa | Errors, performance, session replay, profiling | iOS, macOS, tvOS, watchOS, visionOS | Largest platform breadth; auto-instrumentation for URLSession |
| Datadog iOS RUM | RUM, traces, logs, error tracking | iOS, tvOS | Strong OTel bridge; custom view timing |
| Firebase Crashlytics | Crashes, breadcrumbs | iOS, macOS | Free; Google ecosystem; no RUM performance. ANR is Android-only; iOS Hang metric is a separate concept handled by MetricKit / Embrace |
| Embrace | Startup, Hang (iOS) / ANR (Android), network spans, session replay | iOS | Startup-time focus; per-session timeline view |
| `opentelemetry-swift` | Traces, metrics (experimental) | iOS, macOS | OTel-native; no auto-instrumentation yet |

### Android SDKs

| SDK | Coverage | Notes |
|-----|----------|-------|
| Sentry Android | Errors, performance, ANR, profiling, session replay | Auto-instrumentation via Gradle plugin; OkHttp + Retrofit |
| Datadog Android RUM | RUM, traces, logs, error tracking | Kotlin-first; OTel exporter available |
| Firebase Crashlytics | Crashes, ANR, breadcrumbs | Free; no RUM performance |
| Embrace | Startup, ANR, network, session replay | App Standby bucket reporting |
| `opentelemetry-android` | Traces, metrics (experimental) | OTel-native; auto-instrumentation via Bytecode plugin (experimental) |

Vendor category cross-reference: `../../vendor-categories.md §Crash Analytics` for full category taxonomy and how-to-choose criteria.

---

## 8. React Native and Flutter

### React Native

React Native spans both the JS bridge and native layers. A single SDK must handle JS errors, native crashes, and the bridge overhead.

| SDK | Coverage |
|-----|----------|
| Sentry React Native | JS errors, native crashes, performance, session replay. Source map upload + native dSYM (iOS) / ProGuard (Android) combined via `sentry-cli`. |
| Datadog RN SDK | RUM, traces, logs. OTel bridge for trace propagation. |

Symbolication: JS source maps + native symbols must both be uploaded per release build. See `crash-analytics.md` for the full symbolication pipeline.

### Flutter

Flutter compiles to native ARM; the Dart VM's `--split-debug-info` flag separates debug symbols for upload.

| SDK | Coverage |
|-----|----------|
| Sentry Flutter | Dart errors, native crashes, performance. `--split-debug-info` + source map + native symbol upload via `sentry-dart-plugin`. |
| Firebase Crashlytics Flutter plugin | Dart and native crashes. Free; no RUM performance metrics. |

---

## 9. W3C Trace Context Propagation: Mobile to Backend

Mobile apps are the root of distributed traces that span multiple backend services. Injecting `traceparent` on outbound requests enables end-to-end correlation.

**Propagation flow:**

```
Mobile App (root span, SpanKind.CLIENT)
  |-- traceparent header injected on HTTP/gRPC call
  v
Backend API Gateway (continues trace, creates child spans)
  |-- downstream services inherit trace context
  v
Database / Queue / External API
```

**Implementation rules:**

1. The mobile SDK intercepts outbound HTTP requests and injects `traceparent` (and optionally `tracestate`) automatically. Configure the propagator in SDK init, not at call sites.
2. `tracestate` carries the mobile sampling decision. Backend tail samplers MUST respect the `sampled` flag to avoid dropping traces that the mobile side intends to keep.
3. The mobile span is `SpanKind.CLIENT`. The backend root span receiving the call is `SpanKind.SERVER`. This pairing is required by OTel span kind semantics.
4. **Session span challenge:** A mobile user session may span minutes or hours with many backend requests. Each HTTP call starts a new child span under the session root span. Store the session root span context in memory (not disk) to avoid stale context on OS-kill and restart.
5. Cross-reference: `../../boundaries/cross-application.md` for the full propagator matrix across service, domain, and vendor boundaries.

W3C Trace Context reference: `../../standards.md §W3C Trace Context`.

---

## 10. Performance Metrics (Mobile-Native)

| Metric | Definition | Target |
|--------|-----------|--------|
| Cold start time | Process launch to first interactive frame | < 2 s (iOS Instruments: `os_signpost`; Android: `reportFullyDrawn`) |
| Warm start time | App resumed from background to interactive | < 1 s |
| Hot start time | Activity recreated (Android) | < 500 ms |
| Frame rendering | Frame drop rate (jank) | iOS ≥ 60 fps (`CADisplayLink`); Android `FrameMetricsAggregator` Jank < 0.1% |
| ANR rate | Application Not Responding (main thread blocked > 5 s) | Android-specific; < 0.47% (Play Console threshold) |
| iOS Hang rate | Non-responsive main thread > 250 ms | < 0.1% (MetricKit `MXHangDiagnostic`) |
| Network latency | Per-request round-trip time | Monitor p50/p95/p99 histograms; no universal target |
| Crash-Free Session Rate | Sessions without a crash / total sessions | Cross-reference `crash-analytics.md §CFR` |

Semconv for mobile startup spans: use `app.lifecycle.event = "cold_start"` with `app.startup.duration_ms` as a custom span attribute until OTel mobile semconv stabilizes.

---

## 11. Matrix Cells (L7 Row, Mobile Slice)

These cells supplement the L7 rows in `../../matrix.md` with mobile-specific caveats:

| Cell | Symbol | Detail |
|------|--------|--------|
| L7 × multi-tenant × metrics | PARTIAL | `tenant.id` on mobile is typically derived from the authenticated user ID, not a direct tenant header. Map `user.id → tenant.id` server-side in the pipeline; avoid attaching `tenant.id` as a metric label on-device (cardinality risk). |
| L7 × cross-application × traces | PASS | Mobile → backend `traceparent` propagation; mobile span is the root CLIENT span. See Section 9. |
| L7 × slo × metrics | PASS | Mobile-side SLIs: cold start time p95, ANR rate, Hang rate. Define SLO targets in OpenSLO YAML referencing mobile metrics ingested via OTLP. Cross-reference `../../boundaries/slo.md`. |
| L7 × privacy × logs | PASS | On-device queued events contain PII at rest. Encryption via Keychain/Keystore required. Field-level masking of card numbers, emails, and passwords before write. Cross-reference `../../signals/privacy.md`. |

---

## 12. Anti-Patterns (Candidates for `../../anti-patterns.md §Section G`)

| Anti-pattern | Risk | Fix |
|-------------|------|-----|
| Telemetry queue stored unencrypted on device | PII at rest exposed if device is lost or compromised | Encrypt queue using iOS Keychain / Android Keystore before writing |
| Heavy RUM SDK flush on every event without batching | Battery drain → user complaints → app uninstall | Batch flushes; align with `BGAppRefreshTask` / `WorkManager` cadence |
| No event TTL on queue | Stale events (hours or days old) arrive at backend; mislead dashboards | Set TTL (24–72 h); drop on exceed before send |
| Missing `traceparent` injection on outbound HTTP | Client-server correlation broken; mobile sessions invisible in backend traces | Configure SDK HTTP interceptor at init; verify with network proxy tool |
| Session replay without PII field masking | Card numbers, emails, passwords captured in replay | Enable SDK masking rules on input fields; test with automated replay review |
| Using server receipt time as event timestamp | Clock skew between device and server corrupts waterfall ordering | Store `observed_time_unix_nano` (device local time) on every event at emit time |

---

## References


- OTel Swift SDK: <https://opentelemetry.io/docs/languages/swift/>
- OTel Android SDK: <https://github.com/open-telemetry/opentelemetry-android>
- Sentry Cocoa: <https://docs.sentry.io/platforms/apple/>
- Sentry Android: <https://docs.sentry.io/platforms/android/>
- Datadog iOS RUM: <https://docs.datadoghq.com/real_user_monitoring/ios/>
- Datadog Android RUM: <https://docs.datadoghq.com/real_user_monitoring/android/>
- Firebase Crashlytics: <https://firebase.google.com/docs/crashlytics>
- iOS Background Tasks: <https://developer.apple.com/documentation/backgroundtasks>
- Android App Standby / Doze: <https://developer.android.com/topic/performance/appstandby>
- W3C Trace Context L1: <https://www.w3.org/TR/trace-context/>
- Battery Historian: <https://developer.android.com/topic/performance/power/battery-historian>

### Primary sources

- <https://opentelemetry.io/docs/languages/swift/>
- <https://github.com/open-telemetry/opentelemetry-android>
