---
otel_semconv: "1.27.0 (2024-11)"
---

# Crash Analytics: L7 Application Layer

## 1. Scope and Why Separate from RUM

Crash analytics is not RUM. The distinction matters for pipeline design, KPI selection, and vendor choice.

| Dimension | RUM | Crash Analytics |
|-----------|-----|----------------|
| Session type | Successful sessions (UX measured) | Abnormal terminations (failed sessions) |
| Primary KPI | Core Web Vitals / INP / startup time | Crash-Free Rate (CFR), ANR Rate |
| Pipeline | Beacon / SDK event stream → backend | Crash report queued on device → uploaded on restart |
| Symbolication | N/A (JS source maps for errors only) | dSYM / ProGuard / R8 / Dart symbols required |
| Offline behavior | Events dropped if offline | Report queued locally; uploaded on next launch |

This file documents the common crash pipeline shared across mobile native, web, and backend uncaught exceptions. Platform-specific symbolication flows are cross-referenced:

- Mobile native detail: `mobile-rum.md §Crash + ANR`
- Web JS error detail: `web-rum.md §Error Boundary and window.onerror`

Platforms in scope: iOS native, Android native, React Native, Flutter, web (JS), backend uncaught exceptions.

---

## 2. Core Metrics

### Crash-Free Rate (CFR)

CFR is the primary stability KPI for mobile and web applications.

```
Session CFR = (sessions without crash) / (total sessions)
User CFR    = (users not experiencing any crash) / (total active users)
```

Targets by application category:

| Category | Minimum Session CFR | Notes |
|----------|--------------------|----|
| Consumer mobile | 99.5% | Google Play bad-behavior threshold: < 99.0% triggers review |
| Enterprise / B2B mobile | 99.7% | Internal SLA baseline |
| Regulated (fintech, health) | 99.9%+ | Near-zero tolerance; compliance-linked |
| Web SPA | 99.8%+ | JS errors below `Error Boundary` counted as crashes |

### ANR Rate and Hang Rate

Near-crash experiences that degrade user perception without a fatal signal:

- **Android ANR Rate** = ANRs / sessions. Two variants per `ApplicationExitInfo`: main-thread input dispatch > 5 s, or broadcast receiver not completing within 200 ms (foreground) / 60 s (background). Google Play flags apps exceeding 0.47% of sessions.
- **iOS Hang Rate** = main-thread hangs > 250 ms / sessions. Visible in Xcode Organizer → Hangs and via MetricKit `MXHangDiagnostic` (iOS 14+).

### Supporting Metrics

| Metric | Definition |
|--------|-----------|
| Affected Users % | `(users who hit crash X) / (DAU)`; scope of a specific crash group |
| Time-to-Symbolication | Duration from crash report ingestion to readable stack trace |
| Crash Volume | Raw crash event count; use with CFR, not instead of it |

---

## 3. Platform-Specific Symbolication Pipelines

Each platform requires a distinct artifact upload strategy. Failure to automate upload results in unreadable stack traces in production.

| Platform | Symbol artifact | Upload trigger | Key complexity |
|----------|----------------|---------------|----------------|
| iOS | dSYM bundle (per build UUID) | Xcode archive → CI upload | Bitcode deprecated in Xcode 14; no re-symbolication |
| Android (JVM) | `mapping.txt` (ProGuard / R8) | Gradle task post-minify | Must match exact build; obfuscation mapping is one-way |
| Android (NDK) | `.so` files with debug info stripped separately | CI symbol upload | Requires `--build-id` linkage; `.so` from APK lacks debug info |
| React Native | JS sourcemap + iOS dSYM + Android mapping.txt | Three uploads per release | Hermes bytecode adds an extra sourcemap layer |
| Flutter | Dart symbol files from `--split-debug-info` | CI upload after `flutter build` | Both iOS dSYM and Android mapping still needed for platform channels |
| Web | JS sourcemap | CI deploy pipeline | `window.onerror` + `onunhandledrejection`; React/Vue Error Boundary |
| Backend | N/A (source available) | OTel `exception.*` attributes | Sentry Python/Node/Ruby SDK for crash-like grouping |

### iOS

dSYM (Debug Symbols) is the binary artifact that maps obfuscated addresses to function names and line numbers. Each Xcode build generates a UUID-stamped dSYM. The UUID must match the crash report exactly; re-using a dSYM from a different build produces wrong or partial symbolication.

Bitcode was deprecated in Xcode 14. Prior to Xcode 14, Apple re-compiled bitcode for App Store distribution, requiring vendors (Firebase, Sentry, Datadog) to download the re-compiled dSYM from App Store Connect via the API. Xcode 14+ removes this complexity; the Xcode-generated dSYM is final.

Apple App Store Connect crash API: Firebase Crashlytics, Sentry, and Datadog can pull crash data from App Store Connect for devices where the user has opted into analytics sharing.

### Android

ProGuard / R8 generates `mapping.txt` which maps obfuscated class/method names back to source. The NDK layer produces native `.so` libraries; the release variant strips debug info into a separate symbol file. Upload both `mapping.txt` and NDK symbol `.so` files for full stack traces covering Java/Kotlin and C++ frames.

Play Console crashes API: available for complementary data, not a replacement for vendor SDK.

### React Native

Three separate symbol artifacts per release:

1. JS bundle sourcemap (Metro bundler output)
2. iOS dSYM (for native frames)
3. Android `mapping.txt` (for native frames)

Hermes JS engine compiles JS to bytecode. The sourcemap chain is: Hermes bytecode address → JS sourcemap → original TS/JS source. Some vendors (Sentry, Datadog) have Hermes-aware symbolication; verify vendor documentation before assuming standard sourcemap upload suffices.

### Flutter

`flutter build --split-debug-info=<dir> --obfuscate` produces Dart symbol files. These must be uploaded to the crash vendor's symbol storage. Platform-channel crashes (e.g., method channel calls into Kotlin/Swift) still require iOS dSYM and Android `mapping.txt` in addition to Dart symbols.

### Web

Cross-reference `web-rum.md §Error tracking`. Key patterns:
- `window.onerror` catches uncaught synchronous errors
- `window.addEventListener('unhandledrejection', ...)` catches unhandled Promise rejections
- React `ErrorBoundary`, Vue `app.config.errorHandler`, Svelte `<svelte:component this={ErrorBoundary}>` prevent full-page crashes and capture component-tree context

### Backend

Backend services rarely produce binary crash reports. OTel `exception.*` attributes (Stable semconv) capture the exception type, message, and stacktrace on spans. Sentry Python / Node.js / Ruby SDKs provide crash-like aggregation with grouping, release tracking, and breadcrumbs that mirror the mobile crash experience.

---

## 4. Release Tracking Integration

Release tracking is the most commonly omitted step in crash pipelines. Without it, a crash spike cannot be correlated to a specific deploy.

**CI must set `service.version` on every build.** The OTel semconv `service.version` attribute (Stable) is the canonical release identifier. All crash events, spans, and metrics must carry this attribute.

**Symbol upload must be automated per release.** Manual upload is an anti-pattern (see Section 12). Use:

- Sentry CLI: `sentry-cli releases propose-version` + `sentry-cli releases files upload-sourcemaps`
- Fastlane plugin: `fastlane-plugin-sentry` for iOS/Android
- Gradle plugin: `io.sentry.android.gradle` for Android automatic upload
- Firebase: `firebase crashlytics:symbols:upload` for dSYM (Fastlane or Gradle)

**Release marker events** are structured log records written at deploy time (see `../../boundaries/release.md`). Crash vendors use these to draw vertical lines on CFR trend charts, enabling before/after comparison. Flagger and Argo Rollouts canary analysis can use CFR as a promotion/rollback gate signal; cross-reference `../../boundaries/release.md §Canary analysis`.

---

## 5. Vendor Comparison Matrix (as of 2026-Q2)

Cross-reference `../../vendor-categories.md §Category (j): Crash Analytics`.

| Vendor | iOS | Android | Web JS | Backend | OSS/Free | Key differentiator |
|--------|-----|---------|--------|---------|----------|--------------------|
| Firebase Crashlytics | Yes | Yes | No | No | Free | Google-owned; zero cost; mobile-only; dSYM auto-upload via Xcode plugin |
| Sentry | Yes | Yes | Yes | Yes | OSS + SaaS | Full-stack (mobile + web + backend); release tracking; OTel trace correlation |
| Bugsnag | Yes | Yes | Yes | Yes | Commercial | Stability score; smart grouping noise reduction; breadcrumb API |
| Embrace | Yes | Yes | No | No | Commercial | ANR/startup crash specialist; session replay; network body capture |
| Datadog Error Tracking | Yes | Yes | Yes | Yes | Commercial | Native RUM + APM integration; unified with Datadog traces |

**Key differentiators:**

- **Firebase Crashlytics**: best choice when budget is zero and stack is mobile-only. No web or backend support. Google ecosystem (Analytics, BigQuery export).
- **Sentry**: strongest cross-platform story. The only vendor in the matrix with meaningful OSS options. Release health dashboard is the most mature. OTel `trace_id` in crash context enables backend span lookup.
- **Bugsnag**: stability metrics and smart grouping reduce alert fatigue. Breadcrumb API is the most flexible for custom event recording.
- **Embrace**: purpose-built for mobile ANR, startup crash, and user-journey session replay. Not a general-purpose crash tool.
- **Datadog Error Tracking**: optimal when Datadog APM is already in use. RUM + APM linkage is native; no separate SDK integration.

---

## 6. Crash Pipeline Flow

```
App crashes → crash report generated
  (stack trace + breadcrumbs + device context + user context)
      |
      v
  Queued locally on device (if offline — see mobile-rum.md §offline queuing)
      |
      v
  Uploaded to vendor on next launch or when network available
      |
      v
  Vendor-side symbolication
  (dSYM / ProGuard / sourcemap matched by build UUID / version)
      |
      v
  Deduplication by stack fingerprint
  (vendor groups similar crashes into one issue)
      |
      v
  Aggregation: Affected Users %, Event Count, CFR delta
      |
      v
  Alert + ticket creation
  (PagerDuty / OpsGenie for CFR breach; JIRA / Linear for issue triage)
```

---

## 7. Breadcrumbs and Context

Breadcrumbs are the sequence of events recorded before the crash. They are the primary debugging context beyond the stack trace.

| Category | Examples |
|----------|---------|
| User actions | Button taps, navigation events, form submissions |
| Network calls | HTTP request URL, status code, duration |
| Log events | Application log statements recorded by SDK |
| State changes | Authentication events, feature flag changes |

Context attached to every crash report:

| Field | Source | Notes |
|-------|--------|-------|
| `device.model` | OS API | Device model identifier |
| `os.version` | OS API | OS version string |
| `service.version` | Build config | Release version; must match symbol upload |
| `screen_resolution` | OS API | Pixel dimensions |
| `free_memory_mb` | OS API | Available RAM at crash time |
| `user.id` | App session | Hash or opaque ID only; see Section 8 |

Custom keys (e.g., `order.id`, `tenant.id`, `experiment.variant`) enable correlation with business context. Restrict to non-PII values or apply redaction before send.

---

## 8. Privacy and PII in Crash Reports

Stack traces frequently contain PII because exception messages capture method arguments, query strings, and HTTP headers verbatim.

**Common PII leaks in crash reports:**

- SQL query in exception message containing `user.email` or SSN
- HTTP response body logged in a network error containing `Authorization: Bearer <token>`
- URL query parameters with `?email=user@example.com`
- Custom key set with `user.email` directly

**Vendor SDK redaction hooks:**

Sentry `beforeSend` callback (JavaScript/TypeScript example):

```typescript
Sentry.init({
  dsn: "...",
  beforeSend(event) {
    // Strip email from exception values
    if (event.exception?.values) {
      event.exception.values = event.exception.values.map((ex) => ({
        ...ex,
        value: ex.value?.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, "[email]"),
      }));
    }
    // Remove auth headers from request context
    if (event.request?.headers) {
      delete event.request.headers["Authorization"];
      delete event.request.headers["Cookie"];
    }
    return event;
  },
});
```

- **Firebase Crashlytics**: use `setCustomKey` with an explicit allowlist; never pass user.email or tokens as custom keys.
- **Datadog**: `beforeSend` callback mirrors Sentry's pattern; apply to both RUM and Error Tracking initializations.

Cross-reference: `../../signals/privacy.md §Crash report redaction` and `../../anti-patterns.md §Section A Privacy`.

---

## 9. CI Integration for Symbol Upload

Symbols must never be stored in git. Upload to the vendor on every CI build that produces a release artifact.

Conceptual GitHub Actions workflow (adapt per vendor):

```yaml
# .github/workflows/release.yml (conceptual — adapt version and paths)
- name: Build iOS release
  run: xcodebuild archive -scheme MyApp -archivePath build/MyApp.xcarchive

- name: Upload dSYM to Sentry
  env:
    SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}
    SENTRY_ORG: my-org
    SENTRY_PROJECT: my-app-ios
  run: |
    sentry-cli releases propose-version
    sentry-cli upload-dif build/MyApp.xcarchive/dSYMs \
      --include-sources

- name: Tag release with service.version
  run: |
    sentry-cli releases new "${GITHUB_REF_NAME}"
    sentry-cli releases set-commits "${GITHUB_REF_NAME}" --auto
    sentry-cli releases finalize "${GITHUB_REF_NAME}"
```

For Android, replace the dSYM upload step with a Gradle task invocation (`./gradlew :app:uploadSentryProguardMappings`) or `sentry-cli upload-proguard --android`.

Cross-reference `../../observability-as-code.md` for observability-as-code patterns and repository structure for symbol upload configuration.

---

## 10. ANR and Hang Detection

ANR and hangs are near-crash events that degrade UX without generating a fatal crash report. They require distinct detection and alerting strategies.

### Android ANR

- Definition: main thread unresponsive for more than 5 seconds during input dispatch or 200 ms during broadcast.
- Detection sources:
  - `ApplicationExitInfo` API (Android 11+): query ANR reasons from the OS after the fact.
  - `ANRWatchDog` library: third-party watchdog thread that detects main-thread stalls in-process.
  - Sentry, Embrace, Datadog: vendor SDK ANR detection via watchdog thread.
- Alert threshold: Google Play considers an ANR rate >= 0.47% of sessions as bad behavior, which can suppress the app in search results or trigger a Play Console warning.

### iOS Hang

- Definition: main thread blocked for more than 250 ms (Xcode Organizer threshold).
- Detection sources:
  - Xcode Organizer → Hangs report (aggregated from devices with diagnostic sharing enabled).
  - Sentry App Hang Detection SDK: configurable threshold, default 2 seconds; reports hangs as separate issue type.
  - MetricKit: `MXHangDiagnostic` (iOS 14+) for on-device hang diagnostics.
- Alert threshold: no Google Play equivalent; apply product-specific target (e.g., < 0.1% of sessions for consumer apps).

---

## 11. Matrix Cross-References (L7 Row)

These cells from `../../matrix.md` are directly informed by this file:

| Matrix cell | Artifact | Note |
|-------------|---------|------|
| L7 × multi-tenant × logs | Per-tenant crash rate segmentation | `tenant.id` as custom key in crash context |
| L7 × release × traces | Release-tagged crash correlates canary rollback | `service.version` on crash event + span `trace_id` linkage |
| L7 × privacy × logs | Crash report redaction before send | `beforeSend` hook; allowlist custom keys |
| L7 × cross-application × traces | Backend `trace_id` in crash context | Sentry / Datadog: attach active `trace_id` to crash report for backend span lookup |

---

## 12. Anti-Patterns

Candidates for `../../anti-patterns.md §Section G — Crash Analytics`:

| Anti-pattern | Impact | Remediation |
|-------------|--------|------------|
| Crash report contains `user.email`, auth tokens, or card numbers without redaction filter | GDPR / PIPA breach; PII in vendor SaaS storage | Implement `beforeSend` allowlist; cross-ref Section 8 |
| Symbol upload not automated in CI | Stack traces unreadable in production | Add symbol upload step to release pipeline; cross-ref Section 9 |
| No release marker → cannot correlate deploy to crash spike | Crash spike investigation requires manual git bisect | Set `service.version` on every build; emit release event at deploy |
| ANR rate unmonitored on Android | Google Play app suppression; degraded store ranking | Add ANR rate metric; alert at 0.47% of sessions |
| Single aggregate CFR target ignoring device-tier or OS-version distribution | p10 device users (low-end hardware) masked by p90 average | Segment CFR by `device.model`, `os.version`, and network type |
| dSYM stored in git LFS instead of vendor symbol storage | Git LFS cost; symbol-build UUID drift | Use vendor upload; never commit dSYMs |

---

## References




### Primary sources

- Firebase Crashlytics: <https://firebase.google.com/docs/crashlytics>
- Sentry iOS: <https://docs.sentry.io/platforms/apple/>
- Sentry Android: <https://docs.sentry.io/platforms/android/>
- Sentry React Native: <https://docs.sentry.io/platforms/react-native/>
- Sentry Flutter: <https://docs.sentry.io/platforms/flutter/>
- Xcode Organizer Hangs: <https://developer.apple.com/documentation/xcode/understanding-user-interface-responsiveness>
- Android ANR: <https://developer.android.com/topic/performance/vitals/anr>
- Android ApplicationExitInfo: <https://developer.android.com/reference/android/app/ApplicationExitInfo>
- OTel `exception.*` semconv (Stable): <https://opentelemetry.io/docs/specs/semconv/exceptions/>
