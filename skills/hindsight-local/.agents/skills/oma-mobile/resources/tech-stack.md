# Mobile Agent - Tech Stack Index

Per-platform tech stacks live under `variants/{platform}/`, mirroring
`oma-backend/variants/`. Each variant owns its own `stack.yaml` (SSOT),
`tech-stack.md`, `snippets.md`, and API template. This file holds only the
**cross-platform guidance** shared by every variant plus a pointer index.

## Platform Variants

| Platform | Stack manifest | Tech stack | Snippets | API template |
|----------|----------------|-----------|----------|--------------|
| Swift (iOS native) | `../variants/swift-ios/stack.yaml` | `../variants/swift-ios/tech-stack.md` | `../variants/swift-ios/snippets.md` | `../variants/swift-ios/api-template.swift` |
| Flutter | `../variants/flutter/stack.yaml` | `../variants/flutter/tech-stack.md` | `../variants/flutter/snippets.md` | `../variants/flutter/api-template.dart` |
| React Native | `../variants/react-native/stack.yaml` | `../variants/react-native/tech-stack.md` | `../variants/react-native/snippets.md` | `../variants/react-native/api-template.ts` |

Stack selection: detect from project files (`Package.swift` / `pubspec.yaml` /
`package.json` + `react-native`), or run `/stack-set` to generate a
project-specific `stack/` from the matching variant baseline. The variant
`stack.yaml` is the default; `/stack-set` may override any field per project.

## Stack at a glance

| | Swift iOS | Flutter | React Native |
|---|---|---|---|
| Language | Swift 5.9+ / Swift 6 | Dart 3.3+ | TypeScript (strict) |
| UI | SwiftUI | Flutter / Material 3 | React Native |
| State | Observation (`@Observable`) | Riverpod | Zustand |
| Navigation | NavigationStack | GoRouter | React Navigation v6 |
| HTTP transport | URLSession | Dio | Axios |
| API/data layer | swift-openapi-generator | Repository (Dio) | TanStack Query + `api/` |
| Response cache (mandatory, repo layer) | hyperoslo/Cache | Drift offline-first repo | TanStack Query |
| Durable storage | SwiftData / Keychain | Drift / flutter_secure_storage | MMKV / secure-store |
| Unit test | XCTest / Swift Testing | flutter_test + mocktail | jest + RNTL |
| E2E | XCUITest | Maestro | Maestro |

## Mandatory: repository-layer response cache

Every variant **mandates a response cache at the Repository / data layer** — the
same philosophy across platforms, different idiomatic tooling:

- Cache **decoded domain models**, never raw transport bytes / `HTTPBody` /
  raw HTTP responses.
- **Stale-while-revalidate** on reads: serve cached immediately, revalidate in
  the background, update state.
- **Invalidate on write**: every create/update/delete drops the affected cache
  keys so the next read repopulates.
- Cache key = operation + params (never URLs); explicit TTL (never infinite).
- The cache is for transient, server-owned data only. Durable user-owned data
  and secrets stay in the platform's durable store / secure store.

See each variant's `tech-stack.md` "Response Cache" section and `snippets.md`
for the platform-specific implementation.

## Cross-platform guidelines (shared)

- **Clean Architecture**: domain → data → presentation (or App/Core/Features/Shared
  on Swift). Business logic never lives in UI widgets/views.
- **Platform design**: Material Design 3 on Android, iOS Human Interface
  Guidelines on iOS. Use platform checks for platform-specific behavior.
- **State management**: no raw `setState`/ad-hoc state for complex logic — use
  the variant's state solution.
- **Networking**: transport-layer interceptors handle auth/retry/logging only;
  domain-model caching is a repository concern (see above). Handle offline
  gracefully.
- **Lifecycle**: dispose controllers / cancel tasks to prevent leaks.
- **Performance**: 60fps target; test on both platforms.
- **E2E**: Maestro for critical user flows.
