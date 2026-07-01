# Mobile Agent - Self-Verification Checklist

Run through every item before submitting your work.

## Architecture
- [ ] Clean Architecture layers: domain -> data -> presentation
- [ ] Entities are pure Dart (no framework dependencies)
- [ ] Repository pattern with interface + implementation
- [ ] Riverpod/Bloc for state management (no setState in complex widgets)

## Platform
- [ ] Material Design 3 for Android
- [ ] iOS Human Interface Guidelines followed
- [ ] Platform-specific code guarded with `Platform.isIOS`/`Platform.isAndroid`
- [ ] Tested on both iOS and Android (emulator or device)
- [ ] Dark mode supported

## Performance
- [ ] 60fps scrolling (no jank)
- [ ] Controllers disposed in `dispose()` method
- [ ] No memory leaks (listeners, subscriptions cleaned up)
- [ ] Images cached and sized appropriately
- [ ] Cold start < 2s

## API Integration
- [ ] Dio with interceptors (auth, error handling)
- [ ] Loading states shown during API calls
- [ ] Error states with retry action
- [ ] Offline handling (graceful degradation or offline-first)

## Testing
- [ ] Unit tests for domain logic and providers
- [ ] Widget tests for key screens
- [ ] E2E tests with Maestro for critical user flows
- [ ] Edge cases: empty lists, error states, offline mode
- [ ] Tests pass on both platforms

## Swift Native (iOS)
> Applies when the project is Swift native (`Package.swift` / `.xcodeproj` present). Skip for Flutter/RN.
- [ ] `App/Core/Features/Shared` layout respected (App = entry/DI, Core = networking/generated client, Features = view+`@Observable` VM slices, Shared = reusable UI/util)
- [ ] API access goes through the generated `Client` from `swift-openapi-generator` — no hand-rolled `URLRequest`/`JSONDecoder` for spec-covered endpoints
- [ ] OpenAPI document present at `Core/Networking/openapi.yaml` and synced from the backend before build
- [ ] SwiftUI state via `@Observable` (Observation framework); `Task`s cancelled in `deinit` to avoid leaks
- [ ] Loading / error (with retry) / empty / data states handled in views
- [ ] iOS Human Interface Guidelines followed
- [ ] Push routes registered via a `swipeBackDestination` wrapper (not bare `navigationDestination`), so nav-bar-hidden screens keep edge swipe-back; guarded pops (unsaved edits) override explicitly — see `variants/swift-ios/snippets.md` §9
- [ ] `swift build` succeeds (runs the generator plugin) and `swift test` passes
- [ ] XCTest/XCUITest coverage for critical flows
