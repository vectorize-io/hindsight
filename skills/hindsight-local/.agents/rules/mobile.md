---
description: Flutter/Dart and cross-platform mobile development standards
globs: "**/*.{dart,swift,kt}"
alwaysApply: false
---

# Mobile Development Standards

## Core Rules

1. **Clean Architecture**: domain -> data -> presentation
2. **State Management**: Riverpod/Bloc for state management (no raw setState for complex logic)
3. **Design Guidelines**: Material Design 3 (Android) + iOS HIG (iOS)
4. **Resource Cleanup**: All controllers disposed in `dispose()` method
5. **Networking**: Dio with interceptors for API calls; handle offline gracefully
6. **Performance**: 60fps target; test on both platforms
7. **E2E Testing**: Use Maestro for end-to-end UI testing
