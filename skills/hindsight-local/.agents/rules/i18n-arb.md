---
description: ARB-based localization standards — ARB as the single cross-platform source of truth for UI strings
globs: "**/*.arb"
alwaysApply: false
---

# ARB-Based Localization

When ARB files exist in the project, they are the **single source of truth** for all user-facing strings — regardless of target platform. A build step generates the platform-specific artifacts from ARB (e.g. JSON for web, generated Dart for Flutter).

1. **Never hardcode UI strings** — all user-visible text must come from ARB files (`*.arb`)
2. **Edit ARB first** — when adding or changing UI text, update the ARB file, then run the i18n build
3. **Build after changes** — run the project's i18n build task (e.g. `mise i18n:build`, or `dart run build_runner build` in a Flutter package) to regenerate localized output (JSON for web, generated Dart, etc.). Prefer the `mise` task when one exists.
4. **Generated output is never hand-edited** — change the ARB and rebuild; treat the emitted JSON/Dart as derived artifacts
5. **Base locale** — the primary ARB file (e.g., `app_en.arb`) is the reference; other locales derive from it
6. **Key naming** — use `camelCase` keys describing the purpose, not the content: `loginButton` not `clickHere`
7. **Placeholders** — use ICU message syntax for interpolation: `"greeting": "Hello, {name}!"`
8. **Do not translate keys** — ARB keys are identifiers, always in English
