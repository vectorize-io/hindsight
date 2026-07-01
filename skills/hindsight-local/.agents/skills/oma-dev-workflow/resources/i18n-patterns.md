# i18n Build Patterns

Internationalization workflows for monorepos.

## Shared i18n Package

```toml
# packages/i18n/mise.toml
[tasks.build]
description = "Build i18n files for all platforms"
depends = ["build:web", "build:mobile"]

[tasks.build:web]
description = "Build for web (next-intl JSON)"
run = "bun scripts/build-web.js"

[tasks.build:mobile]
description = "Build for mobile (Flutter ARB)"
run = "bun scripts/build-mobile.js"

[tasks.watch]
description = "Watch for changes and rebuild"
run = "bun scripts/watch.js"
```

## Usage in Apps

```bash
# Build i18n for all apps
mise run //packages/i18n:build

# Watch for changes during development
mise run //packages/i18n:watch
```

## Web Integration

```toml
# apps/web/mise.toml
[tasks.dev]
description = "Start Next.js dev server"
depends = ["//packages/i18n:build"]
run = "bun dev"

[tasks.build]
description = "Production build"
depends = ["//packages/i18n:build"]
run = "bun run build"
```

## Mobile Integration

```toml
# apps/mobile/mise.toml
[tasks.gen:l10n]
description = "Generate Flutter localizations"
run = "flutter gen-l10n"

[tasks.dev]
description = "Run Flutter on device"
depends = ["gen:l10n"]
run = "flutter run"
```
