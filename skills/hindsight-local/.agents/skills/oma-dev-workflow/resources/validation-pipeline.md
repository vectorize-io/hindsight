# Validation Pipeline

Complete validation pipeline with local git hooks and CI/CD integration.

## Commitlint Configuration

```json
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'ci', 'infra']
    ],
    'scope-enum': [
      2,
      'always',
      ['web', 'api', 'mobile', 'worker', 'shared', 'infra', 'deps']
    ]
  }
};
```

## Local Validation (Git Hooks)

### Mise Hooks Setup

```toml
# mise.toml
[hooks]
postinstall = '''
  mkdir -p .git/hooks

  # commit-msg hook
  cat > .git/hooks/commit-msg <<'EOF'
#!/bin/sh
exec mise run git:commit-msg -- "$1"
EOF
  chmod +x .git/hooks/commit-msg

  # pre-commit hook
  cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
exec mise run git:pre-commit
EOF
  chmod +x .git/hooks/pre-commit

  # pre-push hook
  cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
exec mise run git:pre-push
EOF
  chmod +x .git/hooks/pre-push
'''
```

### Git Hook Tasks

```toml
[tasks."git:commit-msg"]
description = "Validate commit message using commitlint"
run = "bunx @commitlint/cli@20 --edit $1"

[tasks."git:pre-commit"]
description = "Run lint on changed files"
run = '''
#!/usr/bin/env bash
changed=$(git diff --cached --name-only)

if echo "$changed" | grep -q "^apps/api/"; then
    echo "[pre-commit] apps/api detected, running lint..."
    mise //apps/api:lint || exit 1
fi

if echo "$changed" | grep -q "^apps/web/"; then
    echo "[pre-commit] apps/web detected, running lint..."
    mise //apps/web:lint || exit 1
fi

if echo "$changed" | grep -q "^apps/mobile/"; then
    echo "[pre-commit] apps/mobile detected, running lint..."
    mise //apps/mobile:lint || exit 1
fi
'''

[tasks."git:pre-push"]
description = "Validate branch name and run tests"
run = '''
#!/usr/bin/env bash
# Validate branch name
bunx @gracefullight/validate-branch || exit 1

# Get changed files
changed=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1)

if echo "$changed" | grep -q "^apps/api/"; then
    echo "[pre-push] apps/api detected, running test..."
    mise //apps/api:test || exit 1
fi

if echo "$changed" | grep -q "^apps/web/"; then
    echo "[pre-push] apps/web detected, running test..."
    mise //apps/web:test || exit 1
fi

if echo "$changed" | grep -q "^apps/mobile/"; then
    echo "[pre-push] apps/mobile detected, running test..."
    mise //apps/mobile:test || exit 1
fi
'''
```

## CI/CD Validation (GitHub Actions)

### Full Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: mise run install
      - run: mise run lint
      - run: mise run typecheck
      - run: mise run test
```

### Change-Based Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      web: ${{ steps.changes.outputs.web }}
      api: ${{ steps.changes.outputs.api }}
      mobile: ${{ steps.changes.outputs.mobile }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: changes
        with:
          filters: |
            web:
              - 'apps/web/**'
            api:
              - 'apps/api/**'
            mobile:
              - 'apps/mobile/**'

  lint-web:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.web == 'true' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: mise run //apps/web:lint

  lint-api:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.api == 'true' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: mise run //apps/api:lint

  test-web:
    needs: [detect-changes, lint-web]
    if: ${{ needs.detect-changes.outputs.web == 'true' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: mise run //apps/web:test
```

## Reusable Tasks

```toml
# Root mise.toml
[tasks.lint:changed]
description = "Lint only changed apps"
run = '''
#!/usr/bin/env bash
changed_files=$(git diff --name-only HEAD~1)

if echo "$changed_files" | grep -q "^apps/web/"; then
  echo "→ Linting web..."
  mise run //apps/web:lint || exit 1
fi

if echo "$changed_files" | grep -q "^apps/api/"; then
  echo "→ Linting api..."
  mise run //apps/api:lint || exit 1
fi

if echo "$changed_files" | grep -q "^apps/mobile/"; then
  echo "→ Linting mobile..."
  mise run //apps/mobile:lint || exit 1
fi
'''

[tasks.test:changed]
description = "Test only changed apps"
run = '''
#!/usr/bin/env bash
changed_files=$(git diff --name-only HEAD~1)

if echo "$changed_files" | grep -q "^apps/web/"; then
  echo "→ Testing web..."
  mise run //apps/web:test || exit 1
fi

if echo "$changed_files" | grep -q "^apps/api/"; then
  echo "→ Testing api..."
  mise run //apps/api:test || exit 1
fi

if echo "$changed_files" | grep -q "^apps/mobile/"; then
  echo "→ Testing mobile..."
  mise run //apps/mobile:test || exit 1
fi
'''

[tasks.validate:changed]
description = "Validate only changed apps"
depends = ["lint:changed", "test:changed"]
```

## Complete Root mise.toml Example

```toml
[tools]
node = "24"
python = "3.12"
flutter = "3"
bun = "latest"

[hooks]
postinstall = '''
  mkdir -p .git/hooks

  cat > .git/hooks/commit-msg <<'EOF'
#!/bin/sh
exec mise run git:commit-msg -- "$1"
EOF
  chmod +x .git/hooks/commit-msg

  cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
exec mise run git:pre-commit
EOF
  chmod +x .git/hooks/pre-commit

  cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
exec mise run git:pre-push
EOF
  chmod +x .git/hooks/pre-push
'''

[tasks.dev]
description = "Start all development services"
depends = ["//apps/api:dev", "//apps/web:dev"]

[tasks.lint]
description = "Lint all apps"
depends = ["//apps/api:lint", "//apps/web:lint", "//apps/mobile:lint"]

[tasks.test]
description = "Test all apps"
depends = ["//apps/api:test", "//apps/web:test", "//apps/mobile:test"]

[tasks."git:commit-msg"]
description = "Validate commit message"
run = "bunx @commitlint/cli@20 --edit $1"

[tasks."git:pre-commit"]
description = "Run lint on changed files"
run = '''
#!/usr/bin/env bash
changed=$(git diff --cached --name-only)

if echo "$changed" | grep -q "^apps/web/"; then
    mise //apps/web:lint || exit 1
fi

if echo "$changed" | grep -q "^apps/api/"; then
    mise //apps/api:lint || exit 1
fi
'''

[tasks."git:pre-push"]
description = "Run tests + branch validation"
run = '''
#!/usr/bin/env bash
bunx @gracefullight/validate-branch || exit 1

changed=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1)

if echo "$changed" | grep -q "^apps/web/"; then
    mise //apps/web:test || exit 1
fi

if echo "$changed" | grep -q "^apps/api/"; then
    mise //apps/api:test || exit 1
fi
'''

[tasks.lint:changed]
description = "Lint only changed apps"
run = '''
#!/usr/bin/env bash
changed_files=$(git diff --name-only HEAD~1)

if echo "$changed_files" | grep -q "^apps/web/"; then
  mise run //apps/web:lint || exit 1
fi

if echo "$changed_files" | grep -q "^apps/api/"; then
  mise run //apps/api:lint || exit 1
fi
'''

[tasks.test:changed]
description = "Test only changed apps"
run = '''
#!/usr/bin/env bash
changed_files=$(git diff --name-only HEAD~1)

if echo "$changed_files" | grep -q "^apps/web/"; then
  mise run //apps/web:test || exit 1
fi

if echo "$changed_files" | grep -q "^apps/api/"; then
  mise run //apps/api:test || exit 1
fi
'''
```
