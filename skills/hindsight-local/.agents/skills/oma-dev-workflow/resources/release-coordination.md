# Release Coordination

Release Please workflow for automated versioning and releases.

## Overview

This project uses Google's [Release Please](https://github.com/googleapis/release-please) for automated versioning and releases based on conventional commits.

## How It Works

1. **Release PR Creation**: When conventional commits are merged to main, Release Please creates a release PR
2. **CHANGELOG Generation**: Automatically generates CHANGELOG.md from commits
3. **Version Bumping**: Updates version in package.json and other files
4. **Release on Merge**: When the release PR is merged, a GitHub release is created

## GitHub Actions Workflow

```yaml
# .github/workflows/release-please.yml
name: Release Please

on:
  push:
    branches: [main]

permissions:
  contents: write
  pull-requests: write

jobs:
  release-please:
    runs-on: ubuntu-latest
    steps:
      - uses: googleapis/release-please-action@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          release-type: node
```

## Mise Tasks for Release Management

```toml
# Root mise.toml
[tools]
node = "24"
bun = "latest"

[tasks.release:check]
description = "Check release-please status"
run = '''
#!/usr/bin/env bash
echo "Checking release status..."
echo ""
echo "Open release PRs:"
gh pr list --label "autorelease: pending" --state open
echo ""
echo "Latest releases:"
gh release list --limit 5
'''

[tasks.release:manifest]
description = "Show current release manifest"
run = "cat .release-please-manifest.json"

[tasks.release:changelog]
description = "Preview changelog (local dry-run)"
run = '''
#!/usr/bin/env bash
echo "Recent conventional commits:"
git log --pretty=format:"%s" $(git describe --tags --abbrev=0)..HEAD | grep -E "^(feat|fix|docs|style|refactor|test|chore|ci)(\(.+\))?:"
'''
```

## Conventional Commit Format

Release Please recognizes these commit types:

| Type | Version Bump | Description |
|------|--------------|-------------|
| `feat:` | minor | New feature |
| `fix:` | patch | Bug fix |
| `docs:` | patch | Documentation |
| `style:` | patch | Code style (no logic) |
| `refactor:` | patch | Code refactoring |
| `test:` | patch | Tests |
| `chore:` | patch | Maintenance |
| `feat!:` | major | Breaking change |
| `fix!:` | major | Breaking bug fix |

## Release Configuration

```json
// .release-please-config.json
{
  "packages": {
    ".": {
      "release-type": "node",
      "draft": false,
      "prerelease": false
    }
  }
}
```

```json
// .release-please-manifest.json
{
  ".": "1.0.0"
}
```

## Release Flow

### 1. Develop Features
```bash
# Make changes with conventional commits
git commit -m "feat(cli): add infrastructure skills category"
git commit -m "fix(cli): correct skill installation path"
git commit -m "docs: update installation guide"
```

### 2. Merge to Main
```bash
git push origin main
```

### 3. Release Please Creates PR
Automatically creates a PR like:
- Title: `chore(main): release 1.1.0`
- Includes updated CHANGELOG.md
- Includes version bump in package.json

### 4. Review and Merge
```bash
# Check the release PR
gh pr list --label "autorelease: pending"

# Review changes
git fetch origin
git diff origin/main...release-please--branches--main

# Merge via GitHub (creates release)
```

### 5. Verify Release
```toml
[tasks.release:verify]
description = "Verify latest release"
run = '''
#!/usr/bin/env bash
echo "Latest release:"
gh release view --json tagName,name,createdAt

echo ""
echo "Assets:"
gh release view --json assets
'''
```

## Manual Release Trigger

If needed, manually trigger release-please:

```toml
[tasks.release:trigger]
description = "Manually trigger release-please (CI will handle)"
run = '''
#!/usr/bin/env bash
echo "Triggering release-please..."
echo "Push to main will trigger the workflow automatically"
echo ""
echo "Current status:"
mise run release:check
'''
```

## Post-Release Tasks

```toml
[tasks.release:cleanup]
description = "Cleanup after release"
run = '''
#!/usr/bin/env bash
echo "Post-release cleanup..."

# Pull latest changes with tags
git pull origin main --tags

# Verify local state
echo ""
echo "Local version:"
cat package.json | grep '"version"' | cut -d'"' -f4

echo ""
echo "Latest tag:"
git describe --tags --abbrev=0

echo ""
echo "Release cleanup complete"
'''
```

## Troubleshooting

```toml
[tasks.release:debug]
description = "Debug release issues"
run = '''
#!/usr/bin/env bash
echo "Release Debugging"
echo ""

echo "1. Conventional commits since last tag:"
git log --pretty=format:"%s" $(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~10")..HEAD | grep -E "^(feat|fix|docs|style|refactor|test|chore|ci)(\(.+\))?:" || echo "  No conventional commits found"

echo ""
echo "2. Release Please config files:"
ls -la .release-please*.json 2>/dev/null || echo "  Config files not found"

echo ""
echo "3. GitHub Actions status:"
gh run list --workflow=release-please.yml --limit 5 2>/dev/null || echo "  No runs found"

echo ""
echo "4. Open release PRs:"
gh pr list --label "autorelease: pending" 2>/dev/null || echo "  No release PRs"
'''
```

## Setup Requirements

1. **GitHub Actions workflow** (`.github/workflows/release-please.yml`)
2. **Release Please config** (`.release-please-config.json`)
3. **Release Please manifest** (`.release-please-manifest.json`)
4. **Conventional commits** (already configured via commitlint)
5. **GitHub CLI** (`gh`) installed for the release tasks — `brew install gh` or `mise use gh` (it is a Go binary, not an npm package; `bunx gh` resolves an unrelated deprecated package)

## Benefits

- Automated versioning based on commits
- Auto-generated CHANGELOG.md
- GitHub releases with notes
- No manual version bumping
- Integrated with GitHub PR workflow
