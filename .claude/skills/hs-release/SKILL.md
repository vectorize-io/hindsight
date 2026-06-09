---
name: hs-release
description: Cut a core Hindsight release (vX.Y.Z) and open the changelog + blog PR. Use when asked to cut/start a release, bump the version, or publish a new Hindsight version.
user_invocable: true
---

# Hindsight Release

Cut a **core** Hindsight release and open the accompanying changelog/blog PR. This is for the core
product version (API, clients, CLI, control plane, Helm). **Integrations are versioned
independently** — use `scripts/release-integration.sh` for those, not this skill.

The release is **irreversible and outward-facing**: it tags a version and pushes it straight to
`main`, which triggers CI that publishes packages to PyPI / npm / Helm. Confirm the version number
and that the intended fixes are already merged to `main` before you start.

## Step 0 — Pre-flight

1. **Decide the base.** A release is cut from the latest `origin/main`, never from a feature
   branch. `git fetch origin --tags` first. Confirm the "couple of fixes" the user means are
   actually merged to `main` (`git log v<prev>..origin/main --oneline`).
2. **Find where `main` is checked out.** `main` is often already checked out in a sibling worktree
   (`git worktree list`). You **cannot** check out `main` in a second worktree — run the release in
   the worktree that already holds it. If that worktree is dirty with throwaway cruft
   (`.next-*` tsconfig paths, screenshots), `git stash push -u`, fast-forward to `origin/main`,
   run the release, then `git stash pop`.
3. **Pitfall:** never pipe the checkout in an `&&` chain like
   `git checkout main 2>&1 | tail && git reset --hard ...` — the pipe's exit status is `tail`'s
   (always 0), so a failed checkout won't stop the chain and the `reset` fires on the **wrong
   branch**. Check out as its own command and verify `git branch --show-current` before resetting.

## Step 1 — Cut the release

Run from the worktree on a clean `main`:

```bash
./scripts/release.sh <version>     # e.g. 0.8.1  (no leading v)
```

`release.sh` bumps the version in every component, regenerates the OpenAPI spec + all client SDKs,
updates docs versioning, commits `Release v<version>`, tags `v<version>`, and **pushes the commit
and tag directly to `main`**. The push triggers the `Release` GitHub Actions workflow that builds
and publishes the packages. It is **not** a PR.

Verify after: `gh run list --limit 5` should show the `Release v<version>` workflow running, and
`git ls-remote --tags origin v<version>` should return the tag.

## Step 2 — Changelog + blog PR (separate)

Done **after** the tag exists, as its own PR (precedent: v0.8.0 = #2053, v0.8.1 = #2080). Work on a
branch off the new `main`:

```bash
git checkout -b docs-changelog-<version> origin/main
```

Only spin up a separate worktree (`git worktree add ../hindsight-changelog-<version> -b
docs-changelog-<version> origin/main`) if you can't get a clean checkout otherwise — e.g. `main` is
held in another worktree and the current one has work you don't want to disturb.

**Branch naming:** use the `docs-` (hyphen) convention, e.g. `docs-changelog-0.8.1`. A remote
branch literally named `docs` exists, so any `docs/...` branch is rejected on push with
`directory file conflict`.

### Changelog

```bash
uv run --directory hindsight-dev generate-changelog <version>
```

LLM-summarizes the commits between the previous tag and `v<version>` and prepends an entry to
`hindsight-docs/src/pages/changelog/index.md`. Requires `OPENAI_API_KEY` (already in the repo
`.env`). It excludes `hindsight-integrations/` source, but new integrations whose commits also
touched docs will still appear — that matches precedent, leave them in the **changelog**.

### Blog post

Hand-write `hindsight-docs/blog/YYYY-MM-DD-version-X-Y-Z.md` (mirror an existing one; patch
releases are short — see `2026-06-02-version-0-7-2.md`). Guidance:

- **Explain user impact, not internals/mechanism.** Lead with what the user can now do and what to
  set. Config/env-var names are fine (developer-facing), code symbols and internals are not.
- **Do not list integrations in the release blog.** The core blog covers core engine / API /
  ops changes; each integration ships its own changelog. (Integrations may still appear in the
  generated `changelog/index.md` — that's fine; just keep them out of the blog.)
- Call out an upgrade recommendation when there are operational/data-integrity fixes.
- Validate formatting: `npx prettier --check <blog file>`.

### Sync the docs skill

```bash
./scripts/generate-docs-skill.sh
```

Refreshes `skills/hindsight-docs/references/changelog/index.md`. It will also bump
`skills/hindsight-docs/references/openapi.json` by one version — `release.sh` regenerates the skill
*before* bumping OpenAPI, so the skill copy lags a version in the release commit; this step syncs
it. Expect a one-line `version` diff there; keep it.

### Commit, push, PR

```bash
git add -A
git commit --no-verify -m "docs: changelog and blog post for v<version>"
git push -u origin docs-changelog-<version>
gh pr create --base main --title "docs: changelog and blog post for v<version>" --body "..."
```

Expected files in the PR: the changelog entry, the new blog post, the regenerated skill changelog
mirror, and the skill `openapi.json` version sync.

## Cleanup

If you created a temporary worktree, remove it once the PR is up
(`git worktree remove ../hindsight-changelog-<version>`; the branch stays on origin). Restore any
stash you popped in Step 0.
