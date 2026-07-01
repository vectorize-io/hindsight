# PR review: `process --diff` for CI gating

Use direct mode when you want a fast, scoped read of the files changed in a PR rather than a whole-repo audit.

```bash
bunx deepsec process --diff origin/main
```

## How direct mode differs from a full scan

| Step | What it looks at | What it produces |
|---|---|---|
| Resolve files | `--diff` / `--diff-staged` / `--diff-working` / `--files` / `--files-from` | POSIX-relative file list under `rootPath` |
| Scoped scan | Only the listed files | Candidates as **prompt signals** (best-effort) |
| Always-process | The same listed files | AI findings, including files no matcher hit |

Files with no regex hits still get a record and still get investigated as a holistic review.

## Diff sources (mutually exclusive)

| Flag | Meaning |
|---|---|
| `--diff <ref|range>` | `git diff --name-only <ref>` (e.g. `origin/main`, `HEAD~1..HEAD`) |
| `--diff-staged` | Index vs HEAD |
| `--diff-working` | Uncommitted + untracked |
| `--files <csv>` | Explicit comma-separated list |
| `--files-from <path>` | Newline-delimited list (or `-` for stdin) |

Other knobs:

| Flag | Effect |
|---|---|
| `--no-ignore` | Bypass the default ignore filter (test files, `dist/`, `node_modules/`, …) |
| `--comment-out <path>` | Write a PR-comment-shaped markdown summary to `<path>` (only when findings exist) |
| `--project-id <id>` | Override project id (auto-derived from `rootPath` basename otherwise) |
| `--root <path>` | Override project root |

The usual `--agent`, `--model`, `--concurrency`, `--batch-size`, `--max-turns` flags work the same as in standard mode.

## Auto-created projects

You do not need to run `deepsec init` first. With a direct-mode flag, `process` will:

1. Use `--project-id` if you pass one (if declared in `deepsec.config.ts`, the declared root is used; otherwise `--root` or cwd).
2. Otherwise derive the id from the resolved root's basename.
3. Write `data/<id>/project.json` if absent.

Auto-creation is one-line and non-destructive. It never modifies your `deepsec.config.ts`.

## Exit codes (gating contract)

| Code | Meaning |
|---|---|
| `0` | No findings produced in this run |
| `1` | At least one **net-new** finding produced |
| other | Runtime error (bad input, missing credentials, …) |

**Net-new findings only** count toward the exit code. Re-running on a file with existing findings does not fail the build unless something new is surfaced. Pre-existing findings on touched files are intentionally excluded.

## PR-comment markdown

`--comment-out <path>` writes a markdown body summarizing the **net-new** findings only (same scope as the exit-code gate). Descriptions and recommendations are truncated (600 / 400 chars) to stay under GitHub's 65 KiB comment limit; full text remains in `data/<id>/files/`.

The file is only written when there are findings, so a green run leaves nothing on disk and your "post comment" step can short-circuit on `if: hashFiles('comment.md') != ''`.

## Two-job CI pattern (recommended)

Keep PR-controlled code in a no-write job; let a second, code-free job post the comment.

```yaml
name: deepsec

on: pull_request

permissions:
  contents: read

jobs:
  analyze:
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # need history for `git diff origin/<base>`

      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24, cache: pnpm }

      - run: pnpm install --frozen-lockfile
      - run: npm install -g @anthropic-ai/claude-code

      - id: deepsec
        env:
          AI_GATEWAY_API_KEY: ${{ secrets.AI_GATEWAY_API_KEY }}
          CLAUDE_CODE_EXECUTABLE: claude
        run: |
          pnpm deepsec process \
            --diff origin/${{ github.event.pull_request.base.ref }} \
            --comment-out comment.md

      - if: always() && hashFiles('comment.md') != ''
        uses: actions/upload-artifact@v4
        with:
          name: deepsec-comment
          path: comment.md
          retention-days: 1

  comment:
    needs: analyze
    if: always() && needs.analyze.result == 'failure'
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
      pull-requests: write
    steps:
      - id: dl
        continue-on-error: true
        uses: actions/download-artifact@v4
        with:
          name: deepsec-comment

      - if: steps.dl.outcome == 'success'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: fs.readFileSync('comment.md', 'utf8'),
            });
```

> Swap `pnpm deepsec` for `bunx deepsec` / `npx -y deepsec` / `yarn deepsec` to match the project's package manager. If using `bun`, replace the `pnpm/action-setup` + `setup-node(cache: pnpm)` block with `oven-sh/setup-bun` and `bun install --frozen-lockfile`.

### Why the split

- **`analyze`** runs PR-controlled code (the user's `pnpm install`, their config, their source) with the AI gateway secret in scope but **no write permissions on the repo**.
- **`comment`** has `pull-requests: write` but never runs any PR code; it consumes only the sanitized `comment.md` artifact.
- A malicious PR cannot combine "execute arbitrary code" with "write to the repository" in a single privileged step.

## Threat-model notes

- **Do not grant `pull-requests: write` to a job that runs PR code.** A PR can add arbitrary code to its own `package.json` postinstall scripts or to a project config the CLI loads. Both run before any of your steps.
- **Pin actions to full SHAs in production.** The example uses major-version tags for readability. Swap each tag for the action's full commit SHA so a compromised tag cannot pivot into your secret-bearing job. (See GitHub's hardening guide.)
- **Same-repo-only gate** (`if: github.event.pull_request.head.repo.full_name == github.repository`) skips fork PRs, which already do not receive secrets under `pull_request`. Pure UX cleanup.
- **The AI gateway secret still flows through PR code** in `analyze`. The `author_association` / same-repo gate is what prevents that from being a vulnerability. For defense-in-depth, run `analyze` only after a label is applied:
  ```yaml
  if: contains(github.event.pull_request.labels.*.name, 'review-ok')
  ```

## Cost notes

Wide diffs are expensive: every file pays for an AI investigation.

- For PRs against `main`, scope to the merge base (`origin/main`), **not** the entire branch ancestry.
- Drop generated / fixture files via `--files-from`:
  ```bash
  git diff --name-only origin/main \
    | grep -v '^generated/' \
    | bunx deepsec process --files-from -
  ```
- Add stable noise paths to ignore patterns in `data/<id>/config.json:ignorePaths` so they never enter the diff.

## When NOT to use direct mode

- **Initial sweep of a large repo.** Full `scan` + `process` orders by noise tier, parallelizes better, and benefits from whole-repo signal in matcher gating. Direct mode is for incremental review.
- **Revalidating existing findings.** Use `revalidate` with its own filters.
