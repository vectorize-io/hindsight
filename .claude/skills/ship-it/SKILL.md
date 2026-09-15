---
name: ship-it
description: Take a PR from review to merged — run the repo's code-review skill on it, apply ALL fixes on the PR branch, wait for CI green, then squash-merge. Use when asked to "ship it", "ship PR #N", or "review, fix and merge" a PR.
user_invocable: true
---

# Ship It

Input: a PR number or URL, or nothing — then use the current branch's PR (`gh pr view` with no
argument resolves it). If the current branch has no PR yet, push it and open one with
`gh pr create` (maintainer branch only), then continue. Goal: the PR merged with every code-review finding fixed and CI green.

## 0. Blocker rule (applies at every step)

A **blocker** is a finding whose fix would go *against the PR's goal* — the code-review
principles say the thing the PR sets out to do is wrong (e.g. the PR's whole point is a raw-SQL
handler, a new advisory lock, a tuple-returning API, a test that asserts via SQL, dropping bank
scoping), or fixing it would mean rewriting/removing the feature rather than polishing it.

On a blocker: **stop immediately.** Don't push, don't merge, don't partially fix. Tell the user
what the PR is trying to do, which principle it conflicts with (quote the `code-review` section),
and the options you see — then ask for confirmation with AskUserQuestion. Resume only on an explicit answer.

Everything else (must fix, should fix, nits) is not a blocker: fix it all.

## 1. Load the PR

Read the PR — title, body, linked issue, diff — so you know its goal; the blocker rule depends on it.
If it's `CONFLICTING`, rebase onto `origin/main` first: a conflicting PR gets **zero CI runs**, silently.

Work out whether the author is a maintainer (write/maintain/admin on the repo, e.g. `nicoloboschi`).

## 2. Check out the PR branch

Work in the **current worktree** — don't create a new worktree or clone. Before switching, note its
state (current branch/commit and any uncommitted changes) so you can restore it in step 6; set
uncommitted work aside with a WIP commit or a uniquely tagged stash (never a bare `git stash` — the
stash stack is shared across worktrees). Then check out the PR's branch (skip if you're already on
it). On current-branch input, ask the user if uncommitted work isn't obviously part of the PR.

- **Maintainer PR:** fixes are committed and pushed to *this* branch — the PR's own head. Never open a separate fix PR.
- **External (fork) PR:** push to the fork branch too if maintainers can modify it; otherwise stop
  and ask the user how to proceed (follow-up PR vs. ask the contributor).

Before every push, make sure the author hasn't pushed meanwhile. If they have, read their new
commits and rebase yours onto their head — never force-push over work you haven't read.

## 3. Review

Read and follow the repo's code-review skill at its absolute path,
`<repo>/.claude/skills/code-review/SKILL.md` (`<repo>` = `git rev-parse --show-toplevel`), against the PR's changes
(base = `origin/<baseRefName>`). Collect every finding: must fix, should fix, and nits.

Classify each against step 0. Any blocker → stop and ask.

## 4. Apply ALL fixes

- Fix every finding, not just the must-fixes. Stay in the PR's scope — don't refactor neighbouring code the review didn't flag.
- Run `./scripts/hooks/lint.sh` and the tests covering the touched code (see CLAUDE.md for commands).
  Regenerate OpenAPI/clients/docs-skill if the change requires it.
- Re-run `code-review` on the result; repeat until it's clean (or only blockers remain → stop and ask).
- Commit with a message that lists what was fixed, then push to the PR branch.

Don't post a review comment on a maintainer's PR — findings go in the chat reply and the commit
message. On an external PR, one short comment summarising the fixes you pushed is fine.

## 5. Wait for CI green

The repo has **no required status checks**: `gh pr merge --auto` merges immediately, and
`gh pr checks` reports only the checks registered so far (a lone early Strix check reads as
"green"). Gate on the **CI workflow run for the pushed head SHA**:

```bash
SHA=$(gh pr view <N> --json headRefOid --jq .headRefOid)
gh run list --branch <headRefName> --workflow CI --json databaseId,status,conclusion,headSha \
  | jq --arg s "$SHA" 'map(select(.headSha==$s)) | .[0] // empty'
```

No run yet → keep waiting (give up and report after ~15 min with no run). Once `completed`, judge
**per job** (`gh run view <id> --json jobs`): zero `failure`. If the PR touches the API/engine, `test-api (1..3/3)`,
`Core LLM tests`, and the LLM acceptance matrix must be `success`, not `skipped`; for changes outside
those paths the change filter skips them legitimately. The oracle-client jobs are known-flaky and may
be `cancelled` — ignore those.

**Fork PRs skip `test-api` and the LLM jobs** (no secrets). Run the full suite on an upstream branch
before merging — merging to main runs no CI at all:

```bash
git push origin HEAD:ci/pr-<N>-verify
gh workflow run CI --ref ci/pr-<N>-verify   # exactly ONCE — a second dispatch cancels the first
```

A failure caused by the PR/fixes → fix, push, back to step 5. A known flake (check the job log isn't
touching the PR's code) → `gh run rerun <id> --failed` once the run has completed.
If CI still fails for reasons you can't attribute, stop and report — don't merge red.

## 6. Restore the worktree, then merge

First restore the worktree to how you found it: switch back to the original branch/commit and
re-apply any work you set aside (undo the WIP commit / apply-then-drop your tagged stash). Do the
same if you stop early on a blocker or failure. Restoring first also keeps `--delete-branch` from
trying to check out `main`, which another worktree may hold.

```bash
gh pr merge <N> --squash --delete-branch
git push origin --delete ci/pr-<N>-verify   # if you created it
```

Report: PR link, the findings fixed (one line each), anything deliberately left, and the CI run
that gated the merge.
