---
name: ship-it
description: Take a PR from review to merged — run the repo's code-review skill on it in a loop (review, fix, re-review) until nothing is left to fix, applying ALL fixes on the PR branch, wait for CI green, then squash-merge. Use when asked to "ship it", "ship PR #N", or "review, fix and merge" a PR.
user_invocable: true
---

# Ship It

Input: a PR number or URL, or nothing — then use the current branch's PR (`gh pr view` with no
argument resolves it). If the current branch has no PR yet, push it and open one with
`gh pr create` (maintainer branch only), then continue. Goal: the PR merged with every code-review finding fixed and CI green.

## 0b. A figure in the description, when the change has a shape

If the PR changes how something *flows* — a request path, a background job, what a feature does step
by step — a picture explains it faster than the diff does. Use the `figure` skill: write a JSON
spec, render one animated SVG, and put it at the top of the PR description. It plays in the
description with no upload and no click.

Reference it by **commit SHA**, not branch name, or the image dies when the branch is deleted at
merge: `https://raw.githubusercontent.com/<owner>/<repo>/<sha>/<path>.svg`. If the SVG is not part
of the change itself, don't commit it to the repo — render it, commit it on the PR branch only if it
belongs there, and otherwise keep the description's image pinned to the commit that carried it.

Skip it for a fix with no shape: a one-line guard, a dependency bump, a typo.

## 0. Blocker rule (applies at every step)

A **blocker** is a finding whose fix would go *against the PR's goal* — the code-review
principles say the thing the PR sets out to do is wrong (e.g. the PR's whole point is a raw-SQL
handler, a new advisory lock, a tuple-returning API, a test that asserts via SQL, dropping bank
scoping), or fixing it would mean rewriting/removing the feature rather than polishing it.

On a blocker: **stop immediately.** Don't push, don't merge, don't partially fix. Tell the user
what the PR is trying to do, which principle it conflicts with (quote the `code-review` section),
and the options you see — then ask for confirmation with AskUserQuestion. Resume only on an explicit answer.

Everything else (must fix, should fix, nits) is not a blocker: fix it all.

## 0c. Track everything you start (applies at every step)

Whatever you launch to verify the PR — API, worker, control plane, docs server, a pg0 instance —
you must stop yourself at the end (step 7), without touching anyone else's. Other Hindsight
worktrees, servers and pytest runs share this machine.

- Start background processes so you know their PID (`run_in_background`, or `cmd & echo $!`) and
  write each one down. Note the ports you picked.
- Need a database? Give it a **new, unique pg0 instance name** (e.g.
  `HINDSIGHT_API_DATABASE_URL=pg0://ship-it-pr-<N>`) and write down that you created it. If you reuse
  an instance that already existed (check `pg0 list` first), write down that it is **shared**.
- Never `pkill -f hindsight`, `pkill -f pytest`, `killall`, or kill by port without checking the
  PID is one of yours. Those hit other sessions.

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
- **Loop until clean.** One review pass is never enough: fixes introduce new findings, and the
  review only sees what the last pass changed. So repeat — review → fix everything → review again —
  until a full `code-review` pass returns **zero** findings of any severity (must fix, should fix,
  nits). Don't stop at "only nits left", don't stop because the last pass found fewer things, and
  don't declare done on a pass you didn't actually re-run. The only early exit is a blocker (step 0)
  → stop and ask.
- **Follow-up work is done here, not later.** If the PR needs more to be complete — a missing
  piece the review or you found, a doc page, a client regen, a test — do it in this ship-it, on
  this branch. "Follow-up PR" is not an option for work that belongs to this change. Only work
  that is truly a different problem goes to an issue (step 7).
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

## 7. Clean up, file issues, report

Do this also when you stop early (blocker, red CI).

**Stop what you started — only that.** Kill each PID you wrote down in step 0c (`kill <pid>`; check
it's gone, `kill -9 <pid>` only if it hangs). Before killing, confirm with `ps -p <pid> -o command=`
that it is still your process. For each pg0 instance:

- **Created for this ship-it** → `pg0 drop --name <name> -f` (stops it and deletes its data).
- **Shared / already existed** → don't touch it. Not stop, not drop.

Then check nothing of yours is left: `lsof -nP -iTCP:<port> -sTCP:LISTEN` for each port you used.

**File issues for real project problems you ran into.** While reviewing, testing or waiting on CI
you may have hit problems that are *not* this PR's job and *not* your local setup: a flaky test, a
bug in nearby code, a broken script, wrong docs, a CI job that is always red. For each one:

1. Search first so you don't duplicate: `gh issue list --state open --search "<keywords>"`.
2. If none exists, create it: `gh issue create --title ... --body ...`. Body: what happens, how to
   see it (command or steps), what should happen, and where in the code. Keep it short.

Local problems (your env, missing tools, a port in use, a stale `.env`) are not issues — mention
them in the report only if the user needs to act.

**Report:**

- PR link and the CI run that gated the merge.
- The findings fixed (one line each), and anything deliberately left, with why.
- **Issues found on the project** — a clearly marked section listing each issue you created (link +
  one line), and any existing issue you found instead of creating one. Say "none" if none.
- What you stopped and dropped, and any shared pg0 instance you left alone.

## Note on `gh pr edit`

It fails on this repo — a GraphQL "Projects (classic)" deprecation aborts it and nothing changes,
silently. To edit a description use the REST API instead:

```bash
gh api -X PATCH repos/<owner>/<repo>/pulls/<N> -F body=@body.md
```
