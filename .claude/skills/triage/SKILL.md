---
name: triage
description: Triage a GitHub issue or a reported problem — check it is real and not by design (code, git history, PRs, Hindsight memory), find the code and patterns to reuse, build a proof-of-concept fix and prove it solves the problem, then explain the problem and the fix in behaviours before doing the full fix. Use when asked to "triage #N", "look into this issue", or "is this a real bug".
user_invocable: true
---

# Triage

Input: a GitHub issue (number or URL) or a problem described in words. Goal: the user knows whether
the problem is real, what it does, and what a fix would change — backed by a working POC — and
decides before you do the full fix.

**Don't do the full fix in this skill.** It ends at step 5, with the user's answer.

## 1. Is it real, and is it a bug?

Read the issue fully (`gh issue view <N> --comments`): what the reporter saw, what they expected,
version, config. Then check both sides:

- **Reproduce it.** Find the smallest way to make it happen: a test, a script, an API call. If you
  can't reproduce it, say so and say what you tried.
- **Check whether it's by design.** The code shows what happens, not why. Look for the decision:
  - `git log -S '<key string>' --oneline` / `git log -L` / `git blame` on the lines involved, then
    read those commits.
  - The PRs behind them: `gh pr list --state all --search "<keywords>"`, `gh pr view <N> --comments`.
  - Related or duplicate issues: `gh issue list --state all --search "<keywords>"`.
  - Hindsight memory, if the plugin is installed (the `hindsight_*` tools are available): search the
    knowledge pages and, when you need the *why*, run a reflect. Treat what it says as a past
    record — check it against the code.

## 2. If it makes no sense, stop here

If it's not real, already fixed, a duplicate, by design, a local setup problem, or not worth
changing: tell the user plainly why, with the evidence (commit, PR, issue, line of code). Propose
closing it with a short, kind comment, written out ready to post, e.g.:

> This is by design: <one-line reason> (see #<PR>). <What to do instead, if anything.> Closing — feel free to reopen if I missed something.

Ask before posting or closing — never do it on your own.

## 3. Explore the code

Before writing anything, find what is already there:

- Every caller of the code involved, and every path that can hit the same problem — fix it where
  they all route through, not only on the path the issue names.
- Similar code that already solves the same kind of problem, and helpers, types and patterns you can
  reuse. Read `.claude/skills/code-review/SKILL.md` so the POC already follows the repo's rules.
- The tests that cover this area, and how they assert.

## 4. Build a POC fix and prove it

Work on a new branch off `origin/main` (`triage/<issue-number-or-short-slug>`), so the POC never
mixes with other work and can become the full fix later.

**No workarounds.** The POC — and the plan built on it — must be the long-term fix: the one you'd
want in the codebase in a year. Not a retry around a race, a special case for the reporter's input,
a bigger timeout, a swallowed error, a flag to turn the problem off, or a guard at one call site
when the cause sits underneath. Think it through first: why does the code allow this at all, what's
the right place and design to fix it, and does the fix still hold when the data grows or a new
caller shows up? If the right fix is big, the POC can cover the core of it — but it must be that
fix, not a stand-in. If you see a quick patch too, you may mention it, but only as "not
recommended", and say why.

Then prove it:

- The reproduction from step 1 fails before the change and passes after it.
- Nearby tests still pass.
- It fixes the whole problem, not one symptom. Try the other paths you found in step 3.

**Performance issue:** recreate the reporter's situation at the same or a smaller scale (same data
shape, same operation, same config), measure it before the change, then after, the same way, a few
runs each. Use the repo's perf tools when they fit (`./scripts/benchmarks/`,
`hindsight-dev/benchmarks/`), otherwise a small script. Show numbers side by side (before / after /
change). No numbers, no plan.

**UI change:** start the API and the control plane so the user can see it. Use free ports (check
with `lsof -nP -iTCP:<port> -sTCP:LISTEN` — other worktrees often hold 8888/9999) and a new,
unique pg0 instance (`HINDSIGHT_API_DATABASE_URL=pg0://triage-<issue-number-or-short-slug>`).
Write down each PID and the pg0 instance. Seed the data needed to show the change, open it in the browser if browser tools are
available, and give the user the link.

**Services you start:** stop only your own — kill your PIDs (never `pkill -f hindsight`/`pytest`),
and `pg0 drop --name <name> -f` only on instances you created; never touch a shared one. Leave the
UI running while the user looks; stop everything once they are done or the triage ends.

## 5. Explain, then ask

Tell the user, in plain words and in **behaviours** (what a user or agent sees and does, not
function names):

1. **The problem** — what happens today, step by step, and when.
2. **The consequences** — who it hits, how badly, what goes wrong because of it (wrong answers, lost
   data, slow calls, cost).
3. **The fix** — what the POC changes and why that removes the cause, not just the symptom.
4. **What behaviour changes** — everything that will act differently after the fix, including
   things that aren't the bug (a new default, a slower path, a different error). Say "nothing else"
   only if you checked.
5. **The evidence** — the reproduction before/after, test results, benchmark numbers, the UI link.
6. **What the full fix still needs** — tests, docs, migrations, client regen, other call sites.

Then ask whether to go ahead with the full fix (AskUserQuestion). Don't start it before they answer.
