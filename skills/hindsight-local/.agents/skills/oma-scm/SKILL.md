---
name: oma-scm
description: "SCM (software configuration management) and Git: branching, merges, conflicts, worktrees, baselines, audit readiness, plus Conventional Commits and safe staging."
---

# Software configuration management: SCM (`oma-scm`)

## Scheduling

### Goal
Manage Git and software configuration management safely: commits, branches, merges, worktrees, releases, baselines, audit posture, CODEOWNERS, and Conventional Commits.

### Intent signature
- User asks to commit, stage, branch, merge, rebase, cherry-pick, tag, release, resolve conflicts, manage worktrees, inspect SCM posture, or apply Conventional Commits.
- User needs safe Git operations with explicit file staging, secret awareness, and CM governance.

This skill is the **single** place for **configuration management (CM)** on a software repo and for **Conventional Commits** / safe staging.

### When to use

- **Commits:** “commit this”, `/scm`, message type/scope, splitting staged changes into multiple commits.
- **CM / Git:** branching (gitflow, GitHub Flow, GitLab Flow, trunk-based), protected branches, merge queue, merge conflicts, rebase, cherry-pick, worktrees, submodules/subtrees, tags and releases.
- **Governance:** issue/ADR links, breaking-change footers, changelog or release-tool alignment.
- **Audit posture:** signed commits, CI before merge, secret-sensitive paths.

### When NOT to use

- Implementing product or application code -> use the relevant domain skill
- Debugging runtime failures without a Git or CM operation -> use `oma-debug`
- Security, performance, or accessibility review -> use `oma-qa`
- Planning feature requirements or decomposing work -> use `oma-pm`

### Expected inputs
- Git task, desired branch/commit/release operation, and affected files
- Current worktree status, staged diff, branch tracking, config files, and governance constraints
- Optional issue/ADR/PR/release context

### Expected outputs
- Safe commit, branch, merge/rebase guidance, conflict plan, status accounting, or CM audit findings
- Conventional Commit message and explicit staged paths when committing
- Risk notes for shared history, secrets, CODEOWNERS, CI, and release evidence

### Dependencies
- Git CLI and repository metadata
- `config/commit-config.yaml`, `config/cm-config.yaml`, Conventional Commit references, onboarding-risk and CODEOWNERS playbooks

### Control-flow features
- Branches by quick commit path versus full CM/governance path
- Reads Git state and diffs; may write commits, branches, tags, or conflict resolutions
- Requires explicit approval for broad staging, shared-history rewrite, production-destructive operations, or secret-risk paths

## Structural Flow

### Entry
1. Inspect Git status, branch, staged/unstaged changes, and user intent.
2. Choose Quick Path for ordinary commits or Full CM Path for governance/risky history work.
3. Read commit and CM config before enforcing project-specific rules.

### Scenes
1. **PREPARE**: Determine operation type, risk, and affected files.
2. **ACQUIRE**: Read status, diff, logs, config, ownership, and release context.
3. **REASON**: Split changes, choose message/scope, identify CM controls and risks.
4. **ACT**: Stage explicit paths, commit, branch, resolve, or provide CM action plan.
5. **VERIFY**: Check status, staged diff, CI expectations, signatures, secrets, and audit evidence.
6. **FINALIZE**: Report operation result and remaining SCM tasks.

### Transitions
- If user intent is commit-only, follow Quick Path and stop after safe commit.
- If branching/history/release/governance is involved, run Full CM Path.
- If shared history rewrite is requested, require maintainer approval.
- If changes span independent features, split commits unless user requests one commit.

### Failure and recovery
- If worktree is dirty in unrelated files, avoid touching unrelated changes.
- If conflicts exist, resolve markers, test, and preserve target-branch context.
- If secrets are detected or suspected, stop before staging/committing.

### Exit
- Success: requested SCM operation is complete or a safe, auditable plan is delivered.
- Partial success: blockers such as conflicts, missing approval, CI, or secret risk are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read Git state | `READ` | `git status`, diff, log, config |
| Select SCM path | `SELECT` | Quick Path vs Full CM Path |
| Compare change scopes | `COMPARE` | Split by type/scope/feature |
| Validate commit/governance rules | `VALIDATE` | Config and CM controls |
| Stage explicit files | `CALL_TOOL` | `git add <specific-files>` |
| Commit or manage refs | `CALL_TOOL` | Git commit/branch/merge/rebase/tag |
| Write audit notes | `WRITE` | Commit message or CM report |
| Report result | `NOTIFY` | Final SCM summary |

### Tools and instruments
- Git CLI and repository metadata
- Commit/CM config, Conventional Commit guide, CODEOWNERS playbook, onboarding-risk signals

### Canonical command path
```bash
git status -sb
git diff --staged
git log --oneline -5
```

Stage and commit only explicit paths:
```bash
git add <specific-files>
git commit -m "$(cat <<'EOF'
<type>(<scope>): <description>

[optional body]
EOF
)"
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Tracked files, diffs, conflicts, CODEOWNERS |
| `LOCAL_FS` | Git metadata, config files, commit message temp files |
| `PROCESS` | Git commands and verification commands |
| `CREDENTIALS` | Secret-sensitive files must not be staged or committed |

### Preconditions
- Repository and Git intent are identifiable.
- User has authorized the requested SCM operation.

### Effects and side effects
- May stage files, create commits, branches, tags, worktrees, or history operations.
- Can affect shared repository history if unsafe commands are used, so approvals matter.

### Guardrails

1. Choose Quick Path for ordinary commits and Full CM Path for branching, history, release, or governance work.
2. Read `config/commit-config.yaml` and `config/cm-config.yaml` before applying project-specific commit or CM rules.
3. Stage only explicit files; never use broad staging unless the user explicitly approves it.
4. Do not rewrite shared history without maintainer approval.
5. Never stage or commit likely-secret material.
6. **Response language follows `oma-config.yaml` `language`**: user-facing SCM output (status summaries, conflict explanations, CM audit notes, action plans) is localized. Per `i18n-guide.md`, commit messages, PR titles/body, branch names, and status keywords stay in English regardless of the setting.

### Configuration

| File | Role |
|------|------|
| `config/commit-config.yaml` | Conventional Commit types, branch prefixes, message rules |
| `config/cm-config.yaml` | CM pointers (documented process, branching model, baselines, changelog) |

### Operating mode (choose first)

### Quick Path (commit-focused, default)

Use this when the user intent is mainly "commit this safely."

1. Follow **Conventional Commits** section only
2. Stage explicit files only
3. Validate message type/scope/length from `commit-config.yaml`
4. Stop after safe commit unless user asks CM/governance operations

### Full CM Path (repo governance / risky history operations)

Use this when the user asks about branching strategy, merges, rebase/cherry-pick, worktrees, release refs, CODEOWNERS, or audit posture.

1. Run CM workflows in order (Planning -> Identification -> Control -> Status accounting -> Verification)
2. Add onboarding risk scan when inheriting or auditing a repository
3. Include commit governance from Conventional Commits when creating commits
4. For large-scope merge operations, use risk scoring and Ask Gate criteria from `../../workflows/scm.md`

### CM process map (software)

| CM function | Intent | Typical artefacts / actions |
|-------------|--------|------------------------------|
| **Management & planning** | Agreed rules | `CONTRIBUTING.md`, `SECURITY.md`, `cm-config.yaml` |
| **Configuration identification** | What is managed, naming | Branch/tag rules, version files, `.gitattributes`, LFS |
| **Configuration control** | Reviewed change | PRs, checks, issue links, `BREAKING CHANGE` footers |
| **Status accounting** | As-built truth | `main` / release refs, `CHANGELOG`, tags, CI status |
| **Verification & audit** | Evidence | CI logs, signed commits, lockfiles / SBOM policy |

### CM workflows (use before risky history operations)

### 1) Planning

1. Read `cm-config.yaml` and files listed under `documented_process`.
2. If missing, infer from `CONTRIBUTING.md` / `README`; state assumptions.
3. Confirm **branching model** and whether **force-push** on shared branches is allowed (default: not without explicit approval).

### 2) Identification

1. Canonical refs: default branch, release branches/tags, version sources (`package.json`, etc.).
2. `.gitattributes` / LFS for binaries and generated assets.
3. Branch names vs `commit-config.yaml` `branch_prefixes` when the project uses them.

### 3) Control

1. Small, reviewable units; align commits with PR / issue intent.
2. **Conflicts:** `merge-base`, `git status`, resolve markers, tests; suggest `rerere` when conflicts repeat.
3. **Worktrees:** `git worktree add`; merge/rebase from the **target branch’s** checkout; all worktrees share one object database.
4. Do not rewrite **shared** history without maintainer approval; prefer `--force-with-lease` if force-push is unavoidable.

### 4) Status accounting

1. `git status -sb`: branch, remote tracking, ahead/behind, merge state.
2. Relate last tag / release branch to `CHANGELOG` or tooling (semantic-release, release-please, changesets) if present.

### 5) Verification & audit

1. Required CI and `merge_group` when merge queue applies.
2. Never stage/commit secrets (`.env`, keys, raw tokens).
3. Call out signed-commit expectations when the org cares about verification badges.

#### CODEOWNERS maintenance checklist

1. Validate CODEOWNERS file exists (prefer `.github/CODEOWNERS`).
2. Ensure critical paths are explicitly owned (not only fallback `*`).
3. Ensure owners are active and mapped to current teams.
4. Confirm branch protection requires CODEOWNERS review where needed.
5. Flag overlapping/ambiguous rules that can hide intended owners.

Read `change_governance.require_codeowners` and `ownership.*` in `cm-config.yaml` when present.

### 6) Onboarding risk scan (optional, recommended)

Use this quick scan when joining or inheriting a repository to identify risky areas before major changes.

1. High churn files in `lookback` window.
2. Ownership concentration / bus-factor signals.
3. Bug hotspot files from fix-related history.
4. Velocity trend by month.
5. Revert/hotfix/emergency frequency.

Read thresholds from `cm-config.yaml` `onboarding_metrics` when present and cite caveats:
- squash merge teams can distort ownership metrics,
- weak commit labeling reduces hotspot accuracy,
- monorepo commit counts can bias subsystem interpretation.

---

### Conventional Commits

### Commit types

| Type | Description | Branch Prefix |
|------|-------------|---------------|
| feat | New feature | feature/ |
| fix | Bug fix | fix/ |
| refactor | Code improvement | refactor/ |
| docs | Documentation changes | docs/ |
| test | Test additions/modifications | test/ |
| chore | Build, configuration, etc. | chore/ |
| style | Code style changes | style/ |
| perf | Performance improvements | perf/ |

### Commit format

```
<type>(<scope>): <description>

[optional body]

Co-Authored-By: First Fluke <our.first.fluke@gmail.com>
```

### Commit workflow

#### Step 1: Analyze changes

```bash
git status
git diff --staged
git log --oneline -5
```

#### Step 1.5: Split by feature (if needed)

If changes span multiple features/domains, **split commits by feature**.

**Split when:** different scopes, different types, logically independent work.

**Do not split when:** one feature, few files (≤5), or user asked for a single commit.

#### Step 2: Determine type

- New capability → `feat` · Bug fix → `fix` · Structure-only → `refactor` · Docs only → `docs` · Tests → `test` · Build/config → `chore`

#### Step 3: Scope

Use module/component: `feat(auth):`, `fix(api):`, or omit: `chore: update dependencies`

#### Step 4: Description

≤72 chars (per `commit-config.yaml`), imperative mood, lowercase start, no trailing period.

#### Step 5: Execute commit

Show the message, then commit with explicit paths:

```bash
git add <specific-files>
git commit -m "$(cat <<'EOF'
<type>(<scope>): <description>

[optional body]
EOF
)"
```

If HEREDOC is unstable in your shell (or body is long), use file-based commit input:

```bash
git add <specific-files>
cat > /tmp/oma-commit-msg.txt <<'EOF'
<type>(<scope>): <description>

[optional body]
EOF
git commit -F /tmp/oma-commit-msg.txt
```

Use HEREDOC by default, and switch to `-F` for long or flaky terminal sessions.

## References

- `config/commit-config.yaml`
- `config/cm-config.yaml`
- `resources/conventional-commits.md`
- `resources/onboarding-risk-signals.md`
- `resources/codeowners-playbook.md`
- Observability handoff: `../oma-observability/SKILL.md` §Integrations — release markers (`service.version`), revert baseline diff

### Important notes

- **NEVER** `git add -A` or `git add .` without explicit user permission.
- **NEVER** commit likely-secret material.
- **ALWAYS** stage by explicit paths; tie non-trivial CM work to the five CM rows above, even briefly.
