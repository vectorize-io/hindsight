# Skill Routing Map

Routing rules for oma-orchestrator and oma-coordination to assign tasks to the correct agent.

## Progressive Disclosure

Skills use two-stage loading to optimize context usage:

1. **Stage 1 (always loaded)**: `name` and `description` from SKILL.md frontmatter
2. **Stage 2 (on explicit invocation)**: Full SKILL.md body loaded only when skill is explicitly requested via /command or agent skills field

Skills are explicitly loaded via /command invocation or agent skills field. Load full instructions only for explicitly requested skills.

---

## Skill → Agent Mapping

| Skill Domain | Primary Skill | Notes |
|----------------------|---------------|-------|
| API, endpoint, REST, GraphQL, database, migration | **oma-backend** | |
| auth, JWT, login, register, password | **oma-backend** | Auth UI task can also be created for frontend |
| UI, component, page, form, screen (web) | **oma-frontend** | |
| style, Tailwind, responsive, CSS | **oma-frontend** | |
| mobile, iOS, Android, Flutter, React Native, app | **oma-mobile** | |
| offline, push notification, camera, GPS | **oma-mobile** | |
| architecture, system design, software design, module boundary, service boundary, tradeoff, ADR, ATAM, CBAM, quality attribute | **oma-architecture** | Consult before planning when the structure itself is undecided |
| bug, error, crash, broken, slow | **oma-debug** | |
| review, security, performance | **oma-qa** | |
| accessibility, WCAG, a11y | **oma-qa** | |
| UI design, design system, landing page, DESIGN.md, color palette, typography, glassmorphism, responsive design | **oma-design** | |
| brainstorm, ideate, design, explore, idea, concept | **oma-brainstorm** | Run before oma-pm |
| plan, breakdown, task, sprint | **oma-pm** | |
| automatic, parallel, orchestrate | **oma-orchestrator** | |
| workflow, guide, manual, step-by-step | **oma-coordination** | |
| configuration management, SCM, CM, git, commit, gitflow, GitHub Flow, GitLab Flow, trunk-based branching, merge conflict, rebase, worktree, baseline, tag, release branch, signed commits, merge queue, conventional commits | **oma-scm** | SCM + Conventional Commits in one skill |

---

## Complex Request Routing

| Request Pattern | Execution Order |
|----------------|-----------------|
| "Create a fullstack app" | oma-pm → (oma-backend + oma-frontend) parallel → oma-qa |
| "Create a mobile app" | oma-pm → (oma-backend + oma-mobile) parallel → oma-qa |
| "Fullstack + mobile" | oma-pm → (oma-backend + oma-frontend + oma-mobile) parallel → oma-qa |
| "Help me choose the system architecture" | oma-architecture → oma-pm |
| "Review this architecture before we build" | oma-architecture → oma-pm → oma-qa |
| "Fix bug and review" | oma-debug → oma-qa |
| "Add feature and test" | oma-pm → relevant agent → oma-qa |
| "I have an idea for a feature" | oma-brainstorm → oma-pm → relevant agents → oma-qa |
| "Let's design something new" | oma-brainstorm → oma-pm → relevant agents → oma-qa |
| "Do everything automatically" | oma-orchestrator (internally oma-pm → agents → oma-qa) |
| "I'll manage manually" | oma-coordination |
| "Design and build a landing page" | oma-design → oma-frontend |
| "Design, build, and review" | oma-design → oma-frontend → oma-qa |
| "Redesign based on this URL" | oma-design (Phase 2 EXTRACT) → oma-frontend |

---

## Inter-Agent Dependency Rules

### Parallel Execution Possible (No Dependencies)
- oma-backend + oma-frontend (when API contract is pre-defined)
- oma-backend + oma-mobile (when API contract is pre-defined)
- oma-frontend + oma-mobile (independent of each other)

### Sequential Execution Required
- oma-architecture → oma-pm (architecture decision comes before task decomposition)
- oma-brainstorm → oma-pm (design comes before planning)
- oma-pm → all other agents (planning comes first)
- implementation agent → oma-qa (review after implementation complete)
- implementation agent → oma-debug (debugging after implementation complete)
- oma-backend → oma-frontend/oma-mobile (when executing parallel without API contract)

### QA Is Always Last
- oma-qa runs after all implementation tasks are complete
- Exception: Can run immediately if user requests review of specific files only

---

## Escalation Rules

| Situation | Escalation Target |
|-----------|------------------|
| Agent finds bug in different domain | Create task for oma-debug |
| QA finds CRITICAL issue | Re-run relevant domain agent |
| Architecture change needed | oma-architecture → oma-pm |
| Performance issue found (during implementation) | Current agent fixes, oma-debug if severe |
| API contract mismatch | oma-orchestrator re-runs oma-backend |

---

## Turn Limit Guide by Agent

| Agent | Default Turns | Max Turns (including retries) |
|-------|--------------|------------------------------|
| oma-pm | 10 | 15 |
| oma-backend | 20 | 30 |
| oma-frontend | 20 | 30 |
| oma-mobile | 20 | 30 |
| oma-architecture | 12 | 18 |
| oma-debug | 15 | 25 |
| oma-qa | 15 | 20 |
