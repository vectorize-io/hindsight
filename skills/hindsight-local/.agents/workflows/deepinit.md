---
name: deepinit
description: Initialize project harness with AGENTS.md as table of contents, ARCHITECTURE.md as domain map, and a structured docs/ knowledge base
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 0 in order. Explicitly report completion of each step before proceeding.
- **You MUST use MCP tools throughout the entire workflow.** This is NOT optional.
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `search_for_pattern`, `list_dir`) for code exploration.
  - Use file writing tools to generate all output files.
- **Exclude directories:**
  - Respect the project's `.gitignore` as the source of truth for excluding directories.
  - Automatically skip framework-generated cross-platform build/project directories (e.g., Flutter/React Native's `android`, `ios`, `macos`, `linux`, `windows`, `web` folders).

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native file exploration, code analysis, and file writing tools.

---

## Core Philosophy

**AGENTS.md is a table of contents, not an encyclopedia.**

Rather than one giant instruction file, the repository's knowledge base lives in a structured `docs/` directory.
A short AGENTS.md (~100 lines) serves as a map with pointers to deeper sources of truth.

Three categories of documentation:

1. **Maps**: ARCHITECTURE.md, system topology, domain boundaries
2. **Plans**: active and completed work plans (status field tracked per file), tech debt tracking
3. **Design Specifications**: indexed architectural decisions, core beliefs, product specs

Agents can discover file listings and directory structures via tools.
What agents CANNOT discover from code alone, and what the harness must provide:

- Why architectural decisions were made
- Which patterns are forbidden and why
- Dependency direction rules between layers/modules
- Product sense, design principles, quality expectations
- How to work correctly in each domain

---

## Target Output Structure

```
AGENTS.md                           ← table of contents (~100 lines)
ARCHITECTURE.md                     ← top-level domain map and package layering
docs/
├── design-docs/                    ← indexed, verified architectural decisions
│   ├── index.md
│   ├── core-beliefs.md             ← agent-first operating principles
│   └── {decision-name}.md
├── plans/                          ← plan artifacts (local working notes; add to the target repo's .gitignore)
│   ├── designs/                    ← permanent design references (Status: Approved/Draft)
│   │   └── {NNN}-{name}.md
│   └── work/                       ← execution plans (Status: Active/Completed)
│       ├── {NNN}-{name}.md
│       └── tech-debt-tracker.md    ← known debt with priority and rationale
├── generated/                      ← auto-generated docs (DB schema, API specs, etc.)
│   └── db-schema.md
├── product-specs/                  ← product specifications
│   ├── index.md
│   └── {feature-name}.md
├── references/                     ← external library docs reformatted for LLMs
│   └── {library}-llms.txt
├── DESIGN.md                       ← design system, UI/UX principles
├── FRONTEND.md                     ← frontend domain guide
├── PLANS.md                        ← planning process and conventions
├── PRODUCT-SENSE.md                ← product thinking, user mental models
├── QUALITY-SCORE.md                ← quality grades per domain/layer, gap tracking
├── RELIABILITY.md                  ← reliability standards, SLOs, error budgets
├── SECURITY.md                     ← security policies, threat model, auth patterns
└── CODE-REVIEW.md                  ← code review standards and checklist
```

Not all files are required. Generate only what is **discoverable and relevant** to the project.

---

## Step 0: Preparation

1. Check if `AGENTS.md`, `ARCHITECTURE.md`, or `docs/` already exists. If so, this is an **update run** (see Step 6).

---

## Step 1: Analyze Codebase

**Goal:** Understand the project's architecture, domains, patterns, and implicit rules.

1. **Identify project type and structure** using `list_dir` at root and key directories.
   - Monorepo? Single app? Library?
   - What packages/apps/services exist?
   - What tech stacks are used?

2. **Discover architectural patterns** using `get_symbols_overview` and `search_for_pattern`:
   - Layer structure (e.g., controllers → services → repositories)
   - Module boundaries and dependency direction
   - Naming conventions in use
   - Test organization strategy

3. **Identify implicit rules** (patterns consistently followed but not documented):
   - Import restrictions, export patterns
   - Error handling conventions
   - State management approach
   - Code organization patterns per domain

4. **Assess domains** (which areas of the codebase need domain-specific guidance):
   - Frontend, backend, mobile, infra, etc.
   - Design system, product flows
   - Security-sensitive areas
   - Reliability-critical paths

5. **Detect boundaries** (where boundary `AGENTS.md` files are needed):
   - Root (always)
   - Each package/app in monorepo
   - Major architectural boundaries
   - NOT every subdirectory

Report findings to the user before proceeding.

---

## Step 2: Generate ARCHITECTURE.md

**Top-level domain map and package layering.**

### Content

- Domain map: what major domains exist, what each owns
- Package/module layering with dependency direction (e.g., `Types → Config → Repo → Service → Runtime → UI`)
- Key integration points between domains
- Data flow between major components
- Infrastructure topology (if applicable)

### Rules

- Keep under 200 lines.
- Focus on **topology and relationships**, not implementation details.
- Use diagrams (mermaid/ascii) where they clarify relationships.
- This is the first thing an agent reads to understand "what connects to what."

---

## Step 3: Generate `docs/` Knowledge Base

Generate only the files that are **relevant and discoverable** from the codebase.

### `docs/design-docs/`

**Indexed, verified architectural decisions.**

- **`index.md`**: catalogue of all design docs with status (draft/verified/superseded)
- **`core-beliefs.md`**: agent-first operating principles for this project. What defines how work is done here. Examples:
  - "Agents write all code; humans review and set direction"
  - "Every change must be verifiable by CI alone"
  - "Prefer explicit over implicit; no magic"
- **`{decision-name}.md`**: individual architectural decisions with context, options considered, rationale, and consequences

### `docs/plans/`

**Structured plan artifacts (local working notes; add `docs/plans/` to the target repo's `.gitignore` if not already ignored).**

Folder = type. Status field = lifecycle. Filenames use a 3-digit zero-padded sequential prefix per folder.

- **`designs/{NNN}-{name}.md`**: permanent design references (architecture, API specs, tradeoffs)
- **`work/{NNN}-{name}.md`**: execution plans with progress, decision log, and `Status` header (`Active` → `Completed`)
- **`work/tech-debt-tracker.md`**: known tech debt with priority, rationale, and proposed resolution

Initially: create `designs/` and `work/` subdirectories + `work/tech-debt-tracker.md` with any debt discovered in Step 1.

### `docs/generated/`

**Auto-generated documentation.**

- Database schemas, API specs, dependency graphs (anything derivable from code that is expensive to re-derive).
- Mark each file with generation method and timestamp.
- Initially: create directory. Populate only if generation sources exist (e.g., Prisma schema → db-schema.md).

### `docs/product-specs/`

**Product specifications.**

- **`index.md`**: catalogue of product specs
- **`{feature-name}.md`**: user-facing feature specs with acceptance criteria
- Initially: create directory + index.md. Populate from discovered product-facing code.

### `docs/references/`

**External library docs reformatted for LLMs.**

- Format: `{library}-llms.txt` (plain text, LLM-optimized)
- Contains key API surfaces, patterns, and gotchas for external dependencies the project relies on heavily.
- Initially: create directory. Identify key external deps from package.json/requirements.txt and note which ones would benefit from LLM-formatted references.

### Domain Docs (root-level in `docs/`)

Generate only those relevant to the project:

| File | When to Generate | Content |
|------|-----------------|---------|
| `DESIGN.md` | Project has UI/design system | Design system principles, component patterns, visual language |
| `FRONTEND.md` | Project has frontend | Frontend architecture, rendering strategy, state management, routing |
| `PLANS.md` | Always | Planning process conventions, how to write plans, template |
| `PRODUCT-SENSE.md` | User-facing product | Product thinking, user mental models, prioritization framework |
| `QUALITY-SCORE.md` | Always | Quality grades per domain/layer with gap tracking over time |
| `RELIABILITY.md` | Has backend/infra | Reliability standards, SLOs, error budgets, incident response |
| `SECURITY.md` | Has auth/data handling | Security policies, threat model, auth patterns, data handling rules |
| `CODE-REVIEW.md` | Always | Code review standards, checklist, and review behavior configuration |

### `docs/CODE-REVIEW.md`

**Code review standards referenced from AGENTS.md.**

Ensures consistent review behavior across all agents and contributors. Content:

- Review checklist (what to check in every PR)
- Severity levels (blocker, major, minor, nit)
- Domain-specific review focus (e.g., security-sensitive areas need extra scrutiny)
- Anti-patterns to flag during review
- When to request human review vs auto-approve

Reference this file from AGENTS.md so review agents load it automatically.

### Generation Rules

- Write **only what was discovered** in Step 1. Do not fabricate rules.
- If a pattern is unclear, note it as "observed but unconfirmed" for human review.
- Mark sections needing human input with `<!-- TODO: confirm this rule -->`.
- Keep each file focused on one concern per file.
- Use concrete examples from the actual codebase.

---

## Step 4: Generate Root AGENTS.md

**~100 lines. Table of contents pointing to docs/.**

### Template

```markdown
# {Project Name}

> {One-line project description}

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full domain map.

## Documentation
- [Design Docs](docs/design-docs/index.md) — architectural decisions and core beliefs
- [Plans](docs/plans/) — design references (`designs/`) and execution plans (`work/`)
- [Product Specs](docs/product-specs/index.md) — feature specifications
- [References](docs/references/) — external library docs for LLMs

## Domain Guides
{Only list docs that were actually generated}
- [Design](docs/DESIGN.md) — design system and UI principles
- [Frontend](docs/FRONTEND.md) — frontend architecture and patterns
- [Reliability](docs/RELIABILITY.md) — SLOs, error budgets, incident response
- [Security](docs/SECURITY.md) — security policies and threat model

## Quality & Planning
- [Quality Score](docs/QUALITY-SCORE.md) — per-domain quality grades
- [Code Review](docs/CODE-REVIEW.md) — review standards and checklist
- [Plans](docs/PLANS.md) — planning conventions
- [Tech Debt](docs/plans/work/tech-debt-tracker.md) — known debt tracker
- [Product Sense](docs/PRODUCT-SENSE.md) — product thinking framework

## Project Structure
{Brief layout — packages, apps, key directories}
{Point to boundary AGENTS.md files if they exist}

## Quick Rules
{3-5 most critical rules that every agent must know, inlined here}

<!-- MANUAL: Notes below this line are preserved on regeneration -->
```

### Rules

- **No file listings.** Agents can `list_dir`.
- **Every line should point somewhere or state a rule.** No filler.
- Only list docs that were actually generated. No dead links.

---

## Step 5: Generate Boundary AGENTS.md Files

Only at **package/app boundaries** in monorepos.

### When to Create

- Each workspace package in a monorepo
- Each deployable app/service
- Major architectural boundary (e.g., shared library, SDK)
- **NOT** for subdirectories within a package

### Template

```markdown
<!-- Parent: ../AGENTS.md -->

# {Package/App Name}

> {One-line purpose}

## Constraints
{Rules specific to this boundary}

## Working Here
{How to correctly add/modify code in this area}

## Dependencies
- Depends on: {internal deps}
- Depended on by: {reverse deps}

<!-- MANUAL: Notes below this line are preserved on regeneration -->
```

### Rules

- **Max 50 lines.** Point to `docs/` for details.
- Content must be **specific to this boundary**. Don't repeat root-level rules.

---

## Step 6: Verify After Updates (Delegated)

This step is now handled by `oma-docs`. After deepinit completes, the user can run:

- `/oma-docs verify`: check generated harness docs against the current codebase.
- Set `docs.auto_verify: true` in `oma-config.yaml` to run verify automatically at the end of `/scm`, `/work`, and `/ultrawork` workflows.

deepinit no longer detects drift on update runs; it only generates 0→1 bootstrap content. The legacy `<!-- REVIEW: this rule may be outdated -->` marker behavior is replaced by the broken-ref report from `oma-docs verify`.

---

## Step 7: Validate

1. All files referenced in AGENTS.md actually exist (no dead links).
2. All `<!-- Parent: -->` references in boundary AGENTS.md resolve correctly.
3. ARCHITECTURE.md is consistent with actual package/module structure.
4. No `docs/` file contains information agents can derive from code (file listings, symbol enumerations).
5. Root AGENTS.md is under 120 lines.
6. ARCHITECTURE.md is under 200 lines.
7. Each boundary AGENTS.md is under 60 lines.
8. `docs/design-docs/index.md` lists all design docs that exist.
9. `docs/product-specs/index.md` lists all product specs that exist.

Report validation results to the user.
