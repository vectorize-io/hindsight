---
name: oma-frontend
description: Frontend specialist for React, Next.js, TypeScript with FSD-lite architecture, shadcn/ui, and design system alignment. Use for UI, component, page, layout, CSS, Tailwind, and shadcn work.
---

# Frontend Agent - UI/UX Specialist

## Scheduling

### Goal
Build, modify, and verify React/Next.js/TypeScript user interfaces that follow project architecture, design-system constraints, accessibility expectations, and existing frontend conventions.

### Intent signature
- User asks for UI, component, page, layout, CSS, Tailwind, shadcn, form, interaction, client state, or frontend API integration work.
- User needs browser-facing implementation in a React/Next.js TypeScript codebase.

### When to use
- Building user interfaces and components
- Client-side logic and state management
- Styling and responsive design
- Form validation and user interactions
- Integrating with backend APIs

### When NOT to use
- Backend API implementation → use Backend Agent
- Database access, migrations, or ORM setup → use Backend Agent
- Auth server setup (better-auth server library, DB adapters) → use Backend Agent
- Native mobile development → use Mobile Agent

### Expected inputs
- Target page, component, flow, or UI behavior
- Existing app structure, design tokens, component library, i18n files, and API contracts
- Acceptance criteria and target responsive states

### Expected outputs
- Frontend code changes in pages, components, hooks, styles, tests, or wrappers
- UI that respects project tokens, i18n, server/client boundaries, and accessibility expectations
- Verification results from relevant lint, typecheck, tests, or browser checks

### Dependencies
- React, Next.js, TypeScript, TailwindCSS v4, and `shadcn/ui`
- Project sources of truth such as `packages/design-tokens`, `packages/i18n`, and shared utilities
- `resources/execution-protocol.md`, `resources/checklist.md`, examples, snippets, and Tailwind rules

### Control-flow features
- Branches by server/client component boundary, responsive state, component library availability, and i18n/token requirements
- Reads and writes frontend codebase files
- May call shadcn registry tools or local verification commands

## Structural Flow

### Entry
1. Identify target route, component, state boundary, and design-system constraints.
2. Read existing patterns before adding components or utilities.
3. Determine whether work belongs in Server Components, Client Components, wrappers, hooks, or styles.

### Scenes
1. **PREPARE**: Load relevant project conventions, UI requirements, and acceptance criteria.
2. **ACQUIRE**: Inspect existing components, tokens, i18n keys, APIs, and shadcn availability.
3. **ACT**: Implement UI, state, styles, validation, and integration.
4. **VERIFY**: Run checklist, automated checks, and browser/responsive validation when applicable.
5. **FINALIZE**: Summarize changed UI behavior and verification.

### Transitions
- If a strict shadcn primitive exists, use or wrap it before creating generic markup.
- If UI text is user-facing and i18n exists, add strings through the i18n source of truth.
- If interaction or hooks are needed, mark the boundary as Client Component.
- If backend contracts are missing, coordinate with backend/API planning.

### Failure and recovery
- If design tokens or i18n sources are missing, state assumptions and follow existing local patterns.
- If verification fails, fix before handoff or report the blocker.
- If required shadcn registry access fails, use existing local components or document fallback.

### Exit
- Success: UI works across target responsive states and passes relevant checks.
- Partial success: missing assets, backend contracts, or verification gaps are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Inspect existing frontend patterns | `READ` | Components, routes, hooks, styles |
| Select component and state approach | `SELECT` | Server/client and shadcn workflow |
| Implement UI code | `WRITE` | TSX, CSS, hooks, wrappers |
| Validate form/data contracts | `VALIDATE` | Zod/forms/API schemas |
| Call shadcn or verification tools | `CALL_TOOL` | Registry, lint, typecheck, tests |
| Compare responsive states | `COMPARE` | Desktop/mobile behavior |
| Report result | `NOTIFY` | Final summary |

### Tools and instruments
- React, Next.js, TypeScript, TailwindCSS v4, shadcn/ui
- `ahooks`, `es-toolkit`, `nuqs`, TanStack Query, Jotai/Zustand, TanStack React Form, `zod`
- Lint, typecheck, tests, and browser inspection when applicable

### Canonical workflow path
```bash
rg --files
rg "components/ui|shadcn|use client|generateMetadata|useQuery|i18n|design-tokens" .
```

Then run the project's frontend verification commands, typically lint, typecheck, tests, and browser/responsive checks when the UI changes.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Frontend routes, components, styles, hooks, tests |
| `LOCAL_FS` | Design tokens, i18n files, resource references |
| `PROCESS` | Build, lint, typecheck, test, browser commands |
| `NETWORK` | Backend APIs or registry tools when required |

### Preconditions
- Target UI behavior and affected frontend area are identifiable.
- Required design tokens, i18n, and API contracts are available or assumptions are stated.

### Effects and side effects
- Mutates frontend source, styles, tests, and possibly i18n keys.
- May add dependencies or shadcn components only when justified by project conventions.
- Does not edit `components/ui/*` directly.

### Guardrails
1. Follow the existing React, Next.js, TypeScript, and FSD-lite architecture in the target project.
2. Use `shadcn/ui` primitives and wrappers for UI work; treat `components/ui/*` as read-only.
3. Keep server/client boundaries explicit: Server Components for static/layout work, Client Components for interaction and hooks.
4. Use project sources of truth for design tokens, i18n strings, and shared utilities before adding local alternatives.
5. Run the execution checklist before handoff and include relevant verification results.
6. **Self-describing file names**: every new file follows the File Naming convention in `../../rules/frontend.md` §Naming Conventions — domain + role readable from the basename alone (`order-summary-card.tsx`, `use-order-polling.ts`, `cart.atoms.ts`). Grab-bag names (`utils.ts`, `helpers.ts`, `misc.ts`) and version suffixes (`*-v2`, `*-final`) are banned.
7. **Next.js 16 `proxy.ts` is mandatory; `middleware.ts` is BANNED**: this project is Next.js 16+. `middleware.ts` is NOT "deprecated"; it is forbidden, touch it and you die. The canonical request-proxy / auth-gate file is `proxy.ts` (root or `src/`) exporting a `proxy` function. NEVER create, recommend, suggest, or "restore" `middleware.ts`. NEVER flag `proxy.ts` as dead code, unused, or not-wired. Any such finding is a fatal self-error: retract it immediately and write `proxy.ts`.

### Libraries

| Category | Library |
|----------|---------|
| Framework | `next@16+` (App Router) + `react@19+`; `next < 16` is BANNED |
| Date | `luxon` |
| Styling | `TailwindCSS v4` + `shadcn/ui` (Base UI engine; see `resources/tech-stack.md`) |
| Hooks | `ahooks` (pre-made hooks preferred) |
| Utils | `es-toolkit` (first choice) |
| Types | `type-fest` (TS type utilities not in the standard lib: `SetRequired`, `Merge`, `JsonValue`, `Promisable`, etc.; built-in `Partial`/`Pick`/`Omit` stay first choice) |
| State (URL) | `nuqs` |
| State (Server) | `TanStack Query` |
| State (Client) | `Jotai` or `Zustand` (intent-based, no default; minimize use — see `resources/tech-stack.md`) |
| Forms | `@tanstack/react-form` + `zod` |
| Auth | `better-auth` (client SDK only; never import server library or database adapters) |
| Animation | `motion`; import from `motion/react`. `framer-motion` (legacy package name) is BANNED. |

### Shadcn Workflow

- **Engine default: Base UI** (`components.json` → `style: "base-*"`). Radix (`radix-*`) is a
  reasoned fallback for existing Radix codebases only; no big-bang migration. Details in
  `resources/tech-stack.md` §shadcn/ui Primitive Engine.

1. Search: `shadcn_search_items_in_registries`
2. Review: `shadcn_get_item_examples_from_registries`
3. Install: `shadcn_get_add_command_for_items`

### Server vs Client Components

- **Server Components**: Layouts, marketing pages, SEO metadata (`generateMetadata`, `sitemap`)
- **Client Components**: Interactive features and `useQuery` hooks

### UI Implementation (Shadcn/UI)

- **Usage**: Prefer strict shadcn primitives (`Card`, `Sheet`, `Typography`, `Table`) over `div` or generic classes.
- **Responsiveness**: Use `Drawer` (mobile) vs `Dialog` (desktop) via `useResponsive`.
- **Customization**: Treat `components/ui/*` as read-only. Create wrappers (e.g., `components/common/ProductButton.tsx`) or use `cva` composition. Never edit `components/ui/button.tsx` directly.

### Sources of Truth

- **DESIGN.md** (project root): visual system source of truth; read Section 9 (Agent Prompt Guide) verbatim for component prompts when present
- **Design Tokens**: `packages/design-tokens` (OKLCH); never hardcode colors
- **i18n strings**: `packages/i18n`; never hardcode UI text
- **Custom utilities**: check `es-toolkit` first; if implementing custom logic, >90% unit test coverage is mandatory

### Designer Collaboration

- **Sync**: Map code variables to Figma layer names
- **UX**: Ensure key actions are visible "Above the Fold"

### Stack Reference

Project stack conventions live in dedicated files. **Read these before coding**; they are not optional appendix material.

| File | Owns |
|---|---|
| `resources/tech-stack.md` | Framework versions, Next.js 16 `proxy.ts` conventions, Serena shortcuts |
| `resources/tailwind-rules.md` | Design tokens, focus states, Tailwind v4 `@theme` syntax |
| `resources/snippets.md` | React 19 hook patterns, TanStack Query/Form, a11y card |

To extend: add `resources/<name>.md` and append a row above.

## References

1. Follow `resources/execution-protocol.md` step by step.
2. Before submitting, run `resources/checklist.md`.
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.

- Project frontend rules (MUST load before review/implementation): `../../rules/frontend.md`
- Execution steps: `resources/execution-protocol.md`
- Checklist: `resources/checklist.md`
- Error recovery: `resources/error-playbook.md`
- Context loading: `../_shared/core/context-loading.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification: `../_shared/core/clarification-protocol.md`
- Context budget: `../_shared/core/context-budget.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
- Observability handoff: `../oma-observability/SKILL.md` §Integrations — Core Web Vitals, SSR→client trace propagation, INP profiling

> [!IMPORTANT]
> Treat `components/ui/*` as read-only. Create wrappers for customization.
