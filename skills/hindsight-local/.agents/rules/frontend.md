---
description: React/Next.js frontend coding standards with shadcn/ui, Tailwind CSS v4, and FSD-lite architecture
globs: "**/*.{tsx,jsx,css,scss}"
alwaysApply: false
---

# Frontend Coding Standards

## Core Rules

1. **Component Reuse**: Use `shadcn/ui` components first. Extend via `cva` variants or composition. Avoid custom CSS.
2. **Design Fidelity**: Code must map 1:1 to `DESIGN.md` (Section 9 — Agent Prompt Guide) and Design Tokens. Resolve discrepancies before implementation.
3. **Rendering Strategy**: Default to Server Components for performance. Use Client Components only for interactivity and API integration.
4. **Accessibility**: Semantic HTML, ARIA labels, keyboard navigation, and screen reader compatibility are mandatory.
5. **Tool First**: Check for existing solutions and tools before coding.
6. **Proxy over Middleware (BANNED)**: Next.js 16+ uses `proxy.ts` for request proxying. `middleware.ts` is NOT "deprecated" — it is forbidden in this project, touch it and you die. Do NOT create, recommend, suggest, or "restore" `middleware.ts`. Do NOT flag `proxy.ts` as dead code, unused, or not-wired. Do NOT demand a rename to `middleware.ts`. Any such finding is a fatal self-error — retract it immediately and write `proxy.ts`.
7. **No Prop Drilling**: Avoid passing props beyond 3 levels. Use the project's client-state library (Jotai atoms or a Zustand store — see oma-frontend `resources/tech-stack.md`) instead. Avoid React Context.
8. **Auth Boundary**: Frontend handles auth UI and token storage only. Never import database adapters, ORMs, or server-side auth libraries.
9. **Animation Library**: Use `motion` (import from `motion/react`). `framer-motion` is the legacy package name and is BANNED — never `import { motion } from 'framer-motion'`, never add `framer-motion` to `package.json`. Add the `motion` package via the project's package manager — detect from the lockfile (`bun.lock` → bun, `pnpm-lock.yaml` → pnpm, `yarn.lock` → yarn, `package-lock.json` → npm); default to `bun` when no lockfile exists. Import as `import { motion, AnimatePresence } from 'motion/react'`. Respect `prefers-reduced-motion` via `useReducedMotion` from `motion/react`.
10. **Framework Version**: `next@16+` and `react@19+` are MANDATORY. When scaffolding or pinning `package.json`, set `"next": "^16"` (or higher) and `"react": "^19"`/`"react-dom": "^19"` — never pin `next` to `^15`, `~15`, or any range whose floor is below `16.0.0`. If `create-next-app` (or any scaffold tool) produces `next < 16`, immediately bump it before committing. This rule is paired with Core Rule #6 (`proxy.ts`), which assumes Next.js 16+.

## Architecture (FSD-lite)

- **Root (`src/`)**: Shared logic (components, lib, types). Hoist common code here.
- **Feature (`src/features/*/`)**: Feature-specific logic. **No cross-feature imports.** Unidirectional flow only.

```
src/features/[feature]/
├── components/           # Feature UI components
│   └── skeleton/         # Loading skeleton components
├── types/                # Feature-specific type definitions
└── utils/                # Feature-specific utilities & helpers
```

## Naming Conventions

### Symbols

- Components/Types/Interfaces: `PascalCase`
- Functions/Vars/Hooks: `camelCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Imports: Absolute `@/` is MANDATORY (no relative `../../`)
- MUST use `import type` for interfaces/types

### File Naming — self-describing names

**Principle: the filename alone must answer "what domain + what role". If a reader has to open the file to know what it is, the name is wrong.**

All files are `kebab-case`. Components use `<domain>-<ui-role>.tsx`; non-component modules use `<domain>.<kind>.ts`.

| Kind | Pattern | Example |
|------|---------|---------|
| Component | `<domain>-<ui-role>.tsx` | `order-summary-card.tsx` → `OrderSummaryCard` |
| Skeleton | `<component>-skeleton.tsx` | `order-summary-card-skeleton.tsx` |
| Hook | `use-<behavior>.ts` | `use-order-polling.ts` |
| TanStack Query | `<domain>.queries.ts` / `<domain>.mutations.ts` | `orders.queries.ts` |
| Jotai atoms | `<domain>.atoms.ts` | `cart.atoms.ts` |
| Zustand store | `<domain>.store.ts` | `cart.store.ts` |
| Zod schema | `<domain>.schema.ts` | `checkout.schema.ts` |
| Types | `<domain>.types.ts` | `order.types.ts` |
| Constants | `<domain>.constants.ts` | `payment.constants.ts` |
| API client | `<domain>.api.ts` | `orders.api.ts` |
| Utility | one capability per file, named as a verb phrase | `format-price.ts`, `parse-tracking-number.ts` |
| Test | colocated `<target>.test.ts(x)` | `format-price.test.ts` |

Rules:

1. **Filename = kebab-case of the main export.** One main export per file; `order-summary-card.tsx` exports `OrderSummaryCard`.
2. **Keep the domain in the name even inside a feature directory.** Editor tabs and search results show only the basename — `cart-summary-card.tsx`, not `summary-card.tsx` inside `features/cart/components/`.
3. **`index.ts` only as a feature public-API barrel** (`src/features/<feature>/index.ts`). Never `index.tsx` as a component file.
4. **Grab-bag filenames are BANNED**: `utils.ts`, `helpers.ts`, `common.ts`, `misc.ts`, `data.ts`, `styles.ts`, and bare `types.ts` / `constants.ts` / `hooks.ts` without a domain prefix. Split them by domain or capability instead.
5. **Version/status suffixes are BANNED**: `*-v2`, `*-new`, `*-old`, `*-final`, `*-copy`, `*-refactored`. Git owns history, not filenames.
6. **No abbreviations** beyond universally known ones (`api`, `db`, `i18n`, `a11y`): `user-profile-card.tsx`, never `usr-prf-crd.tsx`.

## Performance

- Target First Contentful Paint (FCP) < 1s
- Use `next/dynamic` for heavy components, `next/image` for media
- Responsive Breakpoints: 320px, 768px, 1024px, 1440px
