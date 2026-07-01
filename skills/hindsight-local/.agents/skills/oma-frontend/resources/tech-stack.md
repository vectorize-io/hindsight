# Frontend Agent - Tech Stack Reference

## Core Framework
- **Framework**: Next.js 16+ (App Router), React 19+
- **Language**: TypeScript (strict mode)
- **Testing**: Vitest, React Testing Library, Playwright
- **UI**: `shadcn/ui` on the **Base UI** engine (see below)

## shadcn/ui Primitive Engine — Base UI vs Radix

shadcn/ui ships on two interchangeable headless engines: **Radix UI** and **Base UI**
(by the MUI team). Every component has parity docs/examples on both, and the public
component API is identical — only the underlying primitive changes.

### How the engine is selected

The engine is **not** a standalone `components.json` field. It is encoded in the **`style`**
prefix:

| `style` value | Engine |
|---|---|
| `base-*` (e.g. `base-vega`) | **Base UI** |
| `radix-*` (e.g. `radix-nova`) | Radix UI |

```jsonc
// components.json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "base-vega",        // <- engine + style; base-* = Base UI
  "rsc": true,
  "tsx": true,
  "tailwind": { "css": "src/styles/globals.css", "baseColor": "neutral", "cssVariables": true },
  "iconLibrary": "lucide",
  "aliases": { /* ... */ }
}
```

Bootstrap with `npx shadcn create` (prompts for the engine) or `npx shadcn init`.

### Project default: **Base UI**

1. **New projects MUST default to Base UI** (`style: "base-*"`). Rationale: Radix slowed after
   the WorkOS acquisition; Base UI is under active development with smaller bundles and is the
   more future-proof bet. The API is identical, so there is no DX cost.
2. **Radix is an allowed fallback** — keep `radix-*` for an existing Radix codebase, or when a
   needed component is only stable on Radix. State the reason when choosing Radix.
3. **Do NOT big-bang migrate** an existing project to Base UI just because it is the default.
   Migrate component-by-component, with reason; due to API differences a global swap is risky.
4. To switch engines: change the `style` prefix, then re-add base components with
   `npx shadcn add button card dialog … --overwrite`. Treat `components/ui/*` as read-only
   otherwise (customize via wrappers / `cva`).

## Next.js 16 Conventions

### Proxy replaces Middleware

`middleware.ts` is **BANNED** in this project. It is NOT merely deprecated; touch it and you die. No exceptions.

- File: `middleware.ts` → `proxy.ts` (root or `src/`)
- Exported function: `middleware` → `proxy`
- Config flags: `skipMiddlewareUrlNormalize` → `skipProxyUrlNormalize`, etc.
- `src/proxy.ts` is the canonical request-proxy / auth-gate location

Forbidden actions (any of these is a fatal self-error; retract immediately):

- Creating a new `middleware.ts`
- Suggesting a rename of `proxy.ts` back to `middleware.ts`
- Flagging `proxy.ts` as dead code, unused, or not-wired

Reference: https://nextjs.org/docs/messages/middleware-to-proxy

## Client State — Jotai vs Zustand

Client state has **no project default**; pick per intent. Both are allowed, but **only one
per project** unless there is a strong reason to mix. The overriding rule is to *minimize
client state* in the first place: server state belongs in **TanStack Query**, URL state in
**nuqs**. Reach for a client-state library only for genuine client-owned state.

| Pick | When |
|---|---|
| **Jotai** | Atomic / bottom-up. Fine-grained, derived, or near-local shared state; many small independent atoms; state that maps onto React's render graph and RSC-split boundaries. |
| **Zustand** | Store / top-down. A single cohesive global store with action methods; needs non-React access (`store.getState()` / `subscribe` outside render); simpler mental model for app-wide state. |

Guidance:

1. **State the choice** in the project (e.g. a short note in the app README or DESIGN.md) so
   contributors don't introduce the other library ad hoc.
2. **Don't mix** Jotai and Zustand in one app without justification — two client-state models
   fragment the codebase.
3. **Default-free does not mean optional analysis**: if neither atomic nor single-store clearly
   fits, the state probably belongs in TanStack Query (server) or nuqs (URL), not here.

## Serena MCP Shortcuts
- `find_symbol("ComponentName")`: locate existing component
- `get_symbols_overview("src/components")`: list all components
- `find_referencing_symbols("Button")`: find usages before changes
