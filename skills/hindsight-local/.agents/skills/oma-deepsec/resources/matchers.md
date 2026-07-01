# Matchers: author project-specific entry-point coverage

The default matcher set covers common CWE shapes (SQL injection, SSRF, path traversal, …) and a handful of popular framework shapes (Next.js, Prisma, Express, Hono, FastAPI, Django, Laravel, Rails, Gin/Echo/Fiber/Chi, …). It will miss patterns specific to your codebase: an internal RPC framework, a less common language, a custom auth helper, a non-default route layout. Custom matchers fill those gaps.

The intended loop:

```
scan (fast, wide) → process (AI, slow + expensive) → revalidate → write better matchers
```

## When to write one

- A revalidated true-positive needs a matcher to catch siblings on future scans.
- A cluster of `other-*` slugs in `bunx deepsec metrics` points at a real category deepsec has no name for.
- The target repo has **entry points the default matchers do not see**. Check `https://github.com/vercel-labs/deepsec/blob/main/docs/supported-tech.md` first; the framework may already be covered.
- You have an **organization-specific** pattern (internal auth helper, internal SDK call, custom middleware).

## Where matchers live

```
.deepsec/
├── deepsec.config.ts                # inline plugin lists the matchers
└── matchers/
    ├── my-route-no-auth.ts
    └── my-internal-rpc.ts
```

`deepsec.config.ts`:

```ts
import { defineConfig, type DeepsecPlugin } from "deepsec/config";
import { myRouteNoAuth } from "./matchers/my-route-no-auth.js";
import { myInternalRpc } from "./matchers/my-internal-rpc.js";

const myPlugin: DeepsecPlugin = {
  name: "my-app",
  matchers: [myRouteNoAuth, myInternalRpc],
};

export default defineConfig({
  projects: [{ id: "my-app", root: ".." }],
  plugins: [myPlugin],
});
```

Slugs are unique. **If your slug collides with a built-in, your matcher wins.** This is useful for swapping in a tighter org-specific version.

If a matcher is genuinely reusable across orgs (a CWE shape or a public-framework shape), consider upstreaming to https://github.com/vercel-labs/deepsec instead.

## Workflow

### 1. Run `scan` + `process` first

You want real `data/` to point the agent at.

```bash
bunx deepsec scan
bunx deepsec process --limit 50          # upstream-recommended calibration pass (deepsec docs)
bunx deepsec revalidate --min-severity HIGH
```

### 2. Hand the workspace to the agent

Open the **parent repo** (the codebase being scanned) in your coding agent so it can read both source and `.deepsec/data/`. Then prompt:

> I want to add custom matchers to deepsec for this repo. deepsec is already installed at `.deepsec/node_modules/deepsec/` and `.deepsec/data/<projectId>/` has at least one scan + process pass.
>
> **Read these first to understand the contract:**
> - `.deepsec/node_modules/deepsec/dist/config.d.ts` defines the `MatcherPlugin` interface and the `regexMatcher` helper signature.
> - `.deepsec/node_modules/deepsec/dist/samples/webapp/matchers/webapp-debug-flag.ts` is a small `normal`-tier matcher.
> - `.deepsec/node_modules/deepsec/dist/samples/webapp/matchers/webapp-route-no-rate-limit.ts` is a slightly larger matcher with a negative pre-check.
> - `.deepsec/node_modules/deepsec/dist/samples/webapp/deepsec.config.ts` shows how the inline plugin wires matchers into the config.
>
> **Then do the analysis:**
> 1. Walk `.deepsec/data/<projectId>/files/` and look at what the default matchers already cover. Note which `vulnSlug`s show up in `candidates[]` and where the AI's `findings[]` ended up landing after revalidation.
> 2. Compare against the **target repository** (root above `.deepsec/`). Identify the **major entry points**: public HTTP handlers, RPC entry points, queue consumers, cron jobs, CLI commands, anything that takes untrusted input from the outside. Walk route/handler/api directories and framework config files (`next.config.*`, `wrangler.toml`, `serverless.yml`, `Procfile`, `main.go`, `app.py`, …) to figure out the entry-point shape.
> 3. Decide which entry points the default matchers **do not reach**. Common gaps:
>    - Frameworks deepsec does not ship a glob for (Hono, Elysia, Cloudflare Workers, Bun, Deno, FastAPI, Rails controllers, Go `chi`/`gin`, internal RPC).
>    - Languages with thin built-in coverage (Go, Python, Ruby, Lua, shell, Terraform, SQL).
>    - Custom org-specific wrappers (auth middleware, rate-limit wrappers, request-validation helpers) where deepsec's generic regexes do not know the convention.
> 4. **Then write matchers that cover those gaps.** Prefer one matcher per concern. For each:
>    - **Slug** (kebab-case, names what it flags, e.g. `hono-route-no-auth`, `worker-fetch-handler`).
>    - **Noise tier**: `precise` | `normal` | `noisy` (see below).
>    - **`filePatterns`** as tight as you can make them (language- or directory-anchored).
>    - **Regex(es)** that match the shape. Skip test files (`.test.`, `.spec.`, `__tests__`, `_test.go`, …).
>    - Save to `.deepsec/matchers/<slug>.ts`. Import types from `"deepsec/config"`.
> 5. Wire the new matchers into the inline plugin in `.deepsec/deepsec.config.ts` (create the plugin if it does not exist yet).
> 6. Run `bunx deepsec scan --matchers <slug1>,<slug2>,…` from `.deepsec/` and report how many candidates each matcher fired. Open 3 candidates per matcher to spot-check the regex is not producing obvious false positives.
>
> Bias toward `precise` when you can describe the bug exactly. Use `noisy` deliberately when the goal is **entry-point coverage**: you would rather the AI look at every `**/api/**/route.ts` than rely on a regex to predict which ones are vulnerable.
>
> Generalize the *shape* of the pattern, not specific identifiers. If the repo's auth helper is `requireSession()`, the matcher should catch any handler that does not call any session/auth helper, not the literal string `requireSession`.

### 3. Tune and ship

```bash
bunx deepsec scan --matchers <new-slug>
```

Watch the candidate count:

| Tier | Sweet spot |
|---|---|
| `precise` | 1–20 hits per 1k files |
| `normal` | 5–100 hits per 1k files |
| `noisy` | ≈ entry-point count of the targeted framework (10s, not 1000s) |

0 hits → too strict (loosen). >100 hits in a small repo → too loose (tighten).

When happy, commit `.deepsec/deepsec.config.ts` and `.deepsec/matchers/`. The next full scan picks them up automatically.

## Noise tiers

| Tier | When | Example |
|---|---|---|
| `precise` | Pattern is unambiguous. | `prisma-raw-sql`: `\$queryRawUnsafe\s*\(` matches only the unsafe API. |
| `normal` | Pattern is broader; AI disambiguates. | `auth-bypass`: flags admin checks and skip-auth strings; AI judges. |
| `noisy` | Every file matching a glob should be reviewed by the AI. | `service-entry-point`: every `**/api/**/route.ts` becomes a candidate. |

Tier also influences ordering. `precise` candidates are processed first because they have the highest signal per token.

## File globs

Set `filePatterns` tightly. A noisy matcher with `**/*.{ts,tsx}` wedges the scanner on a 100k-file repo. Prefer:

- Language-specific: `**/*.go`, `**/*.lua`, `**/*.tf`
- Directory-anchored: `**/api/**/*.ts`, `**/services/**/handlers/*.ts`
- Combined: `**/services/**/*.{ts,go}`

## Worked example: covering missing entry points (FastAPI)

A team scans a FastAPI service. After a `process` pass, `data/<id>/files/` shows the default matchers fired plenty on `requirements.txt` and a few `*.sql` files but barely touched `app/routers/*.py`, where the actual HTTP handlers live. The default glob set is tilted toward TypeScript/Next.js.

1. **Inspect coverage.** Walk `data/<id>/files/app/routers/`. Most `FileRecord`s have empty `candidates[]`; the AI never picks them up.
2. **Identify entry points.** Each router decorates handlers with `@router.get("/…")`, `@router.post("/…")`, etc. The team's convention: authenticated handlers depend on a `current_user: User = Depends(get_current_user)` parameter.
3. **Add a noisy entry-point matcher.** Slug `fastapi-route`, `noiseTier: "noisy"`, `filePatterns: ["app/routers/**/*.py", "app/api/**/*.py"]`, regex `/@\w+\.(get|post|put|delete|patch)\s*\(/`. Every router file becomes a candidate; the AI reads them on the next `process` pass.
4. **Add a precise auth-shape matcher.** Slug `fastapi-route-no-auth`, `noiseTier: "precise"`, same globs, regex sweep for `@\w+\.(get|post|...)` whose subsequent `def`/`async def` signature lacks `Depends(get_current_user)` or `Depends(require_*)`.

Result on the next scan: the AI investigates every router file, and the precise matcher flags handlers that skip the auth dependency.

## Generic vs plugin vs upstream contribution

| Catches… | Where |
|---|---|
| An org-specific helper, package, or route layout | Your inline plugin (`.deepsec/matchers/`) |
| A reference to a concrete internal service name | Your inline plugin |
| A CWE shape (path traversal, SSRF, prototype pollution) the public set misses | Consider upstreaming to https://github.com/vercel-labs/deepsec |
| A shape for a popular OSS framework (Hono, FastAPI, Drizzle) | Upstreaming benefits everyone |

For copy-paste starting points, see `.deepsec/node_modules/deepsec/dist/samples/webapp/matchers/`.
