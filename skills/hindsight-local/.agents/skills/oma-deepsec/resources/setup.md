# Setup: install `.deepsec/` and bootstrap `INFO.md`

## 1. Install the workspace

Requires **Node.js 22+**. Run from the **root of the codebase you want to scan**:

```bash
bunx deepsec init                       # creates .deepsec/ and registers this repo
cd .deepsec
bun install                             # installs deepsec from npm

# pnpm / npm / yarn equivalents work the same way:
#   npx deepsec init && cd .deepsec && pnpm install
#   npx deepsec init && cd .deepsec && npm  install
#   npx deepsec init && cd .deepsec && yarn install
```

`init` lays down a minimal scaffold inside `.deepsec/`:

- `package.json`
- `deepsec.config.ts` (one `projects[]` entry pointing at `..`, id derived from the parent dir's basename)
- `data/<id>/INFO.md` (template with section placeholders)
- `data/<id>/SETUP.md` (per-project agent prompt)
- workspace-level `AGENTS.md`
- `.env.local`
- `.gitignore` (keeps `INFO.md`, `SETUP.md`, `deepsec.config.ts` tracked; ignores `data/*/files/`, `data/*/runs/`, etc.)

No custom matchers in the scaffold. Add those only when a real finding shapes one for you.

> To scan another codebase from the same `.deepsec/`: `bunx deepsec init-project <path>` (relative paths resolve against `.deepsec/`'s parent).

## 2. Pick a credential

Open `.deepsec/.env.local` and pick **one**:

| Mode | When | Set |
|---|---|---|
| AI Gateway API key | Anywhere, simplest | `AI_GATEWAY_API_KEY=vck_…` from the Vercel AI Gateway API Keys page |
| Vercel OIDC token | Already linked to a Vercel project (or using Sandbox) | `npx vercel link && npx vercel env pull` writes `VERCEL_OIDC_TOKEN` (12 h expiry; re-pull on auth errors) |
| Direct Anthropic | BYOK / bypass gateway | `ANTHROPIC_AUTH_TOKEN=sk-ant-…` + `ANTHROPIC_BASE_URL=https://api.anthropic.com` |
| Direct OpenAI | Codex backend, BYOK | `OPENAI_API_KEY=sk-…` (+ `OPENAI_BASE_URL` only for proxies) |
| Subscription | Local-only evaluation | `claude login` and/or `codex login` already done; non-sandbox runs reuse the session, no token needed |

`AI_GATEWAY_API_KEY` expands at CLI startup into `ANTHROPIC_AUTH_TOKEN` / `OPENAI_API_KEY` / `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`. Any of those four set explicitly always wins.

> Subscriptions are useful for evaluating deepsec but generally do not have enough headroom for full repo scans. Switch to the gateway once past evaluation.

## 3. Verify the credential

```bash
bunx deepsec scan --limit 20         # cheap, no AI calls
bunx deepsec process --limit 5       # exercises the gateway
```

If the second call returns `Missing AI credentials` or `401`, see `config.md` § Troubleshooting.

## 4. Write `INFO.md` (do not skip)

`INFO.md` is what makes deepsec project-aware. It is injected into the AI prompt for every batch, so vague content here means vague findings.

### Recommended: agent-driven write-up

Open the **parent repo** (the codebase you scanned, **not** `.deepsec/`) in your coding agent and paste the prompt that `deepsec init` printed (also in the project root README):

> Read `.deepsec/node_modules/deepsec/SKILL.md` to understand the tool. Then read `.deepsec/data/<id>/SETUP.md` and follow it: skim this repo's README, any `AGENTS.md` / `CLAUDE.md`, and a handful of representative code files, then replace each section of `.deepsec/data/<id>/INFO.md`.
>
> Keep it SHORT: target 50–100 lines total. Pick 3–5 examples per section, not exhaustive enumeration. Name primitives (auth helpers, middleware) but no line numbers. Skip generic CWE categories; built-in matchers cover those. Cover only what is project-specific. `INFO.md` is injected into every scan batch; verbose context dilutes signal.

### Manual write-up

The processor auto-loads `data/<id>/INFO.md` from the workspace's data dir. Edit it directly; no extra wiring is needed in `deepsec.config.ts`. Even a single tight paragraph noticeably improves the AI's output.

### What goes in `INFO.md`

Project-specific only:

- **What the codebase does** in a few sentences.
- **Auth shape**: names of helpers / middleware / decorators that gate access (`requireSession`, `Depends(get_current_user)`, etc.). Name them, do not quote them.
- **Threat model**: which surfaces matter (public HTTP, internal RPC, queue consumers, cron, CLI) and which are out of scope.
- **Known FP sources**: patterns the AI tends to over-flag in this repo.
- **Project-specific primitives**: internal SDK calls, custom validators, codified secret-loading paths.
- **Out of scope**: directories or file types the AI should ignore.

### What stays out

- Generic CWE category descriptions; built-in matchers cover those.
- Exhaustive enumeration. Pick 3–5 representative examples per section.
- Line numbers. They drift; the AI re-reads files anyway.
- Boilerplate intro paragraphs.

## 5. `.gitignore` hygiene

The scaffold's `.deepsec/.gitignore` already keeps `INFO.md`, `SETUP.md`, and `deepsec.config.ts` tracked (so teammates inherit project context) and ignores generated state. Do **not** unignore `data/*/files/` or `data/*/runs/` unless you have a deliberate reason (e.g. CI cache).

`.env.local` must stay gitignored. Never commit `vck_…`, `sk-ant-…`, `sk-…`, or OIDC tokens.

## 6. Multi-project workspaces

To scan a *different* codebase from the same `.deepsec/`:

```bash
bunx deepsec init-project <path>
```

Each project gets its own `data/<id>/` subdirectory. Pass `--project-id <id>` to disambiguate any subsequent command (auto-resolution only kicks in with exactly one project).

## 7. Sanity check before the first real run

- [ ] `.deepsec/.env.local` has a working credential.
- [ ] `bunx deepsec scan --limit 20` succeeds.
- [ ] `bunx deepsec process --limit 5` succeeds and prints a per-batch cost number.
- [ ] `data/<id>/INFO.md` is filled in (50-100 lines, project-specific).
- [ ] You and the user agree on a calibration scope for the first `process` run (deepsec docs default: `--limit 50 --concurrency 5`).
