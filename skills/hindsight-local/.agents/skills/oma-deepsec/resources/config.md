# Configuration: `deepsec.config.ts`, env vars, plugins, models

deepsec reads `deepsec.config.{ts,mjs,js,cjs}` from the current working directory, walking up. The CLI inherits whatever the file declares.

```ts
import { defineConfig } from "deepsec/config";
import myPlugin from "@my-org/deepsec-plugin-foo";

export default defineConfig({
  projects: [
    { id: "my-app",  root: "../my-app" },
    { id: "service", root: "../service",
      githubUrl: "https://github.com/me/service/blob/main" },
  ],
  plugins: [myPlugin()],
});
```

For a fully-worked example exercising every common field (`infoMarkdown`, `promptAppend`, `priorityPaths`, an inline plugin), see `samples/webapp/deepsec.config.ts` in the deepsec repo.

## Top-level fields

| Field | Type | Purpose |
|---|---|---|
| `projects` | `ProjectDeclaration[]` | Codebases deepsec knows about. |
| `plugins` | `DeepsecPlugin[]` | Loaded in order; later plugins override single-slot capabilities. |
| `matchers` | `{ only?: string[]; exclude?: string[] }` | Filter the matcher set used by `scan`. |
| `defaultAgent` | `"claude" | "codex"` | Default `--agent` value. |
| `dataDir` | `string` | Override the `data/` directory. Defaults to `./data`. |

## `ProjectDeclaration`

| Field | Type | Required | Purpose |
|---|---|---|---|
| `id` | `string` | yes | Used as `--project-id` and the data directory name. |
| `root` | `string` | yes | Absolute or relative path to the codebase. |
| `githubUrl` | `string` | no | `https://github.com/owner/repo/blob/branch` for clickable links in exports. Auto-detected from `git remote` if omitted. |
| `infoMarkdown` | `string` | no | Repo context injected into AI prompts. Overrides `data/<id>/INFO.md` if both are set. |
| `promptAppend` | `string` | no | Free-form text appended to the system prompt for this project. |
| `priorityPaths` | `string[]` | no | Path prefixes to process first. |

## Per-project `data/<id>/config.json`

Optional, read by `scan` and the AI agents. Overrides the same fields on the project declaration if both are present.

```json
{
  "priorityPaths": ["app/api/", "lib/"],
  "promptAppend": "Pay extra attention to the booking flow.",
  "ignorePaths": ["**/legacy/**"]
}
```

## Matcher filtering

```ts
matchers: {
  only:    ["sql-injection", "auth-bypass"],   // run *only* these
  exclude: ["framework-internal-header"],      // skip these
}
```

If `only` is set, `exclude` is ignored. CLI flag `--matchers <slugs>` overrides the config when both are present.

## Plugin order

Plugins are evaluated in array order:

```ts
plugins: [genericPlugin(), orgPlugin()]
```

| Slot | Behavior |
|---|---|
| `matchers`, `notifiers`, `agents` | **Additive.** Both plugins' contributions stack. |
| `ownership`, `people`, `executor` | **Last-write-wins.** `orgPlugin()`'s provider replaces `genericPlugin()`'s. |

A monorepo gating example:

```ts
const projectId = process.argv[process.argv.indexOf("--project-id") + 1];
const isInternal = projectId?.startsWith("internal-") ?? false;

export default defineConfig({
  projects: [
    { id: "internal-api", root: "../api" },
    { id: "open-source-app", root: "../app" },
  ],
  plugins: isInternal ? [orgPlugin()] : [],
});
```

The config file is real TypeScript. Any logic at module-load time works.

## Plugin slots

| Slot | Purpose |
|---|---|
| `matchers` | Additional regex matchers, registered alongside the built-ins. |
| `notifiers` | Where findings get reported (Slack, GitHub Issues, webhooks, …). |
| `ownership` | Map files to owning teams/people (e.g. an internal directory). |
| `people` | Look up a person by email/name (managers, on-call, contact info). |
| `executor` | Run a deepsec command on remote infrastructure. |

```ts
export interface DeepsecPlugin {
  name: string;
  matchers?: MatcherPlugin[];
  notifiers?: NotifierPlugin[];
  ownership?: OwnershipProvider;
  people?: PeopleProvider;
  executor?: ExecutorProvider;
  agents?: AgentPluginRef[];
  commands?: (program: unknown) => void; // commander program
}
```

A single plugin can fill any subset. For details see https://github.com/vercel-labs/deepsec/blob/main/docs/plugins.md.

## Models

| Backend | Default | Used by |
|---|---|---|
| `codex` (default unless `defaultAgent` pins `claude`) | `gpt-5.5` | `process`, `revalidate` |
| `claude` | `claude-opus-4-8` | `process`, `revalidate` |
| `claude` (triage) | `claude-sonnet-4-6` | `triage` |

CLI selection:

```bash
bunx deepsec process --agent claude --model claude-sonnet-4-6   # cheaper Claude
bunx deepsec process --agent codex  --model gpt-5.4              # cheaper Codex
bunx deepsec triage  --model claude-haiku-4-5                    # cheaper triage
```

`--agent` and `--model` are accepted on `process`, `revalidate`, and `triage`. Set the workspace-wide default via `defaultAgent` in `deepsec.config.ts`.

## Environment variables

deepsec reads `.env.local` (auto-loaded by the CLI) or the process environment.

### Required (one of)

| Var | Used by | Purpose |
|---|---|---|
| `AI_GATEWAY_API_KEY` | all AI commands | Shortcut. Expands at startup into `ANTHROPIC_AUTH_TOKEN` / `OPENAI_API_KEY` / `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` (one key covers Claude **and** Codex through Vercel AI Gateway). Any of those four set explicitly always wins. Falls back to `VERCEL_OIDC_TOKEN` when unset. |
| `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` | `process`, `revalidate`, `triage` (Claude) | Direct Anthropic, or BYOK gateway-issued token. |
| `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | `--agent codex` | Codex SDK token. |
| `claude login` / `codex login` session | local non-sandbox runs only | Subscription fallback. Generally lacks headroom for full scans. |

### Optional

| Var | Purpose |
|---|---|
| `DEEPSEC_AGENT_DEBUG` | Set to `1` for verbose agent logging. |
| `DEEPSEC_DATA_ROOT` | Override the data directory (= `dataDir` in config). |
| Plugin-specific | Each plugin documents its own env vars in its README. |

### Vercel Sandbox (optional)

For `bunx deepsec sandbox …`. Pick OIDC for local dev, access token for unattended CI:

```bash
# OIDC (12 h expiry, re-pull when expired)
npx vercel link
npx vercel env pull            # writes VERCEL_OIDC_TOKEN

# Access token (long-lived, headless)
VERCEL_TOKEN=…
VERCEL_TEAM_ID=team_…
VERCEL_PROJECT_ID=prj_…
```

The Sandbox SDK reads these directly from `process.env` at `Sandbox.create()` time. The SDK prefers `VERCEL_OIDC_TOKEN` and falls back to access-token mode otherwise.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Missing AI credentials for --agent claude|codex` | No credential present. | Set `AI_GATEWAY_API_KEY=vck_…` in `.env.local`, or `claude login` / `codex login`. |
| `401 Unauthorized` on `process` / `revalidate` | Credential present but rejected. | OIDC: `vercel env pull` (12 h expiry). API key: regenerate in dashboard. Confirm `.env.local` is in cwd. |
| `Stopped: Vercel AI Gateway credits exhausted` | Gateway balance is $0. | Top up at the printed URL, then re-run the same command; it resumes. |
| `Stopped: Anthropic API credits exhausted` | Direct Anthropic out of credits. | Top up at console.anthropic.com, or switch to the gateway. |
| `Stopped: OpenAI API quota exhausted` | Direct OpenAI out of quota. | Top up in the OpenAI dashboard, or switch to the gateway. |
| `Stopped: Claude Pro/Max subscription exhausted` | Hit weekly / 5-hour cap. | Switch to AI Gateway. |
| `Stopped: ChatGPT subscription exhausted` | Hit ChatGPT Plus / Pro quota. | Switch to AI Gateway. |
| Sandbox spawn fails with auth error | OIDC expired or access-token vars wrong. | `vercel env pull`, or verify the three access-token vars. |
| Findings missing cost in the log | Pricing entry missing for a non-default Codex model. | Add a line to `MODEL_PRICING_USD_PER_M_TOKENS` in `packages/processor/src/agents/codex-sdk.ts` (only matters if you are extending deepsec itself). |
| Persistent refusal on a single file (>5 % of batches) | Hard-to-disambiguate exploit pattern. | Add to `data/<id>/config.json:ignorePaths`, or run with `--batch-size 1`. |

After **any** quota / credit fix, `process` and `revalidate` resume on re-run. No recovery flag, no state to reset. Files already analyzed stay analyzed; only unfinished ones get picked up. Use `--reinvestigate` (process) or `--force` (revalidate) only when you specifically want to redo finished work.

## Security model of deepsec itself

Treat deepsec like a coding agent with full shell access on the machine it runs on. It is designed to run on trusted inputs (your source code), but you may still be concerned about prompt injection from external dependencies or vendored code.

`deepsec sandbox …` substantially limits exposure:

- API keys are injected outside the sandbox and cannot be exfiltrated.
- Worker-sandbox network egress is locked to the configured AI host. (Egress is allowed during bootstrap, before the coding agent starts.)

Use sandbox mode for unfamiliar / vendored / contractor codebases. Local mode is fine for your own first-party code.
