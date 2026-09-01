/**
 * Codebase survey (Option C): on a cold repo, `session-start.ts` (hook harnesses) or
 * `runtime.ts`'s `seedIfCold` (opencode) spawns a DETACHED headless coding agent alongside the
 * git-history backfill (core/seed.ts) to explore the repo's structure and ingest what it finds as
 * memories, via the `hindsight_ingest_document` MCP tool (core/knowledge-tools.ts).
 *
 * A Hindsight knowledge page's body is synthesized server-side from the bank's MEMORIES via its
 * `source_query` — it can't be authored directly. So this survey doesn't write pages itself; it
 * ingests structural findings as documents, and the existing pages (missions.ts PAGES) synthesize
 * from them on the next consolidation, same as git history and chat transcripts do.
 *
 * **Harness-portable.** The survey used to hardcode headless `claude`, so a Codex/Antigravity/opencode
 * user without the `claude` binary silently lost the survey (the git-log seed still ran). It now
 * builds a per-agent headless recipe and runs the survey under **the current harness's own CLI**
 * (falling back to any other available agent), so whichever agent you use can seed its own repo:
 *   - claude : `claude -p … --mcp-config <inline> --disallowedTools …`   (inline MCP, self-contained)
 *   - codex  : `codex exec --sandbox read-only -c mcp_servers.hindsight…` (inline MCP, self-contained)
 *   - antigravity : `agy -p … --mode=plan --cwd <repo>` (read-only planning mode; MCP from the
 *                  global Antigravity customization config)
 *   - opencode: `opencode run --agent hindsight-survey …` (a read-only agent OUR plugin defines —
 *               see SURVEY_AGENT; tools from the loaded plugin, which under HINDSIGHT_DISABLE_HOOKS
 *               registers tools but skips seed/recall/write-back)
 * Each is read-only sandboxed (prompt-injection safety, since the survey reads untrusted repo files)
 * and spawned with HINDSIGHT_DISABLE_HOOKS=1 so the survey's own session doesn't re-fire our hooks.
 *
 * **Dcode is deliberately not on this list**, unlike its otherwise-full harness support. Its
 * headless runtime (`dcode -n`) rejects every MCP tool that is not annotated read-only —
 * "This MCP action requires approval, but the current headless runtime has no approval UI"
 * (`auto_mode.py:HeadlessMCPGuardMiddleware`). `hindsight_ingest_document` writes, so it is gated
 * by design and no flag lifts it: `--yolo`/`-y` are documented as ignored in headless mode. A
 * survey that cannot call the one tool it exists to call would spend a model budget and ingest
 * nothing, so Dcode falls back to another installed agent's CLI below — the same treatment as
 * Cursor, Copilot, Devin, Grok Build, Cline, Kilo and Prime Agent. Verified against
 * deepagents-code 0.1.65; revisit if Dcode gains a headless approval policy.
 *
 * Fire-and-forget and fail-safe throughout, mirroring core/seed.ts's `startBackgroundSeed`: a
 * missing binary or a spawn failure must silently no-op, never crash the caller.
 */
import { spawn as realSpawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { binOnPath } from "./util";

/** Deterministic doc ids of the survey's findings (its fixed titles slugified by
 *  hindsight_ingest_document). Their presence in the bank = the survey actually FINISHED —
 *  the baseline marker only proves it was started. */
export const SURVEY_DOC_IDS = [
  "repository-component-map",
  "repository-core-concepts",
  "repository-conventions-and-patterns",
  "repository-tech-stack-and-features",
] as const;

export type SurveyHarness = "claude-code" | "codex" | "antigravity-cli" | "opencode";

/** Resolve the claude CLI binary for a detached spawn (a shell alias won't apply to child_process).
 *  Order: explicit arg -> env HINDSIGHT_CLAUDE_BIN -> ~/.claude/local/claude (native installer) ->
 *  "claude" (PATH fallback, resolved by the shell/child_process at spawn time). */
export function resolveClaudeBin(explicit?: string): string {
  if (explicit) return explicit;
  if (process.env.HINDSIGHT_CLAUDE_BIN) return process.env.HINDSIGHT_CLAUDE_BIN;
  const nativeInstallPath = join(homedir(), ".claude", "local", "claude");
  try {
    if (existsSync(nativeInstallPath)) return nativeInstallPath;
  } catch {
    /* fall through to PATH lookup */
  }
  return "claude";
}

/**
 * The opencode agent the survey runs under, defined by our own plugin (harness/opencode.ts feeds
 * this to opencode's `config` hook) so the recipe below needs nothing in the user's opencode.json.
 *
 * It replaces the built-in `plan` agent, which was chosen as the read-only boundary but appends a
 * system-reminder to every user message — "CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase
 * … ANY file edits, modifications, or system changes … This ABSOLUTE CONSTRAINT overrides ALL other
 * instructions" — while leaving plugin tools callable (plan denies only `edit`). The survey exists
 * to call hindsight_ingest_document, which reads as a "system change", so completion came down to
 * how literally a model took the reminder: #3450 saw the seed stall on ~half their repos, the agent
 * asking for permission that plan mode cannot grant. Prompt wording cannot win that argument — the
 * reminder claims to override all other instructions.
 *
 * The ruleset is also a TIGHTER boundary than plan's: opencode drops denied tools from the tool
 * list entirely, so the session is offered exactly read/grep/glob plus the one ingest tool — no
 * write, no bash, and no `task` (plan left all three reachable, and `task` reaches a subagent that
 * CAN write). Verified against opencode 1.18.9.
 */
export const SURVEY_AGENT = "hindsight-survey";

export const SURVEY_AGENT_CONFIG = {
  description:
    "One-time read-only structural survey that seeds this repository's Hindsight memory.",
  mode: "primary",
  permission: {
    "*": "deny",
    read: "allow",
    grep: "allow",
    glob: "allow",
    hindsight_ingest_document: "allow",
  },
} as const;

/** Resolve a harness's agent CLI binary: explicit `claudeBin` (claude only) -> `HINDSIGHT_<AGENT>_BIN`
 *  env override -> the bare command name (resolved on PATH at spawn time). */
function resolveAgentBin(harness: SurveyHarness, claudeBin?: string): string {
  switch (harness) {
    case "claude-code":
      return resolveClaudeBin(claudeBin);
    case "codex":
      return process.env.HINDSIGHT_CODEX_BIN || "codex";
    case "antigravity-cli":
      return process.env.HINDSIGHT_ANTIGRAVITY_BIN || "agy";
    case "opencode":
      return process.env.HINDSIGHT_OPENCODE_BIN || "opencode";
  }
}

export const SURVEY_PROMPT =
  "You are performing a one-time structural survey of THIS repository to seed its Hindsight " +
  "memory. Work efficiently — DO NOT read every file; sample enough to understand the " +
  "architecture: the directory layout, entry points, package manifests (package.json / " +
  "pyproject.toml / Cargo.toml / go.mod), the README, and a few representative source files per " +
  "major area. Use Glob (e.g. `**/*`) to see the directory layout — Read takes a FILE path only " +
  "and errors on a directory; never call Read on a bare directory path.\n" +
  "IMPORTANT — DO NOT read, quote, summarize, or ingest agent-instruction files: CLAUDE.md, " +
  "AGENTS.md, GEMINI.md, .cursorrules, .cursor/rules/*, or .github/copilot-instructions.md. These " +
  "are live, user-controlled instructions (not repository knowledge); capturing them as memory " +
  "would let a stale copy override the user's current instructions. Exclude them entirely from " +
  "your survey and from every ingested document.\n" +
  "Then use the hindsight_ingest_document tool to save what you learned as separate documents (one " +
  "call each), with these titles and factual, developer-oriented content grounded in what you " +
  "actually saw:\n" +
  '- "Repository component map": the top-level modules/directories, each one\'s responsibility, ' +
  "and how they connect (data flow / dependencies).\n" +
  '- "Repository core concepts": the domain model and key abstractions a new contributor must ' +
  "understand.\n" +
  '- "Repository conventions and patterns": naming, file/module organization, testing approach, ' +
  "error handling, and other recurring patterns.\n" +
  '- "Repository tech stack and features": languages, frameworks, build/test tooling, and the ' +
  "notable features the project provides.\n" +
  "Keep each a few hundred words. When done, stop.";

/**
 * Tools the survey session must NEVER be able to invoke, no matter what a prompt injection in a
 * repo file (README, source comment, commit message, etc.) tries to talk it into. This is a
 * deny-list, not an allow-list gap: `--allowedTools` alone does not reliably block unlisted tools
 * in headless `-p` mode (empirically verified against `claude` v2.1.218 — the session had full
 * Bash/Write access despite `--allowedTools` omitting them, because `--permission-mode
 * bypassPermissions` overrides the allow-list). `--disallowedTools` DOES reliably block, with or
 * without a permission-mode flag, so it — not bypassPermissions — is the actual sandbox boundary
 * for the claude recipe. (codex uses `--sandbox read-only`; Antigravity `--mode=plan`; opencode
 * the read-only `plan` agent.) See survey.test.ts / the live acceptance test for verification.
 */
const SURVEY_DISALLOWED_TOOLS = [
  "Bash",
  "Write",
  "Edit",
  "NotebookEdit",
  "WebFetch",
  "WebSearch",
  "Task",
];

interface SpawnPlan {
  bin: string;
  args: string[];
  env: NodeJS.ProcessEnv;
}

/** Build the headless survey command for one agent. `bin` is already resolved (and confirmed to
 *  exist by the caller). All recipes run read-only and set HINDSIGHT_DISABLE_HOOKS=1. */
function buildSurveyPlan(
  harness: SurveyHarness,
  bin: string,
  repoDir: string,
  opts: { model?: string; budgetUsd?: number; mcpServerPath: string }
): SpawnPlan {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    // Anti-recursion: the survey's own agent session must not fire our hooks (session-start,
    // UserPromptSubmit/BeforeAgent, Stop/SessionEnd) or re-seed — see the HINDSIGHT_DISABLE_HOOKS
    // guard in hook.ts / retain-hook.ts / session-start.ts / runtime.ts.
    HINDSIGHT_DISABLE_HOOKS: "1",
    // Bank scoping for the MCP server the survey agent spawns.
    HINDSIGHT_MCP_PROJECT_CWD: repoDir,
  };

  switch (harness) {
    case "claude-code": {
      const mcpConfig = JSON.stringify({
        mcpServers: {
          hindsight: {
            command: "node",
            args: [opts.mcpServerPath],
            // HINDSIGHT_MCP_HARNESS names the agent whose CLI this recipe drives — the survey's
            // ingests are its writes. mcp-server.js REQUIRES it (it used to default to
            // "claude-code", which is how the codex recipe below silently stamped its findings
            // harness:claude-code and wrote them to Claude Code's bank — #3603).
            env: { HINDSIGHT_MCP_PROJECT_CWD: repoDir, HINDSIGHT_MCP_HARNESS: "claude-code" },
          },
        },
      });
      return {
        bin,
        args: [
          "-p",
          SURVEY_PROMPT,
          "--model",
          opts.model ?? "haiku",
          "--mcp-config",
          mcpConfig,
          "--strict-mcp-config",
          "--allowedTools",
          "Read",
          "Glob",
          "Grep",
          "mcp__hindsight__hindsight_ingest_document",
          "--disallowedTools",
          ...SURVEY_DISALLOWED_TOOLS,
          "--max-budget-usd",
          String(opts.budgetUsd ?? 2),
        ],
        env,
      };
    }
    case "codex": {
      // `codex exec` is non-interactive; `--sandbox read-only` is the injection boundary (read-only
      // filesystem, no network). MCP server injected inline via `-c` overrides so the recipe is
      // self-contained (doesn't require the user's config.toml to already have hindsight). Model is
      // left to Codex's configured default — Codex model ids differ from Claude's, and the read-only
      // sandbox + the bounded prompt are the cost/safety controls (Codex has no per-run budget flag).
      return {
        bin,
        args: [
          "exec",
          "--sandbox",
          "read-only",
          "-c",
          `mcp_servers.hindsight.command="node"`,
          "-c",
          `mcp_servers.hindsight.args=["${opts.mcpServerPath}"]`,
          "-c",
          `mcp_servers.hindsight.env.HINDSIGHT_MCP_PROJECT_CWD="${repoDir}"`,
          "-c",
          `mcp_servers.hindsight.env.HINDSIGHT_MCP_HARNESS="codex"`,
          SURVEY_PROMPT,
        ],
        env,
      };
    }
    case "antigravity-cli": {
      // Antigravity's documented `-p` mode supports one-shot surveys. `--mode=plan` is the
      // read-only boundary, and the installed MCP server comes from agy's current
      // ~/.gemini/config/mcp_config.json compatibility location.
      return {
        bin,
        args: ["-p", SURVEY_PROMPT, "--mode=plan"],
        env,
      };
    }
    case "opencode": {
      // `opencode run` under the survey agent OUR OWN PLUGIN defines (harness/opencode.ts), which
      // is both the injection boundary and the reason the seed completes: the built-in `plan` agent
      // used to fill that slot, but its read-only system-reminder talked models out of the ingest
      // call the survey exists to make (#3450). The hindsight tools come from the loaded plugin;
      // under HINDSIGHT_DISABLE_HOOKS it registers tools but skips seed/recall/write-back
      // (runtime.ts), so this survey run doesn't re-seed itself. Model left to opencode's
      // configured default (its `provider/model` format differs per setup).
      return {
        bin,
        args: ["run", "--agent", SURVEY_AGENT, SURVEY_PROMPT],
        env,
      };
    }
  }
}

/**
 * Spawn a DETACHED headless agent to survey `repoDir` and ingest structural findings via the
 * `hindsight_ingest_document` tool. Runs the survey under the current harness's own CLI when
 * available, else falls back to any available agent (claude → codex → antigravity → opencode — claude and
 * codex first because their inline-MCP recipes are self-contained). Fire-and-forget; never throws.
 */
export function startCodebaseSurvey(
  repoDir: string,
  opts: {
    harness?: SurveyHarness;
    model?: string;
    budgetUsd?: number;
    mcpServerPath?: string;
    claudeBin?: string;
    spawn?: typeof realSpawn;
    exists?: (bin: string) => boolean; // seam for tests
  } = {}
): void {
  try {
    const spawnFn = opts.spawn ?? realSpawn;
    const exists = opts.exists ?? binOnPath;
    const mcpServerPath =
      opts.mcpServerPath ?? join(dirname(fileURLToPath(import.meta.url)), "mcp-server.js");

    // Survey-agent priority: the current harness's own CLI first, then any other available agent.
    const preferred = opts.harness ?? "claude-code";
    const order: SurveyHarness[] = [
      preferred,
      "claude-code",
      "codex",
      "antigravity-cli",
      "opencode",
    ];
    const seen = new Set<SurveyHarness>();

    for (const harness of order) {
      if (seen.has(harness)) continue;
      seen.add(harness);
      const bin = resolveAgentBin(harness, opts.claudeBin);
      if (!exists(bin)) continue;

      const plan = buildSurveyPlan(harness, bin, repoDir, {
        model: opts.model,
        budgetUsd: opts.budgetUsd,
        mcpServerPath,
      });
      const child = spawnFn(plan.bin, plan.args, {
        cwd: repoDir,
        detached: true,
        stdio: "ignore",
        windowsHide: true,
        env: plan.env,
      });
      // spawn() failures (binary not found, EACCES, sandboxed environments) often arrive
      // ASYNCHRONOUSLY as an 'error' event on the child, not as a synchronous throw — an unhandled
      // 'error' event would crash the caller. Swallow it: fire-and-forget best-effort.
      child.on("error", () => {});
      child.unref();
      return; // one survey agent is enough
    }
    // No capable agent found — fail open (the git-log seed already ran; the survey is a bonus).
  } catch {
    /* best-effort: a failed spawn must not break the caller */
  }
}
