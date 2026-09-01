/**
 * Harness registry. Add a coding agent by implementing HarnessAdapter (one file in this dir) and
 * registering it here — the backfill's --harness flag resolves through getHarness().
 *
 * getHarness() never statically OR dynamically imports opencode.ts, even for "opencode": backfill
 * (its only real caller — see core/config.ts's note that the top-level `harness` key just selects
 * the backfill session formatter) only ever needs a harness's chatReader, and every harness's
 * chatReader — opencode included — is the same normalized-JSON reader (jsonChatReader below).
 * opencode.ts's plugin-specific createRuntime (the only part that needs @opencode-ai/plugin) is
 * wired up directly by src/index.ts, the opencode plugin entrypoint, bypassing this registry
 * entirely. That keeps this file, and everything bundled from it (in particular backfill.js), free
 * of the @opencode-ai/plugin dependency.
 */
import { readFileSync } from "node:fs";
import type { ChatSession, HarnessAdapter } from "../core/types";

/** Every harness ingests past sessions through the same normalized JSON interchange format. */
export const jsonChatReader = (harness: string) => ({
  describe:
    `${harness} sessions via a normalized JSON export ` +
    "(--conversations file: [{ id, turns:[{role,text,timestamp?}] }])",
  async read(opts: { conversations?: string }): Promise<ChatSession[]> {
    if (!opts.conversations) return [];
    return JSON.parse(readFileSync(opts.conversations, "utf8")) as ChatSession[];
  },
});

/**
 * A harness whose runtime this registry never constructs — either because it's a per-prompt HOOK
 * binary rather than a persistent plugin (core/hook.ts), or because (opencode) its runtime is built
 * directly by its own entrypoint, not via this registry.
 */
const noRuntimeAdapter = (name: string, hint: string): HarnessAdapter => ({
  name,
  chatReader: jsonChatReader(name),
  createRuntime() {
    throw new Error(hint);
  },
});

export const HARNESS_NAMES = [
  "opencode",
  // opencode v2 (binary `opencode2`) — a ground-up rewrite of the plugin API, loaded from the
  // package's root index.js (src/opencode2.ts). Like opencode it has NO hook binary.
  "opencode2",
  // Kilo CLI is an opencode fork loaded as a persistent plugin (src/kilo.ts), so like opencode it
  // has NO hook binary — deliberately absent from HOOK_BINS below.
  "kilo",
  // Cline CLI loads dist/cline.js through its native plugin manager; file hooks cannot inject.
  "cline-cli",
  // DeepSeek Harness loads dist/dsh.js as a native Cordis plugin (src/dsh.ts). Its Claude Code /
  // Codex hook bridges are optional packages, so there is no hook binary to install either.
  "dsh",
  // Prime Agent loads dist/prime-agent.js as an extension (src/prime-agent.ts); no hook binary.
  "prime-agent",
  "claude-code",
  "cursor-cli",
  "codex",
  "dcode",
  "antigravity-cli",
  "devin-cli",
  "copilot-cli",
  "grok-build",
  "qwen-code",
];

const HOOK_BINS: Record<string, string> = {
  "claude-code": "hindsight-claude-hook",
  "cursor-cli": "hindsight-cursor-hook",
  codex: "hindsight-codex-hook",
  dcode: "hindsight-dcode-hook",
  "antigravity-cli": "hindsight-antigravity-hook",
  "devin-cli": "hindsight-devin-hook",
  "copilot-cli": "hindsight-copilot-hook",
  "grok-build": "hindsight-grok-hook",
  "qwen-code": "hindsight-qwen-hook",
  // more hook harnesses: add a HookSpec entry point (see src/cursor-hook.ts) + a registration here.
};

/** Where each persistent-plugin harness's runtime is actually built (see the branch below). */
const PLUGIN_ENTRYPOINTS: Record<string, string> = {
  opencode: "src/index.ts",
  opencode2: "src/opencode2.ts",
  kilo: "src/kilo.ts",
  "cline-cli": "src/cline.ts",
  "prime-agent": "src/prime-agent.ts",
  dsh: "src/dsh.ts",
};

export async function getHarness(name: string): Promise<HarnessAdapter> {
  // The persistent-plugin harnesses: their runtime is built by their own plugin entrypoint (which
  // owns the @opencode-ai/plugin dependency this file must stay free of), never via this registry.
  // Backfill still resolves them here for the chatReader.
  const entry = PLUGIN_ENTRYPOINTS[name];
  if (entry) {
    return noRuntimeAdapter(
      name,
      `${name}'s runtime is built by ${entry} (the ${name} plugin entrypoint), not via the harness registry`
    );
  }
  const bin = HOOK_BINS[name];
  if (!bin) {
    throw new Error(`unknown harness '${name}'. available: ${HARNESS_NAMES.join(", ")}`);
  }
  return noRuntimeAdapter(
    name,
    `'${name}' has no persistent plugin runtime — install its hook binary (${bin}) instead`
  );
}
