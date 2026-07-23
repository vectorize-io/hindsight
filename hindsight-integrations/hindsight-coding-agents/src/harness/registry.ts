/**
 * Harness registry. Add a coding agent by implementing HarnessAdapter (one file in this dir) and
 * registering it here — the backfill's --harness flag and the runtime's `harness` config key both
 * resolve through getHarness().
 */
import { readFileSync } from "node:fs";
import type { ChatSession, HarnessAdapter } from "../core/types";
import { opencodeAdapter } from "./opencode";

/** Every harness ingests past sessions through the same normalized JSON interchange format. */
const jsonChatReader = (harness: string) => ({
  describe:
    `${harness} sessions via a normalized JSON export ` +
    "(--conversations file: [{ id, turns:[{role,text,timestamp?}] }])",
  async read(opts: { conversations?: string }): Promise<ChatSession[]> {
    if (!opts.conversations) return [];
    return JSON.parse(readFileSync(opts.conversations, "utf8")) as ChatSession[];
  },
});

/** A harness whose runtime is a per-prompt HOOK binary, not a persistent plugin (core/hook.ts). */
const hookAdapter = (name: string, bin: string): HarnessAdapter => ({
  name,
  chatReader: jsonChatReader(name),
  createRuntime() {
    throw new Error(
      `'${name}' has no persistent plugin runtime — install its hook binary (${bin}) instead`
    );
  },
});

export const HARNESSES: Record<string, HarnessAdapter> = {
  opencode: opencodeAdapter,
  "claude-code": hookAdapter("claude-code", "hindsight-claude-hook"),
  "cursor-cli": hookAdapter("cursor-cli", "hindsight-cursor-hook"),
  codex: hookAdapter("codex", "hindsight-codex-hook"),
  // more agents: persistent-plugin harnesses implement HarnessAdapter fully; hook harnesses add a
  // HookSpec entry point (see src/cursor-hook.ts) + a hookAdapter registration here.
};

export const HARNESS_NAMES = Object.keys(HARNESSES);

export function getHarness(name: string): HarnessAdapter {
  const a = HARNESSES[name];
  if (!a) {
    throw new Error(`unknown harness '${name}'. available: ${HARNESS_NAMES.join(", ")}`);
  }
  return a;
}
