/**
 * hindsight-coding-agents — long-term memory for coding agents (reflect + INJECT), harness-pluggable.
 *
 * READ: on a task, the plugin asks the project's memory (Hindsight `reflect`) about the symptom and
 * PUSHES the synthesized root-cause answer into the system prompt.
 * WRITE (opt-in): with `retainSessions` on, it binds the live session into memory — every few turns it
 * upserts the user/assistant transcript (tool calls dropped) under a stable per-session id.
 *
 * The reflect/inject/write-back logic is a harness-agnostic RuntimeCore; a per-agent adapter (selected
 * by config `harness`, default "opencode") binds it to that agent's plugin API. This file is the
 * opencode entrypoint: opencode loads the default export as a Plugin.
 *
 * All configuration comes from the JSON file ~/.hindsight/coding-agent.json (no environment variables) —
 * see core/config.ts for the shape and defaults.
 */
import type { Plugin } from "@opencode-ai/plugin";
import { loadConfig } from "./core/config";
import { HindsightClient } from "./core/hindsight";
import { RuntimeCore } from "./core/runtime";
import { getHarness } from "./harness/registry";

const HindsightCodingAgentsPlugin: Plugin = async (input) => {
  const cfg = loadConfig();
  if (cfg.disabled) return {}; // inert: same agent, no memory (baseline parity)

  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: cfg.bankId,
  });
  const core = new RuntimeCore(client, {
    retainSessions: cfg.retainSessions,
    retainEveryTurns: cfg.retainEveryTurns,
    reflectTimeoutMs: cfg.reflectTimeoutMs,
    gitSync: cfg.gitSync.enabled,
    gitSyncRef: cfg.gitSync.ref,
    gitSyncFetch: cfg.gitSync.fetch,
  });

  // config.harness selects the runtime adapter; this entry is loaded BY opencode, so opencode is
  // the default and its adapter returns an opencode Plugin hooks object.
  const harness = getHarness(cfg.harness);
  const runtime = harness.createRuntime(core) as Awaited<ReturnType<Plugin>>;

  // Keep the bank current: on load, async + best-effort, retain commits new since the backfill (or the
  // last run). opencode's plugin input carries the repo path; fire-and-forget so it never blocks startup.
  void core.syncGitOnce(input?.worktree || input?.directory);
  return runtime;
};

export default HindsightCodingAgentsPlugin;
export { HindsightCodingAgentsPlugin };
