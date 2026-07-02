/**
 * hindsight-coding-agents — long-term memory for coding agents (reflect + INJECT), harness-pluggable.
 *
 * READ: on a task, the plugin asks the project's memory (Hindsight `reflect`) about the symptom and
 * PUSHES the synthesized root-cause answer into the system prompt.
 * WRITE (opt-in): with HINDSIGHT_RETAIN_SESSIONS on, it binds the live session into memory — every few
 * turns it upserts the user/assistant transcript (tool calls dropped) under a stable per-session id.
 *
 * The reflect/inject/write-back logic is a harness-agnostic RuntimeCore; a per-agent adapter (selected
 * by HINDSIGHT_HARNESS, default "opencode") binds it to that agent's plugin API. This file is the
 * opencode entrypoint: opencode loads the default export as a Plugin.
 *
 * Env: HINDSIGHT_API_URL (default http://localhost:8888), HINDSIGHT_BANK_ID, HINDSIGHT_API_TOKEN,
 *      HINDSIGHT_HARNESS (default opencode), HINDSIGHT_DISABLED (hard off-switch),
 *      HINDSIGHT_RETAIN_SESSIONS (enable live write), HINDSIGHT_RETAIN_EVERY_TURNS (default 5),
 *      HINDSIGHT_REFLECT_TIMEOUT_MS (default 120000).
 */
import type { Plugin } from "@opencode-ai/plugin";
import { HindsightClient } from "./core/hindsight";
import { RuntimeCore } from "./core/runtime";
import { getHarness } from "./harness/registry";

const env = (k: string, d = "") => process.env[k] ?? d;
const boolEnv = (k: string) => ["1", "true"].includes(env(k).toLowerCase());

const HindsightCodingAgentsPlugin: Plugin = async () => {
  if (env("HINDSIGHT_DISABLED")) return {}; // inert: same agent, no memory (baseline parity)

  const client = new HindsightClient({
    apiUrl: env("HINDSIGHT_API_URL", "http://localhost:8888"),
    apiToken: env("HINDSIGHT_API_TOKEN") || undefined,
    bank: env("HINDSIGHT_BANK_ID", "coding"),
  });
  const core = new RuntimeCore(client, {
    retainSessions: boolEnv("HINDSIGHT_RETAIN_SESSIONS"),
    retainEveryTurns: Number(env("HINDSIGHT_RETAIN_EVERY_TURNS")) || 5,
    reflectTimeoutMs: Number(env("HINDSIGHT_REFLECT_TIMEOUT_MS")) || 120000,
  });

  // HINDSIGHT_HARNESS selects the runtime adapter; this entry is loaded BY opencode, so opencode is
  // the default and its adapter returns an opencode Plugin hooks object.
  const harness = getHarness(env("HINDSIGHT_HARNESS", "opencode"));
  return harness.createRuntime(core) as Awaited<ReturnType<Plugin>>;
};

export default HindsightCodingAgentsPlugin;
export { HindsightCodingAgentsPlugin };
