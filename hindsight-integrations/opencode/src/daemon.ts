/**
 * Hindsight-embed daemon lifecycle management.
 *
 * Manages three connection modes:
 *   1. External API: user provides hindsightApiUrl (skip daemon entirely)
 *   2. Existing local server: check health on configured port
 *   3. Auto-managed daemon: start hindsight-embed via uvx
 *
 * Mirrors the Claude Code integration's daemon management pattern,
 * adapted for Bun's subprocess APIs.
 */

import { $ } from "bun";
import { platform } from "os";
import type { HindsightConfig } from "./config";
import { debugLog } from "./config";
import { HindsightClient } from "./client";

const PROFILE_NAME = "opencode";

function detectLlmEnv(config: HindsightConfig): Record<string, string> {
  const env: Record<string, string> = {};

  if (config.llmProvider) {
    env.HINDSIGHT_API_LLM_PROVIDER = config.llmProvider;
  }
  if (config.llmModel) {
    env.HINDSIGHT_API_LLM_MODEL = config.llmModel;
  }

  // Auto-detect provider from available API keys
  const provider = config.llmProvider;
  if (provider === "openai" || (!provider && process.env.OPENAI_API_KEY)) {
    env.HINDSIGHT_API_LLM_PROVIDER = env.HINDSIGHT_API_LLM_PROVIDER || "openai";
    env.HINDSIGHT_API_LLM_API_KEY = process.env.OPENAI_API_KEY || "";
  } else if (provider === "anthropic" || (!provider && process.env.ANTHROPIC_API_KEY)) {
    env.HINDSIGHT_API_LLM_PROVIDER = env.HINDSIGHT_API_LLM_PROVIDER || "anthropic";
    env.HINDSIGHT_API_LLM_API_KEY = process.env.ANTHROPIC_API_KEY || "";
  } else if (provider === "gemini" || (!provider && process.env.GEMINI_API_KEY)) {
    env.HINDSIGHT_API_LLM_PROVIDER = env.HINDSIGHT_API_LLM_PROVIDER || "gemini";
    env.HINDSIGHT_API_LLM_API_KEY = process.env.GEMINI_API_KEY || "";
  } else if (provider === "groq" || (!provider && process.env.GROQ_API_KEY)) {
    env.HINDSIGHT_API_LLM_PROVIDER = env.HINDSIGHT_API_LLM_PROVIDER || "groq";
    env.HINDSIGHT_API_LLM_API_KEY = process.env.GROQ_API_KEY || "";
  }

  // Force CPU on macOS to avoid MPS issues
  if (platform() === "darwin") {
    env.HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU = "1";
    env.HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU = "1";
  }

  return env;
}

export async function resolveApiUrl(
  config: HindsightConfig,
  allowDaemonStart = false,
): Promise<string | null> {
  // Mode 1: External API
  if (config.hindsightApiUrl) {
    debugLog(config, `Using external API: ${config.hindsightApiUrl}`);
    return config.hindsightApiUrl;
  }

  // Mode 2: Check existing local server
  const baseUrl = `http://127.0.0.1:${config.apiPort}`;
  const client = new HindsightClient(baseUrl);
  if (await client.healthy()) {
    debugLog(config, `Existing server healthy on port ${config.apiPort}`);
    return baseUrl;
  }

  // Mode 3: Auto-start daemon
  if (!allowDaemonStart) {
    debugLog(config, `No server on port ${config.apiPort}, daemon start not allowed in this hook`);
    return null;
  }

  debugLog(config, `No server on port ${config.apiPort}, attempting daemon start`);

  try {
    await startDaemon(config);
    return baseUrl;
  } catch (err) {
    debugLog(config, `Daemon start failed: ${err}`);
    return null;
  }
}

async function startDaemon(config: HindsightConfig): Promise<void> {
  const llmEnv = detectLlmEnv(config);
  if (!llmEnv.HINDSIGHT_API_LLM_API_KEY && !["ollama", "lmstudio", "claude-code", "openai-codex"].includes(llmEnv.HINDSIGHT_API_LLM_PROVIDER || "")) {
    throw new Error(
      "No LLM API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or configure llmProvider in ~/.hindsight/opencode.json",
    );
  }

  const embedCmd = config.embedPackagePath
    ? `uv run --directory ${config.embedPackagePath} hindsight-embed`
    : `uvx hindsight-embed@${config.embedVersion || "latest"}`;

  // Configure profile
  const envArgs = Object.entries(llmEnv)
    .filter(([, v]) => v)
    .map(([k, v]) => `--env ${k}=${v}`)
    .join(" ");

  const idleTimeout = config.daemonIdleTimeout || 300;
  const envArgsWithTimeout = `${envArgs} --env HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=${idleTimeout}`;

  await $`${embedCmd} profile create ${PROFILE_NAME} --merge --port ${config.apiPort} ${envArgsWithTimeout}`.quiet();

  // Start daemon
  await $`${embedCmd} daemon --profile ${PROFILE_NAME} start`.quiet();

  // Wait for ready
  const client = new HindsightClient(`http://127.0.0.1:${config.apiPort}`);
  for (let attempt = 0; attempt < 30; attempt++) {
    if (await client.healthy()) {
      debugLog(config, `Daemon ready after ${attempt + 1} attempts`);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  throw new Error("Daemon failed to become ready within 30 seconds");
}

export async function stopDaemon(config: HindsightConfig): Promise<void> {
  if (config.hindsightApiUrl) return;

  const embedCmd = config.embedPackagePath
    ? `uv run --directory ${config.embedPackagePath} hindsight-embed`
    : `uvx hindsight-embed@${config.embedVersion || "latest"}`;

  try {
    await $`${embedCmd} daemon --profile ${PROFILE_NAME} stop`.quiet();
    debugLog(config, "Daemon stopped");
  } catch (err) {
    debugLog(config, `Daemon stop error: ${err}`);
  }
}
