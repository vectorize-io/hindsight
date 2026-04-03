/**
 * Configuration management for the opencode-hindsight plugin.
 *
 * Loads settings from ~/.hindsight/opencode.json merged with environment
 * variable overrides. Follows the same layering pattern as the Claude Code
 * integration: built-in defaults < user config file < env vars.
 */

import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";

export interface HindsightConfig {
  // Retain
  autoRetain: boolean;
  retainContext: string;
  retainToolCalls: boolean;
  retainMinChars: number;
  retainSkipSubagent: boolean;
  retainTags: string[];
  retainMetadata: Record<string, string>;

  // Recall
  autoRecall: boolean;
  recallBudget: "low" | "mid" | "high";
  recallMaxTokens: number;
  recallTypes: string[];
  recallPromptPreamble: string;

  // Connection
  hindsightApiUrl: string;
  hindsightApiToken: string | null;
  apiPort: number;
  daemonIdleTimeout: number;
  embedVersion: string;
  embedPackagePath: string | null;

  // Bank
  bankId: string;
  bankIdPrefix: string;
  dynamicBankId: boolean;
  dynamicBankGranularity: string[];
  bankMission: string;
  retainMission: string | null;

  // LLM (for daemon mode)
  llmProvider: string | null;
  llmModel: string | null;
  llmApiKeyEnv: string | null;

  // Misc
  debug: boolean;
}

const DEFAULTS: HindsightConfig = {
  autoRetain: true,
  retainContext: "opencode",
  retainToolCalls: false,
  retainMinChars: 200,
  retainSkipSubagent: true,
  retainTags: [],
  retainMetadata: {},

  autoRecall: true,
  recallBudget: "mid",
  recallMaxTokens: 1024,
  recallTypes: ["world", "experience"],
  recallPromptPreamble:
    "Relevant memories from past coding sessions (prioritize recent when " +
    "conflicting). Only use memories that are directly useful; ignore the rest:",

  hindsightApiUrl: "",
  hindsightApiToken: null,
  apiPort: 9077,
  daemonIdleTimeout: 0,
  embedVersion: "latest",
  embedPackagePath: null,

  bankId: "opencode",
  bankIdPrefix: "",
  dynamicBankId: false,
  dynamicBankGranularity: ["project"],
  bankMission:
    "I am a memory system for a software developer's coding sessions with AI assistants. " +
    "I track architectural decisions, tool preferences, coding patterns, project context, " +
    "and technical discussions. I prioritize decisions and their rationale, recurring patterns, " +
    "and project-specific knowledge that would be useful in future sessions.",
  retainMission: null,

  llmProvider: null,
  llmModel: null,
  llmApiKeyEnv: null,

  debug: false,
};

const ENV_OVERRIDES: Record<string, { key: keyof HindsightConfig; type: "string" | "bool" | "int" }> = {
  HINDSIGHT_API_URL: { key: "hindsightApiUrl", type: "string" },
  HINDSIGHT_API_TOKEN: { key: "hindsightApiToken", type: "string" },
  HINDSIGHT_BANK_ID: { key: "bankId", type: "string" },
  HINDSIGHT_AUTO_RECALL: { key: "autoRecall", type: "bool" },
  HINDSIGHT_AUTO_RETAIN: { key: "autoRetain", type: "bool" },
  HINDSIGHT_RECALL_BUDGET: { key: "recallBudget", type: "string" },
  HINDSIGHT_RECALL_MAX_TOKENS: { key: "recallMaxTokens", type: "int" },
  HINDSIGHT_API_PORT: { key: "apiPort", type: "int" },
  HINDSIGHT_DAEMON_IDLE_TIMEOUT: { key: "daemonIdleTimeout", type: "int" },
  HINDSIGHT_EMBED_VERSION: { key: "embedVersion", type: "string" },
  HINDSIGHT_EMBED_PACKAGE_PATH: { key: "embedPackagePath", type: "string" },
  HINDSIGHT_DYNAMIC_BANK_ID: { key: "dynamicBankId", type: "bool" },
  HINDSIGHT_BANK_MISSION: { key: "bankMission", type: "string" },
  HINDSIGHT_LLM_PROVIDER: { key: "llmProvider", type: "string" },
  HINDSIGHT_LLM_MODEL: { key: "llmModel", type: "string" },
  HINDSIGHT_DEBUG: { key: "debug", type: "bool" },
};

function castEnv(value: string, type: "string" | "bool" | "int"): unknown {
  if (type === "bool") return value.toLowerCase() === "true" || value === "1";
  if (type === "int") {
    const n = parseInt(value, 10);
    return isNaN(n) ? undefined : n;
  }
  return value;
}

function loadJsonFile(path: string): Record<string, unknown> {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return {};
  }
}

export function loadConfig(): HindsightConfig {
  const config = { ...DEFAULTS };

  // User config file: ~/.hindsight/opencode.json
  const userConfigPath = join(homedir(), ".hindsight", "opencode.json");
  const userConfig = loadJsonFile(userConfigPath);
  for (const [key, value] of Object.entries(userConfig)) {
    if (key in config && value != null) {
      (config as Record<string, unknown>)[key] = value;
    }
  }

  // Environment variable overrides
  for (const [envName, { key, type }] of Object.entries(ENV_OVERRIDES)) {
    const val = process.env[envName];
    if (val !== undefined) {
      const cast = castEnv(val, type);
      if (cast !== undefined) {
        (config as Record<string, unknown>)[key] = cast;
      }
    }
  }

  return config;
}

export function debugLog(config: HindsightConfig, ...args: unknown[]): void {
  if (config.debug) {
    console.error("[Hindsight]", ...args);
  }
}
