/**
 * Configuration management for the Hindsight OpenCode plugin.
 *
 * Loading order (later entries win):
 *   1. Built-in defaults
 *   2. User config file (~/.hindsight/opencode.json)
 *   3. Plugin options (from opencode.json plugin tuple)
 *   4. Environment variable overrides
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

/** Default API URL used when no override is supplied via env, file, or plugin options. */
export const DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io";

export interface HindsightConfig {
  // Recall
  autoRecall: boolean;
  recallBudget: string;
  recallMaxTokens: number;
  recallTypes: string[];
  recallContextTurns: number;
  recallMaxQueryChars: number;
  recallPromptPreamble: string;
  recallTags: string[];
  recallTagsMatch: "any" | "all" | "any_strict" | "all_strict";

  // Retain
  autoRetain: boolean;
  retainMode: string;
  retainEveryNTurns: number;
  retainOverlapTurns: number;
  retainContext: string;
  retainTags: string[];
  retainMetadata: Record<string, string>;

  // Connection
  hindsightApiUrl: string | null;
  hindsightApiToken: string | null;
  /**
   * Per-agent API tokens. When `dynamicApiKey` is enabled (default) and the
   * running agent's name is a key in this map, the corresponding token is used
   * for that agent's requests instead of `hindsightApiToken`. Agent names with
   * no entry fall back to `hindsightApiTokens[agentName]` (config default) then
   * to `hindsightApiToken`.
   */
  hindsightApiTokens: Record<string, string>;
  /** When true (default), resolve the API token per-agent from `hindsightApiTokens`. */
  dynamicApiKey: boolean;

  // Bank
  bankId: string | null;
  bankIdPrefix: string;
  dynamicBankId: boolean;
  dynamicBankGranularity: string[];
  bankMission: string;
  retainMission: string | null;
  agentName: string;
  /**
   * Per-agent bank IDs. When the running agent's name is a key in this map, the
   * corresponding bank ID is used for that agent's requests instead of the
   * derived bank ID. The `bankIdPrefix` is applied to the mapped value.
   * Agent names with no entry fall back to `hindsightBankIds[agentName]` (the
   * configured default agent) then to the normal `deriveBankId` result (static
   * `bankId` or dynamic-granularity composition).
   */
  hindsightBankIds: Record<string, string>;

  // Misc
  debug: boolean;
}

const DEFAULTS: HindsightConfig = {
  // Recall
  autoRecall: true,
  recallBudget: "mid",
  recallMaxTokens: 1024,
  recallTypes: ["world", "experience"],
  recallContextTurns: 1,
  recallMaxQueryChars: 800,
  recallTags: [],
  recallTagsMatch: "any",
  recallPromptPreamble:
    "Relevant memories from past conversations (prioritize recent when " +
    "conflicting). Only use memories that are directly useful to continue " +
    "this conversation; ignore the rest:",

  // Retain
  autoRetain: true,
  retainMode: "full-session",
  retainEveryNTurns: 3,
  retainOverlapTurns: 2,
  retainContext: "opencode",
  retainTags: [],
  retainMetadata: {},

  // Connection
  hindsightApiUrl: DEFAULT_HINDSIGHT_API_URL,
  hindsightApiToken: null,
  hindsightApiTokens: {},
  dynamicApiKey: true,

  // Bank
  bankId: null,
  bankIdPrefix: "",
  dynamicBankId: false,
  dynamicBankGranularity: ["agent", "project"],
  bankMission: "",
  retainMission: null,
  agentName: "opencode",
  hindsightBankIds: {},

  // Misc
  debug: false,
};

/** Env var → config key + type mapping */
const ENV_OVERRIDES: Record<string, [keyof HindsightConfig, "string" | "bool" | "int"]> = {
  HINDSIGHT_API_URL: ["hindsightApiUrl", "string"],
  HINDSIGHT_API_TOKEN: ["hindsightApiToken", "string"],
  HINDSIGHT_DYNAMIC_API_KEY: ["dynamicApiKey", "bool"],
  HINDSIGHT_BANK_ID: ["bankId", "string"],
  HINDSIGHT_AGENT_NAME: ["agentName", "string"],
  HINDSIGHT_AUTO_RECALL: ["autoRecall", "bool"],
  HINDSIGHT_AUTO_RETAIN: ["autoRetain", "bool"],
  HINDSIGHT_RETAIN_MODE: ["retainMode", "string"],
  HINDSIGHT_RECALL_BUDGET: ["recallBudget", "string"],
  HINDSIGHT_RECALL_MAX_TOKENS: ["recallMaxTokens", "int"],
  HINDSIGHT_RECALL_MAX_QUERY_CHARS: ["recallMaxQueryChars", "int"],
  HINDSIGHT_RECALL_CONTEXT_TURNS: ["recallContextTurns", "int"],
  HINDSIGHT_DYNAMIC_BANK_ID: ["dynamicBankId", "bool"],
  HINDSIGHT_BANK_MISSION: ["bankMission", "string"],
  HINDSIGHT_BANK_ID_PREFIX: ["bankIdPrefix", "string"],
  HINDSIGHT_RETAIN_EVERY_N_TURNS: ["retainEveryNTurns", "int"],
  HINDSIGHT_RETAIN_OVERLAP_TURNS: ["retainOverlapTurns", "int"],
  HINDSIGHT_RECALL_TAGS: ["recallTags", "string"],
  HINDSIGHT_RETAIN_TAGS: ["retainTags", "string"],
  HINDSIGHT_RECALL_TAGS_MATCH: ["recallTagsMatch", "string"],
  HINDSIGHT_RECALL_PROMPT_PREAMBLE: ["recallPromptPreamble", "string"],
  HINDSIGHT_RETAIN_CONTEXT: ["retainContext", "string"],
  // NOTE: `debug` is intentionally NOT an env override. It is a proper config
  // option set via opencode.json plugin options or ~/.hindsight/opencode.json,
  // because env vars are unreliable to set for OpenCode's plugin runtime
  // (notably on Windows).
};

function castEnv(value: string, typ: "string" | "bool" | "int"): string | boolean | number | null {
  if (typ === "bool") return ["true", "1", "yes"].includes(value.toLowerCase());
  if (typ === "int") {
    const n = parseInt(value, 10);
    return isNaN(n) ? null : n;
  }
  return value;
}

function loadSettingsFile(path: string): Record<string, unknown> {
  try {
    const raw = readFileSync(path, "utf-8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

export function loadConfig(pluginOptions?: Record<string, unknown>): HindsightConfig {
  // 1. Start with defaults
  const config: Record<string, unknown> = { ...DEFAULTS };

  // 2. User config file (~/.hindsight/opencode.json)
  const userConfigPath = join(homedir(), ".hindsight", "opencode.json");
  const fileConfig = loadSettingsFile(userConfigPath);
  for (const [key, value] of Object.entries(fileConfig)) {
    if (value !== null && value !== undefined) {
      config[key] = value;
    }
  }

  // 3. Plugin options (from opencode.json: ["@vectorize-io/opencode-hindsight", { ... }])
  if (pluginOptions) {
    for (const [key, value] of Object.entries(pluginOptions)) {
      if (value !== null && value !== undefined) {
        config[key] = value;
      }
    }
  }

  // 4. Environment variable overrides (highest priority)
  for (const [envName, [key, typ]] of Object.entries(ENV_OVERRIDES)) {
    const val = process.env[envName];
    if (val !== undefined) {
      const castVal = castEnv(val, typ);
      if (castVal !== null) {
        config[key] = castVal;
      }
    }
  }

  // Array env vars (comma-separated)
  const recallTagsEnv = process.env["HINDSIGHT_RECALL_TAGS"];
  if (recallTagsEnv !== undefined) {
    config["recallTags"] = recallTagsEnv
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
  }
  const recallTagsMatchEnv = process.env["HINDSIGHT_RECALL_TAGS_MATCH"];
  if (recallTagsMatchEnv !== undefined) {
    config["recallTagsMatch"] = recallTagsMatchEnv;
  }

  const retainTagsEnv = process.env["HINDSIGHT_RETAIN_TAGS"];
  if (retainTagsEnv !== undefined) {
    config["retainTags"] = retainTagsEnv
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
  }

  // Per-agent API token map (JSON object: { "agentName": "token", ... }).
  const apiTokensEnv = process.env["HINDSIGHT_API_TOKENS"];
  if (apiTokensEnv !== undefined) {
    try {
      const parsed = JSON.parse(apiTokensEnv);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const tokens: Record<string, string> = {};
        for (const [k, v] of Object.entries(parsed)) {
          if (typeof v === "string") tokens[k] = v;
        }
        config["hindsightApiTokens"] = tokens;
      } else {
        console.error(
          `[Hindsight] HINDSIGHT_API_TOKENS must be a JSON object — ignoring.`
        );
      }
    } catch (e) {
      console.error(
        `[Hindsight] Failed to parse HINDSIGHT_API_TOKENS as JSON — ignoring. ${
          String(e).split("\n")[0]
        }`
      );
    }
  }

  // Per-agent bank ID map (JSON object: { "agentName": "bank-id", ... }).
  const bankIdsEnv = process.env["HINDSIGHT_BANK_IDS"];
  if (bankIdsEnv !== undefined) {
    try {
      const parsed = JSON.parse(bankIdsEnv);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const ids: Record<string, string> = {};
        for (const [k, v] of Object.entries(parsed)) {
          if (typeof v === "string") ids[k] = v;
        }
        config["hindsightBankIds"] = ids;
      } else {
        console.error(
          `[Hindsight] HINDSIGHT_BANK_IDS must be a JSON object — ignoring.`
        );
      }
    } catch (e) {
      console.error(
        `[Hindsight] Failed to parse HINDSIGHT_BANK_IDS as JSON — ignoring. ${
          String(e).split("\n")[0]
        }`
      );
    }
  }

  const result = config as unknown as HindsightConfig;

  // Validate enum-like fields to catch typos early
  const VALID_RETAIN_MODES = ["full-session", "last-turn"];
  if (!VALID_RETAIN_MODES.includes(result.retainMode)) {
    console.error(
      `[Hindsight] Unknown retainMode "${result.retainMode}" — ` +
        `valid: ${VALID_RETAIN_MODES.join(", ")}. Falling back to "full-session".`
    );
    result.retainMode = "full-session";
  }

  const VALID_TAGS_MATCH = ["any", "all", "any_strict", "all_strict"];
  if (!VALID_TAGS_MATCH.includes(result.recallTagsMatch)) {
    console.error(
      `[Hindsight] Unknown recallTagsMatch "${result.recallTagsMatch}" — ` +
        `valid: ${VALID_TAGS_MATCH.join(", ")}. Falling back to "any".`
    );
    result.recallTagsMatch = "any";
  }

  const VALID_BUDGETS = ["low", "mid", "high"];
  if (!VALID_BUDGETS.includes(result.recallBudget)) {
    console.error(
      `[Hindsight] Unknown recallBudget "${result.recallBudget}" — ` +
        `valid: ${VALID_BUDGETS.join(", ")}. Falling back to "mid".`
    );
    result.recallBudget = "mid";
  }

  return result;
}

/**
 * Resolve the Hindsight API token for a given agent name.
 *
 * Resolution order (first non-empty wins):
 *   1. `hindsightApiTokens[agentName]` — when `dynamicApiKey` is enabled and an
 *      explicit entry exists for the running agent.
 *   2. `hindsightApiTokens[config.agentName]` — the entry for the configured
 *      default agent name.
 *   3. `hindsightApiToken` — the single static token (legacy behavior).
 *
 * `agentName` is the name of the OpenCode agent currently driving the session
 * (e.g. "build", "code-reviewer"), as reported by OpenCode's tool/session
 * context. Pass `null`/`undefined` when the agent name is unknown.
 */
export function resolveApiKey(
  config: HindsightConfig,
  agentName?: string | null
): string | null {
  if (config.dynamicApiKey && agentName) {
    const perAgent = config.hindsightApiTokens?.[agentName];
    if (perAgent) return perAgent;
  }
  if (config.dynamicApiKey && config.hindsightApiTokens?.[config.agentName]) {
    return config.hindsightApiTokens[config.agentName]!;
  }
  return config.hindsightApiToken;
}
