/**
 * Configuration for the Hindsight Zed integration.
 *
 * Settings layer (later wins): built-in defaults -> ~/.hindsight/zed.json ->
 * environment variables. Resolved into a plain config object.
 *
 * The integration is configuration-only: it wires Zed's MCP `context_servers` to
 * the Hindsight MCP endpoint and writes a recall/retain rule into Zed's global
 * instructions file. Memory operations happen through the MCP server at runtime,
 * so there is no daemon or direct API client here.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { isFile } from "./fsutil.js";

// Cross-integration cloud-default convention.
export const DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io";
export const DEFAULT_BANK_ID = "zed";

export const USER_CONFIG_FILE = join(homedir(), ".hindsight", "zed.json");

// user-config file key -> config attribute
const FILE_KEYS = {
  hindsightApiUrl: "hindsightApiUrl",
  hindsightApiToken: "hindsightApiToken",
  bankId: "bankId",
};

// env var -> config attribute
const ENV_KEYS = {
  HINDSIGHT_API_URL: "hindsightApiUrl",
  HINDSIGHT_API_TOKEN: "hindsightApiToken",
  HINDSIGHT_ZED_BANK_ID: "bankId",
};

/** Load and resolve configuration from file then environment. */
export function loadConfig({ configFile = USER_CONFIG_FILE, env = process.env } = {}) {
  const cfg = {
    hindsightApiUrl: DEFAULT_HINDSIGHT_API_URL,
    hindsightApiToken: null,
    bankId: DEFAULT_BANK_ID,
  };

  if (isFile(configFile)) {
    let data = {};
    try {
      data = JSON.parse(readFileSync(configFile, "utf-8"));
    } catch {
      data = {};
    }
    if (data && typeof data === "object" && !Array.isArray(data)) {
      for (const [key, attr] of Object.entries(FILE_KEYS)) {
        const value = data[key];
        if (value) cfg[attr] = String(value);
      }
    }
  }

  for (const [key, attr] of Object.entries(ENV_KEYS)) {
    const value = env[key];
    if (value) cfg[attr] = String(value);
  }

  if (!cfg.hindsightApiUrl) cfg.hindsightApiUrl = DEFAULT_HINDSIGHT_API_URL;
  if (!cfg.bankId) cfg.bankId = DEFAULT_BANK_ID;

  return cfg;
}
