/**
 * Bank ID derivation and mission management.
 *
 * Port of Claude Code plugin's bank.py, adapted for OpenCode's context model.
 *
 * Dimensions for dynamic bank IDs:
 *   - agent   → configured name or "opencode"
 *   - project → derived from working directory basename
 */

import { basename } from "node:path";
import type { HindsightConfig } from "./config.js";
import { debugLog } from "./config.js";
import type { HindsightClient } from "@vectorize-io/hindsight-client";

const DEFAULT_BANK_NAME = "opencode";
const VALID_FIELDS = new Set(["agent", "project", "channel", "user"]);

/**
 * Derive a bank ID from context and config.
 *
 * Static mode: returns config.bankId or DEFAULT_BANK_NAME.
 * Dynamic mode: composes from granularity fields joined by '::'.
 */
export function deriveBankId(config: HindsightConfig, directory: string): string {
  const prefix = config.bankIdPrefix;

  if (!config.dynamicBankId) {
    const base = config.bankId || DEFAULT_BANK_NAME;
    return prefix ? `${prefix}-${base}` : base;
  }

  const fields = config.dynamicBankGranularity?.length
    ? config.dynamicBankGranularity
    : ["agent", "project"];

  for (const f of fields) {
    if (!VALID_FIELDS.has(f)) {
      console.error(
        `[Hindsight] Unknown dynamicBankGranularity field "${f}" — ` +
          `valid: ${[...VALID_FIELDS].sort().join(", ")}`
      );
    }
  }

  const channelId = process.env.HINDSIGHT_CHANNEL_ID || "";
  const userId = process.env.HINDSIGHT_USER_ID || "";

  const fieldMap: Record<string, string> = {
    agent: config.agentName || "opencode",
    project: directory ? basename(directory) : "unknown",
    channel: channelId || "default",
    user: userId || "anonymous",
  };

  // bank_id is stored as-is server-side; HTTP path encoding is the client layer's job.
  const segments = fields.map((f) => fieldMap[f] || "unknown");
  const baseBankId = segments.join("::");

  return prefix ? `${prefix}-${baseBankId}` : baseBankId;
}

/**
 * Set bank mission on first use, skip if already set.
 * Uses an in-memory Set (plugin is long-lived, unlike Claude Code's ephemeral hooks).
 */
export async function ensureBankMission(
  client: HindsightClient,
  bankId: string,
  config: HindsightConfig,
  missionsSet: Set<string>
): Promise<void> {
  const mission = config.bankMission;
  if (!mission?.trim()) return;
  if (missionsSet.has(bankId)) return;

  try {
    await client.createBank(bankId, {
      reflectMission: mission,
      retainMission: config.retainMission || undefined,
    });
    missionsSet.add(bankId);
    // Cap tracked banks
    if (missionsSet.size > 10000) {
      const keys = [...missionsSet].sort();
      for (const k of keys.slice(0, keys.length >> 1)) {
        missionsSet.delete(k);
      }
    }
    debugLog(config, `Set mission for bank: ${bankId}`);
  } catch (e) {
    // Don't fail if mission set fails — bank may not exist yet
    debugLog(config, `Could not set bank mission for ${bankId}: ${e}`);
  }
}
