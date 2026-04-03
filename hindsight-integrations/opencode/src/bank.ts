/**
 * Bank ID derivation and mission management.
 *
 * Supports static bank IDs ("opencode") or dynamic per-project IDs
 * derived from the working directory. Ensures the bank exists with the
 * configured mission before the first retain/recall operation.
 */

import { createHash } from "crypto";
import { basename } from "path";
import type { HindsightConfig } from "./config";
import type { HindsightClient } from "./client";
import { debugLog } from "./config";

const ensuredBanks = new Set<string>();

export function deriveBankId(config: HindsightConfig, directory?: string): string {
  if (!config.dynamicBankId) {
    const base = config.bankId || "opencode";
    return config.bankIdPrefix ? `${config.bankIdPrefix}-${base}` : base;
  }

  const parts: string[] = [];

  for (const field of config.dynamicBankGranularity) {
    if (field === "project" && directory) {
      parts.push(basename(directory));
    } else if (field === "agent") {
      parts.push("opencode");
    }
  }

  if (parts.length === 0) parts.push("opencode");

  const slug = parts.join("-").toLowerCase().replace(/[^a-z0-9-]/g, "-");
  const bankId = config.bankIdPrefix ? `${config.bankIdPrefix}-${slug}` : slug;

  // Bank IDs have a max length in Hindsight; hash if too long
  if (bankId.length > 64) {
    const hash = createHash("sha256").update(bankId).digest("hex").slice(0, 12);
    return bankId.slice(0, 51) + "-" + hash;
  }

  return bankId;
}

export async function ensureBankMission(
  client: HindsightClient,
  bankId: string,
  config: HindsightConfig,
): Promise<void> {
  if (ensuredBanks.has(bankId)) return;

  try {
    const exists = await client.bankExists(bankId);
    if (!exists) {
      debugLog(config, `Creating bank "${bankId}"`);
      await client.createOrUpdateBank({
        bankId,
        name: `OpenCode - ${bankId}`,
        mission: config.bankMission,
        retainMission: config.retainMission || undefined,
      });
    }
    ensuredBanks.add(bankId);
  } catch (err) {
    debugLog(config, `Failed to ensure bank: ${err}`);
  }
}
