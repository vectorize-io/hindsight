/**
 * Bank ID derivation and mission management.
 *
 * Port of Claude Code plugin's bank.py, adapted for OpenCode's context model.
 *
 * Dimensions for dynamic bank IDs:
 *   - agent      → configured name or "opencode"
 *   - project    → derived from the working directory basename
 *   - gitProject → derived from the main worktree's basename when inside a
 *                  git repository (so all linked worktrees of the same repo
 *                  share a single memory bank). Falls back to the working
 *                  directory basename when git is unavailable or the
 *                  directory is not a repo.
 */

import { basename, dirname } from "node:path";
import { execFileSync } from "node:child_process";
import type { HindsightConfig } from "./config.js";
import { debugLog } from "./config.js";
import type { HindsightClient } from "@vectorize-io/hindsight-client";

const DEFAULT_BANK_NAME = "opencode";
const VALID_FIELDS = new Set(["agent", "project", "gitProject", "channel", "user"]);

/**
 * Resolve the main worktree root for a directory inside a git repository.
 *
 * Uses `git rev-parse --path-format=absolute --git-common-dir`, which always
 * points to the .git directory of the *main* worktree, even when invoked from
 * a linked worktree (created with `git worktree add`). The parent of that path
 * is the main worktree root, so all linked worktrees of the same repo resolve
 * to the same root and end up sharing one memory bank.
 *
 * Returns `null` when git is unavailable, the directory is not a repo, or the
 * git invocation fails for any other reason.
 */
function getProjectRootFromGit(directory: string): string | null {
  if (!directory) return null;
  try {
    const commonDir = execFileSync(
      "git",
      ["rev-parse", "--path-format=absolute", "--git-common-dir"],
      {
        cwd: directory,
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 1000,
      }
    ).trim();
    if (!commonDir) return null;
    // For typical clones and `git worktree add`, common-dir is `<root>/.git`,
    // so the parent is the main worktree root. For bare repos, common-dir is
    // the bare directory itself (e.g. `myrepo.git`); use it directly.
    if (basename(commonDir) === ".git") {
      return dirname(commonDir);
    }
    return commonDir;
  } catch {
    return null;
  }
}

function deriveGitProjectName(directory: string): string {
  const projectRoot = getProjectRootFromGit(directory);
  if (projectRoot) return basename(projectRoot);
  return directory ? basename(directory) : "unknown";
}

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

  // Lazy resolution so we don't spawn `git` for `gitProject` when the field
  // isn't part of the configured granularity.
  const fieldResolvers: Record<string, () => string> = {
    agent: () => config.agentName || "opencode",
    project: () => (directory ? basename(directory) : "unknown"),
    gitProject: () => deriveGitProjectName(directory),
    channel: () => channelId || "default",
    user: () => userId || "anonymous",
  };

  // bank_id is stored as-is server-side; HTTP path encoding is the client layer's job.
  const segments = fields.map((f) => fieldResolvers[f]?.() || "unknown");
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
