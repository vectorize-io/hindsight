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

import { basename, dirname, join } from "node:path";
import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, statSync } from "node:fs";
import { homedir } from "node:os";
import type { HindsightConfig } from "./config.js";
import { Logger } from "./logger.js";
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
  missionsSet: Set<string>,
  logger: Logger = new Logger({ silent: true })
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
    logger.debug(`Set mission for bank: ${bankId}`);
  } catch (e) {
    // Don't fail if mission set fails — bank may not exist yet
    logger.debug(`Could not set bank mission for ${bankId}`, { error: String(e) });
  }
}

/**
 * Search locations for agent definition files, in priority order.
 * Project-scoped files take precedence over global ones (mirroring how
 * OpenCode merges configs).
 */
const AGENT_FILE_DIRS = (directory: string): string[] => [
  join(directory, ".opencode", "agent"),
  join(directory, ".opencode", "agents"),
  join(homedir(), ".config", "opencode", "agent"),
  join(homedir(), ".config", "opencode", "agents"),
];

/** Cache: agent name → { mtime, bankId } so we re-read only when the file changes. */
const agentFileCache = new Map<string, { mtime: number; bankId: string | null }>();

/**
 * Extract the `bankid` scalar from YAML frontmatter without a full YAML parser.
 * Handles `bankid: value`, `"value"`, and `'value'` forms. Only top-level keys
 * (column 0) are matched, so nested keys with the same name are ignored.
 */
function extractBankIdFromFrontmatter(content: string): string | null {
  // Find frontmatter block between the first pair of `---` lines.
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return null;
  const body = match[1];
  for (const line of body.split("\n")) {
    // Top-level key only (no leading whitespace).
    const m = line.match(/^bankid:\s*(.*)$/i);
    if (m) {
      let val = m[1].trim();
      // Strip surrounding quotes.
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      return val || null;
    }
  }
  return null;
}

/**
 * Read the `bankid` field from an agent's `.md` definition file.
 *
 * Searches the standard agent file locations (project then global) and returns
 * the frontmatter `bankid` value if present. Results are cached per agent name
 * keyed on file mtime, so repeated calls during a session are cheap.
 *
 * Returns `null` when the agent file is not found, has no frontmatter, or the
 * frontmatter does not declare a `bankid`.
 */
export function readAgentBankId(
  agentName: string | null | undefined,
  directory: string
): string | null {
  if (!agentName) return null;

  const dirs = AGENT_FILE_DIRS(directory);
  let filePath: string | null = null;
  for (const dir of dirs) {
    const candidate = join(dir, `${agentName}.md`);
    if (existsSync(candidate)) {
      filePath = candidate;
      break;
    }
  }
  if (!filePath) return null;

  let mtime: number;
  try {
    mtime = statSync(filePath).mtimeMs;
  } catch {
    return null;
  }

  const cached = agentFileCache.get(filePath);
  if (cached && cached.mtime === mtime) {
    return cached.bankId;
  }

  let bankId: string | null = null;
  try {
    const raw = readFileSync(filePath, "utf-8");
    bankId = extractBankIdFromFrontmatter(raw);
  } catch {
    bankId = null;
  }

  agentFileCache.set(filePath, { mtime, bankId });
  return bankId;
}

/**
 * Resolve the bank ID for a given agent name, applying the per-agent
 * `hindsightBankIds` map with a fallback to the normal `deriveBankId` result.
 *
 * Resolution order (first defined value wins):
 *   1. Agent `.md` frontmatter `bankid` field — read from the agent's
 *      definition file. This takes precedence over all other sources.
 *      The `bankIdPrefix` is applied.
 *   2. `hindsightBankIds[agentName]` — explicit entry for the running agent.
 *      The `bankIdPrefix` is applied.
 *   3. `hindsightBankIds[config.agentName]` — entry for the configured
 *      default agent name. The `bankIdPrefix` is applied.
 *   4. `deriveBankId(config, directory)` — the legacy derivation (static
 *      `bankId` or dynamic-granularity composition). Already applies the prefix.
 *
 * `agentName` is the name of the OpenCode agent currently driving the session
 * (e.g. "build", "code-reviewer"). Pass `null`/`undefined` when the agent
 * name is unknown — resolution then skips straight to the fallback.
 */
export function resolveBankId(
  config: HindsightConfig,
  directory: string,
  agentName?: string | null
): string {
  const prefix = config.bankIdPrefix;
  const applyPrefix = (base: string) => (prefix ? `${prefix}-${base}` : base);

  // 1. Agent .md frontmatter (highest precedence)
  const agentFileBankId = readAgentBankId(agentName, directory);
  if (agentFileBankId) {
    return applyPrefix(agentFileBankId);
  }

  // 2. Per-agent map
  if (agentName && config.hindsightBankIds?.[agentName]) {
    return applyPrefix(config.hindsightBankIds[agentName]);
  }
  // 3. Default agentName entry
  if (config.hindsightBankIds?.[config.agentName]) {
    return applyPrefix(config.hindsightBankIds[config.agentName]);
  }
  // 4. Legacy derivation
  return deriveBankId(config, directory);
}

/** Anything that can resolve a bank ID for a given agent name. */
export interface BankResolver {
  forAgent(agentName?: string | null): string;
}

/**
 * Normalize a `string | BankResolver` into a `BankResolver`. A bare string
 * (e.g. a fixed bank ID in unit tests) is wrapped so `forAgent()` always
 * returns it, regardless of agent — keeping existing call sites unchanged.
 */
export function toBankResolver(bankIdOrResolver: string | BankResolver): BankResolver {
  const maybe = bankIdOrResolver as Partial<BankResolver>;
  if (typeof maybe.forAgent === "function") {
    return bankIdOrResolver as BankResolver;
  }
  const bankId = bankIdOrResolver as string;
  return { forAgent: () => bankId };
}

/**
 * Build a `BankResolver` that resolves the bank ID per-agent from
 * `hindsightBankIds`, falling back to the legacy `deriveBankId` derivation.
 */
export function createBankResolver(
  config: HindsightConfig,
  directory: string
): BankResolver {
  return {
    forAgent: (agentName?: string | null) => resolveBankId(config, directory, agentName),
  };
}
