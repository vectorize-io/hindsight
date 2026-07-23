/**
 * Dynamic bank resolution — which memory bank does THIS directory belong to?
 *
 * Port of the family convention (claude-code bank.py -> omo/cline/opencode): coding memory is
 * per-REPOSITORY, so by default the bank id is derived from the git repo the working directory
 * lives in, worktree-aware — every linked worktree of a repo resolves to the main worktree's
 * basename and therefore shares one bank. An explicit `bankId` in config keeps today's static
 * behavior (used by the benchmark harness, single-bank setups, CI).
 *
 * Resolution order:
 *   1. `directoryBankMap` — exact working-directory -> bank mapping (collision escape hatch)
 *   2. static — when `dynamicBankId` is false, or left unset WITH an explicit `bankId`
 *   3. dynamic — `dynamicBankGranularity` fields (default ["gitProject"]) joined by "::"
 * `bankIdPrefix` is prepended to whatever the above produced.
 *
 * The default granularity is ["gitProject"] (not agent::project): the point of the multi-harness
 * plugin is that opencode and claude share ONE memory per repo — add "agent" to the granularity
 * to split per agent instead.
 */
import { execFileSync } from "node:child_process";
import { basename, dirname, normalize } from "node:path";

export interface BankConfig {
  bankId?: string;
  bankIdPrefix?: string;
  dynamicBankId?: boolean;
  dynamicBankGranularity?: string[];
  directoryBankMap?: Record<string, string>;
  agentName?: string;
  resolveWorktrees?: boolean; // default true: worktrees share the main repo's bank
}

const DEFAULT_BANK_NAME = "coding";
const VALID_FIELDS = new Set(["agent", "project", "gitProject", "channel", "user"]);

/** Main-worktree root for a directory inside a git repo (worktree- and bare-repo-aware), else null. */
export function getProjectRootFromGit(directory: string): string | null {
  if (!directory) return null;
  try {
    const commonDir = execFileSync(
      "git",
      ["rev-parse", "--path-format=absolute", "--git-common-dir"],
      { cwd: directory, encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"], timeout: 1000 }
    ).trim();
    if (!commonDir) return null;
    // clones + `git worktree add`: common-dir is `<main root>/.git`; bare repos: the dir itself.
    return basename(commonDir) === ".git" ? dirname(commonDir) : commonDir;
  } catch {
    return null;
  }
}

function gitProjectName(directory: string, resolveWorktrees: boolean): string {
  if (resolveWorktrees) {
    const root = getProjectRootFromGit(directory);
    if (root) return basename(root);
  }
  return directory ? basename(directory) : "unknown";
}

/** Derive the bank id for a working directory (see module doc for the resolution order). */
export function deriveBankId(config: BankConfig, directory: string): string {
  const prefix = config.bankIdPrefix;
  const withPrefix = (base: string) => (prefix ? `${prefix}-${base}` : base);

  const map = config.directoryBankMap ?? {};
  if (directory && Object.keys(map).length) {
    const cwd = normalize(directory);
    for (const [dir, bank] of Object.entries(map)) {
      if (normalize(dir) === cwd) return withPrefix(bank);
    }
  }

  // dynamic by default — but an explicit bankId (without dynamicBankId: true) means "static".
  const dynamic = config.dynamicBankId ?? !config.bankId;
  if (!dynamic) return withPrefix(config.bankId || DEFAULT_BANK_NAME);

  const fields = config.dynamicBankGranularity?.length ? config.dynamicBankGranularity : ["gitProject"];
  for (const f of fields) {
    if (!VALID_FIELDS.has(f)) {
      console.error(
        `hindsight: unknown dynamicBankGranularity field "${f}" — valid: ${[...VALID_FIELDS].sort().join(", ")}`
      );
    }
  }
  const resolvers: Record<string, () => string> = {
    agent: () => config.agentName || "coding",
    project: () => (directory ? basename(directory) : "unknown"),
    gitProject: () => gitProjectName(directory, config.resolveWorktrees ?? true),
    channel: () => process.env.HINDSIGHT_CHANNEL_ID || "default",
    user: () => process.env.HINDSIGHT_USER_ID || "anonymous",
  };
  return withPrefix(fields.map((f) => resolvers[f]?.() || "unknown").join("::"));
}
