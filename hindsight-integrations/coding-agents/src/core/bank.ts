/**
 * Dynamic bank resolution — which memory bank does THIS directory belong to?
 *
 * Coding memory is per-REPOSITORY: by default the bank id is derived from the git repo the
 * working directory lives in, worktree-aware — every linked worktree of a repo resolves to the
 * main worktree's basename and therefore shares one bank.
 *
 * Resolution order:
 *   1. `mapPathToBank` — absolute path -> bank; LONGEST matching prefix wins, so mapping a
 *      repo root covers every subdirectory (and worktree paths can be pinned individually).
 *      Overrides everything, including an explicit bankId.
 *   2. static — when `dynamicBankId` is false, or left unset WITH an explicit `bankId`
 *      (the benchmark harness and single-bank setups).
 *   3. dynamic — `bankIdTemplate` (default "coding-agent::{gitProject}") with placeholders:
 *        {gitProject}  worktree-aware repo name (all worktrees share it; outside a repo: the
 *                      basename of the directory the SESSION started in, not the agent's live cwd)
 *        {project}     working-directory basename (no git involved)
 *        {harness}     the entry point asking ("opencode", "claude-code", "codex", "antigravity-cli", ...)
 *        {channel}     $HINDSIGHT_CHANNEL_ID or "default"
 *        {user}        $HINDSIGHT_USER_ID or "anonymous"
 *      e.g. "hindsight-{gitProject}" or "{harness}-{gitProject}" to split per agent. The default
 *      is harness-neutral "coding-agent::{gitProject}" so every coding agent shares ONE memory per repo.
 */
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, normalize, sep } from "node:path";
import { applyTemplate } from "./template";

export interface BankConfig {
  bankId?: string;
  dynamicBankId?: boolean;
  bankIdTemplate?: string;
  mapPathToBank?: Record<string, string>;
  resolveWorktrees?: boolean; // default true: worktrees share the main repo's bank
  optInOnly?: boolean; // memory runs ONLY where opted in (see isOptedIn)
  optInPaths?: string[]; // directories opted in, matched as prefixes
}

const DEFAULT_BANK_NAME = "coding";
// Harness-NEUTRAL default so every coding agent (Claude, Codex, Cursor, opencode) shares ONE bank
// per repo — switch agents, keep your memory. Namespaced with `coding-agent::` to identify these
// banks and avoid collisions with other Hindsight banks. Deliberately NOT `{harness}::…` (that would
// split memory per agent, defeating cross-agent sharing).
const DEFAULT_TEMPLATE = "coding-agent::{gitProject}";

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

/** Worktree-aware repo name for DOCUMENT IDS (gitlog:<name>, commit context): all worktrees of a
 *  repo must produce the SAME name, or each worktree writes its own gitlog document into the
 *  shared bank (seen in the wild: gitlog:hindsight-wt7 next to gitlog:memory-poc). */
export function projectNameOf(directory: string, sessionRoot?: string): string {
  return gitProjectName(directory, true, sessionRoot);
}

/**
 * Project roots a harness exports into its hook process. Only Claude Code is known to export one,
 * so this is deliberately a list of ONE rather than a guess at nine names — the mechanism that
 * covers every harness is the ancestor walk in `nearestExistingDir`, not this.
 *
 * It earns its place for the case the walk cannot reach: a LINKED worktree (a sibling directory,
 * not a child of the repo) that has been deleted. Walking up from it lands outside the repository
 * entirely, while this variable still names it. The session root passed by the hook runtimes covers
 * the same case harness-neutrally when a session is what is being resolved.
 */
const PROJECT_ROOT_ENV = ["CLAUDE_PROJECT_DIR"] as const;

/**
 * The nearest ancestor of `directory` that still exists, or "" if none does.
 *
 * A hook runs after the fact, so the directory it reports can already be gone — an ephemeral
 * worktree removed once the task finished, a checkout moved or deleted mid-session. git can only
 * answer about a path that exists, so probing the vanished leaf fails and its basename — a
 * throwaway name like `agent-a33c4d63` — becomes the project identity, scattering memory into
 * orphan banks (#3110). Walking up finds the repository that contained it. Harness-agnostic: no
 * harness has to export anything for this to work.
 *
 * When the directory does exist this returns it unchanged, so the common path is untouched and no
 * existing bank moves.
 */
function nearestExistingDir(directory: string): string {
  let current = directory;
  while (current) {
    if (existsSync(current)) return current;
    const parent = dirname(current);
    if (parent === current) return "";
    current = parent;
  }
  return "";
}

/** Basename of a directory, or "unknown" — never the empty string. `basename("/")` is "", which
 *  would otherwise produce a bank id like `coding-agent::` that names nothing. */
function dirName(directory: string): string {
  return (directory && basename(directory)) || "unknown";
}

function gitProjectName(directory: string, resolveWorktrees: boolean, sessionRoot = ""): string {
  if (resolveWorktrees) {
    // The directory itself first (via the walk, which is a no-op when it exists), so anything git
    // can still resolve keeps its historical answer and no existing bank moves. The session root
    // and the exported roots are a last rescue, not a new source of truth.
    const candidates = [
      nearestExistingDir(directory),
      sessionRoot,
      ...PROJECT_ROOT_ENV.map((v) => process.env[v] || ""),
    ];
    for (const candidate of candidates) {
      const root = candidate ? getProjectRootFromGit(candidate) : null;
      if (root) return basename(root);
    }
  }
  // Nothing git can name. `directory` is the agent's LIVE working directory and it moves during
  // normal work; inside a repo that was harmless because every subdirectory resolved back to the
  // root above, but a plain directory tree has no root to resolve to, so the bank id followed the
  // agent and one session was retained into a bank per directory it stepped into (#3563). Name the
  // directory the session STARTED in instead — a subdirectory earns its own bank by starting a
  // session there, not by being visited.
  return dirName(sessionRoot || directory);
}

/** A configured directory, `~`-expanded and normalised, without a trailing separator. */
function configuredDir(dir: string): string {
  const expanded = dir === "~" || dir.startsWith("~/") ? join(homedir(), dir.slice(1)) : dir;
  return normalize(expanded).replace(new RegExp(`\\${sep}+$`), "");
}

/** Whether `directory` IS `configured`, or lives under it. */
function isWithin(directory: string, configured: string): boolean {
  return directory === configured || directory.startsWith(configured + sep);
}

/** Longest-prefix match of `directory` against the map's absolute paths (exact or ancestor). */
function mapLookup(map: Record<string, string>, directory: string): string | undefined {
  const cwd = normalize(directory);
  let best: { len: number; bank: string } | undefined;
  for (const [dir, bank] of Object.entries(map)) {
    const p = configuredDir(dir);
    if (isWithin(cwd, p)) {
      if (!best || p.length > best.len) best = { len: p.length, bank };
    }
  }
  return best?.bank;
}

/**
 * Whether memory may run for this directory at all.
 *
 * Off by default: without `optInOnly` every project gets memory, which is what makes the plugin
 * zero-setup. With it, the plugin stays inert — no bank, no retain, no seed — unless the directory
 * was named on purpose, which means one of:
 *
 *   - it is under an `optInPaths` entry (prefix-matched, so approving a directory approves the
 *     repos beneath it while each keeps its own dynamic bank), or
 *   - it is under a `mapPathToBank` entry, since routing a path to a named bank is already a
 *     deliberate declaration of that project.
 *
 * A bare `bankId` deliberately does NOT approve anything: it names a bank, not a project, so it
 * cannot express which work is allowed to be remembered. Under `optInOnly` an unlisted project is
 * inert even then — a privacy switch has to fail closed.
 */
export function isOptedIn(config: BankConfig, directory: string): boolean {
  if (!config.optInOnly) return true;
  if (!directory) return false;
  const cwd = normalize(directory);
  if ((config.optInPaths ?? []).some((dir) => dir && isWithin(cwd, configuredDir(dir))))
    return true;
  return Boolean(config.mapPathToBank && mapLookup(config.mapPathToBank, directory));
}

/** Derive the bank id for a working directory (see module doc for the resolution order). */
export function deriveBankId(
  config: BankConfig,
  directory: string,
  harness = "coding",
  /** Where the SESSION started, when the caller knows it (hook runtimes do — see
   *  `sessionRootDir`). Used only where the live directory yields no project: see gitProjectName. */
  sessionRoot?: string
): string {
  const mapped =
    directory && config.mapPathToBank ? mapLookup(config.mapPathToBank, directory) : undefined;
  if (mapped) return mapped;

  // dynamic by default — but an explicit bankId (without dynamicBankId: true) means "static".
  const dynamic = config.dynamicBankId ?? !config.bankId;
  if (!dynamic) return config.bankId || DEFAULT_BANK_NAME;

  const resolvers: Record<string, () => string> = {
    harness: () => harness,
    project: () => dirName(directory),
    gitProject: () => gitProjectName(directory, config.resolveWorktrees ?? true, sessionRoot),
    channel: () => process.env.HINDSIGHT_CHANNEL_ID || "default",
    user: () => process.env.HINDSIGHT_USER_ID || "anonymous",
  };
  return applyTemplate(config.bankIdTemplate || DEFAULT_TEMPLATE, resolvers, "bankIdTemplate");
}
