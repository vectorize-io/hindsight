/**
 * Dynamic bank resolution — which memory bank does THIS directory belong to?
 *
 * Coding memory is per-REPOSITORY: by default the bank id is derived from the git repo the
 * working directory lives in, worktree-aware — every linked worktree of a repo resolves to the
 * main worktree's basename and therefore shares one bank.
 *
 * Resolution order:
 *   1. `mapPathToBank` — absolute path -> bank; LONGEST matching prefix wins, so mapping a
 *      repo root covers every subdirectory and linked worktree of that repo.
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
 *
 * The repository probe (core/git-layout.ts) reads the repository layout off disk and answers one
 * of three things: this repo, no repo, or "could not tell". Only "no repo" reaches the basename
 * fallback — a probe that FAILED makes resolution throw `BankResolutionError`, and the lifecycle
 * hooks skip the session (`deriveBankIdOrSkip`). Guessing there is how a linked worktree ended up
 * with a permanent bank of its own (#3950): a skipped session is recoverable, a scattered one is not.
 */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, normalize, sep } from "node:path";
import { diag } from "./diag";
import { probeGitLayout } from "./git-layout";
import { log } from "./log";
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

/**
 * The repository a directory belongs to — or WHY there is no answer.
 *
 * "This is not a repository" and "the probe did not complete" are different answers and only the
 * first may reach a fallback: mapping both to `null` is what forked linked worktrees into their
 * own banks (#3950).
 */
export type ProjectRoot =
  | { status: "resolved"; root: string }
  | { status: "absent" }
  | { status: "failed"; reason: string };

/** Thrown when a repository could not be identified because the PROBE failed. Callers on the
 *  retain path skip the session rather than invent a bank: a skipped session is recoverable, a
 *  session scattered into a bank nobody reads is not. */
export class BankResolutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BankResolutionError";
  }
}

/** Main-worktree root for a directory inside a git repo (worktree- and bare-repo-aware). */
export function resolveProjectRoot(directory: string): ProjectRoot {
  if (!directory) return { status: "absent" };
  const layout = probeGitLayout(directory);
  if (layout.status !== "resolved") return layout;
  const commonDir = layout.commonDir;
  // Clones + `git worktree add`: common-dir is `<main root>/.git`.
  if (basename(commonDir) === ".git") return { status: "resolved", root: dirname(commonDir) };

  // A bare-hub keeps its bare repository in a hidden plumbing directory (usually `.bare`), while a
  // standalone bare clone uses its directory name as the project identity.
  if (basename(commonDir).startsWith(".") && layout.bare) {
    return { status: "resolved", root: dirname(commonDir) };
  }

  // Preserve the historical name for standalone bare repositories such as `myrepo.git`.
  return { status: "resolved", root: commonDir };
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

/**
 * The main-worktree root for a location, or null — ONE cascade, shared by the two things that ask
 * it: which repo names this bank (`gitProjectName`) and which repo's approval and mapping this
 * directory inherits (`lookupDirectories`). They used to walk their own copies, and the copies
 * disagreed: bank identity consulted the session root while approval did not, so a directory could
 * be named after a repo it was not allowed to be remembered for.
 *
 * The directory itself comes first (via the walk, a no-op when it exists), so anything git can
 * still resolve keeps its historical answer and no existing bank moves. The session root and the
 * exported roots are a last rescue, not a new source of truth — and both name the CURRENT
 * session's own project, so neither can reach a repo this session was not already working in.
 */
function mainWorktreeRoot(directory: string, sessionRoot = ""): ProjectRoot {
  const candidates = [
    nearestExistingDir(directory),
    sessionRoot,
    ...PROJECT_ROOT_ENV.map((v) => process.env[v] || ""),
  ];
  // A failure anywhere in the cascade is remembered but not returned early: a later candidate may
  // still answer, and only when NONE does does the difference between "no repository here" and
  // "could not tell" matter.
  let failure: ProjectRoot | null = null;
  for (const candidate of candidates) {
    if (!candidate) continue;
    const projectRoot = resolveProjectRoot(candidate);
    if (projectRoot.status === "resolved") return projectRoot;
    if (projectRoot.status === "failed") failure = projectRoot;
  }
  return failure ?? { status: "absent" };
}

function gitProjectName(directory: string, resolveWorktrees: boolean, sessionRoot = ""): string {
  if (resolveWorktrees) {
    const projectRoot = mainWorktreeRoot(directory, sessionRoot);
    if (projectRoot.status === "resolved") return basename(projectRoot.root);
    // The fallback below is right for a directory outside any repository and WRONG for a failed
    // probe inside one — for a linked worktree it names the worktree instead of the repo, forking
    // it into a bank of its own, forever and silently (#3950). Refuse to guess instead.
    if (projectRoot.status === "failed") {
      throw new BankResolutionError(
        `git probe failed for ${directory} (${projectRoot.reason}) — refusing to guess a bank id`
      );
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

/** Current directory first, then its main Git root. Keeping the literal path first preserves an
 * explicit worktree-specific mapping while letting an approved checkout carry that approval to
 * linked worktrees outside the configured directory tree. Same cascade the bank name resolves
 * through, session root included, so approval and identity cannot disagree about which repo a
 * directory belongs to. */
function lookupDirectories(config: BankConfig, directory: string, sessionRoot = ""): string[] {
  const directories = [normalize(directory)];
  if (config.resolveWorktrees ?? true) {
    // Deliberately failure-tolerant, unlike bank naming: approval and mapping fall back to the
    // literal directory, which can only ever be narrower than the repo-wide answer.
    const projectRoot = mainWorktreeRoot(directory, sessionRoot);
    const normalizedRoot = projectRoot.status === "resolved" ? normalize(projectRoot.root) : "";
    if (normalizedRoot && normalizedRoot !== directories[0]) directories.push(normalizedRoot);
  }
  return directories;
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
  const directories = lookupDirectories(config, directory);
  if (
    directories.some((candidate) =>
      (config.optInPaths ?? []).some(
        (configured) => configured && isWithin(candidate, configuredDir(configured))
      )
    )
  )
    return true;
  const pathMap = config.mapPathToBank;
  return Boolean(pathMap && directories.some((candidate) => mapLookup(pathMap, candidate)));
}

/** Derive the bank id for a working directory (see module doc for the resolution order). */
export function deriveBankId(
  config: BankConfig,
  directory: string,
  harness = "coding",
  /** Where the SESSION started, when the caller knows it (hook runtimes do — see
   *  `sessionRootDir`). A rescue for a live directory git cannot name: it resolves both the
   *  project the bank is named after (gitProjectName) and the checkout whose `mapPathToBank`
   *  entry this directory inherits (lookupDirectories). */
  sessionRoot?: string
): string {
  const pathMap = config.mapPathToBank;
  const mapped =
    directory && pathMap
      ? lookupDirectories(config, directory, sessionRoot)
          .map((candidate) => mapLookup(pathMap, candidate))
          .find((bank) => bank !== undefined)
      : undefined;
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

/**
 * `deriveBankId`, or null when the repository could not be identified — with a `warn` and a diag
 * event so the miss is diagnosable in minutes rather than by cross-referencing bank creation
 * timestamps against a log file.
 *
 * The lifecycle hooks (SessionStart, prompt, retain) use this: skipping a session loses one
 * session's memory and heals on the next hook, while guessing a bank id scatters it permanently.
 */
export function deriveBankIdOrSkip(
  config: BankConfig,
  directory: string,
  harness = "coding",
  sessionRoot?: string
): string | null {
  try {
    return deriveBankId(config, directory, harness, sessionRoot);
  } catch (error) {
    if (!(error instanceof BankResolutionError)) throw error;
    log.warn(harness, "bank unresolved: skipping (repository could not be identified)", {
      directory,
      error: error.message,
    });
    diag(harness, "bank_unresolved", { directory, error: error.message });
    return null;
  }
}
