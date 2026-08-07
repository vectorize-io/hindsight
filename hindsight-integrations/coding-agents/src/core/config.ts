/**
 * ONE config file: ~/.hindsight/coding-agent.json (JSON, no environment variables — the sole
 * exception is HINDSIGHT_DIAG_FILE for the diagnostics path).
 *
 * Layering, later wins per field:
 *   1. built-in defaults
 *   2. the config file's top level
 *   3. its `harnesses.<name>` section for the asking harness
 *
 * There is deliberately NO project-local config: a second, repo-carried file was both a security
 * surface (untrusted repos influencing memory behavior) and a second place to look. Per-repo bank
 * routing is `mapPathToBank`; per-agent differences are `harnesses.<name>`.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { DEFAULT_SEED_LIMIT } from "./seed";

/** Default config-file path: ~/.hindsight/coding-agent.json */
export // HINDSIGHT_CONFIG joins the two env exceptions (diag/log files): it points at THE config file,
// for containers and test harnesses where $HOME isn't the right anchor. Still one file.
const CONFIG_PATH =
  process.env.HINDSIGHT_CONFIG || join(homedir(), ".hindsight", "coding-agent.json");

/** Daemon-mode defaults. The port matches the old per-agent Claude Code plugin so a machine that
 *  already ran that daemon keeps using the same one. */
export const DEFAULT_DAEMON_PORT = 9077;
export const DEFAULT_DAEMON_IDLE_TIMEOUT = 300;
export const DEFAULT_DAEMON_PROFILE = "coding-agent";

/** Incremental git-sync settings (see core/sync.ts). */
/** The config file's shape — every field optional; omitted fields take the documented default. */
export interface RawConfig {
  /** Where memory lives. All three modes speak the same HTTP API; they differ only in who runs it:
   *   "cloud"       — Hindsight Cloud (the default `apiUrl`)
   *   "self-hosted" — a Hindsight server you run; set `apiUrl` to it
   *   "daemon"      — a local `hindsight-embed` daemon this plugin starts on demand, at
   *                   127.0.0.1:{apiPort}. `apiUrl` is ignored in this mode.
   * Omitted, it is inferred: "daemon" is never assumed, so an existing config keeps talking to
   * whatever `apiUrl` it already names. */
  serverMode?: "cloud" | "self-hosted" | "daemon";
  apiUrl?: string; // Hindsight API base URL (default https://api.hindsight.vectorize.io — Cloud; set to http://localhost:8888 for a local server)
  apiToken?: string; // bearer token (optional)
  /** Daemon mode: port the local daemon listens on (default 9077).
   *  NOT 8888 — that is the conventional port for a server you run yourself, and a daemon must
   *  never squat on it. A healthy server already on this port is adopted rather than restarted. */
  apiPort?: number;
  /** Daemon mode: seconds of inactivity before the daemon exits (default 300).
   *  This is how the daemon shuts down. There is deliberately no stop-on-session-end: one daemon
   *  is shared by every agent and repo on the machine, so ending one session must not cut memory
   *  out from under another. */
  daemonIdleTimeout?: number;
  /** Daemon mode: `hindsight-embed` profile name, i.e. which local database is used (default
   *  "coding-agent"). Separate profiles keep unrelated setups from sharing memory. */
  daemonProfile?: string;
  /** Daemon mode: which `hindsight-embed` release to run via uvx (default "latest"). */
  embedVersion?: string;
  /** Daemon mode: local `hindsight-embed` checkout to run instead of a published release (dev). */
  embedPackagePath?: string;
  bankId?: string; // EXPLICIT memory bank id — set = static bank; unset = per-repo dynamic (core/bank.ts)
  dynamicBankId?: boolean; // force dynamic resolution even when bankId is set (default: dynamic iff no bankId)
  bankIdTemplate?: string; // dynamic bank id format (default "coding-agent::{gitProject}" — harness-neutral) —
  //   placeholders: {gitProject} {project} {harness} {channel} {user} (see core/bank.ts)
  mapPathToBank?: Record<string, string>; // absolute path -> bank; longest prefix wins; overrides everything
  resolveWorktrees?: boolean; // {gitProject}: worktrees share the main repo's bank (default true)
  harness?: string; // runtime adapter (default "opencode")
  disabled?: boolean; // hard off-switch — inert plugin, for a no-memory baseline (default false)
  retainSessions?: boolean; // opencode plugin write-back (default true; set false to opt out). Hook harnesses always write back on Stop and ignore this flag.
  retainEveryTurns?: number; // write-back cadence in user turns (default 1: async upsert every turn)
  reflectTimeoutMs?: number; // session-start reflect timeout (default 120000; hooks cap lower internally)
  autoReflect?: boolean; // inject a one-time reflect synthesis on the session's first prompt (default true; false = the agent reflects only via the hindsight_reflect tool, and the tool guide tells it to do so on new goals)
  pageRefreshEveryTurns?: number; // knowledge-page refresh cadence in user turns (default 10)
  autoSeed?: boolean; // SessionStart: auto-seed a cold repo's bank from git history (default true)
  seedLimit?: number; // SessionStart auto-seed: most-recent-N-commits cap (default 300)
  codebaseSurvey?: boolean; // SessionStart: spawn a headless claude to survey a cold repo's structure (default true)
  surveyModel?: string; // model passed to the headless survey's `claude -p --model` (default "haiku")
  surveyBudgetUsd?: number; // spend cap passed to the headless survey's `claude -p --max-budget-usd` (default 2)
  /** Plugin log verbosity ("debug" | "info" | "warn" | "error", default "info");
   *  HINDSIGHT_LOG_LEVEL overrides for ad-hoc debugging. */
  logLevel?: "debug" | "info" | "warn" | "error";
  surveyRefreshCommits?: number; // re-run the survey at SessionStart once this many commits have accrued since the last one, so structural pages track an evolving architecture (default 20; 0 = cold-seed only)
  /** How git history feeds memory — seeding AND keeping current use the same engine:
   *  "message" = commit messages only (cheap aggregated doc, re-upserted when HEAD moves);
   *  "full"    = messages + every recent commit's full diff (progressive batches, newest first);
   *  "none"    = git ingestion off entirely. Default "message" (cheap by default; opt into depth). */
  gitIngest?: "message" | "full" | "none";
  /** Per-harness overrides of any of the fields above, keyed by harness name ("opencode",
   *  "claude-code", ...). Lets one config file give each agent its own bank/settings. */
  harnesses?: Record<string, Omit<RawConfig, "harnesses">>;
  /** Per-BANK overrides, applied AFTER bank resolution — the per-repo opt-in/out surface, keyed
   *  by the RESOLVED bank id (run a session once or check the banner to see it). Any behavioral
   *  field, plus `bank` to RENAME the destination (single hop: the section is selected by the
   *  resolved id, the rename is literal — several ids may converge on one shared bank).
   *  Bank-resolution fields (bankId/template/map/...) are ignored here. Example:
   *    "banks": { "coding-agent::secret-client": { "disabled": true },
   *               "coding-agent::old-name": { "bank": "team::shared" },
   *               "coding-agent::big-mono": { "gitIngest": "full", "retainSessions": false } } */
  banks?: Record<string, Omit<RawConfig, "banks" | "harnesses"> & { bank?: string }>;
}

/** Fully-resolved config: every field present. */
export interface Config {
  serverMode: "cloud" | "self-hosted" | "daemon";
  /** The EFFECTIVE base URL. In daemon mode this is already 127.0.0.1:{apiPort}, so every caller
   *  that builds a client keeps working without knowing which mode is active. */
  apiUrl: string;
  apiToken?: string;
  apiPort: number;
  daemonIdleTimeout: number;
  daemonProfile: string;
  embedVersion?: string;
  embedPackagePath?: string;
  bankId?: string; // resolved per-directory via deriveBankId(cfg, dir) — see core/bank.ts
  dynamicBankId?: boolean;
  bankIdTemplate?: string;
  mapPathToBank?: Record<string, string>;
  resolveWorktrees?: boolean;
  harness: string;
  disabled: boolean;
  retainSessions: boolean;
  retainEveryTurns: number;
  reflectTimeoutMs: number;
  autoReflect: boolean;
  pageRefreshEveryTurns: number;
  autoSeed: boolean;
  seedLimit: number;
  codebaseSurvey: boolean;
  surveyModel: string;
  surveyBudgetUsd: number;
  surveyRefreshCommits: number;
  gitIngest: "message" | "full" | "none";
  banks: Record<string, Omit<RawConfig, "banks" | "harnesses"> & { bank?: string }>;
  logLevel: "debug" | "info" | "warn" | "error";
}

/** Apply defaults to a raw (file) config. Pure — the single place the defaults live. */
export function resolveConfig(raw: RawConfig = {}): Config {
  const serverMode = ["cloud", "self-hosted", "daemon"].includes(raw.serverMode as string)
    ? (raw.serverMode as "cloud" | "self-hosted" | "daemon")
    : "cloud";
  const apiPort = raw.apiPort || DEFAULT_DAEMON_PORT;
  return {
    serverMode,
    // Daemon mode resolves the URL HERE rather than at each call site: every entry point already
    // builds its client from cfg.apiUrl, so collapsing the mode into that one field means the
    // daemon needs no plumbing through eight separate constructors.
    apiUrl:
      serverMode === "daemon"
        ? `http://127.0.0.1:${apiPort}`
        : (raw.apiUrl ?? "https://api.hindsight.vectorize.io"),
    apiToken: raw.apiToken || undefined,
    apiPort,
    daemonIdleTimeout: raw.daemonIdleTimeout ?? DEFAULT_DAEMON_IDLE_TIMEOUT,
    daemonProfile: raw.daemonProfile || DEFAULT_DAEMON_PROFILE,
    embedVersion: raw.embedVersion || undefined,
    embedPackagePath: raw.embedPackagePath || undefined,
    bankId: raw.bankId,
    dynamicBankId: raw.dynamicBankId,
    bankIdTemplate: raw.bankIdTemplate,
    mapPathToBank: raw.mapPathToBank,
    resolveWorktrees: raw.resolveWorktrees,
    harness: raw.harness ?? "opencode",
    disabled: raw.disabled ?? false,
    retainSessions: raw.retainSessions ?? true, // opencode: write back by default (parity with hook-harness Stop)
    retainEveryTurns: raw.retainEveryTurns || 1,
    reflectTimeoutMs: raw.reflectTimeoutMs || 120000,
    autoReflect: raw.autoReflect ?? true,
    pageRefreshEveryTurns: raw.pageRefreshEveryTurns || 10,
    autoSeed: raw.autoSeed ?? true,
    seedLimit: raw.seedLimit || DEFAULT_SEED_LIMIT,
    codebaseSurvey: raw.codebaseSurvey ?? true,
    surveyModel: raw.surveyModel || "haiku",
    surveyBudgetUsd: raw.surveyBudgetUsd || 2,
    surveyRefreshCommits: raw.surveyRefreshCommits ?? 20,
    gitIngest: ["message", "full", "none"].includes(raw.gitIngest as string)
      ? (raw.gitIngest as "message" | "full" | "none")
      : "message",
    banks: raw.banks && typeof raw.banks === "object" ? raw.banks : {},
    logLevel: ["debug", "info", "warn", "error"].includes(raw.logLevel as string)
      ? (raw.logLevel as "debug" | "info" | "warn" | "error")
      : "info",
  };
}

function readRaw(path: string): RawConfig {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as RawConfig;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      console.error(`hindsight: ignoring invalid config at ${path}: ${(e as Error)?.message || e}`);
    }
    return {};
  }
}

/** Shallow-merge b over a; `harnesses` never survives into a layer; `banks` merges by bank id. */
function mergeRaw(a: RawConfig, b: RawConfig): RawConfig {
  const { harnesses: _drop, ...flat } = b;
  return { ...a, ...flat, banks: { ...(a.banks ?? {}), ...(b.banks ?? {}) } };
}

/**
 * Keys a PROJECT-LOCAL (untrusted, repo-supplied) config may NOT set. These control where the user's
 * credential + prompts/transcripts are sent (`apiUrl`, `apiToken`) or remap banks globally
 * (`mapPathToBank`, which "overrides everything"). A repo can still set its own per-repo bank via
 * `bankId`/`bankIdTemplate` — just not the network endpoint, token, or global path→bank map. The
 * user-global config is trusted and unrestricted.
 */
export interface LoadOptions {
  /** Which harness is asking ("opencode", "claude-code", ...) — applies its `harnesses.<name>` overrides. */
  harness?: string;
  /** Explicit global-config path (default ~/.hindsight/coding-agent.json). */
  path?: string;
}

/** Apply one config layer (its top level, then its `harnesses.<name>` override) over `raw`. */
function applyLayer(raw: RawConfig, layer: RawConfig, harness?: string): RawConfig {
  let out = mergeRaw(raw, layer);
  const perHarness = harness ? layer.harnesses?.[harness] : undefined;
  if (perHarness) out = mergeRaw(out, perHarness);
  return out;
}

/**
 * Env var backing each scalar setting: `HINDSIGHT_` + the field in SCREAMING_SNAKE.
 *
 * These are a FALLBACK, not an override — the config file still wins wherever it sets a value, so
 * an existing file behaves exactly as before. They exist for the places a file is awkward:
 * containers, CI, and secret managers that inject `HINDSIGHT_API_TOKEN` rather than writing a
 * credential to disk.
 *
 * The map-valued settings (mapPathToBank, harnesses, banks) are deliberately absent: they are
 * nested structures whose whole point is per-repo/per-harness branching, which does not survive
 * flattening into one env var. They stay file-only.
 */
const ENV_KEYS = {
  serverMode: "HINDSIGHT_SERVER_MODE",
  apiUrl: "HINDSIGHT_API_URL",
  apiToken: "HINDSIGHT_API_TOKEN",
  // These four keep the names the old per-agent Claude Code plugin used, so a user migrating from
  // it can carry their existing environment over unchanged.
  apiPort: "HINDSIGHT_API_PORT",
  daemonIdleTimeout: "HINDSIGHT_DAEMON_IDLE_TIMEOUT",
  embedVersion: "HINDSIGHT_EMBED_VERSION",
  embedPackagePath: "HINDSIGHT_EMBED_PACKAGE_PATH",
  daemonProfile: "HINDSIGHT_DAEMON_PROFILE",
  bankId: "HINDSIGHT_BANK_ID",
  dynamicBankId: "HINDSIGHT_DYNAMIC_BANK_ID",
  bankIdTemplate: "HINDSIGHT_BANK_ID_TEMPLATE",
  resolveWorktrees: "HINDSIGHT_RESOLVE_WORKTREES",
  harness: "HINDSIGHT_HARNESS",
  disabled: "HINDSIGHT_DISABLED",
  retainSessions: "HINDSIGHT_RETAIN_SESSIONS",
  retainEveryTurns: "HINDSIGHT_RETAIN_EVERY_TURNS",
  reflectTimeoutMs: "HINDSIGHT_REFLECT_TIMEOUT_MS",
  autoReflect: "HINDSIGHT_AUTO_REFLECT",
  pageRefreshEveryTurns: "HINDSIGHT_PAGE_REFRESH_EVERY_TURNS",
  autoSeed: "HINDSIGHT_AUTO_SEED",
  seedLimit: "HINDSIGHT_SEED_LIMIT",
  codebaseSurvey: "HINDSIGHT_CODEBASE_SURVEY",
  surveyModel: "HINDSIGHT_SURVEY_MODEL",
  surveyBudgetUsd: "HINDSIGHT_SURVEY_BUDGET_USD",
  surveyRefreshCommits: "HINDSIGHT_SURVEY_REFRESH_COMMITS",
  logLevel: "HINDSIGHT_LOG_LEVEL",
  gitIngest: "HINDSIGHT_GIT_INGEST",
} as const satisfies Partial<Record<keyof RawConfig, string>>;

/** Fields parsed as booleans/numbers; everything else is taken as a string. */
const ENV_BOOLEANS = new Set<keyof RawConfig>([
  "dynamicBankId",
  "resolveWorktrees",
  "disabled",
  "retainSessions",
  "autoReflect",
  "autoSeed",
  "codebaseSurvey",
]);
const ENV_NUMBERS = new Set<keyof RawConfig>([
  "apiPort",
  "daemonIdleTimeout",
  "retainEveryTurns",
  "reflectTimeoutMs",
  "pageRefreshEveryTurns",
  "seedLimit",
  "surveyBudgetUsd",
  "surveyRefreshCommits",
]);

/**
 * Read the env fallback layer. An unset or empty var contributes nothing (so it can't mask a file
 * value), and a malformed number is ignored with a warning rather than silently becoming NaN —
 * which would otherwise disable a timeout or cadence in a way that is very hard to trace.
 */
export function readEnvConfig(env: NodeJS.ProcessEnv = process.env): RawConfig {
  const out: Record<string, unknown> = {};
  for (const [key, name] of Object.entries(ENV_KEYS) as [keyof RawConfig, string][]) {
    const value = env[name];
    if (value === undefined || value.trim() === "") continue;
    if (ENV_BOOLEANS.has(key)) {
      out[key] = ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
    } else if (ENV_NUMBERS.has(key)) {
      const n = Number(value);
      if (Number.isFinite(n)) out[key] = n;
      else console.error(`hindsight: ignoring ${name}=${value} — not a number`);
    } else {
      out[key] = value;
    }
  }
  return out as RawConfig;
}

/** Load + resolve config from THE config file, with env as a fallback. Missing file -> defaults. */
export function loadConfig(opts: LoadOptions | string = {}): Config {
  const o: LoadOptions = typeof opts === "string" ? { path: opts } : opts; // legacy: loadConfig(path)
  // Env first so the FILE wins on any field it sets.
  const withEnv = applyLayer({}, readEnvConfig(), o.harness);
  return resolveConfig(applyLayer(withEnv, readRaw(o.path ?? CONFIG_PATH), o.harness));
}

/** Bank-resolution fields are meaningless inside a `banks.<id>` section (they can't change the id
 *  that selected it) — strip them so a typo there can't silently re-route memory. */
const BANK_OVERRIDE_EXCLUDED = [
  "bankId",
  "bankIdTemplate",
  "mapPathToBank",
  "dynamicBankId",
  "resolveWorktrees",
  "harness",
] as const;

/**
 * Apply the `banks.<bankId>` override section (if any) to an already-resolved config — the step
 * that runs AFTER bank resolution, giving per-repo opt-in/out (disable, retainSessions, gitIngest,
 * survey settings, ...) from the ONE global config file.
 */
export function applyBankConfig(cfg: Config, resolvedId: string): { cfg: Config; bankId: string } {
  const section = cfg.banks[resolvedId];
  if (!section) return { cfg, bankId: resolvedId };
  const safe: Record<string, unknown> = { ...section };
  for (const k of BANK_OVERRIDE_EXCLUDED) delete safe[k];
  delete safe.banks;
  // `bank` renames the destination — single hop, selected by the ORIGINAL resolved id (the
  // target's own banks section, if any, is deliberately NOT consulted: no chaining).
  const bankId = typeof safe.bank === "string" && safe.bank ? (safe.bank as string) : resolvedId;
  delete safe.bank;
  return { cfg: { ...cfg, ...resolvePartial(cfg, safe as RawConfig) }, bankId };
}

/** Resolve just the fields present in `patch`, defaulting against the CURRENT cfg (not the global
 *  defaults) so an override only changes what it names. */
function resolvePartial(cfg: Config, patch: RawConfig): Partial<Config> {
  const full = resolveConfig(patch);
  const out: Partial<Config> = {};
  for (const key of Object.keys(patch) as (keyof RawConfig)[]) {
    if (key in full)
      (out as Record<string, unknown>)[key] = (full as unknown as Record<string, unknown>)[key];
  }
  return out;
}
