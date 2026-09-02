/**
 * ONE config file: ~/.hindsight/coding-agent.json.
 *
 * Layering, later wins per field:
 *   1. built-in defaults
 *   2. environment variables (ENV_KEYS below) — a FALLBACK for containers, CI and secret managers
 *   3. the config file's top level
 *   4. its `harnesses.<name>` section for the asking harness
 *
 * There is deliberately NO project-local config: a second, repo-carried file was both a security
 * surface (untrusted repos influencing memory behavior) and a second place to look. Per-repo bank
 * routing is `mapPathToBank`; per-agent differences are `harnesses.<name>`.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { DEFAULT_SEED_LIMIT } from "./seed";
import { isOptedIn } from "./bank";
import { log } from "./log";
import { DEFAULT_OBSERVATION_SCOPES, type ObservationScopes } from "./hindsight";

/** Default config-file path: ~/.hindsight/coding-agent.json */
export // HINDSIGHT_CONFIG joins the two env exceptions (diag/log files): it points at THE config file,
// for containers and test harnesses where $HOME isn't the right anchor. Still one file.
const CONFIG_PATH =
  process.env.HINDSIGHT_CONFIG || join(homedir(), ".hindsight", "coding-agent.json");

/** Daemon-mode defaults. The port matches the old per-agent Claude Code plugin so a machine that
 *  already ran that daemon keeps using the same one. */
export const DEFAULT_DAEMON_PORT = 9077;
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
  /** Daemon mode: seconds of inactivity before the daemon exits. Unset means it never exits on
   *  its own — `hindsight-embed`'s own default, which every other Hindsight integration also
   *  ships. This used to default to 300 here, which made the plugin the only thing on the machine
   *  opting a shared daemon into an auto-exit nobody asked for. There is deliberately no
   *  stop-on-session-end either: one daemon is shared by every agent and repo on the machine, so
   *  ending one session must not cut memory out from under another. */
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
  /** Run memory ONLY in projects that were opted in (default false: every project gets memory,
   *  which is what makes the plugin zero-setup). With it on, an unlisted project is inert — no
   *  bank is created, nothing is retained, no seed runs — so unrelated work leaves no trace. */
  optInOnly?: boolean;
  /** Directories opted in, matched as PREFIXES with `~` expanded: approving a directory approves
   *  the repos beneath it, and each still gets its own dynamic bank. A `mapPathToBank` entry counts
   *  as opted in too, since routing a path to a named bank already declares that project. */
  optInPaths?: string[];
  harness?: string; // runtime adapter (default "opencode")
  disabled?: boolean; // hard off-switch — inert plugin, for a no-memory baseline (default false)
  /** Session transcript write-back (default true; set false to opt out). Honored by every
   *  harness: the hook harnesses' Stop write-back and the persistent plugins' per-turn
   *  cadence alike. Gates ONLY the transcript — recall, git ingest, seeding and the memory
   *  tools keep working (that is `disabled`'s job). */
  retainSessions?: boolean;
  /** Cap on concurrent retain-related requests the client sends to the API (default 10):
   *  drain()'s per-operation polls and deepen's chat/git retain pools. A single request returning
   *  200 while bursts get 429s means the server is rate-limiting concurrency, not total volume —
   *  lower this rather than raising it. */
  maxParallelRetains?: number;
  reflectTimeoutMs?: number; // session-start reflect timeout (default 120000; hooks cap lower internally)
  /** Timeout for the agent-invoked `hindsight_reflect` tool (default 330000). Deliberately its own
   *  knob and much larger than `reflectTimeoutMs`: that one bounds an automatic hook that must fit
   *  the host's 25s window, whereas this one bounds a call the agent made on purpose and waits on,
   *  whose `budget: "high"` synthesis on a populated bank can run for minutes. The default sits
   *  ABOVE the server's own reflect wall timeout (HINDSIGHT_API_REFLECT_WALL_TIMEOUT, 300s) so the
   *  server decides when to give up, not an arbitrary client deadline (#3590). Unset, it inherits
   *  an explicitly-raised `reflectTimeoutMs` — a user who raised that meant "let reflect run". */
  reflectToolTimeoutMs?: number;
  /** Reflect budget for the `hindsight_reflect` tool: "low" | "mid" | "high" (default "high").
   *  Drop to "mid"/"low" on a large bank where high-budget synthesis exceeds the server's wall
   *  timeout. The automatic session-start reflect is NOT affected — it always uses "low" to fit
   *  its hook window. */
  reflectBudget?: "low" | "mid" | "high";
  /** Inject one reflect synthesis on the session's first prompt (default true). False suppresses
   *  automatic synthesis; the tool guide routes new goals through pages before optional reflection. */
  autoReflect?: boolean;
  pageRefreshEveryTurns?: number; // knowledge-page refresh cadence in user turns (default 10)
  /** What it COSTS to keep this project's knowledge pages current — the trigger stamped on every
   *  page this plugin creates (the seeded taxonomy and each captured initiative):
   *    "auto-refresh" (default) — refresh after every consolidation that produced new material
   *    "cron"                   — refresh on `pageTriggerCron` only, and only when actually stale
   *    "manual"                 — never refresh on its own; the tools and control plane still can
   *  Auto-refresh is both the most current and the most expensive: one LLM synthesis per page per
   *  consolidation, which adds up fast across auto-surveyed repos (#3506). Existing pages keep the
   *  trigger they were created with — this changes what NEW pages get. */
  pageTriggerType?: "auto-refresh" | "cron" | "manual";
  /** Schedule for `pageTriggerType: "cron"` — UTC, standard 5-field cron, e.g. "0 3 * * *". */
  pageTriggerCron?: string;
  autoSeed?: boolean; // SessionStart: auto-seed a cold repo's bank from git history (default true)
  seedLimit?: number; // SessionStart auto-seed: most-recent-N-commits cap (default 300)
  codebaseSurvey?: boolean; // SessionStart: spawn a headless claude to survey a cold repo's structure (default true)
  surveyModel?: string; // model passed to the headless survey's `claude -p --model` (default "haiku")
  surveyBudgetUsd?: number; // spend cap passed to the headless survey's `claude -p --max-budget-usd` (default 2)
  /** Plugin log verbosity ("debug" | "info" | "warn" | "error", default "info");
   *  HINDSIGHT_LOG_LEVEL overrides for ad-hoc debugging. */
  logLevel?: "debug" | "info" | "warn" | "error";
  /** Keep the installed runtime current by itself (default true). Once a day a session start asks
   *  npm for the published version and, when it is newer, re-stages ~/.hindsight/coding-agents in
   *  the background — the copy every wired agent's hooks already point at. It rewires no host
   *  config, so a release that adds a NEW hook entry point still needs a manual `install`.
   *  Set false to pin the installed version (air-gapped machines, or a deliberate downgrade);
   *  updating is then `npx @vectorize-io/hindsight-coding-agents install` again, as before. */
  autoUpdate?: boolean;
  surveyRefreshCommits?: number; // re-run the survey at SessionStart once this many commits have accrued since the last one, so structural pages track an evolving architecture (default 20; 0 = cold-seed only)
  /** How git history feeds memory — seeding AND keeping current use the same engine:
   *  "message" = commit messages only (cheap aggregated doc, re-upserted when HEAD moves);
   *  "full"    = messages + every recent commit's full diff (progressive batches, newest first);
   *  "none"    = git ingestion off entirely. Default "message" (cheap by default; opt into depth). */
  gitIngest?: "message" | "full" | "none";
  /** Extra tags stamped on every session write-back, e.g. ["project:{gitProject}", "env:work"].
   *  Placeholders: {gitProject} {project} {harness} {bankId} {sessionId} {timestamp} {channel}
   *  {user} (see core/retain-stamp.ts). Mainly for a SHARED bank, where the bank id no longer says
   *  which repo a memory came from. Built-in tags win on conflict. */
  retainTags?: string[];
  /** Extra metadata stamped on every session write-back, e.g. {"repo": "{gitProject}"}. Same
   *  placeholders as retainTags; built-in metadata (harness attribution) wins on conflict. */
  retainMetadata?: Record<string, string>;
  /** Let the plugin shape the bank's own configuration — the retain strategies it writes under,
   *  the `knowledge` entity-label group, and (on a bank that has none) the missions (default true).
   *
   *  Writing is strictly ADDITIVE: the plugin adds what the bank does not already define and never
   *  overwrites an existing value, so an edit made in the control plane survives (#3927). Set false
   *  to keep it out of the bank's configuration entirely — for a bank you shape yourself, or share
   *  with non-coding work. That bank should then define the strategies this plugin retains under
   *  (`git`, `gitlog`, `conversation`, `document`, `survey`): the server does not reject a retain
   *  naming a strategy the bank lacks, it logs a warning and extracts with the bank's own config
   *  instead — so a diff, a transcript and a survey marker all get the same generic treatment.
   *  Knowledge pages are seeded either way (see `pageTriggerType`). */
  manageBankConfig?: boolean;
  /** How consolidation groups the observations this plugin's memories feed (default "shared" — one
   *  global scope per bank, so every agent working a repo builds ONE set of beliefs; see
   *  DEFAULT_OBSERVATION_SCOPES). "combined" restores the server default of one scope per distinct
   *  tag set, "per_tag" one per tag, "all_combinations" one per subset; a string[][] declares the
   *  scopes literally. Anything else falls back to the default. */
  observationScopes?: ObservationScopes;
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
  /** Undefined = never told the daemon to auto-exit; see RawConfig above. */
  daemonIdleTimeout?: number;
  daemonProfile: string;
  embedVersion?: string;
  embedPackagePath?: string;
  bankId?: string; // resolved per-directory via deriveBankIdOrSkip — see core/bank.ts
  dynamicBankId?: boolean;
  bankIdTemplate?: string;
  mapPathToBank?: Record<string, string>;
  resolveWorktrees?: boolean;
  optInOnly: boolean;
  optInPaths: string[];
  harness: string;
  disabled: boolean;
  retainSessions: boolean;
  maxParallelRetains: number;
  reflectTimeoutMs: number;
  reflectToolTimeoutMs: number;
  reflectBudget: "low" | "mid" | "high";
  autoReflect: boolean;
  pageRefreshEveryTurns: number;
  pageTriggerType: "auto-refresh" | "cron" | "manual";
  pageTriggerCron?: string;
  autoSeed: boolean;
  seedLimit: number;
  codebaseSurvey: boolean;
  surveyModel: string;
  surveyBudgetUsd: number;
  surveyRefreshCommits: number;
  gitIngest: "message" | "full" | "none";
  retainTags: string[];
  retainMetadata: Record<string, string>;
  manageBankConfig: boolean;
  observationScopes: ObservationScopes;
  banks: Record<string, Omit<RawConfig, "banks" | "harnesses"> & { bank?: string }>;
  logLevel: "debug" | "info" | "warn" | "error";
  autoUpdate: boolean;
}

/**
 * Which page-refresh trigger a raw config asks for.
 *
 * `"cron"` without a `pageTriggerCron` is a broken config, not a request to stop refreshing: the
 * API rejects a cron trigger with no expression, which would fail page creation outright. Fall
 * back to the default and say so — a user who wants pages to stop refreshing writes "manual".
 */
function resolvePageTriggerType(raw: RawConfig): "auto-refresh" | "cron" | "manual" {
  if (raw.pageTriggerType === "manual") return "manual";
  if (raw.pageTriggerType === "cron") {
    if (raw.pageTriggerCron?.trim()) return "cron";
    log.warn(
      "config",
      'pageTriggerType "cron" needs pageTriggerCron (UTC 5-field, e.g. "0 3 * * *") — ' +
        'falling back to "auto-refresh"'
    );
  }
  return "auto-refresh";
}

/** Default timeout for the agent-invoked `hindsight_reflect` tool — see RawConfig.reflectToolTimeoutMs. */
export const DEFAULT_REFLECT_TOOL_TIMEOUT_MS = 330_000;

const REFLECT_BUDGETS = ["low", "mid", "high"] as const;

/**
 * Which reflect budget the `hindsight_reflect` tool should ask for.
 *
 * An unrecognized value takes the default rather than travelling: the API rejects an unknown budget
 * outright, which would turn a typo here into a hard tool failure on every call.
 */
function resolveReflectBudget(raw: RawConfig): "low" | "mid" | "high" {
  const value: unknown = raw.reflectBudget;
  if (value === undefined) return "high";
  if (typeof value === "string" && (REFLECT_BUDGETS as readonly string[]).includes(value))
    return value as "low" | "mid" | "high";
  log.warn("config", `ignoring reflectBudget=${JSON.stringify(value)} — expected low|mid|high`);
  return "high";
}

/** The server's scalar scoping modes; anything else in this field has to be an explicit scope list. */
// `per_source` is this plugin's own mode, resolved per document in `resolveRetainScopes` and never
// sent to the server; the rest are the server's.
const OBSERVATION_SCOPE_MODES = [
  "shared",
  "combined",
  "per_tag",
  "all_combinations",
  "per_source",
] as const;

/**
 * Validate `observationScopes`, falling back to the default on anything unrecognized.
 *
 * A typo here would otherwise reach the API as an unknown scoping mode and change how a whole
 * bank's observations are grouped, so an unusable value takes the default rather than travelling.
 * An empty list is unusable too, and specifically so: `[]` declares ZERO scopes, which the server
 * reads as "no spec" and silently treats as `combined` — the opposite of what writing this field
 * at all was meant to express.
 */
function resolveObservationScopes(raw: RawConfig["observationScopes"]): ObservationScopes {
  // Widened deliberately: this arrives from a hand-edited JSON file, so the declared type says what
  // is meant, not what is there.
  const value: unknown = raw;
  if (typeof value === "string")
    return (OBSERVATION_SCOPE_MODES as readonly string[]).includes(value)
      ? (value as ObservationScopes)
      : DEFAULT_OBSERVATION_SCOPES;
  if (Array.isArray(value)) {
    const scopes = (value as unknown[])
      .filter((scope): scope is unknown[] => Array.isArray(scope))
      .map((scope) => scope.filter((t): t is string => typeof t === "string" && t.trim() !== ""));
    if (scopes.length) return scopes;
  }
  return DEFAULT_OBSERVATION_SCOPES;
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
    daemonIdleTimeout: raw.daemonIdleTimeout,
    daemonProfile: raw.daemonProfile || DEFAULT_DAEMON_PROFILE,
    embedVersion: raw.embedVersion || undefined,
    embedPackagePath: raw.embedPackagePath || undefined,
    bankId: raw.bankId,
    dynamicBankId: raw.dynamicBankId,
    bankIdTemplate: raw.bankIdTemplate,
    mapPathToBank: raw.mapPathToBank,
    resolveWorktrees: raw.resolveWorktrees,
    optInOnly: raw.optInOnly ?? false,
    // Same shape as retainTags: a config typo must not become a path that silently approves nothing.
    optInPaths: Array.isArray(raw.optInPaths)
      ? raw.optInPaths.filter((p): p is string => typeof p === "string" && p.trim() !== "")
      : [],
    harness: raw.harness ?? "opencode",
    disabled: raw.disabled ?? false,
    retainSessions: raw.retainSessions ?? true, // write sessions back by default, every harness
    manageBankConfig: raw.manageBankConfig ?? true,
    maxParallelRetains: raw.maxParallelRetains || 10,
    reflectTimeoutMs: raw.reflectTimeoutMs || 120000,
    // Inherit an explicitly-raised reflectTimeoutMs (that is what users reaching for a longer
    // reflect already set), but never let it LOWER the tool below the default — a short window is
    // set to bound the automatic hook, not to cut off a call the agent is waiting on.
    reflectToolTimeoutMs:
      raw.reflectToolTimeoutMs ||
      Math.max(raw.reflectTimeoutMs || 0, DEFAULT_REFLECT_TOOL_TIMEOUT_MS),
    reflectBudget: resolveReflectBudget(raw),
    autoReflect: raw.autoReflect ?? true,
    pageRefreshEveryTurns: raw.pageRefreshEveryTurns || 10,
    pageTriggerType: resolvePageTriggerType(raw),
    pageTriggerCron: raw.pageTriggerCron?.trim() || undefined,
    autoSeed: raw.autoSeed ?? true,
    seedLimit: raw.seedLimit || DEFAULT_SEED_LIMIT,
    codebaseSurvey: raw.codebaseSurvey ?? true,
    surveyModel: raw.surveyModel || "haiku",
    surveyBudgetUsd: raw.surveyBudgetUsd || 2,
    surveyRefreshCommits: raw.surveyRefreshCommits ?? 20,
    gitIngest: ["message", "full", "none"].includes(raw.gitIngest as string)
      ? (raw.gitIngest as "message" | "full" | "none")
      : "message",
    // Hostile input is a config typo, not an attack: keep only string entries so a stray number or
    // nested object cannot reach the API as a tag and fail the whole retain.
    retainTags: Array.isArray(raw.retainTags)
      ? raw.retainTags.filter((t): t is string => typeof t === "string" && t.trim() !== "")
      : [],
    retainMetadata:
      raw.retainMetadata && typeof raw.retainMetadata === "object"
        ? Object.fromEntries(
            Object.entries(raw.retainMetadata).filter(([, v]) => typeof v === "string")
          )
        : {},
    observationScopes: resolveObservationScopes(raw.observationScopes),
    banks: raw.banks && typeof raw.banks === "object" ? raw.banks : {},
    logLevel: ["debug", "info", "warn", "error"].includes(raw.logLevel as string)
      ? (raw.logLevel as "debug" | "info" | "warn" | "error")
      : "info",
    autoUpdate: raw.autoUpdate ?? true,
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
 * The map-valued settings (mapPathToBank, harnesses, banks, retainMetadata) are deliberately
 * absent: they are structures whose whole point is per-repo/per-harness/per-key branching, which
 * does not survive flattening into one env var. They stay file-only.
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
  optInOnly: "HINDSIGHT_OPT_IN_ONLY",
  optInPaths: "HINDSIGHT_OPT_IN_PATHS",
  harness: "HINDSIGHT_HARNESS",
  disabled: "HINDSIGHT_DISABLED",
  retainSessions: "HINDSIGHT_RETAIN_SESSIONS",
  maxParallelRetains: "HINDSIGHT_MAX_PARALLEL_RETAINS",
  reflectTimeoutMs: "HINDSIGHT_REFLECT_TIMEOUT_MS",
  reflectToolTimeoutMs: "HINDSIGHT_REFLECT_TOOL_TIMEOUT_MS",
  reflectBudget: "HINDSIGHT_REFLECT_BUDGET",
  autoReflect: "HINDSIGHT_AUTO_REFLECT",
  pageRefreshEveryTurns: "HINDSIGHT_PAGE_REFRESH_EVERY_TURNS",
  pageTriggerType: "HINDSIGHT_PAGE_TRIGGER_TYPE",
  pageTriggerCron: "HINDSIGHT_PAGE_TRIGGER_CRON",
  autoSeed: "HINDSIGHT_AUTO_SEED",
  seedLimit: "HINDSIGHT_SEED_LIMIT",
  codebaseSurvey: "HINDSIGHT_CODEBASE_SURVEY",
  surveyModel: "HINDSIGHT_SURVEY_MODEL",
  surveyBudgetUsd: "HINDSIGHT_SURVEY_BUDGET_USD",
  surveyRefreshCommits: "HINDSIGHT_SURVEY_REFRESH_COMMITS",
  logLevel: "HINDSIGHT_LOG_LEVEL",
  autoUpdate: "HINDSIGHT_AUTO_UPDATE",
  gitIngest: "HINDSIGHT_GIT_INGEST",
  // Scalar modes only ("shared", "combined", "per_tag", "all_combinations"). An explicit scope
  // list is a list OF lists, which does not survive flattening into one variable — file-only.
  observationScopes: "HINDSIGHT_OBSERVATION_SCOPES",
  // Comma-separated, e.g. HINDSIGHT_RETAIN_TAGS="project:{gitProject},env:work". A LIST rather than
  // a map, so it flattens cleanly; its sibling retainMetadata stays file-only for the reason above.
  retainTags: "HINDSIGHT_RETAIN_TAGS",
  manageBankConfig: "HINDSIGHT_MANAGE_BANK_CONFIG",
} as const satisfies Partial<Record<keyof RawConfig, string>>;

/** Fields parsed as booleans/numbers/comma-separated lists; everything else is taken as a string. */
const ENV_BOOLEANS = new Set<keyof RawConfig>([
  "dynamicBankId",
  "resolveWorktrees",
  "optInOnly",
  "disabled",
  "retainSessions",
  "autoReflect",
  "autoSeed",
  "codebaseSurvey",
  "autoUpdate",
  "manageBankConfig",
]);
const ENV_LISTS = new Set<keyof RawConfig>(["retainTags", "optInPaths"]);
const ENV_NUMBERS = new Set<keyof RawConfig>([
  "apiPort",
  "daemonIdleTimeout",
  "maxParallelRetains",
  "reflectTimeoutMs",
  "reflectToolTimeoutMs",
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
    } else if (ENV_LISTS.has(key)) {
      // Empty entries dropped, so a trailing comma or "a,,b" is a typo rather than an empty tag.
      const items = value
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean);
      if (items.length) out[key] = items;
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
  const raw = applyLayer(withEnv, readRaw(o.path ?? CONFIG_PATH), o.harness);
  // The harness that ASKED is the correct fallback for an unset `harness` field — not a hardcoded
  // "opencode", whose silent default misfiled every other harness's background seed into an
  // `opencode::<project>` bank (#3247). An explicit `harness:` in the config file still wins.
  if (!raw.harness && o.harness) raw.harness = o.harness;
  return resolveConfig(raw);
}

/** Bank-resolution fields are meaningless inside a `banks.<id>` section (they can't change the id
 *  that selected it) — strip them so a typo there can't silently re-route memory. */
const BANK_OVERRIDE_EXCLUDED = [
  "bankId",
  "bankIdTemplate",
  "mapPathToBank",
  "dynamicBankId",
  "resolveWorktrees",
  // Approval is decided BEFORE the bank is resolved, so a `banks.<id>` section naming these could
  // only ever arrive too late to matter — strip them rather than let them read as effective.
  "optInOnly",
  "optInPaths",
  "harness",
] as const;

/**
 * Apply the `banks.<bankId>` override section (if any) to an already-resolved config — the step
 * that runs AFTER bank resolution, giving per-repo opt-in/out (disable, retainSessions, gitIngest,
 * survey settings, ...) from the ONE global config file.
 */
export function applyBankConfig(
  cfg: Config,
  resolvedId: string,
  /** The directory the bank was resolved FROM. Supplying it enforces `optInOnly`; every entry
   *  point already has it to hand, having just passed it to bank resolution. */
  directory?: string
): { cfg: Config; bankId: string } {
  // Rides the `disabled` gate every entry point already checks after bank resolution, rather than
  // adding a second thing eight call sites must remember — and that gate is known to stop the run
  // before anything creates a bank.
  if (directory !== undefined && !isOptedIn(cfg, directory))
    return { cfg: { ...cfg, disabled: true }, bankId: resolvedId };
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
