/**
 * Harness-agnostic configuration — JSON files, NO environment variables.
 *
 * All entry points read this: the runtime adapters (opencode plugin, claude hook) and the backfill
 * CLI (whose --flags override it). Files are optional: missing -> defaults; malformed -> one warning
 * + defaults (memory never breaks the agent).
 *
 * Layering (later wins, per field):
 *   1. built-in defaults
 *   2. ~/.hindsight/coding-agent.json            (user-global)
 *   3.   its `harnesses.<name>` section          (per-agent override, e.g. different bankId for claude)
 *   4. <project>/.hindsight/coding-agent.json    (project-local — natural place for a per-repo bank)
 *   5.   its `harnesses.<name>` section
 *
 * Each runtime entry point knows which harness it IS (the opencode plugin is loaded by opencode, the
 * claude hook by Claude Code) and passes its own name — so one shared config serves several agents
 * side by side. The legacy top-level `harness` key only selects the backfill's session formatter.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/** Default config-file path: ~/.hindsight/coding-agent.json */
export const CONFIG_PATH = join(homedir(), ".hindsight", "coding-agent.json");

/** Incremental git-sync settings (see core/sync.ts). */
export interface GitSyncConfig {
  enabled?: boolean; // keep the bank current with new commits on load (default false: opt-in)
  ref?: string; // target ref (default origin/main; falls back to HEAD)
  fetch?: boolean; // git fetch the ref before diffing (default false: no network side effect)
}

/** The config file's shape — every field optional; omitted fields take the documented default. */
export interface RawConfig {
  apiUrl?: string; // Hindsight API base URL (default http://localhost:8888)
  apiToken?: string; // bearer token (optional)
  bankId?: string; // EXPLICIT memory bank id — set = static bank; unset = per-repo dynamic (core/bank.ts)
  dynamicBankId?: boolean; // force dynamic resolution even when bankId is set (default: dynamic iff no bankId)
  bankIdTemplate?: string; // dynamic bank id format, e.g. "hindsight-{gitProject}" (default "{gitProject}") —
  //   placeholders: {gitProject} {project} {harness} {channel} {user} (see core/bank.ts)
  directoryBankMap?: Record<string, string>; // absolute path -> bank; longest prefix wins; overrides everything
  resolveWorktrees?: boolean; // {gitProject}: worktrees share the main repo's bank (default true)
  harness?: string; // runtime adapter (default "opencode")
  disabled?: boolean; // hard off-switch — inert plugin, for a no-memory baseline (default false)
  retainSessions?: boolean; // enable live write-back (default false)
  retainEveryTurns?: number; // write-back cadence in user turns (default 5)
  reflectTimeoutMs?: number; // reflect timeout (default 120000)
  gitSync?: GitSyncConfig;
  /** Per-harness overrides of any of the fields above, keyed by harness name ("opencode",
   *  "claude-code", ...). Lets one config file give each agent its own bank/settings. */
  harnesses?: Record<string, Omit<RawConfig, "harnesses">>;
}

/** Fully-resolved config: every field present, gitSync fully populated. */
export interface Config {
  apiUrl: string;
  apiToken?: string;
  bankId?: string; // resolved per-directory via deriveBankId(cfg, dir) — see core/bank.ts
  dynamicBankId?: boolean;
  bankIdTemplate?: string;
  directoryBankMap?: Record<string, string>;
  resolveWorktrees?: boolean;
  harness: string;
  disabled: boolean;
  retainSessions: boolean;
  retainEveryTurns: number;
  reflectTimeoutMs: number;
  gitSync: Required<GitSyncConfig>;
}

/** Apply defaults to a raw (file) config. Pure — the single place the defaults live. */
export function resolveConfig(raw: RawConfig = {}): Config {
  const gs = raw.gitSync ?? {};
  return {
    apiUrl: raw.apiUrl ?? "http://localhost:8888",
    apiToken: raw.apiToken || undefined,
    bankId: raw.bankId,
    dynamicBankId: raw.dynamicBankId,
    bankIdTemplate: raw.bankIdTemplate,
    directoryBankMap: raw.directoryBankMap,
    resolveWorktrees: raw.resolveWorktrees,
    harness: raw.harness ?? "opencode",
    disabled: raw.disabled ?? false,
    retainSessions: raw.retainSessions ?? false,
    retainEveryTurns: raw.retainEveryTurns || 5,
    reflectTimeoutMs: raw.reflectTimeoutMs || 120000,
    gitSync: {
      enabled: gs.enabled ?? false, // opt-in: off unless explicitly enabled in config
      ref: gs.ref ?? "origin/main",
      fetch: gs.fetch ?? false,
    },
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

/** Shallow-merge b over a; gitSync merges field-wise; `harnesses` never survives into a layer. */
function mergeRaw(a: RawConfig, b: RawConfig): RawConfig {
  const { harnesses: _drop, ...flat } = b;
  return { ...a, ...flat, gitSync: { ...(a.gitSync ?? {}), ...(b.gitSync ?? {}) } };
}

export interface LoadOptions {
  /** Which harness is asking ("opencode", "claude-code", ...) — applies its `harnesses.<name>` overrides. */
  harness?: string;
  /** Project directory — the NEAREST .hindsight/coding-agent.json at or above it layers over the global file. */
  projectDir?: string;
  /** Explicit global-config path (default ~/.hindsight/coding-agent.json). */
  path?: string;
}

/** Nearest project config at or above `dir`: walk up until a .hindsight/coding-agent.json exists. */
function findProjectConfig(dir: string): string | undefined {
  let d = dir;
  for (let i = 0; i < 64; i++) {
    const candidate = join(d, ".hindsight", "coding-agent.json");
    try {
      readFileSync(candidate);
      return candidate;
    } catch {
      /* keep walking */
    }
    const parent = dirname(d);
    if (parent === d) return undefined; // filesystem root
    d = parent;
  }
  return undefined;
}

/** Load + resolve config from the layered files. Missing files -> silent defaults. */
export function loadConfig(opts: LoadOptions | string = {}): Config {
  const o: LoadOptions = typeof opts === "string" ? { path: opts } : opts; // legacy: loadConfig(path)
  let raw: RawConfig = {};
  for (const file of [
    o.path ?? CONFIG_PATH,
    o.projectDir ? findProjectConfig(o.projectDir) : undefined,
  ]) {
    if (!file) continue;
    const layer = readRaw(file);
    raw = mergeRaw(raw, layer);
    const perHarness = o.harness ? layer.harnesses?.[o.harness] : undefined;
    if (perHarness) raw = mergeRaw(raw, perHarness);
  }
  return resolveConfig(raw);
}
