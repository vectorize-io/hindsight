/**
 * Harness-agnostic configuration — a single JSON file, NO environment variables.
 *
 * Both entry points read this: the runtime plugin (src/index.ts) uses it wholesale, and the backfill
 * CLI (src/backfill.ts) uses it for the shared connection/bank settings with its --flags overriding.
 * The file is optional: a missing file yields all defaults (so the plugin still works out of the box);
 * a present-but-malformed file logs one warning and falls back to defaults (memory never breaks the agent).
 *
 * Location: ~/.hindsight/coding-agent.json
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

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
  bankId?: string; // memory bank id (default "coding")
  harness?: string; // runtime adapter (default "opencode")
  disabled?: boolean; // hard off-switch — inert plugin, for a no-memory baseline (default false)
  retainSessions?: boolean; // enable live write-back (default false)
  retainEveryTurns?: number; // write-back cadence in user turns (default 5)
  reflectTimeoutMs?: number; // reflect timeout (default 120000)
  gitSync?: GitSyncConfig;
}

/** Fully-resolved config: every field present, gitSync fully populated. */
export interface Config {
  apiUrl: string;
  apiToken?: string;
  bankId: string;
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
    bankId: raw.bankId ?? "coding",
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

/** Load + resolve the config file. Missing file -> silent defaults; malformed file -> one warning + defaults. */
export function loadConfig(path: string = CONFIG_PATH): Config {
  let raw: RawConfig = {};
  try {
    raw = JSON.parse(readFileSync(path, "utf8")) as RawConfig;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      console.error(`hindsight: ignoring invalid config at ${path}: ${(e as Error)?.message || e}`);
    }
  }
  return resolveConfig(raw);
}
