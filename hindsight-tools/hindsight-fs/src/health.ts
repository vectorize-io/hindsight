/**
 * Health assessment for a mount — used by `status` (human + `--json`) and
 * exported for programmatic healthchecks/watchdogs.
 *
 * Two orthogonal signals are combined into one verdict:
 *  - liveness: is the daemon process actually alive?
 *  - freshness: did a sync succeed recently (within the stale threshold)?
 */

import { daemonStatus } from "./daemon.js";
import { loadState } from "./state.js";
import type { MountConfig } from "./config.js";

export type HealthStatus = "ok" | "stale" | "failed" | "dead";

export interface HealthReport {
  /** True only when status === "ok". Drives the process exit code. */
  healthy: boolean;
  status: HealthStatus;
  mount: string;
  bank: string;
  apiUrl: string;
  mode: "read-only" | "writable";
  daemon: {
    running: boolean;
    pid: number | null;
    startedAt: string | null;
    intervalSeconds: number | null;
  };
  lastSync: {
    at: string | null;
    ok: boolean;
    /** Seconds since the last sync attempt, or null if it never ran. */
    ageSeconds: number | null;
    error: string | null;
  };
  /** A sync older than this many seconds is considered stale. */
  staleAfterSeconds: number;
  mirroredFiles: number;
}

export interface HealthOptions {
  /** Override the stale threshold; default is max(interval × 3, 15s). */
  staleAfterSeconds?: number;
  /** Injectable clock (epoch ms) for testing. */
  now?: number;
}

export async function computeHealth(
  config: MountConfig,
  opts: HealthOptions = {}
): Promise<HealthReport> {
  const ds = await daemonStatus(config.dir);
  const state = await loadState(config.dir, config.bankId, config.apiUrl);
  const now = opts.now ?? Date.now();

  const intervalSeconds = ds.record?.intervalSeconds ?? config.intervalSeconds;
  const staleAfterSeconds = opts.staleAfterSeconds ?? Math.max(intervalSeconds * 3, 15);

  const ageSeconds = state.lastSyncAt
    ? Math.max(0, Math.round((now - Date.parse(state.lastSyncAt)) / 1000))
    : null;

  let status: HealthStatus;
  if (!ds.running) {
    status = "dead";
  } else if (state.lastSyncAt === null) {
    status = "stale"; // up but hasn't completed a first sync yet
  } else if (!state.lastSyncOk) {
    status = "failed"; // looping but the API keeps erroring
  } else if (ageSeconds === null || ageSeconds >= staleAfterSeconds) {
    status = "stale"; // alive but wedged — no fresh sync (>= so --stale-after 0 = always stale)
  } else {
    status = "ok";
  }

  return {
    healthy: status === "ok",
    status,
    mount: config.dir,
    bank: state.bankId || config.bankId || "",
    apiUrl: state.apiUrl || config.apiUrl,
    mode: config.writable ? "writable" : "read-only",
    daemon: {
      running: ds.running,
      pid: ds.pid,
      startedAt: ds.record?.startedAt ?? null,
      intervalSeconds: ds.record?.intervalSeconds ?? null,
    },
    lastSync: {
      at: state.lastSyncAt,
      ok: state.lastSyncOk,
      ageSeconds,
      error: state.lastError,
    },
    staleAfterSeconds,
    mirroredFiles: Object.keys(state.files).length,
  };
}
