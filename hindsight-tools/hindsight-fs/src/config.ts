/**
 * Configuration resolution for hindsight-fs.
 *
 * Priority (highest first): explicit CLI flags > saved mount config
 * (<dir>/.hindsight-fs/config.json) > environment variables > defaults.
 *
 * The saved config is written when a folder is first mounted so that later
 * commands run inside that folder (status/stop/sync) reuse the same bank and
 * endpoint without re-passing flags.
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";
import { CONTROL_DIR } from "./paths.js";

export interface MountConfig {
  /** Absolute path to the mount directory. */
  dir: string;
  /** Hindsight API base URL (no trailing slash). */
  apiUrl: string;
  /** Bearer token, if the API requires auth. */
  apiToken?: string;
  /** Bank whose knowledge base is mirrored. */
  bankId: string;
  /** Refresh interval in seconds for the sync loop. */
  intervalSeconds: number;
  /**
   * When false (the default), mirrored files are written read-only so agents
   * cannot edit them. Set true to opt into editable files (still one-way: any
   * edit is overwritten on the next refresh).
   */
  writable: boolean;
}

/** Overrides supplied directly on the command line (all optional). */
export interface ConfigOverrides {
  dir?: string;
  apiUrl?: string;
  apiToken?: string;
  bankId?: string;
  intervalSeconds?: number;
  writable?: boolean;
}

export const DEFAULT_API_URL = "http://localhost:8000";
export const DEFAULT_INTERVAL_SECONDS = 30;
export const DEFAULT_DIR = "./hindsight-fs";

function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Partial config persisted alongside a mount. */
interface SavedConfig {
  apiUrl?: string;
  apiToken?: string;
  bankId?: string;
  intervalSeconds?: number;
  writable?: boolean;
}

async function readSavedConfig(dir: string): Promise<SavedConfig> {
  const file = path.join(dir, CONTROL_DIR, "config.json");
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as SavedConfig;
  } catch {
    return {};
  }
}

/**
 * Resolve the effective config for a command.
 *
 * `requireBank` controls whether a missing bank id is a hard error (true for
 * commands that talk to the API; false for read-only local commands).
 */
export async function resolveConfig(
  overrides: ConfigOverrides,
  opts: { requireBank?: boolean } = {}
): Promise<MountConfig> {
  const dir = path.resolve(overrides.dir ?? process.env.HINDSIGHT_FS_DIR ?? DEFAULT_DIR);
  const saved = await readSavedConfig(dir);

  const apiUrl = stripTrailingSlash(
    overrides.apiUrl ?? saved.apiUrl ?? process.env.HINDSIGHT_API_URL ?? DEFAULT_API_URL
  );

  const apiToken =
    overrides.apiToken ?? saved.apiToken ?? process.env.HINDSIGHT_API_TOKEN ?? undefined;

  const bankId = overrides.bankId ?? saved.bankId ?? process.env.HINDSIGHT_BANK_ID ?? "";

  if (opts.requireBank && !bankId) {
    throw new Error(
      "No bank specified. Pass --bank <id>, set HINDSIGHT_BANK_ID, or run inside an already-mounted folder."
    );
  }

  const intervalRaw =
    overrides.intervalSeconds ??
    saved.intervalSeconds ??
    (process.env.HINDSIGHT_FS_INTERVAL ? Number(process.env.HINDSIGHT_FS_INTERVAL) : undefined) ??
    DEFAULT_INTERVAL_SECONDS;
  const intervalSeconds =
    Number.isFinite(intervalRaw) && intervalRaw >= 1
      ? Math.floor(intervalRaw)
      : DEFAULT_INTERVAL_SECONDS;

  const writable = overrides.writable ?? saved.writable ?? false;

  return { dir, apiUrl, apiToken, bankId, intervalSeconds, writable };
}

/** Persist the parts of a config worth remembering for a mounted folder. */
export async function saveConfig(config: MountConfig): Promise<void> {
  const controlDir = path.join(config.dir, CONTROL_DIR);
  await fs.mkdir(controlDir, { recursive: true });
  const saved: SavedConfig = {
    apiUrl: config.apiUrl,
    apiToken: config.apiToken,
    bankId: config.bankId,
    intervalSeconds: config.intervalSeconds,
    writable: config.writable,
  };
  await fs.writeFile(
    path.join(controlDir, "config.json"),
    JSON.stringify(saved, null, 2) + "\n",
    "utf8"
  );
}
