/**
 * Persistent sync state for a mount. Tracks which file mirrors each mental model
 * and a content hash so unchanged models are not rewritten (keeps mtimes stable
 * for editors, watchers, and `ls -la`).
 */

import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import * as path from "node:path";
import { CONTROL_DIR, STATE_FILE } from "./paths.js";

export interface FileEntry {
  /** Filename relative to the mount root. */
  file: string;
  /** sha256 of the rendered document. */
  hash: string;
}

export interface SyncState {
  version: 1;
  bankId: string;
  apiUrl: string;
  lastSyncAt: string | null;
  lastSyncOk: boolean;
  lastError: string | null;
  /** Mental-model id → file entry. */
  files: Record<string, FileEntry>;
}

export function emptyState(bankId: string, apiUrl: string): SyncState {
  return {
    version: 1,
    bankId,
    apiUrl,
    lastSyncAt: null,
    lastSyncOk: false,
    lastError: null,
    files: {},
  };
}

export function hashContent(content: string): string {
  return createHash("sha256").update(content, "utf8").digest("hex");
}

function statePath(dir: string): string {
  return path.join(dir, CONTROL_DIR, STATE_FILE);
}

export async function loadState(dir: string, bankId: string, apiUrl: string): Promise<SyncState> {
  try {
    const raw = JSON.parse(await fs.readFile(statePath(dir), "utf8")) as SyncState;
    if (raw.version === 1 && raw.files) return raw;
  } catch {
    /* fall through to empty */
  }
  return emptyState(bankId, apiUrl);
}

export async function saveState(dir: string, state: SyncState): Promise<void> {
  await fs.mkdir(path.join(dir, CONTROL_DIR), { recursive: true });
  await fs.writeFile(statePath(dir), JSON.stringify(state, null, 2) + "\n", "utf8");
}
