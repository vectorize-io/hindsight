/**
 * JSON-file persistence for the sync index, the CLI's equivalent of the plugin's
 * `data.json`. Kept out of the vault by default (see {@link defaultIndexPath})
 * so Obsidian Sync never propagates it and it can't collide with the plugin's
 * own index on another machine.
 */

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { SyncIndex } from "../sync";

interface IndexFile {
  syncIndex: SyncIndex;
  lastSyncAt: string | null;
}

/** Default out-of-vault index location: `~/.hindsight/obsidian/<vault>.json`. */
export function defaultIndexPath(vaultName: string): string {
  const safe = vaultName.replace(/[^A-Za-z0-9._-]+/g, "_") || "vault";
  return join(homedir(), ".hindsight", "obsidian", `${safe}.json`);
}

/** Load a previously-persisted index, or an empty one if absent/unreadable. */
export async function loadIndex(path: string): Promise<SyncIndex> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    return {}; // first run — no index yet
  }
  try {
    const data = JSON.parse(raw) as Partial<IndexFile>;
    return data.syncIndex ?? {};
  } catch {
    // A corrupt index means a full re-ingest (hash-skips unchanged) rather than
    // a crash; warn because orphan pruning can't run until the index is rebuilt.
    console.warn(`[hindsight] ignoring unreadable sync index at ${path}; starting fresh`);
    return {};
  }
}

/** A `persist` callback for {@link SyncEngine} that atomically writes the index. */
export function makePersist(path: string, nowIso: () => string = () => new Date().toISOString()) {
  return async (index: SyncIndex): Promise<void> => {
    await mkdir(dirname(path), { recursive: true });
    const payload: IndexFile = { syncIndex: index, lastSyncAt: nowIso() };
    const tmp = `${path}.tmp`;
    await writeFile(tmp, `${JSON.stringify(payload, null, 2)}\n`);
    await rename(tmp, path); // atomic replace — never leave a half-written index
  };
}
