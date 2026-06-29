/**
 * The sync engine: fetch a bank's mental models and mirror them as markdown
 * files in the mount directory.
 *
 * The mirror is strictly one-way (API → disk). Two mechanisms enforce that:
 *
 *  1. Mirrored files are written read-only (mode 0444 unless `config.writable`),
 *     so an agent's in-place edit or editor-save fails with EACCES.
 *  2. Every pass compares the *on-disk* bytes against the freshly rendered
 *     content, so any drift — a tampered file, a force-chmod edit, a partial
 *     write — is reverted on the next tick, even when the model is unchanged
 *     server-side. (Change detection looks at disk, not just the last API hash,
 *     precisely so local edits cannot survive.)
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";
import { HindsightFsClient, type MentalModel } from "./client.js";
import { fileNameFor, renderIndex, renderMentalModel } from "./format.js";
import { hashContent, loadState, saveState, type SyncState } from "./state.js";
import { CONTROL_DIR, INDEX_FILE } from "./paths.js";
import type { MountConfig } from "./config.js";

export interface SyncResult {
  total: number;
  /** Files (re)written because they were new, changed, or tampered with. */
  written: number;
  /** Files left untouched because disk already matched the API. */
  unchanged: number;
  /** Files removed because their model no longer exists in the bank. */
  removed: number;
  /** Subset of `written` that were rewritten because the on-disk copy drifted. */
  reverted: number;
  syncedAt: string;
}

const READONLY_MODE = 0o444;
const WRITABLE_MODE = 0o644;

/** Write `content` to `file` atomically (temp file + rename within the same dir). */
async function atomicWrite(file: string, content: string, mode: number): Promise<void> {
  const tmp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(tmp, content, "utf8");
  // rename replaces the destination entry regardless of the old file's mode
  // (only the directory needs to be writable), so this works over a 0444 file.
  await fs.rename(tmp, file);
  await fs.chmod(file, mode);
}

async function readFileOrNull(file: string): Promise<string | null> {
  try {
    return await fs.readFile(file, "utf8");
  } catch {
    return null;
  }
}

/** Re-assert the desired permission bits (cheap; guards against a force-chmod). */
async function enforceMode(file: string, mode: number): Promise<void> {
  try {
    await fs.chmod(file, mode);
  } catch {
    /* file vanished between write and chmod — next tick recreates it */
  }
}

/**
 * Perform a single sync pass.
 *
 * Pruning of files for deleted models happens only after a successful fetch, so
 * a transient API/network error never wipes the existing mirror.
 */
export async function runSync(config: MountConfig): Promise<SyncResult> {
  const syncedAt = new Date().toISOString();
  const mode = config.writable ? WRITABLE_MODE : READONLY_MODE;
  await fs.mkdir(path.join(config.dir, CONTROL_DIR), { recursive: true });

  const state = await loadState(config.dir, config.bankId, config.apiUrl);
  const client = new HindsightFsClient({ apiUrl: config.apiUrl, apiToken: config.apiToken });

  let models: MentalModel[];
  try {
    models = await client.listMentalModels(config.bankId, config.detail);
  } catch (err) {
    state.lastSyncAt = syncedAt;
    state.lastSyncOk = false;
    state.lastError = err instanceof Error ? err.message : String(err);
    await saveState(config.dir, state);
    throw err;
  }

  let written = 0;
  let unchanged = 0;
  let reverted = 0;
  const seenIds = new Set<string>();
  const nextFiles: SyncState["files"] = {};

  for (const model of models) {
    seenIds.add(model.id);
    const fileName = fileNameFor(model.id);
    const expected = renderMentalModel(model);
    const hash = hashContent(expected);
    const prev = state.files[model.id];
    const absPath = path.join(config.dir, fileName);

    // Clean up a renamed file (id kept, slug changed).
    if (prev && prev.file !== fileName) {
      await safeUnlink(path.join(config.dir, prev.file));
    }

    // Source of truth is the bytes on disk — so a local edit is detected and
    // overwritten even when the model is identical to last time.
    const onDisk = await readFileOrNull(absPath);
    if (onDisk === null || onDisk !== expected) {
      await atomicWrite(absPath, expected, mode);
      written++;
      // It drifted (not new, and the API content didn't change) ⇒ tampering.
      if (onDisk !== null && prev && prev.hash === hash) reverted++;
    } else {
      await enforceMode(absPath, mode);
      unchanged++;
    }
    nextFiles[model.id] = { file: fileName, hash };
  }

  // Prune files whose models no longer exist.
  let removed = 0;
  for (const [id, entry] of Object.entries(state.files)) {
    if (!seenIds.has(id)) {
      await safeUnlink(path.join(config.dir, entry.file));
      removed++;
    }
  }

  const newState: SyncState = {
    version: 1,
    bankId: config.bankId,
    apiUrl: config.apiUrl,
    lastSyncAt: syncedAt,
    lastSyncOk: true,
    lastError: null,
    files: nextFiles,
  };
  await saveState(config.dir, newState);

  await atomicWrite(
    path.join(config.dir, CONTROL_DIR, INDEX_FILE),
    renderIndex(models, config),
    mode
  );

  return {
    total: models.length,
    written,
    unchanged,
    removed,
    reverted,
    syncedAt,
  };
}

async function safeUnlink(p: string): Promise<void> {
  try {
    // Drop read-only bit first so the unlink itself isn't blocked on some FS.
    await fs.chmod(p, WRITABLE_MODE).catch(() => {});
    await fs.unlink(p);
  } catch {
    /* already gone */
  }
}
