/**
 * The sync engine: fetch a bank's knowledge base (folder/page tree + page
 * contents) and mirror it as a nested folder of markdown files in the mount
 * directory — folders become directories, pages become `.md` files.
 *
 * The mirror is strictly one-way (API → disk). Two mechanisms enforce that:
 *
 *  1. Mirrored files are written read-only (mode 0444 unless `config.writable`),
 *     so an agent's in-place edit or editor-save fails with EACCES.
 *  2. Every pass compares the *on-disk* bytes against the freshly rendered
 *     content, so any drift — a tampered file, a force-chmod edit, a partial
 *     write — is reverted on the next tick, even when the page is unchanged
 *     server-side.
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";
import { HindsightFsClient } from "./client.js";
import { planMirror, renderIndex } from "./format.js";
import { hashContent, loadState, saveState, type SyncState } from "./state.js";
import { CONTROL_DIR, INDEX_FILE } from "./paths.js";
import type { MountConfig } from "./config.js";

export interface SyncResult {
  /** Pages mirrored. */
  total: number;
  /** Folder directories in the mirror. */
  folders: number;
  /** Files (re)written because they were new, changed, or tampered with. */
  written: number;
  /** Files left untouched because disk already matched the API. */
  unchanged: number;
  /** Files removed because their page no longer exists in the bank. */
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
 * Pruning of files/folders for removed pages happens only after a successful
 * fetch, so a transient API/network error never wipes the existing mirror.
 */
export async function runSync(config: MountConfig): Promise<SyncResult> {
  const syncedAt = new Date().toISOString();
  const mode = config.writable ? WRITABLE_MODE : READONLY_MODE;
  await fs.mkdir(path.join(config.dir, CONTROL_DIR), { recursive: true });

  const state = await loadState(config.dir, config.bankId, config.apiUrl);
  const client = new HindsightFsClient({ apiUrl: config.apiUrl, apiToken: config.apiToken });

  let snapshot;
  try {
    snapshot = await client.loadKnowledge(config.bankId);
  } catch (err) {
    state.lastSyncAt = syncedAt;
    state.lastSyncOk = false;
    state.lastError = err instanceof Error ? err.message : String(err);
    await saveState(config.dir, state);
    throw err;
  }

  const plan = planMirror(snapshot);

  // Create folder directories first (parents before children — plan.dirs is in
  // tree order). The control dir is excluded by the .hindsight-fs prefix.
  for (const dir of plan.dirs) {
    await fs.mkdir(path.join(config.dir, dir), { recursive: true });
  }

  let written = 0;
  let unchanged = 0;
  let reverted = 0;
  const nextFiles: SyncState["files"] = {};

  for (const page of plan.files) {
    const hash = hashContent(page.content);
    const prev = state.files[page.relPath];
    const absPath = path.join(config.dir, page.relPath);
    await fs.mkdir(path.dirname(absPath), { recursive: true });

    // Source of truth is the bytes on disk — so a local edit is detected and
    // overwritten even when the page is identical to last time.
    const onDisk = await readFileOrNull(absPath);
    if (onDisk === null || onDisk !== page.content) {
      await atomicWrite(absPath, page.content, mode);
      written++;
      if (onDisk !== null && prev && prev.hash === hash) reverted++;
    } else {
      await enforceMode(absPath, mode);
      unchanged++;
    }
    nextFiles[page.relPath] = { file: page.relPath, hash };
  }

  // Prune files whose pages no longer exist.
  let removed = 0;
  for (const [rel, entry] of Object.entries(state.files)) {
    if (!nextFiles[rel]) {
      await safeUnlink(path.join(config.dir, entry.file));
      removed++;
    }
  }

  // Prune folder directories that no longer exist (deepest first so they empty
  // out before removal); only remove ones we created and that are now gone.
  const liveDirs = new Set(plan.dirs);
  const goneDirs = state.dirs.filter((d) => !liveDirs.has(d)).sort((a, b) => b.length - a.length);
  for (const dir of goneDirs) {
    await safeRmdir(path.join(config.dir, dir));
  }

  const newState: SyncState = {
    version: 1,
    bankId: config.bankId,
    apiUrl: config.apiUrl,
    lastSyncAt: syncedAt,
    lastSyncOk: true,
    lastError: null,
    files: nextFiles,
    dirs: plan.dirs,
  };
  await saveState(config.dir, newState);

  await atomicWrite(
    path.join(config.dir, CONTROL_DIR, INDEX_FILE),
    renderIndex(snapshot, config),
    mode
  );

  return {
    total: plan.pageCount,
    folders: plan.folderCount,
    written,
    unchanged,
    removed,
    reverted,
    syncedAt,
  };
}

async function safeUnlink(p: string): Promise<void> {
  try {
    await fs.chmod(p, WRITABLE_MODE).catch(() => {});
    await fs.unlink(p);
  } catch {
    /* already gone */
  }
}

async function safeRmdir(p: string): Promise<void> {
  try {
    await fs.rmdir(p);
  } catch {
    /* non-empty (has unmirrored files) or already gone — leave it */
  }
}
