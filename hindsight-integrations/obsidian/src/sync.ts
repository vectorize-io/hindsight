/**
 * SyncEngine — keeps a Hindsight bank in sync with the vault, one way
 * (Obsidian → Hindsight). The vault is always the source of truth; this engine
 * only ever derives Hindsight content from notes (DESIGN.md §0.5).
 *
 * It is written against minimal `SyncVault`/`SyncFile` interfaces (a subset of
 * Obsidian's `Vault`/`TFile`) so it can be unit-tested without the Obsidian
 * runtime.
 */

import { createHash } from "node:crypto";
import type { HindsightClient } from "./client";
import { normalizeNote } from "./frontmatter";

export interface SyncFile {
  path: string;
  stat: { mtime: number; ctime: number };
}

export interface SyncVault {
  getMarkdownFiles(): SyncFile[];
  read(file: SyncFile): Promise<string>;
}

export interface NoteState {
  hash: string;
  mtime: number;
  syncedAt: string;
}

export type SyncIndex = Record<string, NoteState>;

export interface SyncConfig {
  bankId: string;
  /** Empty = include everything (minus excludes). */
  includeFolders: string[];
  excludeFolders: string[];
  vaultName: string;
  /** Prefix document ids with the vault name (for multi-vault shared banks). */
  prefixDocId: boolean;
}

export interface ReconcileSummary {
  added: number;
  updated: number;
  deleted: number;
  unchanged: number;
}

export type IngestOutcome = "created" | "updated" | "skipped";

function underFolder(path: string, folder: string): boolean {
  const f = folder.replace(/^\/+|\/+$/g, "");
  if (!f) return true;
  return path === f || path.startsWith(`${f}/`);
}

function isoFromMillis(ms: number): string {
  return new Date(ms).toISOString();
}

/**
 * `folder:` tags for every ancestor of a note path, so `folder:Work` matches
 * everything under Work/. e.g. "Work/Clients/acme.md" → ["folder:Work",
 * "folder:Work/Clients"].
 */
function folderTags(path: string): string[] {
  const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
  if (!dir) return [];
  const parts = dir.split("/");
  return parts.map((_, i) => `folder:${parts.slice(0, i + 1).join("/")}`);
}

/**
 * Year + month bucket tags, e.g. ["created:2026", "created:2026-03"]. Recall has
 * no hard date-range filter, so date scoping is expressed as an OR over buckets.
 */
function dateTags(prefix: string, ms: number): string[] {
  if (!Number.isFinite(ms)) return [];
  const d = new Date(ms);
  const year = d.getUTCFullYear();
  const month = String(d.getUTCMonth() + 1).padStart(2, "0");
  return [`${prefix}:${year}`, `${prefix}:${year}-${month}`];
}

async function mapLimit<T>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<void>
): Promise<void> {
  const queue = [...items];
  const workerCount = Math.max(1, Math.min(limit, queue.length));
  const workers = Array.from({ length: workerCount }, async () => {
    for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
      await fn(item);
    }
  });
  await Promise.all(workers);
}

export class SyncEngine {
  constructor(
    private readonly client: HindsightClient,
    private readonly vault: SyncVault,
    private config: SyncConfig,
    private index: SyncIndex,
    private readonly persist: (index: SyncIndex) => Promise<void>,
    private readonly nowIso: () => string = () => new Date().toISOString()
  ) {}

  docId(path: string): string {
    return this.config.prefixDocId ? `${this.config.vaultName}/${path}` : path;
  }

  shouldInclude(path: string): boolean {
    if (!path.endsWith(".md")) return false;
    if (this.config.excludeFolders.some((f) => underFolder(path, f))) return false;
    if (this.config.includeFolders.length === 0) return true;
    return this.config.includeFolders.some((f) => underFolder(path, f));
  }

  private hash(content: string): string {
    return `sha256:${createHash("sha256").update(content).digest("hex")}`;
  }

  /**
   * Ingest a single note, upserting it into the bank. Returns whether the note
   * was newly created, updated, or skipped (unchanged / excluded / empty).
   */
  async ingestFile(
    file: SyncFile,
    opts: { force?: boolean; persist?: boolean } = {}
  ): Promise<IngestOutcome> {
    const doPersist = opts.persist ?? true;
    if (!this.shouldInclude(file.path)) return "skipped";

    const prev = this.index[file.path];
    // Cheap pre-filter: same mtime as last sync → nothing to do (no read).
    if (!opts.force && prev && prev.mtime === file.stat.mtime) return "skipped";

    const raw = await this.vault.read(file);
    const hash = this.hash(raw);
    if (!opts.force && prev && prev.hash === hash) {
      // mtime moved but content is identical — refresh the mtime, don't re-ingest.
      this.index[file.path] = { ...prev, mtime: file.stat.mtime };
      if (doPersist) await this.persist(this.index);
      return "skipped";
    }

    const folder = file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/")) : "";
    const note = normalizeNote(raw, folder);
    if (!note.body) return "skipped"; // nothing to ground on

    // Auto-scope tags: implicit scoping derived from normal Obsidian usage, so the
    // user only thinks about scope at recall/reflect time (DESIGN.md §4.6). Recall
    // can then filter by any combination via tag_groups, from the UI or the API.
    const createdMs = note.timestamp ? Date.parse(note.timestamp) : file.stat.ctime;
    const scopeTags = [
      `vault:${this.config.vaultName}`,
      ...folderTags(file.path),
      ...dateTags("created", Number.isFinite(createdMs) ? createdMs : file.stat.ctime),
      ...dateTags("updated", file.stat.mtime),
    ];
    const tags = [...new Set([...note.tags, ...scopeTags])];
    // `path` lets API consumers (automations) map a recall hit back to the note.
    const metadata = { ...note.metadata, vault: this.config.vaultName, path: file.path };

    await this.client.retain(this.config.bankId, this.docId(file.path), note.body, {
      tags,
      metadata,
      timestamp: note.timestamp ?? isoFromMillis(file.stat.ctime),
      updateMode: "replace",
    });

    const outcome: IngestOutcome = prev ? "updated" : "created";
    this.index[file.path] = { hash, mtime: file.stat.mtime, syncedAt: this.nowIso() };
    if (doPersist) await this.persist(this.index);
    return outcome;
  }

  /** Handle a note deletion: remove its document (only if we synced it). */
  async handleDelete(path: string): Promise<void> {
    if (!this.index[path]) return;
    await this.client.deleteDocument(this.config.bankId, this.docId(path));
    delete this.index[path];
    await this.persist(this.index);
  }

  /** Handle a rename: delete the old document, then ingest under the new path. */
  async handleRename(file: SyncFile, oldPath: string): Promise<void> {
    if (this.index[oldPath]) {
      await this.client.deleteDocument(this.config.bankId, this.docId(oldPath));
      delete this.index[oldPath];
      await this.persist(this.index);
    }
    await this.ingestFile(file, { force: true });
  }

  /**
   * Full reconcile: ingest every (drifted) included note, then prune documents
   * we previously synced whose note is now gone or excluded. Self-heals after
   * the plugin was disabled during edits.
   */
  async reconcile(): Promise<ReconcileSummary> {
    const summary: ReconcileSummary = { added: 0, updated: 0, deleted: 0, unchanged: 0 };
    const files = this.vault.getMarkdownFiles().filter((f) => this.shouldInclude(f.path));
    const livePaths = new Set(files.map((f) => f.path));

    await mapLimit(files, 3, async (file) => {
      const outcome = await this.ingestFile(file, { persist: false });
      if (outcome === "created") summary.added++;
      else if (outcome === "updated") summary.updated++;
      else summary.unchanged++;
    });

    // Prune by local index, NOT by listing server documents. Listing would also
    // surface docs we don't own (e.g. opt-in conversation memory under
    // `conversation/…`, or notes from another tool sharing the bank) and we'd
    // wrongly delete them. The trade-off: orphans created while the local index
    // was lost (reinstall) aren't auto-pruned — re-deleting the note fixes that.
    for (const path of Object.keys(this.index)) {
      if (!livePaths.has(path)) {
        await this.client.deleteDocument(this.config.bankId, this.docId(path));
        delete this.index[path];
        summary.deleted++;
      }
    }

    await this.persist(this.index);
    return summary;
  }
}
