/**
 * JSONL-backed retain queue for buffering failed HTTP retains.
 *
 * When the remote Hindsight API is unreachable, retain requests are stashed
 * in a local JSONL file and flushed later. Only used in external API mode —
 * the local daemon handles its own persistence.
 *
 * Zero runtime dependencies; uses only Node built-ins.
 */

import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  existsSync,
  renameSync,
  unlinkSync,
} from "fs";
import { randomBytes } from "crypto";

/** The subset of a retain payload the queue needs to persist and replay. */
export interface QueuedRetainPayload {
  content: string;
  documentId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  updateMode?: "replace" | "append";
}

export interface QueuedRetain {
  id: string;
  bankId: string;
  content: string;
  documentId: string;
  metadata: Record<string, unknown>;
  tags?: string[];
  updateMode?: "replace" | "append";
  createdAt: string; // ISO 8601
}

export interface RetainQueueOptions {
  /** Path to the JSONL queue file. The parent directory must already exist. */
  filePath: string;
  /** Max age in ms for queued items. `-1` (default) keeps items forever. */
  maxAgeMs?: number;
}

export class RetainQueue {
  private readonly filePath: string;
  private readonly maxAgeMs: number;
  private cachedSize: number;

  constructor(opts: RetainQueueOptions) {
    this.filePath = opts.filePath;
    this.maxAgeMs = opts.maxAgeMs ?? -1;
    this.cachedSize = this.readAll().length;
  }

  /** Append a failed retain for later delivery. */
  enqueue(bankId: string, request: QueuedRetainPayload, metadata?: Record<string, unknown>): void {
    const item: QueuedRetain = {
      id: `${Date.now()}-${randomBytes(4).toString("hex")}`,
      bankId,
      content: request.content,
      documentId: request.documentId || "conversation",
      metadata: metadata || request.metadata || {},
      tags: request.tags,
      updateMode: request.updateMode,
      createdAt: new Date().toISOString(),
    };
    appendFileSync(this.filePath, JSON.stringify(item) + "\n", "utf8");
    this.cachedSize++;
  }

  /** Get up to `limit` oldest pending items (FIFO). */
  peek(limit = 50): QueuedRetain[] {
    return this.readAll().slice(0, limit);
  }

  /** Remove a single item by id. */
  remove(id: string): void {
    const items = this.readAll().filter((i) => i.id !== id);
    this.writeAll(items);
  }

  /** Remove multiple items by id in a single file rewrite. */
  removeMany(ids: string[]): void {
    const idSet = new Set(ids);
    const items = this.readAll().filter((i) => !idSet.has(i.id));
    this.writeAll(items);
  }

  /** Number of items waiting (cached, O(1)). */
  size(): number {
    return this.cachedSize;
  }

  /** Drop items older than `maxAgeMs`. No-op when `maxAgeMs < 0`. */
  cleanup(): number {
    if (this.maxAgeMs < 0) return 0;
    const cutoff = Date.now() - this.maxAgeMs;
    const items = this.readAll();
    const kept = items.filter((i) => new Date(i.createdAt).getTime() >= cutoff);
    const removed = items.length - kept.length;
    if (removed > 0) this.writeAll(kept);
    return removed;
  }

  /** No-op — kept for API symmetry with DB-backed queues. */
  close(): void {
    /* nothing to close */
  }

  // -------------------------------------------------------------------------

  private readAll(): QueuedRetain[] {
    if (!existsSync(this.filePath)) return [];
    const content = readFileSync(this.filePath, "utf8").trim();
    if (!content) return [];
    const items: QueuedRetain[] = [];
    for (const line of content.split("\n")) {
      try {
        items.push(JSON.parse(line) as QueuedRetain);
      } catch {
        // skip malformed lines
      }
    }
    return items;
  }

  /** Atomically rewrite the file with the given items. */
  private writeAll(items: QueuedRetain[]): void {
    if (items.length === 0) {
      try {
        unlinkSync(this.filePath);
      } catch {
        /* already gone */
      }
      this.cachedSize = 0;
      return;
    }
    const tmpPath = this.filePath + ".tmp";
    writeFileSync(tmpPath, items.map((i) => JSON.stringify(i)).join("\n") + "\n", "utf8");
    renameSync(tmpPath, this.filePath);
    this.cachedSize = items.length;
  }
}
