import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { PageRef } from "./knowledge-injection";
import type { RetainCursor, RetainCursorStore } from "./retain-cursor";

/** Process-shared state for hook harnesses. SessionStart and prompt hooks are separate Node
 * processes, so this temp-file handoff carries lifecycle decisions without writing user config or
 * bank state. */
export interface SessionCache {
  turns?: number;
  reflectAnswer?: string; // present (even "") = reflect already ran this session
  /** SessionStart saw a new/empty bank; consume this on prompt one, then allow reflect. */
  deferInitialReflect?: boolean;
  pages?: { atTurn: number; list: PageRef[] };
  /** How much of this session's transcript is already in its document (core/retain-cursor.ts). */
  retain?: RetainCursor;
}

export function sessionCacheFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.json`);
}

export function readSessionCache(cacheFile: string): SessionCache {
  try {
    return JSON.parse(readFileSync(cacheFile, "utf8")) as SessionCache;
  } catch {
    return {};
  }
}

export function writeSessionCache(cacheFile: string, cache: SessionCache): void {
  try {
    mkdirSync(dirname(cacheFile), { recursive: true });
    writeFileSync(cacheFile, JSON.stringify(cache));
  } catch {
    /* session state is best-effort */
  }
}

/**
 * Retain cursor kept in the per-session temp file, for the hook harnesses: Stop runs in a fresh
 * process every time, so "what have I already written" cannot live in memory.
 *
 * Losing this file (temp cleanup, reboot) is not a correctness problem — a missing cursor means the
 * next retain replaces the whole document, which is exactly what the plugin did before appends.
 */
export function fileCursorStore(harness: string): RetainCursorStore {
  return {
    read: (sessionId) => readSessionCache(sessionCacheFile(harness, sessionId)).retain,
    write: (sessionId, cursor) => {
      const file = sessionCacheFile(harness, sessionId);
      writeSessionCache(file, { ...readSessionCache(file), retain: cursor });
    },
  };
}
