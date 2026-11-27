import {
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { PageRef } from "./page-injection";
import type { RetainCursor, RetainCursorStore } from "./retain-cursor";
import type { UsageCursorStore } from "./usage";

/** Process-shared state for hook harnesses. SessionStart and prompt hooks are separate Node
 * processes, so this temp-file handoff carries lifecycle decisions without writing user config or
 * bank state. */
export interface SessionCache {
  turns?: number;
  reflectAnswer?: string; // present (even "") = reflect already resolved this session
  /** How many times auto-inject has been ATTEMPTED this session — every source, not just reflect.
   *  A failure leaves `reflectAnswer` unset so a later turn can retry; this bounds that retry (see
   *  HOOK_INJECT_ATTEMPTS). Named for reflect like `reflectAnswer` above, whose name also predates
   *  the other two sources. */
  reflectAttempts?: number;
  /** The user turn the auto-inject source last ran on; `autoInjectEveryTurns` counts from here. */
  injectedAtTurn?: number;
  /** SessionStart saw a new/empty bank; consume this on prompt one, then allow reflect. */
  deferInitialReflect?: boolean;
  pages?: { atTurn: number; list: PageRef[] };
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

/**
 * Replace a state file atomically: write a sibling temp file, then rename over the target.
 *
 * A plain writeFileSync is not atomic — a reader can observe a half-written file, and a process
 * killed mid-write leaves one behind. rename() is atomic on POSIX and replaces the destination on
 * Windows, so a reader sees either the old contents or the new, never a fragment (#3136, which
 * measured this class on the per-agent plugin; its Python state writes already had os.replace()).
 */
function writeFileAtomic(path: string, body: string): void {
  mkdirSync(dirname(path), { recursive: true });
  // The temp name is per-process, so two writers cannot collide on it.
  const tmp = `${path}.${process.pid}.tmp`;
  try {
    writeFileSync(tmp, body);
    renameSync(tmp, path);
  } catch (e) {
    try {
      rmSync(tmp, { force: true });
    } catch {
      /* nothing further to do */
    }
    throw e;
  }
}

export function writeSessionCache(cacheFile: string, cache: SessionCache): void {
  try {
    writeFileAtomic(cacheFile, JSON.stringify(cache));
  } catch {
    /* session state is best-effort */
  }
}

/**
 * Forget what this session has been shown, as if it had just started (#6527). A compaction drops
 * the injected memory and the page roster from the model's context, so the next prompt must
 * deliver both again: with no cache file it is the session's first prompt once more.
 */
export function resetSessionCache(cacheFile: string): void {
  try {
    rmSync(cacheFile, { force: true });
  } catch {
    /* session state is best-effort */
  }
}

/** Context a prompt hook could not deliver itself, waiting for the session's next tool hook (see
 *  HookSpec.toolHook). Its own file: `buildHookOutput` rewrites the session cache every turn. */
function pendingContextFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.pending`);
}

/** Add a block to the session's pending context. A block already waiting is not added again, so a
 *  session that goes several turns without a tool call is not handed the same roster N times. */
export function stashPendingContext(harness: string, sessionId: string, context: string): void {
  const path = pendingContextFile(harness, sessionId);
  let existing = "";
  try {
    existing = readFileSync(path, "utf8");
  } catch {
    /* nothing pending yet */
  }
  if (existing.includes(context)) return;
  try {
    writeFileAtomic(path, existing ? `${existing}\n\n${context}` : context);
  } catch {
    /* best-effort, like the session cache */
  }
}

/** Claim and clear the session's pending context. The rename is the claim: tool hooks can run in
 *  parallel, and only the one whose rename succeeds delivers the block. */
export function takePendingContext(harness: string, sessionId: string): string | undefined {
  const path = pendingContextFile(harness, sessionId);
  const claimed = `${path}.${process.pid}.claim`;
  try {
    renameSync(path, claimed);
    return readFileSync(claimed, "utf8") || undefined;
  } catch {
    return undefined;
  } finally {
    rmSync(claimed, { force: true });
  }
}

/**
 * Claim a user message for recall: true only when `mark` (its transcript position) is past the last
 * one claimed in this session. For a host whose prompt hook fires on every model step and carries no
 * prompt (Antigravity), so a step recalls only a user message newer than the last one recalled (#6944).
 * Its own file, like the pending context: `buildHookOutput` rewrites the session cache every turn.
 */
export function claimPromptMark(harness: string, sessionId: string, mark: number): boolean {
  const path = join(tmpdir(), `hindsight-${harness}`, `${sessionId}.recalled`);
  let last = -1;
  try {
    last = Number(readFileSync(path, "utf8"));
  } catch {
    /* nothing recalled yet this session */
  }
  if (Number.isFinite(last) && mark <= last) return false;
  try {
    writeFileAtomic(path, String(mark));
  } catch {
    /* best-effort: a lost mark costs one repeat recall, never a missed one */
  }
  return true;
}

/** The session root's own file, deliberately NOT the shared session cache — see sessionRootDir. */
function sessionRootFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.root`);
}

/**
 * The directory this session STARTED in: recorded the first time any of its hooks runs — normally
 * SessionStart — and returned unchanged for the rest of the session.
 *
 * Bank resolution needs a directory that does not move. A hook event reports the agent's LIVE
 * working directory, and an agent `cd`s during ordinary work. Inside a git repo that is harmless
 * (every subdirectory resolves back to the repo root), but a plain directory tree has no root to
 * resolve to, so the bank id followed the agent and ONE conversation was retained into a bank per
 * directory it visited — the same document, its facts split across banks (#3563).
 *
 * Deliberately keyed on the session id rather than read from a harness-exported project-root
 * variable: every hook harness reports a session id, only Claude Code exports a root. Same reason
 * `nearestExistingDir` walks the tree instead of guessing at nine env var names.
 *
 * It lives in its OWN file, like the retain cursor and for the same reason: the prompt hook writes
 * a fresh session-cache object rather than merging, so a field there would be dropped on every user
 * prompt — and the very first prompt is where it would first be needed.
 *
 * Best-effort throughout. A temp file that cannot be read or written just yields `cwd`, which is
 * exactly the behaviour before this existed — never an error, never a lost retain.
 */
export function sessionRootDir(
  harness: string,
  sessionId: string | undefined,
  cwd: string
): string {
  if (!sessionId || !cwd) return cwd;
  const file = sessionRootFile(harness, sessionId);
  try {
    const recorded = readFileSync(file, "utf8").trim();
    if (recorded) return recorded;
  } catch {
    /* not recorded yet — this hook is the session's first */
  }
  try {
    writeFileAtomic(file, cwd);
  } catch {
    /* best-effort: an unrecorded root costs stability, never data */
  }
  return cwd;
}

/**
 * How many of a session's turns core/usage.ts has recorded. Its own file for the same reason as
 * the retain cursor below; losing it re-records the session's turns, which the report dedupes.
 */
export function fileUsageCursorStore(harness: string): UsageCursorStore {
  const file = (sessionId: string) =>
    join(tmpdir(), `hindsight-${harness}`, `${sessionId}.usage.json`);
  return {
    read: (sessionId) => {
      try {
        const turns = (JSON.parse(readFileSync(file(sessionId), "utf8")) as { turns?: unknown })
          .turns;
        return typeof turns === "number" ? turns : undefined;
      } catch {
        return undefined;
      }
    },
    write: (sessionId, turns) => {
      try {
        writeFileAtomic(file(sessionId), JSON.stringify({ turns }));
      } catch {
        /* best-effort */
      }
    },
  };
}

const CURSOR_SUFFIX = ".retain.json";

/** The cursor's own file, deliberately NOT the shared session cache — see fileCursorStore. */
function cursorFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}${CURSOR_SUFFIX}`);
}

/** A session still owing a write-back: its last replace was never confirmed (`dirty`) or it holds
 *  appends that never landed (`pending`). */
export interface OrphanedCursor {
  sessionId: string;
  cursor: RetainCursor;
}

/**
 * This harness's sessions whose last write-back failed and that nobody has touched for `idleMs` —
 * the ones no later Stop of their own will ever retry, because the session ended (#6097). Ones
 * older than `maxAgeMs` are left alone: their transcript is likely gone, and retrying them on every
 * Stop forever would cost a request each time for nothing.
 */
export function orphanedCursors(
  harness: string,
  opts: { idleMs: number; maxAgeMs: number; now?: number }
): OrphanedCursor[] {
  const dir = join(tmpdir(), `hindsight-${harness}`);
  const now = opts.now ?? Date.now();
  let names: string[];
  try {
    names = readdirSync(dir).filter((n) => n.endsWith(CURSOR_SUFFIX));
  } catch {
    return [];
  }
  const out: { mtime: number; found: OrphanedCursor }[] = [];
  for (const name of names) {
    try {
      const file = join(dir, name);
      const mtime = statSync(file).mtimeMs;
      if (now - mtime < opts.idleMs || now - mtime > opts.maxAgeMs) continue;
      const cursor = JSON.parse(readFileSync(file, "utf8")) as RetainCursor;
      if (!cursor.dirty && !cursor.pending?.length) continue;
      out.push({
        mtime,
        found: { sessionId: name.slice(0, -CURSOR_SUFFIX.length), cursor },
      });
    } catch {
      /* unreadable or half-written: not ours to guess at */
    }
  }
  // Newest first: the most recent loss is the one most likely to still have its transcript.
  return out.sort((a, b) => b.mtime - a.mtime).map((o) => o.found);
}

/**
 * Retain cursor for every harness. A hook harness's Stop runs in a fresh process every time, so
 * "what have I already written" cannot live in memory; a persistent-plugin host keeps it here too, so
 * a write that failed before the host exited is still owed after it (#6559).
 *
 * ONE store per harness per process: core/chat.ts serializes a session's write-backs per store
 * object, and dsh (a core per workspace) or a replay alongside a live save would otherwise hold two
 * stores for one session's file and race on its document.
 *
 * It lives in its OWN file rather than a field of the session cache. It shared that file until now,
 * and the prompt hook — which writes a fresh `{turns, reflectAnswer, pages}` object rather than
 * merging — dropped the cursor on EVERY user prompt. The next Stop then found none and rewrote the
 * whole document, so the incremental write-back never engaged past a session's first turn on any
 * hook harness. Separate files make that structural: the two writers have different lifecycles and
 * no longer share a record, so neither can clobber the other, and the concurrent read-modify-write
 * that #3136 measured has nothing left to lose here.
 *
 * Losing the file (temp cleanup, reboot) is still not a correctness problem — a missing cursor
 * means the next retain replaces the whole document, exactly as it did before appends existed.
 */
const cursorStores = new Map<string, RetainCursorStore>();

export function fileCursorStore(harness: string): RetainCursorStore {
  let store = cursorStores.get(harness);
  if (!store) cursorStores.set(harness, (store = newFileCursorStore(harness)));
  return store;
}

function newFileCursorStore(harness: string): RetainCursorStore {
  return {
    read: (sessionId) => {
      try {
        return JSON.parse(readFileSync(cursorFile(harness, sessionId), "utf8")) as RetainCursor;
      } catch {
        return undefined;
      }
    },
    write: (sessionId, cursor) => {
      try {
        writeFileAtomic(cursorFile(harness, sessionId), JSON.stringify(cursor));
      } catch {
        /* best-effort: a cursor that cannot be written costs a replace, never data */
      }
    },
  };
}
