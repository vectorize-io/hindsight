/**
 * How much of a live session has already been written to its `conversation:<id>` document.
 *
 * Without a cursor every write-back re-uploads the WHOLE conversation: a long session re-sends the
 * entire transcript on every Stop (and every N turns under the persistent-plugin runtime), which is
 * what makes a runaway session unretainable rather than merely large. With one, a retain sends only
 * the turns added since the last successful write, using the server's `update_mode: "append"`.
 *
 * Append is only correct while our view of the document matches the server's, so this deliberately
 * falls back to a FULL REPLACE — which is idempotent by construction and therefore always safe —
 * whenever that cannot be established:
 *
 *   - no cursor          first write-back of the session, or the cache was evicted.
 *   - fingerprint drift  the transcript was REWRITTEN rather than extended (Claude Code compaction,
 *                        a truncated/rotated rollout), so what we already sent is no longer a prefix
 *                        of what we hold and an append would splice two different conversations.
 *   - dirty              the previous retain failed, or its outcome is unknown (the client aborts at
 *                        15s while the server may well have committed it). Appending on top of an
 *                        unknown state is the one way to DUPLICATE turns inside the document, so we
 *                        rewrite the whole thing instead and let replace re-establish the truth.
 *
 * `dirty` is set BEFORE the request and cleared after it succeeds, so an overlapping retain (the
 * runtime fires them without awaiting) sees the advanced cursor and does not re-send the same slice.
 */
import { createHash } from "node:crypto";
import type { TransportTurn } from "./chat";

export interface RetainCursor {
  /** Turns already written to the document (index into the normalized transcript). */
  turns: number;
  /** Identity of the written prefix — detects a rewritten transcript (see module doc). */
  fingerprint: string;
  /** The bank the document was written to. The cursor is keyed by session, but the bank is derived
   *  per hook invocation from that event's cwd — a session that moves between repos (#3133) keeps
   *  its id and changes bank, and the new bank holds no document to append to. */
  bank: string;
  /** Appends written since the last full document write (see MAX_APPENDS_BEFORE_RESYNC). */
  appends?: number;
  /** A write was started and not confirmed: the next retain must replace, not append. */
  dirty?: boolean;
}

/**
 * Force a full write every so often, however healthy the cursor looks.
 *
 * Retains are async: the server acknowledges the submission and extracts later, so a write can be
 * confirmed to us and still fail server-side afterwards — and a resubmitted `operation_id` replays
 * the original operation whatever its status, so those turns would never be sent again. Replacing
 * everything was self-healing precisely because it re-sent the whole document each time; appending
 * gives that up, and this bounds what a single lost write can cost to the turns since the last
 * re-sync rather than to the rest of the session.
 */
export const MAX_APPENDS_BEFORE_RESYNC = 20;

/**
 * Identity of the first `count` turns — every one of them, hashed incrementally.
 *
 * Sampling the ends is not enough: a transcript whose MIDDLE was rewritten (an edited or redacted
 * turn) still starts and ends the same way, and would then be appended to as though nothing had
 * changed. Hashing the whole prefix costs O(prefix) per retain, far less than the replace it avoids
 * — the same bytes through a digest instead of over the network — and it streams, so nothing here
 * materializes a second copy of the transcript.
 */
export function fingerprintTurns(turns: TransportTurn[], count: number): string {
  const h = createHash("sha1").update(String(count));
  for (let i = 0; i < count; i++) h.update("\n" + JSON.stringify(turns[i]));
  return h.digest("hex");
}

export type RetainPlan =
  | { mode: "replace" }
  | { mode: "append"; fromTurn: number }
  | { mode: "skip" };

/**
 * Decide how to write `turns` given what we last wrote. See the module doc for why every uncertain
 * case resolves to "replace" — the expensive-but-correct option — rather than to an append.
 */
export function planRetain(
  turns: TransportTurn[],
  cursor: RetainCursor | undefined,
  opts: { appendSupported: boolean; bank: string }
): RetainPlan {
  if (!turns.length) return { mode: "skip" };
  if (!opts.appendSupported || !cursor || cursor.dirty) return { mode: "replace" };
  // A different bank holds no document for this session: appending would store the tail alone.
  if (cursor.bank !== opts.bank) return { mode: "replace" };
  if ((cursor.appends ?? 0) >= MAX_APPENDS_BEFORE_RESYNC) return { mode: "replace" };
  // Fewer turns than we wrote: the transcript shrank, so it was rewritten, not extended.
  if (cursor.turns > turns.length) return { mode: "replace" };
  if (fingerprintTurns(turns, cursor.turns) !== cursor.fingerprint) return { mode: "replace" };
  if (cursor.turns === turns.length) return { mode: "skip" }; // nothing new since the last write
  return { mode: "append", fromTurn: cursor.turns };
}

/** Per-session cursor storage. Hook harnesses are a fresh process per event so theirs is file-backed
 *  (core/session-cache.ts); the persistent-plugin runtime keeps its own in memory. */
export interface RetainCursorStore {
  read(sessionId: string): RetainCursor | undefined;
  write(sessionId: string, cursor: RetainCursor): void;
}

/** In-memory store for a long-lived host (opencode/kilo). */
export function memoryCursorStore(): RetainCursorStore {
  const cursors = new Map<string, RetainCursor>();
  return {
    read: (sessionId) => cursors.get(sessionId),
    write: (sessionId, cursor) => void cursors.set(sessionId, cursor),
  };
}
