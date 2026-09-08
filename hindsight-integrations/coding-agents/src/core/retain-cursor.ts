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
 *   - fingerprint drift  the transcript was REWRITTEN rather than extended (an edited or redacted
 *                        turn, a truncated/rotated rollout), so what we already sent is no longer a
 *                        prefix of what we hold and an append would splice two conversations.
 *                        NOT Claude Code compaction, which this used to cite: that appends a
 *                        summary record and leaves every earlier record in place (#3379), so the
 *                        prefix stays intact and the append path keeps working.
 *   - dirty              a REPLACE was started and not confirmed. There is nothing worth replaying
 *                        (another replace re-establishes the same truth from the same transcript),
 *                        so the next write-back simply replaces again.
 *
 * An APPEND that fails does NOT fall back to replace, because that fallback is what made a single
 * outage cost a full re-extraction on every subsequent Stop for the life of the session (#3989).
 * Its bytes are buffered in `pending` instead and replayed on the next write-back, unchanged and
 * under their original operation id — so a write the server actually committed is collapsed into
 * that same operation rather than appending the turns twice, which is the only thing appending onto
 * an unknown state could otherwise get wrong. The buffer is an optimisation, never the source of
 * truth: whenever it cannot be trusted (see `pendingReplayable`) the cursor falls back to replace.
 *
 * The claim (`dirty`, or `pending` plus the advanced position) is written BEFORE the request and
 * settled after it, so an overlapping retain (the runtime fires them without awaiting) sees the
 * advanced cursor and does not re-send the same slice.
 */
import { createHash } from "node:crypto";
import type { TransportTurn } from "./chat";

/** One append that was built and submitted but never confirmed. */
export interface PendingAppend {
  /** Its EXACT bytes. Replayed unchanged — different bytes would be a different operation, and the
   *  server's dedupe (which is what makes a replay safe at all) would not apply to them. */
  content: string;
  /** The operation id it went out under, replayed unchanged for the same reason. */
  operationId: string;
  /** When it was buffered, so a replay cannot outlive the server's operation retention. */
  at: number;
}

/** A buffered append is only replayable while the server still holds the operation it would collapse
 *  into. Operations are kept forever by default (`HINDSIGHT_API_OPERATION_RETENTION_DAYS=0`), but an
 *  operator can prune them, and a replay after that prune would append the same turns a second time
 *  — so buffered bytes expire well inside any retention an operator would plausibly configure. */
export const PENDING_MAX_AGE_MS = 12 * 60 * 60 * 1000;
/** Past this much buffered content a replace costs about the same as the replay and is simpler, so
 *  the buffer stops growing and the cursor falls back to it. Also bounds the cursor file. */
export const PENDING_MAX_BYTES = 256 * 1024;

/** Whether a cursor's buffered appends can still be replayed, or the write-back must replace. */
export function pendingReplayable(cursor: RetainCursor, now: number): boolean {
  const pending = cursor.pending ?? [];
  if (!pending.length) return true;
  if (pending.some((p) => now - p.at > PENDING_MAX_AGE_MS)) return false;
  return pending.reduce((n, p) => n + p.content.length, 0) <= PENDING_MAX_BYTES;
}

export interface RetainCursor {
  /** Turns the document holds once every `pending` entry has been applied — what the next append
   *  must start from, whether those turns are already committed or still buffered. */
  turns: number;
  /** Identity of the written prefix — detects a rewritten transcript (see module doc). */
  fingerprint: string;
  /** The bank the document was written to. The cursor is keyed by session, but the bank is derived
   *  per hook invocation from that event's cwd — a session that moves between repos (#3133) keeps
   *  its id and changes bank, and the new bank holds no document to append to. */
  bank: string;
  /** A REPLACE was started and not confirmed: the next retain must replace, not append. */
  dirty?: boolean;
  /** Appends started and not confirmed, oldest first — replayed before anything new. Their turns
   *  are already counted in `turns`: the cursor covers what is committed OR buffered. */
  pending?: PendingAppend[];
}

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
  opts: { appendSupported: boolean; bank: string; now?: number }
): RetainPlan {
  if (!turns.length) return { mode: "skip" };
  if (!opts.appendSupported || !cursor || cursor.dirty) return { mode: "replace" };
  // Buffered appends we can no longer replay safely: replace subsumes them (and the caller drops
  // them), so the document is rebuilt from the transcript rather than left with a gap.
  if (!pendingReplayable(cursor, opts.now ?? Date.now())) return { mode: "replace" };
  // A different bank holds no document for this session: appending would store the tail alone.
  if (cursor.bank !== opts.bank) return { mode: "replace" };
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
