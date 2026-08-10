/**
 * Harness-agnostic chat memory: the JSON user/assistant transcript schema shared by BOTH the
 * backfill (ingest past sessions) and the live runtime write-back. A leading `system` turn carries
 * the REF-ID tracer; every turn gets an ABSOLUTE timestamp.
 */
import type { HindsightClient } from "./hindsight";
import { fingerprintTurns, planRetain, type RetainCursorStore } from "./retain-cursor";
import type { ChatSession } from "./types";
import { uuidV5 } from "./uuid";
import { pool } from "./util";

export interface TransportTurn {
  role: string;
  content: string;
  timestamp?: string;
}

/** Prepend the REF-ID system turn to a set of already-normalized turns. */
export function withRefId(refId: string, turns: TransportTurn[], baseTs: string): TransportTurn[] {
  return [{ role: "system", content: `REF-ID: ${refId}`, timestamp: baseTs }, ...turns];
}

/**
 * Render normalized turns as a JSONL transcript (ONE turn per line) — the same shape everywhere:
 * live write-back and backfilled chats alike. JSONL beats a JSON array on both ends: appending a
 * turn never rewrites the document, and the server's structured chunker treats each line as an
 * atomic unit (`retain_structured_chunk_size`), so a turn is never split mid-thought. The REF-ID
 * system turn leads; tool activity is already compacted into `role:"action"` turns.
 */
export function renderSessionJsonl(refId: string, turns: TransportTurn[], baseTs: string): string {
  return withRefId(refId, turns, baseTs)
    .map((t) => JSON.stringify(t))
    .join("\n");
}

/** Backfill: ingest past sessions RAW as JSON transcripts under the `conversation` strategy. */
export async function ingestChats(
  client: HindsightClient,
  sessions: ChatSession[],
  opts: { concurrency?: number; log?: (m: string) => void } = {}
): Promise<number> {
  const log = opts.log ?? (() => {});
  if (!sessions.length) {
    log("[chat] no sessions; skipping");
    return 0;
  }
  log(`[chat] ingesting ${sessions.length} chats (RAW, JSONL transcript — one turn per line) …`);
  const NOW = Date.now(); // anchor synthesized times to a real, ABSOLUTE clock (not a fabricated epoch)
  let failures = 0;
  await pool(
    sessions,
    opts.concurrency ?? 8,
    async (s, i) => {
      const id = s.id || `s${i}`;
      // each turn gets an ABSOLUTE timestamp: its own if provided, else synthesized from the real clock,
      // staggered per session + 1 min/turn to preserve ordering. List order is CHRONOLOGICAL (a later
      // chat can amend an earlier one), so the LAST session is the newest — the previous `NOW - i*1h`
      // inverted recency and made an amendment rank older than the decision it superseded.
      const sessBase = NOW - (sessions.length - 1 - i) * 3600000;
      const baseIso = new Date(sessBase).toISOString();
      const turns = withRefId(
        `chat:${id}`,
        (s.turns || []).map((t, j) => ({
          role: t.role,
          content: t.text,
          timestamp: t.timestamp || new Date(sessBase + (j + 1) * 60000).toISOString(),
        })),
        baseIso
      );
      await client.retain(
        turns.map((x) => JSON.stringify(x)).join("\n"),
        "developer chat",
        `chat:${id}`,
        ["source:chat"],
        "conversation",
        {
          timestamp: baseIso,
          metadata: { source: "chat", chat: id, ref_id: `chat:${id}` },
        }
      );
    },
    (i, e) => {
      failures++;
      log(`  ! chat ${i} failed to enqueue: ${(e as Error).message?.slice(0, 120)}`);
    }
  );
  log(`[chat] done: ${sessions.length} chats ingested (JSONL) under strategy 'chat'`);
  return failures;
}

/** Capability probe for the append path. Any failure answers "no": a write-back must never be lost
 *  because we couldn't work out whether the cheaper form of it was available. */
async function supportsAppend(client: HindsightClient): Promise<boolean> {
  try {
    return await client.supportsIdempotentRetain();
  } catch {
    return false;
  }
}

/**
 * One write-back at a time per session, keyed by the store the cursors live in.
 *
 * The persistent-plugin runtime fires retains without awaiting them (a turn-driven one and an
 * idle-driven one can overlap), and reading the cursor is not atomic with claiming it — there is an
 * await in between. Two overlapping calls therefore both planned an append from the SAME position
 * and submitted overlapping slices, duplicating turns inside the document; and even serialising the
 * claim alone would leave an append racing a replace on the wire, where the order they land in
 * decides whether the result is correct. Chaining the whole read-plan-send-confirm cycle is what
 * makes the cursor mean what it says: the second call sees the first call's CONFIRMED position.
 */
const writeBacks = new WeakMap<RetainCursorStore, Map<string, Promise<void>>>();

function serialize(
  cursors: RetainCursorStore,
  sessionId: string,
  write: () => Promise<void>
): Promise<void> {
  const perSession = writeBacks.get(cursors) ?? new Map<string, Promise<void>>();
  writeBacks.set(cursors, perSession);
  // Chain off the previous write-back whether it succeeded or failed — a failure leaves the cursor
  // dirty, which the next call needs to see so it can replace rather than append.
  const next = (perSession.get(sessionId) ?? Promise.resolve()).then(write, write);
  const tail = next.catch(() => {});
  perSession.set(sessionId, tail);
  // Drop the entry once it is settled AND still the tail, so a host that outlives many sessions
  // (opencode runs for days) doesn't accumulate one resolved promise per session id forever.
  void tail.then(() => {
    if (perSession.get(sessionId) === tail) perSession.delete(sessionId);
  });
  return next;
}

/**
 * Live write-back: upsert a running session under a stable document_id, sending only what is new.
 *
 * Given a cursor store, a session that has already been written APPENDS the turns added since the
 * last successful write; the server concatenates them onto the stored document (with "\n", which is
 * why the transcript is JSONL) and re-chunks. Without one — first write, a rewritten transcript, an
 * unconfirmed previous write, or a server too old to be idempotent — it falls back to REPLACING the
 * whole document, which is what this always used to do. See core/retain-cursor.ts for why every
 * uncertain case resolves that way.
 *
 * Uses the same `conversation` strategy as backfilled chats — one strategy for all developer
 * conversations; the mission scales extraction to the substance. The content is a JSON transcript
 * (renderSessionJsonl) whose tool activity is compacted into `role:"action"` turns
 * (see core/transcript*.ts).
 */
export async function retainLiveSession(
  client: HindsightClient,
  sessionId: string,
  turns: TransportTurn[],
  startTs: string,
  harness?: string,
  opts: { cursors?: RetainCursorStore } = {}
): Promise<void> {
  const cursors = opts.cursors;
  if (!cursors) return writeSession(client, sessionId, turns, startTs, harness);
  // Serialised so the plan is made against the previous write-back's CONFIRMED cursor (see above).
  return serialize(cursors, sessionId, () =>
    writeSession(client, sessionId, turns, startTs, harness, cursors)
  );
}

async function writeSession(
  client: HindsightClient,
  sessionId: string,
  turns: TransportTurn[],
  startTs: string,
  harness?: string,
  cursors?: RetainCursorStore
): Promise<void> {
  const refId = `conversation:${sessionId}`;
  const appendSupported = Boolean(cursors) && (await supportsAppend(client));
  const plan = planRetain(turns, cursors?.read(sessionId), { appendSupported, bank: client.bank });
  if (plan.mode === "skip") return;

  const content =
    plan.mode === "append"
      ? turns
          .slice(plan.fromTurn)
          .map((t) => JSON.stringify(t))
          .join("\n")
      : renderSessionJsonl(refId, turns, startTs);

  // Claim the new position BEFORE the write and mark it unconfirmed, so a client that times out on
  // a request the server did commit replaces next time instead of appending the same turns twice.
  const next = {
    turns: turns.length,
    fingerprint: fingerprintTurns(turns, turns.length),
    bank: client.bank,
  };
  cursors?.write(sessionId, { ...next, dirty: true });

  await client.retain(
    content,
    "coding agent session",
    refId,
    ["source:chat", ...(harness ? [`harness:${harness}`] : [])],
    "conversation",
    {
      timestamp: startTs,
      updateMode: plan.mode === "append" ? "append" : undefined,
      // Identity of THIS payload: a resubmission of the same bytes is collapsed server-side into
      // the original operation instead of extracting (or appending) twice.
      operationId: uuidV5(`${client.bank}\n${refId}\n${plan.mode}\n${content}`),
      metadata: {
        source: "chat",
        session_id: sessionId,
        ref_id: refId,
        ...(harness ? { harness } : {}),
      },
    }
  );
  cursors?.write(sessionId, next);
}
