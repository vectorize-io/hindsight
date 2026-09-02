/**
 * Harness-agnostic chat memory: the JSON user/assistant transcript schema shared by BOTH the
 * backfill (ingest past sessions) and the live runtime write-back. A leading `system` turn carries
 * the REF-ID tracer; every turn gets an ABSOLUTE timestamp.
 */
import { RateLimitedError, type HindsightClient } from "./hindsight";
import { fingerprintTurns, planRetain, type RetainCursorStore } from "./retain-cursor";
import type { RetainStamp } from "./retain-stamp";
import type { ChatSession } from "./types";
import { uuidV5 } from "./uuid";
import { pool, sleep } from "./util";

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
  opts: {
    concurrency?: number;
    log?: (m: string) => void;
    stampFor?: (sessionId: string) => RetainStamp;
  } = {}
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
      const stamp = opts.stampFor?.(id);
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
        [...new Set([...(stamp?.tags ?? []), "source:chat"])],
        "conversation",
        {
          timestamp: baseIso,
          metadata: {
            ...stamp?.metadata,
            source: "chat",
            chat: id,
            ref_id: `chat:${id}`,
          },
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

/** Retry window for a caller that did not supply one — a long-lived host with no external clock. */
const DEFAULT_RETRY_WINDOW_MS = 60_000;

/** Attempts a rate-limited write-back may make before giving up. */
const RETAIN_RETRY_ATTEMPTS = 2;

/** What a retry must still leave room for: `req` aborts a request at 15s. Waiting past the point
 *  where the retry itself could not finish buys nothing. */
const REQUEST_BUDGET_MS = 15_000;

/**
 * Submit a write-back, retrying while the API is rate-limiting us AND our write is still the
 * newest one for this session.
 *
 * Retrying is safe rather than duplicative because the payload carries a deterministic
 * `operation_id`: an identical resubmission is collapsed into the original operation server-side.
 *
 * The staleness check is the important half. If another write-back has claimed the cursor since
 * ours — a newer turn, another process — then ours is superseded, and the newer one carries
 * everything we were sending (or replaces the document outright, because our failure left the
 * cursor dirty). Retrying then would spend a hook's remaining time re-sending content that is
 * already on its way.
 */
async function submitWithRetry(
  send: () => Promise<void>,
  isStillNewest: () => boolean,
  retryUntil: number
): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await send();
    } catch (e) {
      const limited = e instanceof RateLimitedError;
      if (!limited || attempt >= RETAIN_RETRY_ATTEMPTS || !isStillNewest()) throw e;
      // Either honour Retry-After in full or do not retry at all: waiting less than the server
      // asked would just earn another 429. Whether it fits is the CALLER's clock, not a constant —
      // a hook process is killed by its host at a known deadline, a long-lived runtime is not.
      const wait = e.retryAfterMs || 1000;
      if (Date.now() + wait + REQUEST_BUDGET_MS > retryUntil) throw e;
      await sleep(wait);
    }
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
  opts: { cursors?: RetainCursorStore; stamp?: RetainStamp; retryUntil?: number } = {}
): Promise<void> {
  const cursors = opts.cursors;
  if (!cursors)
    return writeSession(
      client,
      sessionId,
      turns,
      startTs,
      harness,
      opts.stamp,
      undefined,
      opts.retryUntil
    );
  // Serialised so the plan is made against the previous write-back's CONFIRMED cursor (see above).
  return serialize(cursors, sessionId, () =>
    writeSession(client, sessionId, turns, startTs, harness, opts.stamp, cursors, opts.retryUntil)
  );
}

async function writeSession(
  client: HindsightClient,
  sessionId: string,
  turns: TransportTurn[],
  startTs: string,
  harness?: string,
  stamp?: RetainStamp,
  cursors?: RetainCursorStore,
  /** Absolute time this write-back may keep retrying until; the caller owns its own clock. */
  retryUntil = Date.now() + DEFAULT_RETRY_WINDOW_MS
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

  const claimed = { ...next, dirty: true };
  await submitWithRetry(
    () =>
      client.retain(
        content,
        "coding agent session",
        refId,
        // Configured tags first, built-ins last and deduped: `source:chat` and `harness:<id>` are what
        // the documents list filters and draws its agent logo from, so a template cannot displace them.
        [
          ...new Set([
            ...(stamp?.tags ?? []),
            "source:chat",
            ...(harness ? [`harness:${harness}`] : []),
          ]),
        ],
        "conversation",
        {
          timestamp: startTs,
          updateMode: plan.mode === "append" ? "append" : undefined,
          // Identity of THIS payload: a resubmission of the same bytes is collapsed server-side into
          // the original operation instead of extracting (or appending) twice.
          operationId: uuidV5(`${client.bank}\n${refId}\n${plan.mode}\n${content}`),
          metadata: {
            ...stamp?.metadata, // configured first: the built-ins below win on any key collision
            source: "chat",
            session_id: sessionId,
            ref_id: refId,
            ...(harness ? { harness } : {}),
          },
        }
      ),
    // Still ours to retry only while the cursor holds the claim we wrote above.
    () => {
      if (!cursors) return true;
      const now = cursors.read(sessionId);
      return now?.turns === claimed.turns && now?.fingerprint === claimed.fingerprint;
    },
    retryUntil
  );
  cursors?.write(sessionId, next);
}
