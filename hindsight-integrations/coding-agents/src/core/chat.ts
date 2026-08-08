/**
 * Harness-agnostic chat memory: the JSON user/assistant transcript schema shared by BOTH the
 * backfill (ingest past sessions) and the live runtime write-back. A leading `system` turn carries
 * the REF-ID tracer; every turn gets an ABSOLUTE timestamp.
 */
import type { HindsightClient } from "./hindsight";
import type { ChatSession } from "./types";
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

/**
 * Live write-back: upsert a running session under a stable document_id. Same id => Hindsight
 * reprocesses the FULL conversation, so the settled decision is extracted from the whole thing.
 *
 * Uses the same `conversation` strategy as backfilled chats — one strategy for all developer
 * conversations; the mission scales extraction to the substance. The content is a JSON transcript
 * (renderSessionJson) whose tool activity is compacted into `role:"action"` turns
 * (see core/transcript*.ts).
 */
export async function retainLiveSession(
  client: HindsightClient,
  sessionId: string,
  turns: TransportTurn[],
  startTs: string,
  harness?: string
): Promise<void> {
  const refId = `conversation:${sessionId}`;
  await client.retain(
    renderSessionJsonl(refId, turns, startTs),
    "coding agent session",
    refId,
    ["source:chat", ...(harness ? [`harness:${harness}`] : [])],
    "conversation",
    {
      timestamp: startTs,
      async: true,
      idempotent: true,
      metadata: {
        source: "chat",
        session_id: sessionId,
        ref_id: refId,
        ...(harness ? { harness } : {}),
      },
    }
  );
}
