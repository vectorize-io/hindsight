/**
 * Harness-agnostic chat memory: the JSON user/assistant transcript schema shared by BOTH the
 * backfill (ingest past sessions) and the live runtime write-back. A leading `system` turn carries
 * the REF-ID tracer; every turn gets an ABSOLUTE timestamp.
 */
import type { HindsightClient } from "./hindsight";
import type { ChatSession } from "./types";
import { pool } from "./util";

export interface TransportTurn { role: string; content: string; timestamp?: string }

/** Prepend the REF-ID system turn to a set of already-normalized turns. */
export function withRefId(refId: string, turns: TransportTurn[], baseTs: string): TransportTurn[] {
  return [{ role: "system", content: `REF-ID: ${refId}`, timestamp: baseTs }, ...turns];
}

/** Backfill: ingest past sessions RAW as JSON transcripts under the `chat` strategy. */
export async function ingestChats(
  client: HindsightClient,
  sessions: ChatSession[],
  opts: { concurrency?: number; log?: (m: string) => void } = {},
): Promise<number> {
  const log = opts.log ?? (() => {});
  if (!sessions.length) { log("[chat] no sessions; skipping"); return 0; }
  log(`[chat] ingesting ${sessions.length} chats (RAW, JSON user/assistant transcript) …`);
  const NOW = Date.now(); // anchor synthesized times to a real, ABSOLUTE clock (not a fabricated epoch)
  let failures = 0;
  await pool(sessions, opts.concurrency ?? 8, async (s, i) => {
    const id = s.id || `s${i}`;
    // each turn gets an ABSOLUTE timestamp: its own if provided, else synthesized from the real clock,
    // staggered per session (1h back each) + 1 min/turn to preserve ordering.
    const sessBase = NOW - i * 3600000;
    const baseIso = new Date(sessBase).toISOString();
    const turns = withRefId(
      `chat:${id}`,
      (s.turns || []).map((t, j) => ({
        role: t.role, content: t.text,
        timestamp: t.timestamp || new Date(sessBase + (j + 1) * 60000).toISOString(),
      })),
      baseIso,
    );
    await client.retain(JSON.stringify(turns), "developer chat", `chat:${id}`, ["source:chat"], "chat", {
      timestamp: baseIso, metadata: { source: "chat", chat: id, ref_id: `chat:${id}` },
    });
  }, (i, e) => { failures++; log(`  ! chat ${i} failed to enqueue: ${(e as Error).message?.slice(0, 120)}`); });
  log(`[chat] done: ${sessions.length} chats ingested (JSON) under strategy 'chat'`);
  return failures;
}

/**
 * Live write-back: upsert a running session under a stable document_id. Same id => Hindsight
 * reprocesses the FULL conversation, so the settled decision is extracted from the whole thing.
 */
export async function retainLiveSession(
  client: HindsightClient, sessionId: string, turns: TransportTurn[], startTs: string,
): Promise<void> {
  const withRef = withRefId(`conversation:${sessionId}`, turns, startTs);
  await client.retain(
    JSON.stringify(withRef), "coding agent session", `conversation:${sessionId}`,
    ["source:chat"], "chat",
    { timestamp: startTs, async: true,
      metadata: { source: "chat", session_id: sessionId, ref_id: `conversation:${sessionId}` } },
  );
}
