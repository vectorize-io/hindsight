/**
 * A per-session conversation journal, for hook harnesses whose HOST keeps no durable transcript.
 *
 * Every other hook harness hands its Stop hook a file holding the whole conversation, which is what
 * the incremental write-back needs: `retainLiveSession` re-reads the FULL transcript each time and
 * lets the retain cursor (core/retain-cursor.ts) send only the turns added since the last write. A
 * host that exposes just the latest reply breaks that contract — the turn list would shrink and
 * reorder between Stops, so every write-back would fingerprint-mismatch and replace.
 *
 * ZCode is that host: its `Stop` payload carries the assistant reply (`responseText`) plus an
 * EPHEMERAL, assistant-only transcript that the agent deletes as soon as the hook returns, and it
 * carries no user prompt at all. So the plugin keeps the conversation itself — the prompt hook
 * appends the user turn, the Stop hook appends the reply — and the Stop hook then reads this file
 * exactly as the other harnesses read the host's own transcript. Nothing downstream changes.
 *
 * It lives beside the session cache and the retain cursor in the temp dir, in its own file and for
 * the same reason (see core/session-cache.ts): two writers with different lifecycles must not share
 * a record. Losing it costs the un-retained tail of one session, never correctness — a shorter
 * journal simply replaces the document instead of appending to it.
 *
 * Fail-open throughout: a journal that cannot be written or read yields no turns, which is exactly
 * the behaviour of a harness whose transcript file is missing.
 */
import { appendFileSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { TransportTurn } from "./chat";
import { stripInjectedMemory } from "./transcript-util";

/**
 * Ceiling on one session's journal. Past it the journal stops GROWING rather than being trimmed
 * from the front: dropping the oldest turns would rewrite the prefix the retain cursor fingerprints,
 * turning every later write-back into a full replace of a conversation already too large to resend.
 */
export const JOURNAL_MAX_BYTES = 8 * 1024 * 1024;

/** The journal file for one session. Same directory as the session cache and the retain cursor. */
export function journalPath(harness: string, sessionId: string | undefined): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId || "no-session"}.journal.jsonl`);
}

/**
 * Append one turn, unless it is empty or repeats the turn already at the end of the journal.
 *
 * The dedupe is load-bearing rather than tidy: a host may deliver Stop more than once for the same
 * reply (ZCode fires it again after a cancelled continuation), and the same text appended twice
 * would be retained as two assistant turns that never happened.
 */
export function appendJournalTurn(path: string, turn: TransportTurn): void {
  const content = stripInjectedMemory(turn.content ?? "").trim();
  if (!content) return;
  try {
    if (statSync(path).size >= JOURNAL_MAX_BYTES) return;
  } catch {
    /* no journal yet — this turn opens it */
  }
  const last = readJournalTranscript(path).at(-1);
  if (last && last.role === turn.role && last.content === content) return;
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(
      path,
      JSON.stringify({
        role: turn.role,
        content,
        timestamp: turn.timestamp ?? new Date().toISOString(),
      }) + "\n"
    );
  } catch {
    /* best-effort: an unwritable journal costs the tail of one session, never an error */
  }
}

/** Read a journal back into normalized turns — the `TranscriptReader` shape retain-hook expects. */
export function readJournalTranscript(path: string): TransportTurn[] {
  let body: string;
  try {
    body = readFileSync(path, "utf8");
  } catch {
    return [];
  }
  const turns: TransportTurn[] = [];
  for (const line of body.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue; // a torn final line (the process died mid-append) costs that turn, not the session
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const { role, content, timestamp } = parsed as Record<string, unknown>;
    if (typeof role !== "string" || typeof content !== "string" || !content) continue;
    turns.push({
      role,
      content,
      ...(typeof timestamp === "string" ? { timestamp } : {}),
    });
  }
  return turns;
}
