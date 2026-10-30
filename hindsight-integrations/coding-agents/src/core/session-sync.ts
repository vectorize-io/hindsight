/**
 * The conversation as a Hindsight session (servers with `POST .../sessions/{id}/events`): the
 * plugin only streams the conversation and hands notes over. Saving the conversation into memory
 * and, with `autoInject: "live"`, the live session (what a moment is, what the agent was already
 * told, the reflect question) all happen on the server.
 *
 * One sync reads the transcript's new complete lines, sends them as session messages (plus the
 * working tree's diff when a tool just ran and it changed), and returns the notes the server has
 * not handed out yet. Every step is safe to repeat: messages carry ids from the transcript, so a
 * line sent twice is stored once, and the transcript offset only moves once the server took it.
 *
 * On a server without sessions the first sync answers `unsupported`: the Stop hook then writes the
 * conversation back itself (core/retain-hook.ts), and there is no live session.
 */
import { createHash } from "node:crypto";
import {
  closeSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { conversationDocument } from "./chat";
import { isLive, type Config } from "./config";
import { resolveRetainScopes, type HindsightClient } from "./hindsight";
import { buildRetainStamp } from "./retain-stamp";
import type { EventsOf, SessionMessage, SessionNote, SessionOptions } from "./session-events";
import { sleep } from "./util";
import { renderNote, sourceDiff } from "./live";

/** A crashed or killed holder leaves its lock behind; past this age anyone may take it. */
const STALE_LOCK_MS = 60_000;
/** How long a sync waits for another one to finish: the prompt and turn end it carries
 *  exist only in this process, so it waits rather than drop them. */
const LOCK_WAIT_MS = 5_000;
const APPEND_TIMEOUT_MS = 10_000;
/** Messages per append: a long transcript read for the first time goes in a few requests. */
const BATCH = 200;

export type SessionClient = Pick<HindsightClient, "appendSession">;

/** What the plugin keeps per session between hooks. */
export interface SessionSyncState {
  /** Bytes of the transcript already sent. */
  offset: number;
  /** The last note handed to the hooks to deliver; sent back as `ack`. */
  ack: number;
  /** The working tree's diff last sent, by hash. */
  diffHash: string;
  /** Whether the session's options were sent (the first append creates the session). */
  opened: boolean;
  /** The bank the session was opened in. A session that moves into another repo continues in that
   *  repo's bank, where it is a new session: its options go again, or it starts there with live off
   *  and no harness (seen in beta: sessions that cd'd between repos had 0 checks in the second). */
  bank?: string;
  /** The server has no sessions: no live session, and the plugin writes the conversation back itself. */
  unsupported: boolean;
  /** Prompts the prompt hook already sent, by hash, so the transcript's copy is not sent again. */
  prompts: string[];
  /** Final replies the Stop hook already sent, by hash, likewise. */
  replies: string[];
}

export const emptySyncState = (): SessionSyncState => ({
  offset: 0,
  ack: 0,
  diffHash: "",
  opened: false,
  unsupported: false,
  prompts: [],
  replies: [],
});

const hash = (s: string): string => createHash("sha1").update(s).digest("hex");
/** A prompt or reply as both its hook and the transcript see it: a hook gets it trimmed, the
 *  transcript keeps its newline. */
const textHash = (s: string): string => hash(s.trim());

function loadState(file: string): SessionSyncState {
  try {
    return {
      ...emptySyncState(),
      ...(JSON.parse(readFileSync(file, "utf8")) as Partial<SessionSyncState>),
    };
  } catch {
    return emptySyncState();
  }
}

/** Whether this session already found the server without sessions (read without the lock). */
export function sessionsUnsupported(dir: string): boolean {
  return loadState(join(dir, "session.json")).unsupported;
}

async function lock(path: string, waitMs: number): Promise<boolean> {
  const deadline = Date.now() + waitMs;
  for (;;) {
    try {
      closeSync(openSync(path, "wx"));
      return true;
    } catch {
      try {
        if (Date.now() - statSync(path).mtimeMs > STALE_LOCK_MS) {
          rmSync(path, { force: true });
          continue;
        }
      } catch {
        continue; // released between our two calls: try again at once
      }
      if (Date.now() >= deadline) return false;
      await sleep(50);
    }
  }
}

/** The transcript's new complete lines as messages; a half-written last line waits for next time. */
function readNew(
  transcript: string | undefined,
  state: SessionSyncState,
  eventsOf: EventsOf
): { messages: SessionMessage[]; offset: number } {
  if (!transcript) return { messages: [], offset: state.offset };
  let buf: Buffer;
  try {
    buf = readFileSync(transcript);
  } catch {
    return { messages: [], offset: state.offset };
  }
  // A transcript shorter than what was sent was replaced (a new file under the same path): start over;
  // ids from the lines themselves keep the server from storing what it already has.
  const from = buf.length < state.offset ? 0 : state.offset;
  const chunk = buf.subarray(from);
  const end = chunk.lastIndexOf(0x0a);
  if (end < 0) return { messages: [], offset: from };
  const messages: SessionMessage[] = [];
  let pos = from;
  for (const raw of chunk.subarray(0, end).toString("utf8").split("\n")) {
    const at = pos;
    pos += Buffer.byteLength(raw) + 1;
    let line: unknown;
    try {
      line = JSON.parse(raw);
    } catch {
      continue;
    }
    if (!line || typeof line !== "object") continue;
    for (const m of eventsOf(line as Record<string, unknown>, at)) {
      // The prompt hook sent this request already, under its own id; the Stop hook this reply.
      const seen =
        m.content === undefined
          ? null
          : m.role === "user"
            ? state.prompts
            : m.role === "assistant"
              ? state.replies
              : null;
      const sent = seen ? seen.indexOf(textHash(m.content ?? "")) : -1;
      if (seen && sent >= 0) {
        seen.splice(sent, 1);
        continue;
      }
      messages.push(m);
    }
  }
  return { messages, offset: from + end + 1 };
}

export interface SyncResult {
  /** Rendered notes to deliver, oldest first. */
  notes: string[];
  /** The server has no sessions. */
  unsupported: boolean;
  /** The server is still checking the latest moment: a note may follow. */
  checking?: boolean;
}

/**
 * Send the session what happened since the last sync, and take the notes it has for the agent.
 * The prompt being submitted and the turn's end are what this hook knows and the transcript may
 * not have yet (the prompt is also recorded so its transcript copy is skipped).
 */
export async function syncSession(args: {
  sessionId: string;
  transcriptPath?: string;
  cwd: string;
  dir: string;
  /** The bank this append goes to (the client's); a change reopens the session there. */
  bankId?: string;
  client: SessionClient;
  eventsOf: EventsOf;
  /** How the session is set up; built only for the append that opens it. */
  options: () => SessionOptions;
  prompt?: { text: string };
  turnEnd?: boolean;
  /** The turn's final reply, from the Stop hook: the transcript may not hold it yet when the turn
   *  ends (Claude Code writes it after Stop runs), and it is the message that credits memory. */
  reply?: string;
  diff?: (cwd: string) => string;
}): Promise<SyncResult> {
  const none: SyncResult = { notes: [], unsupported: false };
  mkdirSync(args.dir, { recursive: true });
  const lockFile = join(args.dir, "session.lock");
  const stateFile = join(args.dir, "session.json");
  if (!(await lock(lockFile, LOCK_WAIT_MS))) return none;
  try {
    const state = loadState(stateFile);
    if (state.unsupported) return { ...none, unsupported: true };
    if (args.bankId && state.bank !== args.bankId) {
      // Only what happens from here on goes to the new bank: the offset stays, so the transcript
      // already sent is not sent (and saved, and paid for) a second time.
      if (state.bank) state.opened = false;
      state.bank = args.bankId;
    }
    const read = readNew(args.transcriptPath, state, args.eventsOf);
    const messages = read.messages;
    if (args.prompt?.text.trim()) {
      const h = textHash(args.prompt.text);
      state.prompts = [...state.prompts, h].slice(-20);
      messages.push({
        id: `prompt:${h.slice(0, 16)}:${read.offset}`,
        role: "user",
        content: args.prompt.text,
      });
    }
    // The change so far rides along whenever a tool just ran and it moved: the server checks
    // an edit against the latest one.
    let diffState: { diff: string } | undefined;
    if (messages.some((m) => m.role === "tool")) {
      const diff = (args.diff ?? sourceDiff)(args.cwd);
      const h = hash(diff);
      if (h !== state.diffHash) {
        diffState = { diff };
        state.diffHash = h;
      }
    }
    const reply = args.turnEnd ? args.reply?.trim() : undefined;
    if (reply) {
      const h = textHash(reply);
      const inTranscript = messages.some(
        (m) => m.role === "assistant" && m.content !== undefined && textHash(m.content) === h
      );
      if (!inTranscript) {
        state.replies = [...state.replies, h].slice(-20);
        messages.push({ id: `reply:${h.slice(0, 16)}`, role: "assistant", content: reply });
      }
    }

    const notes: SessionNote[] = [];
    let checking = false;
    // At least one append, even with nothing new: it is also how notes written since are fetched.
    // The diff and the turn end go with the last one, after every message they follow.
    for (let i = 0; i === 0 || i < messages.length; i += BATCH) {
      const last = i + BATCH >= messages.length;
      const r = await args.client.appendSession(
        args.sessionId,
        {
          messages: messages.slice(i, i + BATCH),
          ...(last && diffState ? { state: diffState } : {}),
          ...(last && args.turnEnd ? { turn_end: true } : {}),
          ack: state.ack,
          ...(state.opened ? {} : { session: args.options() }),
        },
        APPEND_TIMEOUT_MS
      );
      if (!r.supported) {
        state.unsupported = true;
        writeFileSync(stateFile, JSON.stringify(state));
        return { ...none, unsupported: true };
      }
      state.opened = true;
      checking = r.checking === true;
      for (const n of r.notes)
        if (n.seq > state.ack && !notes.some((x) => x.seq === n.seq)) notes.push(n);
    }
    // Sent: the transcript is read up to here. Handed over: the notes are this hook's to deliver.
    state.offset = read.offset;
    notes.sort((a, b) => a.seq - b.seq);
    if (notes.length) state.ack = notes[notes.length - 1].seq;
    writeFileSync(stateFile, JSON.stringify(state));
    return { notes: notes.map(renderNote), unsupported: false, checking };
  } finally {
    rmSync(lockFile, { force: true });
  }
}

/**
 * The session as this plugin sets it up: the live session on with `autoInject: "live"`, and the
 * conversation saved at the end of each turn into the document the plugin has always written
 * (`conversation:<session>`), filed exactly as the plugin's own write-back files it — so a bank
 * reads the same whether the plugin or the session wrote it. `retainSessions: false` turns the
 * saving off.
 */
export function sessionOptions(args: {
  cfg: Config;
  harness: string;
  sessionId: string;
  bankId: string;
  cwd: string;
  sessionRoot?: string;
}): SessionOptions {
  const { cfg, harness, sessionId, bankId } = args;
  const doc = conversationDocument(
    sessionId,
    harness,
    buildRetainStamp(cfg, {
      directory: args.cwd,
      sessionRoot: args.sessionRoot,
      harness,
      bankId,
      sessionId,
    })
  );
  return {
    document_id: doc.documentId,
    live: isLive(cfg, harness),
    retain: cfg.retainSessions === false ? "off" : "turns",
    tags: doc.tags,
    metadata: doc.metadata,
    context: doc.context,
    strategy: "conversation",
    observation_scopes: resolveRetainScopes(doc.tags, cfg.observationScopes),
  };
}
