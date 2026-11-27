/**
 * Shared runtime for the Claude Code `Stop` hook: when a session ends, read its transcript,
 * normalize it, and write it back into the bank so the session compounds into memory. The retain
 * half of the plugin (the `UserPromptSubmit` hook in core/hook.ts is the recall half).
 *
 * Write-back is ON by default, and `retainSessions: false` turns it off for the scope it is set on
 * (global, `harnesses.<name>`, or the resolved bank's `banks.<id>` section) — the same flag the
 * opencode persistent-plugin honors for its mid-session cadence (see core/runtime.ts). It gates
 * ONLY the transcript write-back: recall, git ingest, seeding and the memory tools keep working,
 * which is what separates it from the `disabled` kill switch.
 *
 * The pure logic lives in `buildRetain` (path + client in, void out) so it's unit-testable
 * without stdin; `runRetainHook` is thin plumbing around it, mirroring `runHook`/`buildHookOutput`
 * in core/hook.ts.
 */
import { existsSync } from "node:fs";
import { deriveBankIdOrSkip } from "./bank";
import { REQUEST_BUDGET_MS, retainLiveSession } from "./chat";
import { applyBankConfig, loadConfig } from "./config";
import { DAEMON_WAIT_RETAIN_MS, ensureDaemon } from "./daemon";
import { diag, readHookEvent } from "./diag";
import { describeError, log, setLogLevel } from "./log";
import type { Config } from "./config";
import type { ClientOpts } from "./hindsight";
import { HindsightClient } from "./hindsight";
import type { RetainCursorStore } from "./retain-cursor";
import { buildRetainStamp, type RetainStamp } from "./retain-stamp";
import {
  fileCursorStore,
  fileUsageCursorStore,
  orphanedCursors,
  sessionRootDir,
  type OrphanedCursor,
} from "./session-cache";
import { readClaudeTranscript } from "./transcript";
import { appendJournalTurn, journalPath, readJournalTranscript } from "./turn-journal";
import { stripInjectedMemory } from "./transcript-util";
import { recordUsage, type UsageCursorStore } from "./usage";

/** Headroom left before the host's kill: the response still has to come back after the last wait. */
const HOST_DEADLINE_MARGIN_MS = 2000;
import type { TransportTurn } from "./chat";

export interface RetainHookEventFields {
  sessionId?: string;
  transcriptPath?: string;
  cwd?: string;
  /** Dcode's materialized transcript may lag the just-finished assistant response. */
  lastAssistantMessage?: string;
}

/** Read a harness's transcript file into normalized turns. Claude and Codex use different JSONL
 *  schemas, so each harness supplies its own reader (default: Claude). */
export type TranscriptReader = (path: string) => TransportTurn[];

/** Normalize a harness's `lastAssistantMessage` into the same text its transcript reader would
 *  produce for that message. Harness-specific because the field is not always prose: Dcode sends
 *  `str(content)`, a Python repr, whenever the provider returns content blocks. Defaults to
 *  identity for harnesses that send the reply verbatim. */
export type LastMessageReader = (raw: string) => string;

export interface RetainHookSpec {
  /** Harness name — config `harnesses.<name>` section, {harness} template field, diag records. */
  harness: string;
  /** Seconds the HOST allows this hook before killing it — the same number the installer writes
   *  into its hook registration. It varies (60s for Claude Code and Codex, 30s for Cursor and
   *  Antigravity), and it is the only honest basis for deciding how long a rate-limited write-back
   *  may wait: past it the process is killed mid-write, which is worse than deferring. */
  hostTimeoutSec: number;
  /** Read the fields out of the harness's stdin event (shapes differ per harness). */
  parse(event: Record<string, unknown>): RetainHookEventFields;
  /** Optional event gate for one entry point registered on multiple host events. Evaluated before
   *  config loading or daemon startup, so irrelevant notifications stay cheap no-ops. */
  accept?(event: Record<string, unknown>): boolean;
  /** Harness-specific transcript parser. Defaults to the Claude JSONL reader. */
  readTranscript?: TranscriptReader;
  /** Harness-specific decoder for `lastAssistantMessage`. Defaults to identity. */
  readLastMessage?: LastMessageReader;
  /** Which client started the session, read from the transcript — gates `skipOriginators`. Only
   *  Codex records one (core/transcript-codex.ts). */
  originator?(transcriptPath: string): string | undefined;
  /**
   * Hosts that expose no durable transcript retain from the plugin's OWN per-session journal
   * (core/turn-journal.ts) instead of a host file: the prompt hook has already appended the user
   * turn, and this closes the turn with the reply read out of the Stop event.
   *
   * Declaring it here rather than in `parse` is what keeps `parse` pure — resolving the reply is a
   * read of the ephemeral transcript the host is about to delete, and closing the turn is a write.
   */
  journal?: {
    /** The assistant reply for this turn, from the Stop event. "" when the host sent none. */
    assistantText(event: Record<string, unknown>): string;
  };
}

/** Minimal client shape `buildRetain` needs — `HindsightClient` satisfies it structurally. The
 *  capability probe is part of the contract: appending to a document is only safe against a server
 *  that can deduplicate a resubmitted write (see core/retain-cursor.ts), into a bank that keeps
 *  document text to append to (#4613). */
interface RetainClient {
  retain: HindsightClient["retain"];
  supportsAppendRetain: HindsightClient["supportsAppendRetain"];
}

/**
 * Pure retain logic: read the transcript, and if it has any usable turns, upsert the full
 * conversation under `conversation:<sessionId>`. A transcript with no usable turns (e.g. only
 * tool calls / meta lines) is a no-op — nothing worth remembering. Fail-open: never throws.
 */
export async function buildRetain(args: {
  harness: string;
  sessionId: string;
  transcriptPath: string;
  /** Absent when write-back is off: usage is still recorded, nothing is retained (#5391). */
  client?: RetainClient;
  readTranscript?: TranscriptReader;
  lastAssistantMessage?: string;
  readLastMessage?: LastMessageReader;
  /** Configured retainTags/retainMetadata, already resolved for this session (core/retain-stamp.ts). */
  stamp?: RetainStamp;
  /** Injectable for tests; defaults to the per-session temp file (a Stop hook has no memory). */
  cursors?: RetainCursorStore;
  /** Absolute time the host will kill this process; bounds any rate-limit retry. */
  retryUntil?: number;
  /** Resolved bank, recorded on the usage line. */
  bankId?: string;
  /** Injectable for tests; defaults to the per-session temp file. */
  usageCursors?: UsageCursorStore;
}): Promise<void> {
  const { harness, sessionId, transcriptPath, client } = args;
  const readTranscript = args.readTranscript ?? readClaudeTranscript;

  const turns = readTranscript(transcriptPath);
  // Decode BEFORE stripping/trimming: the raw field can be a serialized content-block list rather
  // than prose (see LastMessageReader), and the injected-memory tags live inside its text blocks.
  const decoded = args.lastAssistantMessage
    ? (args.readLastMessage ?? ((raw: string) => raw))(args.lastAssistantMessage)
    : "";
  const lastAssistantMessage = stripInjectedMemory(decoded).trim();
  // Dcode materializes before Stop handlers run, so its final response can be absent from the
  // file. Dedupe by adjacent content because the same response is present after a flush on some
  // runs; this keeps repeated Stop delivery idempotent without dropping a legitimate later reply.
  // The decode above is what makes that compare meaningful — both sides now join text blocks the
  // same way, so an already-flushed reply matches instead of being appended twice.
  if (lastAssistantMessage && turns.at(-1)?.content !== lastAssistantMessage) {
    turns.push({
      role: "assistant",
      content: lastAssistantMessage,
      timestamp: new Date().toISOString(),
    });
  }
  if (turns.length === 0) return;
  // Stop fires once the reply is finished, so every turn in the transcript is complete.
  recordUsage({
    harness,
    sessionId,
    bankId: args.bankId ?? "",
    turns,
    cursors: args.usageCursors ?? fileUsageCursorStore(harness),
    lastTurnComplete: true,
    // The host has usually not flushed this turn's final reply yet — and that is
    // the message carrying the credit line. Re-emit the previous turn so the next
    // Stop, reading a complete transcript, corrects it. See recordUsage.
    reviseLastTurn: true,
  });

  if (!client) return;
  const startTs = turns[0]?.timestamp ?? new Date().toISOString();
  const t0 = Date.now();
  try {
    await retainLiveSession(client as HindsightClient, sessionId, turns, startTs, harness, {
      cursors: args.cursors ?? fileCursorStore(harness),
      stamp: args.stamp,
      retryUntil: args.retryUntil,
      transcriptPath,
    });
    diag(harness, "retain_ok", { ms: Date.now() - t0, turns: turns.length, session: sessionId });
  } catch (e) {
    log.warn(harness, "session write-back failed", {
      error: describeError(e),
    });
    diag(harness, "retain_failed", {
      ms: Date.now() - t0,
      error: describeError(e),
      session: sessionId,
    });
  }
}

/** A session must be this quiet before another one replays its write-back: a session that is
 *  still running retries on its own next Stop, and two processes writing one document race. */
const ORPHAN_IDLE_MS = 30 * 60 * 1000;
/** Past this a failed session is given up on: its transcript is likely gone. */
const ORPHAN_MAX_AGE_MS = 3 * 24 * 60 * 60 * 1000;

/**
 * Re-send the write-backs of this harness's OTHER sessions whose last one failed (#6097).
 *
 * The cursor of a failed session says so (`dirty` or `pending`), but only that session's own next
 * Stop reads it — a session that ended on a failure kept its last turns out of the bank forever.
 * Any later Stop of the same harness now picks it up: the transcript is re-read with this harness's
 * own reader and written exactly as that session's next Stop would have, through the same cursor,
 * so a write that did land collapses into its original operation server-side.
 *
 * ponytail: the idle window is the only guard against replaying a session that is still running in
 * another process; a session resumed at the very moment of a replay can race it. A per-session
 * lock file would close that if it is ever seen.
 */
export async function replayOrphanedSessions(args: {
  harness: string;
  /** The session this Stop is for: its own write-back already ran. */
  currentSessionId: string;
  /** Config before any bank override: each orphan is resolved against its OWN bank. */
  cfg: Config;
  /** Defaults to a plain client: each orphan's bank is only known here. */
  makeClient?: (opts: ClientOpts) => RetainClient;
  readTranscript?: TranscriptReader;
  /** Reads back from the HOST an orphan whose cursor names no transcript file: the persistent-plugin
   *  runtime holds its turns in memory (core/runtime.ts). Absent, such an orphan is skipped. */
  fetchTurns?: (sessionId: string) => Promise<TransportTurn[]>;
  /** The harness's originator reader: an orphan from a `skipOriginators` client stays unwritten. */
  originator?: RetainHookSpec["originator"];
  deadline: number;
  /** Injectable for tests. */
  orphans?: OrphanedCursor[];
  cursors?: RetainCursorStore;
}): Promise<void> {
  const { harness } = args;
  const orphans =
    args.orphans ??
    orphanedCursors(harness, { idleMs: ORPHAN_IDLE_MS, maxAgeMs: ORPHAN_MAX_AGE_MS });
  const cursors = args.cursors ?? fileCursorStore(harness);
  const readTranscript = args.readTranscript ?? readClaudeTranscript;
  for (const { sessionId, cursor } of orphans) {
    if (sessionId === args.currentSessionId) continue;
    // Never start one that cannot finish before the host kills this process.
    if (Date.now() + REQUEST_BUDGET_MS > args.deadline) return;
    // A cursor with no file is a persistent-plugin session: only its host can give its turns back.
    const transcriptPath = cursor.paths?.at(-1);
    if (transcriptPath ? !existsSync(transcriptPath) : !args.fetchTurns) continue;
    // Resolved from the bank id alone: the ended session's working folder is not known here, so a
    // folder-scoped (`paths`) override does not apply — a `banks.<id>` section does.
    const { cfg, bankId } = applyBankConfig(args.cfg, cursor.bank);
    if (cfg.disabled || !cfg.retainSessions) continue;
    if (transcriptPath && cfg.skipOriginators.length) {
      const originator = args.originator?.(transcriptPath);
      if (originator && cfg.skipOriginators.includes(originator)) continue;
    }
    let turns: TransportTurn[];
    try {
      // Guarded: a reader can throw on a damaged file (see transcript-qwen.ts), a host on a session
      // it no longer holds, and one bad old transcript must neither reject this Stop hook nor stop
      // the others from being replayed.
      turns = transcriptPath ? readTranscript(transcriptPath) : await args.fetchTurns!(sessionId);
    } catch (e) {
      diag(harness, "retain_replay_failed", { error: describeError(e), session: sessionId });
      continue;
    }
    if (!turns.length) continue;
    const client = (args.makeClient ?? ((o) => new HindsightClient(o)))({
      apiUrl: cfg.apiUrl,
      apiToken: cfg.apiToken,
      headers: cfg.apiHeaders,
      bank: bankId,
      maxParallelRetains: cfg.maxParallelRetains,
      observationScopes: cfg.observationScopes,
    });
    const t0 = Date.now();
    try {
      await retainLiveSession(
        client as HindsightClient,
        sessionId,
        turns,
        turns[0]?.timestamp ?? new Date().toISOString(),
        harness,
        { cursors, stamp: cursor.stamp, retryUntil: args.deadline, transcriptPath }
      );
      diag(harness, "retain_replayed", { ms: Date.now() - t0, session: sessionId });
    } catch (e) {
      // Still owed: the cursor keeps saying so, and the next Stop tries again. The others would
      // most likely fail the same way, and each try can cost this hook a request's full timeout.
      diag(harness, "retain_replay_failed", { error: describeError(e), session: sessionId });
      return;
    }
  }
}

/** Run one Stop-hook invocation: stdin event in, no stdout output (a Stop hook injects nothing). */
export async function runRetainHook(
  spec: RetainHookSpec,
  makeClient: (opts: ClientOpts) => RetainClient = (o) => new HindsightClient(o)
): Promise<void> {
  // Anti-recursion: the codebase survey's own headless claude session (core/survey.ts) sets this
  // so its hooks are a no-op — it must not retain its own survey session's transcript.
  if (process.env.HINDSIGHT_DISABLE_HOOKS) return;
  // The host started counting when it spawned us, which is near enough to now: everything above
  // is synchronous. A margin keeps the kill from landing between our last wait and its response.
  const hostDeadline = Date.now() + spec.hostTimeoutSec * 1000 - HOST_DEADLINE_MARGIN_MS;

  const ev = readHookEvent(spec.harness);
  if (!ev) return;
  if (spec.accept && !spec.accept(ev)) return;
  const parsed = spec.parse(ev);
  const { sessionId, cwd: rawCwd, lastAssistantMessage } = parsed;
  const cwd = rawCwd || process.cwd();
  let transcriptPath = parsed.transcriptPath;
  let readTranscript = spec.readTranscript;

  let cfg = loadConfig({ harness: spec.harness });
  setLogLevel(cfg.logLevel);
  if (cfg.disabled) return;

  // A journal harness closes the turn HERE, at the first point past the kill switch: the host's
  // copy of the reply is ephemeral (ZCode deletes its Stop transcript the moment this hook
  // returns), so it has to be read before anything that can block or fail. Reading the journal
  // back as the transcript is what makes the rest of this function identical to every other
  // harness's — the same full conversation, planned against the same retain cursor.
  if (spec.journal) {
    transcriptPath = journalPath(spec.harness, sessionId);
    readTranscript = readJournalTranscript;
    appendJournalTurn(transcriptPath, {
      role: "assistant",
      content: spec.journal.assistantText(ev),
    });
  }

  if (!transcriptPath) return;

  const baseCfg = cfg;
  const sessionRoot = sessionRootDir(spec.harness, sessionId, cwd);
  const derived = deriveBankIdOrSkip(cfg, cwd, spec.harness, sessionRoot);
  // Skipping the write-back loses this session; retaining it into a guessed bank loses it AND
  // pollutes the server with a bank nothing ever reads back (#3950).
  if (derived === null) return;
  const resolved = applyBankConfig(cfg, derived, cwd);
  cfg = resolved.cfg;
  const bankId = resolved.bankId;
  if (cfg.disabled) return; // per-bank opt-out (banks.<id> override)
  // Checked only HERE, after the bank is resolved, so a `banks.<id>` section can turn write-back
  // back on for one repo under a global `retainSessions: false` (and vice versa). Before the
  // daemon start below: a session that writes nothing has no reason to bring a server up. It still
  // reads the transcript, because usage is recorded either way — it is a local file (#5391).
  let client: RetainClient | undefined;
  const originator =
    cfg.retainSessions && cfg.skipOriginators.length
      ? spec.originator?.(transcriptPath)
      : undefined;
  if (!cfg.retainSessions) {
    diag(spec.harness, "retain_disabled", { bank: bankId, session: sessionId });
  } else if (originator && cfg.skipOriginators.includes(originator)) {
    diag(spec.harness, "retain_skipped_originator", {
      bank: bankId,
      session: sessionId,
      originator,
    });
  } else {
    // Last chance to get the daemon up: this is the write path, and a session whose daemon never
    // started would otherwise lose its whole conversation. The Stop hook has the longest budget of
    // any hook and nothing is waiting on its result, so it can afford the longer wait.
    // Deliberately NOT gated on the result — retain proceeds either way, so an unreachable daemon
    // produces the same `retain_failed` diagnostic as an unreachable Cloud/self-hosted server.
    await ensureDaemon(cfg, spec.harness, { waitMs: DAEMON_WAIT_RETAIN_MS });
    client = makeClient({
      apiUrl: cfg.apiUrl,
      apiToken: cfg.apiToken,
      headers: cfg.apiHeaders,
      bank: bankId,
      maxParallelRetains: cfg.maxParallelRetains,
      observationScopes: cfg.observationScopes,
    });
  }

  await buildRetain({
    harness: spec.harness,
    sessionId: sessionId || "no-session",
    transcriptPath,
    client,
    readTranscript,
    lastAssistantMessage,
    readLastMessage: spec.readLastMessage,
    retryUntil: hostDeadline,
    bankId,
    stamp: buildRetainStamp(cfg, {
      directory: cwd,
      sessionRoot,
      harness: spec.harness,
      bankId,
      sessionId: sessionId || "no-session",
    }),
  });

  // No client means this Stop saves nothing (write-back off, or a skipped originator), so it
  // deliberately starts no daemon just to retry other sessions' failed saves: the user's own next
  // Stop, which does save, runs that retry.
  if (client)
    await replayOrphanedSessions({
      harness: spec.harness,
      currentSessionId: sessionId || "no-session",
      cfg: baseCfg,
      makeClient,
      readTranscript,
      originator: spec.originator,
      deadline: hostDeadline,
    });
}
