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
import { readFileSync } from "node:fs";
import { deriveBankIdOrSkip } from "./bank";
import { retainLiveSession } from "./chat";
import { applyBankConfig, loadConfig } from "./config";
import { DAEMON_WAIT_RETAIN_MS, ensureDaemon } from "./daemon";
import { diag } from "./diag";
import { describeError, log, setLogLevel } from "./log";
import type { ClientOpts } from "./hindsight";
import { HindsightClient } from "./hindsight";
import type { RetainCursorStore } from "./retain-cursor";
import { buildRetainStamp, type RetainStamp } from "./retain-stamp";
import { fileCursorStore, sessionRootDir } from "./session-cache";
import { readClaudeTranscript } from "./transcript";
import { stripInjectedMemory } from "./transcript-util";

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
  /** Harness-specific transcript parser. Defaults to the Claude JSONL reader. */
  readTranscript?: TranscriptReader;
  /** Harness-specific decoder for `lastAssistantMessage`. Defaults to identity. */
  readLastMessage?: LastMessageReader;
}

/** Minimal client shape `buildRetain` needs — `HindsightClient` satisfies it structurally. The
 *  capability probe is part of the contract: appending to a document is only safe against a server
 *  that can deduplicate a resubmitted write (see core/retain-cursor.ts). */
interface RetainClient {
  retain: HindsightClient["retain"];
  supportsIdempotentRetain: HindsightClient["supportsIdempotentRetain"];
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
  client: RetainClient;
  readTranscript?: TranscriptReader;
  lastAssistantMessage?: string;
  readLastMessage?: LastMessageReader;
  /** Configured retainTags/retainMetadata, already resolved for this session (core/retain-stamp.ts). */
  stamp?: RetainStamp;
  /** Injectable for tests; defaults to the per-session temp file (a Stop hook has no memory). */
  cursors?: RetainCursorStore;
  /** Absolute time the host will kill this process; bounds any rate-limit retry. */
  retryUntil?: number;
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

  const startTs = turns[0]?.timestamp ?? new Date().toISOString();
  const t0 = Date.now();
  try {
    await retainLiveSession(client as HindsightClient, sessionId, turns, startTs, harness, {
      cursors: args.cursors ?? fileCursorStore(harness),
      stamp: args.stamp,
      retryUntil: args.retryUntil,
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

  let ev: Record<string, unknown> = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
  } catch {
    return; // no/invalid event: stay silent
  }
  const { sessionId, transcriptPath, cwd: rawCwd, lastAssistantMessage } = spec.parse(ev);
  const cwd = rawCwd || process.cwd();

  let cfg = loadConfig({ harness: spec.harness });
  setLogLevel(cfg.logLevel);
  if (cfg.disabled) return;

  if (!transcriptPath) return;

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
  // daemon start below: a session that writes nothing has no reason to bring a server up.
  if (!cfg.retainSessions) {
    diag(spec.harness, "retain_disabled", { bank: bankId, session: sessionId });
    return;
  }
  // Last chance to get the daemon up: this is the write path, and a session whose daemon never
  // started would otherwise lose its whole conversation. The Stop hook has the longest budget of
  // any hook and nothing is waiting on its result, so it can afford the longer wait.
  // Deliberately NOT gated on the result — retain proceeds either way, so an unreachable daemon
  // produces the same `retain_failed` diagnostic as an unreachable Cloud/self-hosted server.
  await ensureDaemon(cfg, spec.harness, { waitMs: DAEMON_WAIT_RETAIN_MS });
  const client = makeClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: bankId,
    maxParallelRetains: cfg.maxParallelRetains,
    observationScopes: cfg.observationScopes,
  });

  await buildRetain({
    harness: spec.harness,
    sessionId: sessionId || "no-session",
    transcriptPath,
    client,
    readTranscript: spec.readTranscript,
    lastAssistantMessage,
    readLastMessage: spec.readLastMessage,
    retryUntil: hostDeadline,
    stamp: buildRetainStamp(cfg, {
      directory: cwd,
      sessionRoot,
      harness: spec.harness,
      bankId,
      sessionId: sessionId || "no-session",
    }),
  });
}
