/**
 * Shared runtime for the Claude Code `Stop` hook: when a session ends, read its transcript,
 * normalize it, and write it back into the bank so the session compounds into memory. The retain
 * half of the plugin (the `UserPromptSubmit` hook in core/hook.ts is the recall half).
 *
 * Write-back is ON by default for this hook harness — governed only by `disabled` — unlike the
 * opencode persistent-plugin's `retainSessions` flag, which is a separate opt-in concern for a
 * long-lived process retaining mid-session on a cadence (see core/runtime.ts).
 *
 * The pure logic lives in `buildRetain` (path + client in, void out) so it's unit-testable
 * without stdin; `runRetainHook` is thin plumbing around it, mirroring `runHook`/`buildHookOutput`
 * in core/hook.ts.
 */
import { readFileSync } from "node:fs";
import { deriveBankId } from "./bank";
import { retainLiveSession } from "./chat";
import { applyBankConfig, loadConfig } from "./config";
import { DAEMON_WAIT_RETAIN_MS, ensureDaemon } from "./daemon";
import { diag } from "./diag";
import { log, setLogLevel } from "./log";
import type { ClientOpts } from "./hindsight";
import { HindsightClient } from "./hindsight";
import type { RetainCursorStore } from "./retain-cursor";
import { buildRetainStamp, type RetainStamp } from "./retain-stamp";
import { fileCursorStore } from "./session-cache";
import { readClaudeTranscript } from "./transcript";
import type { TransportTurn } from "./chat";

export interface RetainHookEventFields {
  sessionId?: string;
  transcriptPath?: string;
  cwd?: string;
}

/** Read a harness's transcript file into normalized turns. Claude and Codex use different JSONL
 *  schemas, so each harness supplies its own reader (default: Claude). */
export type TranscriptReader = (path: string) => TransportTurn[];

export interface RetainHookSpec {
  /** Harness name — config `harnesses.<name>` section, {harness} template field, diag records. */
  harness: string;
  /** Read the fields out of the harness's stdin event (shapes differ per harness). */
  parse(event: Record<string, unknown>): RetainHookEventFields;
  /** Harness-specific transcript parser. Defaults to the Claude JSONL reader. */
  readTranscript?: TranscriptReader;
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
  /** Configured retainTags/retainMetadata, already resolved for this session (core/retain-stamp.ts). */
  stamp?: RetainStamp;
  /** Injectable for tests; defaults to the per-session temp file (a Stop hook has no memory). */
  cursors?: RetainCursorStore;
}): Promise<void> {
  const { harness, sessionId, transcriptPath, client } = args;
  const readTranscript = args.readTranscript ?? readClaudeTranscript;

  const turns = readTranscript(transcriptPath);
  if (turns.length === 0) return;

  const startTs = turns[0]?.timestamp ?? new Date().toISOString();
  const t0 = Date.now();
  try {
    await retainLiveSession(client as HindsightClient, sessionId, turns, startTs, harness, {
      cursors: args.cursors ?? fileCursorStore(harness),
      stamp: args.stamp,
    });
    diag(harness, "retain_ok", { ms: Date.now() - t0, turns: turns.length, session: sessionId });
  } catch (e) {
    log.warn(harness, "session write-back failed", {
      error: String((e as Error)?.message || e).slice(0, 200),
    });
    diag(harness, "retain_failed", {
      ms: Date.now() - t0,
      error: String((e as Error)?.message || e).slice(0, 200),
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

  let ev: Record<string, unknown> = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
  } catch {
    return; // no/invalid event: stay silent
  }
  const { sessionId, transcriptPath, cwd: rawCwd } = spec.parse(ev);
  const cwd = rawCwd || process.cwd();

  let cfg = loadConfig({ harness: spec.harness });
  setLogLevel(cfg.logLevel);
  if (cfg.disabled) return;

  if (!transcriptPath) return;

  const resolved = applyBankConfig(cfg, deriveBankId(cfg, cwd, spec.harness));
  cfg = resolved.cfg;
  const bankId = resolved.bankId;
  if (cfg.disabled) return;
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
  });

  await buildRetain({
    harness: spec.harness,
    sessionId: sessionId || "no-session",
    transcriptPath,
    client,
    readTranscript: spec.readTranscript,
    stamp: buildRetainStamp(cfg, {
      directory: cwd,
      harness: spec.harness,
      bankId,
      sessionId: sessionId || "no-session",
    }),
  });
}
