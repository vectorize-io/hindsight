/**
 * Kimi Code wire-log reader.
 *
 * Kimi Code has no chat transcript. A session is a DIRECTORY of append-only agent-loop logs — one
 * `wire.jsonl` per agent under
 * `~/.kimi-code/sessions/wd_<workspace>_<hash>/session_<uuid>/agents/<agentId>/wire.jsonl` — and no
 * hook payload names any of them (the CLI's hook feature has no transcript field at all). So this
 * resolves the location from the Stop event's `session_id` the way transcript-grok.ts resolves
 * Grok's, and the reader is handed the SESSION DIRECTORY as its "transcript path".
 *
 * The conversation lives in a TWO-LEVEL envelope, which is what makes this reader unlike every
 * other one here: `content.part`, `tool.call` and `tool.result` are NOT top-level record types —
 * they are nested inside `{"type":"context.append_loop_event","event":{...}}`. A reader that
 * matches those names at the top level matches nothing and retains an empty session.
 *
 * What is kept, normalized to the same `TransportTurn[]` shape as readClaudeTranscript:
 *   - `turn.prompt` whose `origin.kind` is `user` — the human's prompt.
 *   - `content.part` blocks of `part.type:"text"`, joined per STEP into one assistant turn.
 *     `part.type:"think"` is the model's reasoning (70% of all parts in the local corpus) and is
 *     dropped, like Claude `thinking` and Codex `reasoning`.
 *   - `tool.call` → a compact `role:"action"` turn. Its `args` is already an object (unlike dsh's
 *     raw JSON string), so it goes straight to `actionLine`.
 * `tool.result`, `step.begin`/`step.end`, and the ~25 telemetry record types (`llm.request`,
 * `usage.record`, `staleGuard.recorded`, …) are dropped.
 *
 * `origin.kind` is Kimi's exact analogue of dsh's `source.kind` and Claude's `isMeta`: measured
 * over the local corpus, `turn.prompt` arrives with kind `user` (40), `system_trigger` (65, the
 * swarm dispatcher briefing a subagent), `task` (15) and `skill_activation` (2). Only `user` is a
 * human turn; retaining the rest files Kimi's own scaffolding as if the user had said it. A
 * subagent's work is still captured — its assistant/action turns are read normally — and the task
 * it was given survives on the parent side as the `Agent` action turn that spawned it.
 *
 * `context.append_message` is IGNORED ENTIRELY, and that is load-bearing rather than an omission.
 * It is Kimi's context-ASSEMBLY channel, not a record of the agent's work: it carries the host's
 * own injections (`origin.kind:"injection"`, variants todo_list_reminder / permission_mode /
 * date_change / interruption / …), a duplicate mirror of every `turn.prompt`, and — on a FORKED
 * agent — a full replay of the parent's conversation. Measured over 65 local wire logs, all 259
 * assistant `context.append_message` records belonged to the two forked agents of one session and
 * were byte-identical replays of that session's `main` log. Reading the channel therefore
 * double-ingests a forked session and files host scaffolding as user speech; ignoring it needs no
 * `state.json` `forkedFrom` lookup to get right. It is also the anti-feedback filter: a
 * UserPromptSubmit hook's own injection is appended here with `origin.kind:"hook_result"`, so
 * skipping the channel is what stops a retain→recall loop (`stripInjectedMemory` still runs on
 * every kept turn, for a block that leaked into a genuine prompt).
 *
 * Fail-open throughout: a missing directory, an unreadable log, a malformed line, or a line that
 * parses to a non-object JSON value yields no turns rather than throwing.
 */
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { TransportTurn } from "./chat";
import { diag } from "./diag";
import { readJsonl } from "./jsonl";
import { log } from "./log";
import { actionLine, stripInjectedMemory } from "./transcript-util";

const HARNESS = "kimi-code";
const SESSIONS_DIR = "sessions";
const AGENTS_DIR = "agents";
const WIRE_LOG = "wire.jsonl";
/** The hook payload carries the BARE uuid; the directory (and session_index.jsonl) prefix it. */
const SESSION_PREFIX = "session_";
/** The session's own loop. Every other agent directory is `agent-<n>` (a subagent or a fork). */
const MAIN_AGENT = "main";

/**
 * Most wire-log text one Stop may decode across ALL of a session's agents.
 *
 * `readJsonl` streams and bounds nothing by design (#4380): memory is the largest record plus
 * whatever the reader keeps, and a per-file cap was removed because a sliding tail window shifts
 * every turn's index, which breaks the append cursor. That reasoning holds here too, so this reader
 * reads each wire log WHOLE and bounds what it KEEPS instead.
 *
 * A cap is still needed, because Kimi is the one harness whose transcript is not a file but a
 * DIRECTORY: one wire.jsonl per agent, and a swarm session on this machine has 61 of them under a
 * single session. Bounding the sum is therefore a cross-file concern that no per-file reader can
 * express. The budget is the same order as the old per-file cap — it keeps a hook process near
 * ~100MB — and it is spent in agent order, so `main` is always read whole.
 */
const MAX_SESSION_CHARS = 32 * 1024 * 1024;

/** Structural subset of one nested loop event (`context.append_loop_event.event`). */
interface KimiLoopEvent {
  type?: string;
  /** Groups the parts and calls of one assistant step; absent only on malformed input. */
  stepUuid?: string;
  part?: { type?: string; text?: string };
  name?: string;
  args?: unknown;
}

/** Structural subset of one wire.jsonl record. `time` is epoch milliseconds. */
export interface KimiWireEvent {
  type?: string;
  time?: number;
  /** `turn.prompt` only: the prompt's content blocks. */
  input?: { type?: string; text?: string }[];
  /** `turn.prompt` only: who authored the prompt — see the module doc. */
  origin?: { kind?: string };
  /** `context.append_loop_event` only: the real event. */
  event?: KimiLoopEvent;
}

/**
 * Locate a session's directory from the id Kimi puts on every hook payload.
 *
 * The `wd_<basename>_<hash>` workspace component cannot be derived from `cwd`: this machine has
 * `wd_example-project_9f27d1b6c845` (~/example-project) and `wd_example-project_4b1c9e07a2f3`
 * (/repos/example-project), two different repos whose directories differ only by the hash.
 * So scan for the session instead of composing a path out of the working directory.
 */
export function kimiSessionDir(
  sessionId: string,
  kimiHome = process.env.KIMI_CODE_HOME || join(homedir(), ".kimi-code")
): string | undefined {
  const name = sessionId.startsWith(SESSION_PREFIX) ? sessionId : `${SESSION_PREFIX}${sessionId}`;
  const sessions = join(kimiHome, SESSIONS_DIR);
  try {
    for (const workspace of readdirSync(sessions, { withFileTypes: true })) {
      if (!workspace.isDirectory()) continue;
      const candidate = join(sessions, workspace.name, name);
      if (existsSync(join(candidate, AGENTS_DIR))) return candidate;
    }
  } catch {
    /* no sessions directory yet — reported below, like an id that matched nothing */
  }
  // A session we cannot find retains nothing, which is indistinguishable from an idle session
  // unless it is recorded (the Devin lesson, #3125).
  diag(HARNESS, "session_dir_missing", { sessions, session: name });
  return undefined;
}

/** Kimi stamps epoch milliseconds on the OUTER record; the nested loop event carries no clock. */
function stampOf(event: KimiWireEvent): { timestamp?: string } {
  if (typeof event.time !== "number") return {};
  const at = new Date(event.time);
  return Number.isNaN(at.getTime()) ? {} : { timestamp: at.toISOString() };
}

/** `turn.prompt.input` is a block array; only text blocks carry prose. */
function promptText(input: KimiWireEvent["input"]): string {
  return (input || [])
    .filter((block) => block?.type === "text" && typeof block.text === "string")
    .map((block) => block.text as string)
    .join("\n");
}

/**
 * Normalize ONE agent's wire log into transcript turns: the human prompt, one assistant turn per
 * model step, and a compact `role:"action"` turn per tool call. A pure function over records rather
 * than a file reader — like readDshEvents, and for the same reason: the records are the schema, and
 * the same function serves a file, a stream, or a backfill. Never throws on malformed entries.
 */
export function readKimiWire(events: readonly KimiWireEvent[]): TransportTurn[] {
  const turns: TransportTurn[] = [];
  // One assistant message is streamed as N `content.part` records sharing a stepUuid. Buffer them
  // so a step renders as one turn: the JSONL transcript is chunked a turn per line server-side, and
  // a sentence split across four lines is four fragments to the extractor.
  let step: { uuid: string; texts: string[]; stamp: { timestamp?: string } } | undefined;
  const flushStep = () => {
    if (!step) return;
    const content = stripInjectedMemory(step.texts.join("\n")).trim();
    if (content) turns.push({ role: "assistant", content, ...step.stamp });
    step = undefined;
  };

  for (const event of events || []) {
    if (!event || typeof event !== "object") continue;
    const stamp = stampOf(event);

    if (event.type === "turn.prompt") {
      flushStep();
      // Kimi drives itself through the same record: only `user` is a human — see the module doc.
      if (event.origin?.kind !== "user") continue;
      const content = stripInjectedMemory(promptText(event.input)).trim();
      if (content) turns.push({ role: "user", content, ...stamp });
      continue;
    }
    // Everything else conversational is nested. `context.append_message` is deliberately not a
    // branch here (module doc): it replays a fork's parent and carries the host's own injections.
    if (event.type !== "context.append_loop_event") continue;
    const loop = event.event;
    if (!loop || typeof loop !== "object") continue;

    if (loop.type === "content.part") {
      // `think` parts are the model's reasoning — dropped, like Claude `thinking`.
      if (loop.part?.type !== "text" || typeof loop.part.text !== "string") continue;
      const uuid = typeof loop.stepUuid === "string" ? loop.stepUuid : "";
      if (step && step.uuid !== uuid) flushStep();
      if (!step) step = { uuid, texts: [], stamp };
      step.texts.push(loop.part.text);
    } else if (loop.type === "tool.call") {
      flushStep(); // a step's prose precedes the calls it made
      if (typeof loop.name !== "string" || !loop.name) continue;
      turns.push({ role: "action", content: actionLine(loop.name, loop.args), ...stamp });
    }
    // step.begin / step.end: loop bookkeeping. tool.result: raw tool output, which the shared
    // `actionLine` convention keeps out of the bank (see transcript-util.ts).
  }
  flushStep();
  return turns;
}

/** Agent directories in a stable, readable order: the main loop first, then `agent-<n>` by number
 *  (`agent-2` before `agent-10`, which a lexicographic sort gets backwards). */
function agentIds(sessionDir: string): string[] {
  try {
    return readdirSync(join(sessionDir, AGENTS_DIR), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort(byAgentId);
  } catch {
    return []; // no session directory, or none readable: fail open like every other reader
  }
}

function byAgentId(a: string, b: string): number {
  if (a === b) return 0;
  if (a === MAIN_AGENT) return -1;
  if (b === MAIN_AGENT) return 1;
  const indexA = agentIndex(a);
  const indexB = agentIndex(b);
  return indexA === indexB ? a.localeCompare(b) : indexA - indexB;
}

function agentIndex(id: string): number {
  const suffix = Number(id.slice(id.lastIndexOf("-") + 1));
  return Number.isFinite(suffix) ? suffix : Number.MAX_SAFE_INTEGER;
}

function parseEvent(rawLine: string): KimiWireEvent | undefined {
  const trimmed = rawLine.trim();
  if (!trimmed) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return undefined;
  }
  // JSON.parse accepts non-object top-level values (`null`, numbers, arrays); guard here so a
  // truncated line cannot reach a property access below and throw.
  return typeof parsed === "object" && parsed !== null ? (parsed as KimiWireEvent) : undefined;
}

/**
 * Read a whole Kimi session — every agent's wire log, concatenated in agent order — into one
 * normalized transcript.
 *
 * `path` is the SESSION DIRECTORY, not a file: Kimi's Stop payload has no transcript path, so
 * `retain.parse` resolves the directory with `kimiSessionDir` and passes that (the devin-cli
 * precedent, where the "path" is a session id). One agent's records are parsed at a time, so peak
 * memory is one agent rather than the session.
 */
export function readKimiTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  let remaining = MAX_SESSION_CHARS;
  for (const agentId of agentIds(path)) {
    const wire = join(path, AGENTS_DIR, agentId, WIRE_LOG);
    const events: KimiWireEvent[] = [];
    for (const rawLine of readJsonl(wire)) {
      remaining -= rawLine.length;
      if (remaining < 0) break;
      const event = parseEvent(rawLine);
      if (event) events.push(event);
    }
    for (const turn of readKimiWire(events)) turns.push(turn);
    if (remaining < 0) {
      // Same contract as core/jsonl.ts: over the cap is reported, never dropped in silence.
      log.warn(HARNESS, "session exceeds the wire-log budget — later agents were not read", {
        sessionDir: path,
        stoppedAt: wire,
      });
      break;
    }
  }
  return turns;
}
