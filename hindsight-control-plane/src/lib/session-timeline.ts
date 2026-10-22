/**
 * Pure logic behind the Sessions view (components/session-detail.tsx): turning a
 * session's raw event log into the rows the 3-way conversation renders, and the
 * small rules around it (delivery, latency, the live-session decision path, polling).
 *
 * Kept free of React so every rule here is unit-tested (tests/lib/session-timeline.test.ts).
 */
import type { SessionEvent, SessionSummary } from "@/lib/api";

type CheckOutcome = "note" | "silent" | "repeat" | "skipped" | "error";

export interface LlmCall {
  scope: string;
  duration_ms?: number | null;
  detail?: string | null;
}

/** A `check` entry: one live-session reflect (or a recorded skip) for one moment. */
export interface CheckPayload {
  trigger?: string;
  moment_seq?: number;
  moment?: string;
  outcome?: CheckOutcome;
  reason?: string | null;
  error?: string | null;
  ms?: number;
  question?: string;
  context?: string;
  search_queries?: string[];
  answer?: string;
  abstained?: boolean;
  memory_ids?: string[];
  usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
  llm_calls?: LlmCall[];
  /** Replay only: the check has been running this long and has not answered yet (see replayEvents). */
  running_ms?: number;
}

interface NoteSource {
  document_id?: string | null;
  quote?: string | null;
  mentioned_at?: string | null;
  metadata?: Record<string, unknown> | null;
}

/** A `note` entry: what Hindsight told the agent. */
interface NotePayload {
  text?: string;
  trigger?: string;
  check_seq?: number;
  memory_ids?: string[];
  source?: NoteSource | null;
}

interface SavePayload {
  from_seq?: number;
  to_seq?: number;
  turns?: number;
  document_id?: string;
  operation_id?: string;
}

/** Which family a tool belongs to: picks its icon and how loud its row is. */
export type ToolFamily = "edit" | "command" | "read" | "other";

export interface CheckRow {
  kind: "check";
  key: string;
  seq: number;
  event: SessionEvent;
  check: CheckPayload;
}

export type TimelineRow =
  | {
      kind: "message";
      key: string;
      seq: number;
      event: SessionEvent;
      role: "user" | "assistant" | "system";
      text: string;
      checks: CheckRow[];
    }
  | {
      kind: "tool";
      key: string;
      seq: number;
      /** The call, or null when only its result is known (a result logged without its call). */
      call: SessionEvent | null;
      result: SessionEvent | null;
      tool: string;
      target: string;
      family: ToolFamily;
      isError: boolean;
      checks: CheckRow[];
    }
  | {
      kind: "state";
      key: string;
      seq: number;
      event: SessionEvent;
      diff: string;
      checks: CheckRow[];
    }
  | { kind: "turn_end"; key: string; seq: number; event: SessionEvent; checks: CheckRow[] }
  | {
      kind: "note";
      key: string;
      seq: number;
      event: SessionEvent;
      note: NotePayload;
      checks: CheckRow[];
    }
  | {
      kind: "save";
      key: string;
      seq: number;
      event: SessionEvent;
      save: SavePayload;
      checks: CheckRow[];
    }
  /** A note reaching the agent: the server's `handoff` entry, logged when it first went back. */
  | {
      kind: "handoff";
      key: string;
      seq: number;
      event: SessionEvent;
      note: Extract<TimelineRow, { kind: "note" }>;
      /** The client row whose append's reply carried it; null when a poll fetched it. */
      host: TimelineRow | null;
      checks: CheckRow[];
    }
  /** A check whose moment is not in the log: rendered on its own, where it happened. */
  | (CheckRow & { checks: CheckRow[] })
  | { kind: "other"; key: string; seq: number; event: SessionEvent; checks: CheckRow[] };

const str = (v: unknown): string => (typeof v === "string" ? v : "");

// A tool is known by the words of its name, not by one agent's names: Claude Code's
// Edit/Bash/Read, Codex's apply_patch/shell and Hermes's write_file/terminal/read_file all
// read the same. The server's live-session checker uses the same rule (engine/sessions/watch.py).
const EDIT_WORDS = ["edit", "write", "patch", "replace", "insert", "multiedit", "notebookedit"];
const COMMAND_WORDS = new Set([
  "bash",
  "shell",
  "sh",
  "terminal",
  "exec",
  "execute",
  "run",
  "command",
  "cmd",
]);
const READ_WORDS = new Set([
  "read",
  "grep",
  "glob",
  "ls",
  "view",
  "search",
  "find",
  "list",
  "fetch",
  "get",
  "websearch",
  "webfetch",
]);

function toolFamily(tool: string): ToolFamily {
  const words: string[] =
    tool
      .replace(/([a-z])([A-Z])/g, "$1_$2")
      .toLowerCase()
      .match(/[a-z]+/g) ?? [];
  if (words.includes("todo")) return "read"; // the agent's to-do list: bookkeeping, not code
  if (words.some((w) => EDIT_WORDS.some((e) => w.startsWith(e)))) return "edit";
  if (words.some((w) => COMMAND_WORDS.has(w))) return "command";
  if (words.some((w) => READ_WORDS.has(w))) return "read";
  return "other";
}

/** Keeps the end of a long path: `/work/flagcore/bucket.py` -> `flagcore/bucket.py`. */
function shortPath(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length > 2 ? parts.slice(-2).join("/") : path;
}

/**
 * The one thing a tool call acts on, for its one-line summary:
 * `Edit flagcore/bucket.py`, `Bash pytest -q`, `Grep crc32`.
 */
export function toolTarget(input: unknown): string {
  if (!input || typeof input !== "object") return str(input);
  const i = input as Record<string, unknown>;
  const path = str(i.file_path) || str(i.path) || str(i.notebook_path);
  if (path) return shortPath(path);
  const command = Array.isArray(i.command) ? i.command.join(" ") : str(i.command) || str(i.cmd);
  // A multi-line command (`python3 -c "` + a script) reads as nothing by its first
  // line alone, so its lines are joined and the row truncates the result.
  if (command) return command.replace(/\s*\n\s*/g, " ⏎ ").trim();
  for (const k of ["pattern", "query", "url", "description", "prompt"]) {
    if (str(i[k])) return str(i[k]).split("\n")[0];
  }
  const first = Object.values(i).find((v) => typeof v === "string") as string | undefined;
  return first ? first.split("\n")[0] : "";
}

interface Timeline {
  rows: TimelineRow[];
  /** Row key for any seq it contains (a tool row holds its call's and its result's seq). */
  rowKeyBySeq: Map<number, string>;
  eventBySeq: Map<number, SessionEvent>;
  /** What the agent told the user about each note (by note seq); "used" wins over "unneeded". */
  noteMentions: Map<number, NoteMention>;
}

export type NoteMention = "used" | "unneeded";

// The lines the agent is asked to write about a note (the server counts the same ones:
// hindsight_api/engine/sessions/service.py CREDIT_RE / DISMISS_RE).
const USED_RE = /From Hindsight memory \(live session #(\d+)\)/g;
const UNNEEDED_RE = /Hindsight note \(live session #(\d+)\)/g;

/** The notes the agent's own messages name as used or not needed. */
export function noteMentions(events: SessionEvent[]): Map<number, NoteMention> {
  const out = new Map<number, NoteMention>();
  for (const ev of events) {
    const p = (ev.payload ?? {}) as Record<string, unknown>;
    if (ev.kind !== "message" || p.role !== "assistant" || typeof p.text !== "string") continue;
    for (const m of p.text.matchAll(UNNEEDED_RE)) if (!out.has(+m[1])) out.set(+m[1], "unneeded");
    for (const m of p.text.matchAll(USED_RE)) out.set(+m[1], "used");
  }
  return out;
}

/**
 * Folds the log into rows: a tool result joins its call (by `call_id`), and each
 * check is attached to the row of the moment that caused it (`moment_seq`), so the
 * decision sits next to what it judged rather than wherever the reflect finished.
 */
export function buildTimeline(events: SessionEvent[]): Timeline {
  const rows: TimelineRow[] = [];
  const rowKeyBySeq = new Map<number, string>();
  const eventBySeq = new Map<number, SessionEvent>();
  const rowByKey = new Map<string, TimelineRow>();
  const toolByCallId = new Map<string, Extract<TimelineRow, { kind: "tool" }>>();

  const push = (row: TimelineRow) => {
    rows.push(row);
    rowByKey.set(row.key, row);
    rowKeyBySeq.set(row.seq, row.key);
  };

  for (const ev of events) {
    eventBySeq.set(ev.seq, ev);
    const p = ev.payload;
    const key = `e${ev.seq}`;

    if (ev.source === "hindsight") {
      if (ev.kind === "check") {
        const check: CheckRow = {
          kind: "check",
          key,
          seq: ev.seq,
          event: ev,
          check: p as CheckPayload,
        };
        const momentKey =
          typeof check.check.moment_seq === "number"
            ? rowKeyBySeq.get(check.check.moment_seq)
            : undefined;
        const host = momentKey ? rowByKey.get(momentKey) : undefined;
        if (host) {
          host.checks.push(check);
          rowKeyBySeq.set(ev.seq, host.key);
        } else {
          push({ ...check, checks: [] });
        }
      } else if (ev.kind === "note") {
        push({ kind: "note", key, seq: ev.seq, event: ev, note: p as NotePayload, checks: [] });
      } else if (ev.kind === "handoff") {
        const noteRow = rowByKey.get(rowKeyBySeq.get(Number(p.note_seq)) ?? "");
        if (noteRow?.kind === "note") {
          const withSeqs = Array.isArray(p.with_seqs) ? (p.with_seqs as number[]) : [];
          const last = withSeqs.length ? withSeqs[withSeqs.length - 1] : undefined;
          const host = last !== undefined ? rowByKey.get(rowKeyBySeq.get(last) ?? "") : undefined;
          push({
            kind: "handoff",
            key,
            seq: ev.seq,
            event: ev,
            note: noteRow,
            host: host ?? null,
            checks: [],
          });
        }
      } else if (ev.kind === "save") {
        push({ kind: "save", key, seq: ev.seq, event: ev, save: p as SavePayload, checks: [] });
      } else {
        push({ kind: "other", key, seq: ev.seq, event: ev, checks: [] });
      }
      continue;
    }

    if (ev.kind === "tool_call" || ev.kind === "tool_result") {
      const callId = str(p.call_id);
      const existing = callId ? toolByCallId.get(callId) : undefined;
      const isError = p.is_error === true;
      if (existing) {
        if (ev.kind === "tool_call") {
          existing.call = ev;
          existing.tool = str(p.tool) || existing.tool;
          existing.target = toolTarget(p.input);
          existing.family = toolFamily(existing.tool);
        } else {
          existing.result = ev;
        }
        existing.isError = existing.isError || isError;
        rowKeyBySeq.set(ev.seq, existing.key);
        continue;
      }
      const tool = str(p.tool);
      const row: Extract<TimelineRow, { kind: "tool" }> = {
        kind: "tool",
        key,
        seq: ev.seq,
        call: ev.kind === "tool_call" ? ev : null,
        result: ev.kind === "tool_result" ? ev : null,
        tool,
        target: ev.kind === "tool_call" ? toolTarget(p.input) : "",
        family: toolFamily(tool),
        isError,
        checks: [],
      };
      if (callId) toolByCallId.set(callId, row);
      push(row);
    } else if (ev.kind === "message") {
      const role = p.role === "user" || p.role === "system" ? p.role : "assistant";
      push({ kind: "message", key, seq: ev.seq, event: ev, role, text: str(p.text), checks: [] });
    } else if (ev.kind === "state") {
      const data = (p.data ?? {}) as Record<string, unknown>;
      push({ kind: "state", key, seq: ev.seq, event: ev, diff: str(data.diff), checks: [] });
    } else if (ev.kind === "turn_end") {
      push({ kind: "turn_end", key, seq: ev.seq, event: ev, checks: [] });
    } else {
      push({ kind: "other", key, seq: ev.seq, event: ev, checks: [] });
    }
  }
  return { rows, rowKeyBySeq, eventBySeq, noteMentions: noteMentions(events) };
}

/** Checks that found nothing to say (or never ran): kept quiet, hidden by the "silent checks" toggle. */
export function isQuietCheck(check: CheckPayload): boolean {
  return check.outcome === "silent" || check.outcome === "skipped";
}

export function isDelivered(noteSeq: number, ackedSeq: number): boolean {
  return noteSeq <= ackedSeq;
}

/**
 * How long the agent waited for a note: from the moment's event being logged to the
 * note being written. Null when the chain (note -> check -> moment) is not in the log.
 */
export function noteLatencyMs(
  note: SessionEvent,
  eventBySeq: Map<number, SessionEvent>
): number | null {
  const checkSeq = (note.payload as NotePayload).check_seq;
  const check = typeof checkSeq === "number" ? eventBySeq.get(checkSeq) : undefined;
  const momentSeq = check ? (check.payload as CheckPayload).moment_seq : undefined;
  const moment = typeof momentSeq === "number" ? eventBySeq.get(momentSeq) : undefined;
  if (!moment) return null;
  return Math.max(0, Date.parse(note.created_at) - Date.parse(moment.created_at));
}

/**
 * Where the live-session decision model sent the check (see reflect/agent.py,
 * `_pick_round`): stay silent without an LLM, let a writer word the answer, or let
 * the reflect LLM judge.
 */
export type DecisionRoute = "silent" | "nothing" | "writer" | "llm" | "unknown";

interface DecisionPick {
  /** The decision model's probability that no evidence answers. */
  none: number | null;
  kept: number | null;
  offered: number | null;
  route: DecisionRoute;
  ms: number | null;
}

interface DecisionPath {
  pick: DecisionPick | null;
  /** The LLM calls after the pick (e.g. `abstain_answer`), in order. */
  after: LlmCall[];
}

/** Parses `none=0.02, kept 1 of 8 offered: settled: the writer words the answer` and its siblings. */
export function parseDecisionDetail(detail: string | null | undefined): Omit<DecisionPick, "ms"> {
  const d = detail ?? "";
  const num = (re: RegExp) => {
    const m = d.match(re);
    return m ? Number(m[1]) : null;
  };
  const route: DecisionRoute = d.startsWith("nothing found")
    ? "nothing"
    : d.includes("silent, no LLM")
      ? "silent"
      : d.includes("settled")
        ? "writer"
        : d.includes("unsure")
          ? "llm"
          : "unknown";
  return {
    none: num(/none=([\d.]+)/),
    kept: num(/kept (\d+) of/),
    offered: num(/of (\d+) offered/),
    route,
  };
}

export function decisionPath(calls: LlmCall[] | undefined): DecisionPath {
  const list = calls ?? [];
  const i = list.findIndex((c) => c.scope === "decision_pick");
  if (i < 0) return { pick: null, after: list };
  const c = list[i];
  return {
    pick: { ...parseDecisionDetail(c.detail), ms: c.duration_ms ?? null },
    after: list.filter((_, j) => j !== i),
  };
}

/**
 * Logged something in the last two minutes: shown with a pulsing LIVE marker, otherwise
 * idle. There is no "closed": a session that stopped is idle, and the server saves its
 * last turns once it has been idle a while (HINDSIGHT_API_SESSION_IDLE_MS).
 */
const ACTIVE_WINDOW_MS = 2 * 60 * 1000;

export function isActive(session: Pick<SessionSummary, "last_event_at">, now: number) {
  if (!session.last_event_at) return false;
  return now - Date.parse(session.last_event_at) < ACTIVE_WINDOW_MS;
}

/**
 * How long to wait before the next `events?after=` poll.
 *
 * Fast (1s) while something is happening, slower once the session has gone quiet,
 * slow while the tab is hidden. It never stops: any session can pick up again.
 */
export function nextPollDelay(opts: {
  visible: boolean;
  checking: boolean;
  gotNew: boolean;
  idleMs: number;
}): number {
  if (!opts.visible) return 15_000;
  if (opts.checking || opts.gotNew || opts.idleMs < 60_000) return 1_000;
  if (opts.idleMs < 10 * 60_000) return 3_000;
  return 10_000;
}

export interface RowFilters {
  tools: boolean;
  silent: boolean;
  hindsightOnly: boolean;
  query: string;
}

function rowText(row: TimelineRow): string {
  switch (row.kind) {
    case "message":
      return row.text;
    case "tool":
      return [
        row.tool,
        row.target,
        JSON.stringify(row.call?.payload.input ?? ""),
        str(row.result?.payload.text),
      ].join("\n");
    case "state":
      return row.diff;
    case "note":
      return [row.note.text, row.note.source?.quote].join("\n");
    case "check":
      return checkText(row.check);
    case "handoff":
      return row.note.note.text ?? "";
    default:
      return "";
  }
}

function checkText(c: CheckPayload): string {
  return [c.moment, c.answer, c.reason, c.error, ...(c.search_queries ?? [])].join("\n");
}

/**
 * Applies the view toggles. A row that hosts a visible check is kept even when its
 * own kind is filtered out, so hiding tool calls never hides the decision made on one.
 */
export function filterRows(rows: TimelineRow[], f: RowFilters): TimelineRow[] {
  const q = f.query.trim().toLowerCase();
  const out: TimelineRow[] = [];
  for (const row of rows) {
    const checks = f.silent ? row.checks : row.checks.filter((c) => !isQuietCheck(c.check));
    if (row.kind === "check" && !f.silent && isQuietCheck(row.check)) continue;
    const hindsight =
      row.kind === "note" || row.kind === "save" || row.kind === "check" || row.kind === "handoff";
    const own = f.hindsightOnly
      ? hindsight
      : f.tools || (row.kind !== "tool" && row.kind !== "state");
    if (!own && checks.length === 0) continue;
    if (
      q &&
      !rowText(row).toLowerCase().includes(q) &&
      !checks.some((c) => checkText(c.check).toLowerCase().includes(q))
    ) {
      continue;
    }
    out.push(checks === row.checks ? row : { ...row, checks });
  }
  return out;
}

/**
 * Replay clock: each event's offset from the first, in ms, with dead gaps capped.
 *
 * ponytail: gaps longer than `maxGapMs` are squashed to it so a session that sat
 * idle for an hour still replays in minutes; replaying "real pace" across idle
 * periods would need a skip control instead.
 */
export function replayOffsets(events: SessionEvent[], maxGapMs = 10_000): number[] {
  const out: number[] = [];
  let t = 0;
  for (let i = 0; i < events.length; i++) {
    if (i > 0) {
      const gap = Date.parse(events[i].created_at) - Date.parse(events[i - 1].created_at);
      t += Math.min(Math.max(0, gap), maxGapMs);
    }
    out.push(t);
  }
  return out;
}

/** When a check started and ended on the replay clock: it is logged when it ends, `ms` after it started. */
export interface CheckSpan {
  seq: number;
  start: number;
  end: number;
  outcome?: string;
  trigger?: string;
}

export function checkSpans(events: SessionEvent[], offsets: number[]): CheckSpan[] {
  const offsetBySeq = new Map(events.map((e, i) => [e.seq, offsets[i]]));
  const spans: CheckSpan[] = [];
  events.forEach((ev, i) => {
    if (ev.source !== "hindsight" || ev.kind !== "check") return;
    const c = ev.payload as CheckPayload;
    const end = offsets[i];
    // Never before its moment: idle gaps are squeezed on the replay clock, a check's run time is not.
    const momentAt = typeof c.moment_seq === "number" ? offsetBySeq.get(c.moment_seq) : undefined;
    const start = Math.min(end, Math.max(momentAt ?? 0, end - (c.ms ?? 0)));
    spans.push({ seq: ev.seq, start, end, outcome: c.outcome, trigger: c.trigger });
  });
  return spans;
}

/**
 * The log as it stood at `elapsed` on the replay clock, plus every check that was
 * running then, as a copy without its outcome and with `running_ms` set: the
 * replay shows Hindsight working while the agent works, not only its answers.
 */
export function replayEvents(
  events: SessionEvent[],
  offsets: number[],
  spans: CheckSpan[],
  elapsed: number
): SessionEvent[] {
  let n = 0;
  while (n < offsets.length && offsets[n] <= elapsed) n++;
  const shown = events.slice(0, n);
  const bySeq = new Map(events.map((e) => [e.seq, e]));
  for (const s of spans) {
    if (s.start > elapsed || s.end <= elapsed) continue;
    const ev = bySeq.get(s.seq)!;
    // Not answered yet: no outcome, no answer.
    const payload: CheckPayload = {
      ...(ev.payload as CheckPayload),
      running_ms: elapsed - s.start,
    };
    delete payload.outcome;
    delete payload.answer;
    shown.push({ ...ev, payload: payload as Record<string, unknown> });
  }
  return shown;
}

/** Time since the session started, the same on both sides of the view: "0:41", "12:05", "1:02:09". */
export function formatClock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = String(Math.floor((s % 3600) / 60));
  const ss = String(s % 60).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm.padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
}

/** "1.2s", "850ms", "2m 05s". */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.round(ms / 1000);
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** "3 minutes ago" in the viewer's locale (lib/relative-time.ts is English-only). */
export function relativeTime(dateStr: string, locale: string, now = Date.now()): string {
  const sec = Math.round((Date.parse(dateStr) - now) / 1000);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const abs = Math.abs(sec);
  if (abs < 60) return rtf.format(sec, "second");
  if (abs < 3600) return rtf.format(Math.round(sec / 60), "minute");
  if (abs < 86_400) return rtf.format(Math.round(sec / 3600), "hour");
  return rtf.format(Math.round(sec / 86_400), "day");
}
