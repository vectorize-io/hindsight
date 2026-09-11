/**
 * Hindsight tool usage — is the agent calling our tools, and does it credit what they returned?
 *
 * One JSON line per finished user turn in `~/.hindsight/coding-agents-logs/usage.jsonl`:
 *
 *   {"ts","harness","session","bank","turn":7,"calls":["hindsight_reflect"],"credited":true}
 *
 * - `calls`    the hindsight_* tools the agent called during that turn (other tools are not counted)
 * - `credited` the agent's reply carried the "From Hindsight memory" credit the tool guide asks for
 *              whenever retrieved memory contributed — the cheap proxy for "the result was useful".
 *              The report rates it over turns that called a retrieval tool (see RETRIEVAL_TOOLS)
 *
 * Computed from the NORMALIZED transcript every write-back already reads, not per harness: every
 * transcript reader renders a tool call as a `role:"action"` turn led by the tool name, so one pass
 * covers all of them. Recorded once per turn: a per-session cursor remembers how many turns are
 * already written, because the Stop hook re-reads the whole transcript on every reply. Losing that
 * cursor re-records a session's turns; the report dedupes by (harness, session, turn).
 *
 * Local only — nothing here is sent anywhere. Fail-open throughout.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { TransportTurn } from "./chat";
import { appendLogLine, logsDir } from "./log";

export function usageFilePath(): string {
  return process.env.HINDSIGHT_USAGE_FILE || join(logsDir(), "usage.jsonl");
}

/** A Hindsight tool name at the END of a host's tool name: hosts prefix MCP tools differently
 *  (`mcp__hindsight__hindsight_reflect`, `hindsight.hindsight_reflect`, bare `hindsight_reflect`).
 *  Each segment needs a letter, so `hindsight__hindsight_reflect` cannot match from the first one. */
const HINDSIGHT_TOOL_RE = /hindsight_(?:[a-z]+_)*[a-z]+$/;

/** The credit line the injected tool guide prescribes (core/knowledge-injection.ts). */
const CREDIT_RE = /from hindsight memory/i;

/**
 * The tools that hand memory BACK to the agent — the only calls a credit can follow. The write tools
 * (ingest_document, capture_initiative) and the status probes never do, so counting them in the
 * credit rate's denominator measured how often the agent saved things, not whether retrieval helped:
 * on real sessions it read ~5% because ingest_document dominated.
 */
const RETRIEVAL_TOOLS = new Set([
  "hindsight_search_knowledge_pages",
  "hindsight_list_knowledge_pages",
  "hindsight_read_knowledge_page",
  "hindsight_reflect",
]);

/** Roles a reader uses for the agent's own prose (Gemini-shaped transcripts say "model"). */
const ASSISTANT_ROLES = new Set(["assistant", "model"]);

export interface TurnUsage {
  /** 1-based index of the user turn within the session. */
  turn: number;
  calls: string[];
  credited: boolean;
}

/** Split a transcript into user turns: each `user` turn opens one, and everything up to the next
 *  belongs to it. Anything before the first user turn (a system preamble) belongs to none. */
export function summarizeTurns(turns: TransportTurn[]): TurnUsage[] {
  const out: TurnUsage[] = [];
  for (const t of turns) {
    if (t.role === "user") {
      out.push({ turn: out.length + 1, calls: [], credited: false });
      continue;
    }
    const current = out.at(-1);
    if (!current) continue;
    if (t.role === "action") {
      const tool = t.content.split(" ", 1)[0].match(HINDSIGHT_TOOL_RE)?.[0];
      if (tool) current.calls.push(tool);
    } else if (ASSISTANT_ROLES.has(t.role) && CREDIT_RE.test(t.content)) {
      current.credited = true;
    }
  }
  return out;
}

/** How many of a session's turns are already recorded. File-backed for hook harnesses (a fresh
 *  process per event), in memory for the persistent-plugin runtime. */
export interface UsageCursorStore {
  read(sessionId: string): number | undefined;
  write(sessionId: string, turns: number): void;
}

export function memoryUsageCursorStore(): UsageCursorStore {
  const recorded = new Map<string, number>();
  return {
    read: (sessionId) => recorded.get(sessionId),
    write: (sessionId, turns) => void recorded.set(sessionId, turns),
  };
}

/**
 * Append a usage line for every finished turn not recorded yet.
 *
 * `lastTurnComplete` is false when the transcript was captured while the agent was still answering
 * its last prompt (the persistent-plugin runtime builds it before sending the request): that turn
 * is left for a later call, or it would be recorded with no calls and no credit and never revisited.
 */
export function recordUsage(args: {
  harness: string;
  sessionId: string;
  bankId: string;
  turns: TransportTurn[];
  cursors: UsageCursorStore;
  lastTurnComplete: boolean;
}): void {
  const summary = summarizeTurns(args.turns);
  const finished = args.lastTurnComplete ? summary : summary.slice(0, -1);
  const from = args.cursors.read(args.sessionId) ?? 0;
  // `<=` also covers a transcript that shrank (rewritten): nothing new to say about it.
  if (finished.length <= from) return;
  const ts = new Date().toISOString();
  const lines = finished
    .slice(from)
    .map(
      (u) =>
        JSON.stringify({
          ts,
          harness: args.harness,
          session: args.sessionId,
          bank: args.bankId,
          turn: u.turn,
          calls: u.calls,
          credited: u.credited,
        }) + "\n"
    )
    .join("");
  try {
    appendLogLine(usageFilePath(), lines);
    args.cursors.write(args.sessionId, finished.length);
  } catch {
    /* usage stats must never break the agent */
  }
}

interface UsageLine {
  harness: string;
  session: string;
  turn: number;
  calls: string[];
  credited: boolean;
}

function isUsageLine(v: unknown): v is UsageLine {
  const r = v as UsageLine | null;
  return (
    typeof r === "object" &&
    r !== null &&
    typeof r.harness === "string" &&
    typeof r.session === "string" &&
    typeof r.turn === "number" &&
    Array.isArray(r.calls) &&
    typeof r.credited === "boolean"
  );
}

export interface HarnessUsage {
  harness: string;
  turns: number;
  /** Turns with at least one hindsight_* call. */
  turnsWithCalls: number;
  calls: number;
  /** Turns that called a RETRIEVAL tool — the credit rate's denominator. */
  turnsWithRetrieval: number;
  /** Of `turnsWithRetrieval`, how many replies credited Hindsight memory. */
  creditedTurnsWithRetrieval: number;
  byTool: Record<string, number>;
}

/** Aggregate the usage log (the rotated generation first) per harness, one line per turn. */
export function readUsage(path = usageFilePath()): HarnessUsage[] {
  const latest = new Map<string, UsageLine>();
  for (const file of [`${path}.1`, path]) {
    let text = "";
    try {
      text = readFileSync(file, "utf8");
    } catch {
      continue;
    }
    for (const raw of text.split("\n")) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch {
        continue;
      }
      if (!isUsageLine(parsed)) continue;
      latest.set(`${parsed.harness}\n${parsed.session}\n${parsed.turn}`, parsed);
    }
  }

  const byHarness = new Map<string, HarnessUsage>();
  for (const line of latest.values()) {
    let h = byHarness.get(line.harness);
    if (!h) {
      h = {
        harness: line.harness,
        turns: 0,
        turnsWithCalls: 0,
        calls: 0,
        turnsWithRetrieval: 0,
        creditedTurnsWithRetrieval: 0,
        byTool: {},
      };
      byHarness.set(line.harness, h);
    }
    h.turns++;
    h.calls += line.calls.length;
    for (const tool of line.calls) h.byTool[tool] = (h.byTool[tool] ?? 0) + 1;
    if (line.calls.length) h.turnsWithCalls++;
    if (line.calls.some((tool) => RETRIEVAL_TOOLS.has(tool))) {
      h.turnsWithRetrieval++;
      if (line.credited) h.creditedTurnsWithRetrieval++;
    }
  }
  return [...byHarness.values()].sort((a, b) => b.turns - a.turns);
}

const pct = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : "-");

/** Human-readable report for `hindsight-coding-agents stats`. */
export function formatUsageReport(usage: HarnessUsage[], path = usageFilePath()): string {
  if (!usage.length) return `no Hindsight usage recorded yet (${path})`;
  const out = [`Hindsight tool usage (${path})`, ""];
  for (const h of usage) {
    out.push(
      `${h.harness}: ${h.turns} turns · ${h.turnsWithCalls} with a Hindsight call ` +
        `(${pct(h.turnsWithCalls, h.turns)}) · ${(h.calls / h.turns).toFixed(2)} calls/turn · ` +
        `credited after retrieval ${h.creditedTurnsWithRetrieval}/${h.turnsWithRetrieval} ` +
        `(${pct(h.creditedTurnsWithRetrieval, h.turnsWithRetrieval)})`
    );
    for (const [tool, n] of Object.entries(h.byTool).sort((a, b) => b[1] - a[1])) {
      out.push(`    ${tool}: ${n}`);
    }
  }
  return out.join("\n");
}
