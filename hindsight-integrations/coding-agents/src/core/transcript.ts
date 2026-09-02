/**
 * Claude Code session transcript (JSONL) reader: parses the raw per-line event log into normalized
 * turns for write-back. Keeps the ENGINEERING SUBSTANCE of a coding session — user requests,
 * assistant narration, AND the concrete actions it took: each `tool_use` becomes a compact
 * `role:"action"` turn naming the tool and its primary target (e.g. `Edit boltons/strutils.py`) —
 * no arguments, no outputs. That keeps WHICH files/commands the session touched without burying the
 * decisions in mechanical noise; `tool_result` blocks are dropped entirely.
 *
 * Drops non-message lines (`last-prompt`, `mode`, `summary`, …), `isMeta` lines, compaction
 * summaries (`isCompactSummary`), `isSidechain`
 * (subagent/Task) lines, `thinking` blocks, and turns that render to nothing. Injected recall
 * context (`<hindsight_memories>` / `<hindsight_bank>` / `<relevant_memories>`) is stripped so the
 * write-back can't feed recalled memory back into the bank (a retain→recall feedback loop). A
 * Fail-open: never throws on a missing file, malformed line, or a line that parses to a
 * non-object JSON value (`null`, a number, a boxed primitive, …).
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

interface ContentBlock {
  type?: string;
  text?: string;
  name?: string;
  input?: unknown;
  content?: string | ContentBlock[];
}

interface TranscriptLine {
  type?: string;
  isMeta?: boolean;
  isSidechain?: boolean;
  isCompactSummary?: boolean;
  timestamp?: string;
  message?: {
    content?: string | ContentBlock[];
  };
}

interface RenderedLine {
  role: string;
  content: string;
}

/**
 * Render one message's `content` into turns: text blocks join into one prose turn
 * (injected-memory stripped); each `tool_use` becomes its own compact `role:"action"` turn
 * (name + primary target, via `actionLine`); `tool_result` blocks are dropped.
 */
function renderLine(content: string | ContentBlock[] | undefined, type: string): RenderedLine[] {
  if (typeof content === "string") {
    const text = stripInjectedMemory(content).trim();
    return text ? [{ role: type, content: text }] : [];
  }
  if (!Array.isArray(content)) return [];

  const texts: string[] = [];
  const actions: RenderedLine[] = [];
  for (const b of content) {
    if (!b || typeof b !== "object") continue;
    if (b.type === "text" && typeof b.text === "string") {
      const t = stripInjectedMemory(b.text).trim();
      if (t) texts.push(t);
    } else if (b.type === "tool_use" && typeof b.name === "string") {
      actions.push({ role: "action", content: actionLine(b.name, b.input) });
    }
    // tool_result: dropped — outputs are mechanical noise for extraction
  }

  const out: RenderedLine[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role: type, content: joined });
  out.push(...actions);
  return out;
}

/** Parse a Claude Code transcript JSONL into normalized markdown turns (text + tool calls/results).
 *  Drops thinking blocks, isMeta/isSidechain lines, injected memory, and empty turns.
 *  Never throws on bad lines. */
export function readClaudeTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "claude-code" }).lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    // JSON.parse accepts non-object top-level values (`null`, numbers, booleans, arrays);
    // guard here so a corrupt/truncated line can't reach a property access below and throw.
    if (typeof parsed !== "object" || parsed === null) continue;
    const line = parsed as TranscriptLine;

    if (line.type !== "user" && line.type !== "assistant") continue;
    if (line.isMeta === true) continue;
    if (line.isSidechain === true) continue;
    // Claude Code's own recap of the conversation so far, written when the context window fills.
    // It arrives as a plain type:"user" record with NO isMeta flag, so nothing else filters it and
    // a ~16KB machine-written summary was retained as something the user said. It is also a
    // summary of turns ALREADY retained — compaction appends to the transcript rather than
    // rewriting it — so keeping it extracts the same decisions twice (#3379).
    if (line.isCompactSummary === true) continue;
    if (typeof line.message !== "object" || line.message === null) continue;

    // `type` is validated as "user" | "assistant" above; drive role from it (not the redundant
    // `message.role`). One line can yield a prose turn plus one action turn per tool call.
    for (const rendered of renderLine(line.message.content, line.type)) {
      const turn: TransportTurn = { role: rendered.role, content: rendered.content };
      if (typeof line.timestamp === "string") turn.timestamp = line.timestamp;
      turns.push(turn);
    }
  }

  return turns;
}
