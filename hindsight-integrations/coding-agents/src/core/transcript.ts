/**
 * Claude Code session transcript (JSONL) reader: parses the raw per-line event log into normalized
 * turns for write-back. Keeps the ENGINEERING SUBSTANCE of a coding session — user requests,
 * assistant narration, AND the concrete actions it took: each `tool_use` becomes a compact
 * `role:"action"` turn naming the tool and its primary target (e.g. `Edit boltons/strutils.py`) —
 * no arguments, no outputs — unless `keepToolInputs` names it as a tool the agent speaks through
 * (a Channels reply), when it becomes an assistant turn holding that text. That keeps WHICH
 * files/commands the session touched without burying the decisions in mechanical noise;
 * `tool_result` blocks are dropped, except AskUserQuestion's, which carries the user's answers. A
 * message from a Claude Code Channel (Telegram, …) is kept as a user turn though it is marked meta.
 * A message the human typed while a turn was running (stored only as a `queued_command`
 * attachment) is kept as a user turn at its position. A forked session's copy
 * of its parent's history is skipped: the parent's document already holds it.
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
import { readJsonl } from "./jsonl";
import { stripInjectedMemory, toolTurn } from "./transcript-util";

interface ContentBlock {
  type?: string;
  text?: string;
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  is_error?: boolean;
  content?: string | ContentBlock[];
}

interface TranscriptLine {
  type?: string;
  isMeta?: boolean;
  isSidechain?: boolean;
  isCompactSummary?: boolean;
  /** Where a meta line came from; `channel` is a message from a Claude Code Channel (Telegram, …). */
  origin?: { kind?: string };
  timestamp?: string;
  sessionId?: string;
  requestSessionId?: string;
  attachment?: {
    type?: string;
    prompt?: unknown;
    origin?: { kind?: string };
  };
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
 * (name + primary target, via `toolTurn`); `tool_result` blocks are dropped, except the result
 * of an AskUserQuestion call seen earlier (`questionIds`), which is the user's answer.
 */
function renderLine(
  content: string | ContentBlock[] | undefined,
  type: string,
  questionIds: Set<string>,
  failedIds: Set<string>
): RenderedLine[] {
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
      if (b.name === "AskUserQuestion" && typeof b.id === "string") questionIds.add(b.id);
      // A reply the tool failed to send was never said: it stays the bare action line.
      actions.push(toolTurn(b.name, b.input, typeof b.id === "string" && failedIds.has(b.id)));
    } else if (b.type === "tool_result" && b.tool_use_id && questionIds.has(b.tool_use_id)) {
      // The one tool result that is the USER talking: Claude Code returns the picked answers to
      // AskUserQuestion as its result, so dropping it lost every decision made there (#6421).
      const answer = renderLine(b.content, type, questionIds, failedIds)
        .map((r) => r.content)
        .join("\n");
      if (answer) texts.push(answer);
    }
    // any other tool_result: dropped — outputs are mechanical noise for extraction
  }

  const out: RenderedLine[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role: type, content: joined });
  out.push(...actions);
  return out;
}

/** Parse a Claude Code transcript JSONL into normalized markdown turns (text + tool calls/results,
 *  plus human messages queued mid-turn). Drops thinking blocks, isMeta/isSidechain lines, injected
 *  memory, and empty turns.
 *  Never throws on bad lines. */
export function readClaudeTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  const questionIds = new Set<string>(); // AskUserQuestion calls whose result is the user's answer
  const lines: TranscriptLine[] = [];
  for (const rawLine of readJsonl(path)) {
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
    lines.push(parsed as TranscriptLine);
  }

  // Tool calls whose result is an error: a reply tool that failed delivered nothing (#6481).
  const failedIds = new Set<string>();
  for (const l of lines) {
    const content = l.message?.content;
    if (!Array.isArray(content)) continue;
    for (const b of content) {
      if (b?.type === "tool_result" && b.is_error === true && typeof b.tool_use_id === "string")
        failedIds.add(b.tool_use_id);
    }
  }

  // A forked session (`/fork`, `--fork-session`) starts with a copy of its parent's history, under
  // the fork's own sessionId. That history is already in the parent's document, so saving it again
  // stored and extracted every inherited turn twice (#6642). Claude Code stamps each copied
  // assistant line with the session that produced it (`requestSessionId`); the last line naming
  // another session marks the end of the inherited part, and only what follows is the fork's own.
  let start = 0;
  lines.forEach((l, i) => {
    if (l.requestSessionId && l.sessionId && l.requestSessionId !== l.sessionId) start = i + 1;
  });
  // A session Claude Code continues under a new id starts with a byte-for-byte copy of the old
  // file, its lines keeping the OLD sessionId and no requestSessionId, so the check above misses
  // it and the whole old history was saved again under the new id (#6960). The file's last
  // sessionId is the session being read; a line naming any other was copied in.
  let own: string | undefined;
  for (const l of lines) own = l.sessionId ?? own;

  for (const line of lines.slice(start)) {
    if (line.sessionId && line.sessionId !== own) continue;
    // A message the human typed while a turn was still running is NOT a type:"user" line: Claude
    // Code stores it only as a `queued_command` attachment. Keep the human ones as user turns at
    // their position; other origins (e.g. `task-notification`) are not the human speaking (#6480).
    // A Channels message sent while the agent was busy is queued the same way, still wrapped in its
    // `<channel …>` tag (#6481).
    const queued = line.type === "attachment" ? line.attachment : undefined;
    const kind = queued?.origin?.kind;
    if (queued?.type === "queued_command" && (kind === "human" || kind === "channel")) {
      const prompt = typeof queued.prompt === "string" ? queued.prompt : "";
      const text = stripInjectedMemory(kind === "channel" ? unwrapChannel(prompt) : prompt).trim();
      if (text) {
        const turn: TransportTurn = { role: "user", content: text };
        if (typeof line.timestamp === "string") turn.timestamp = line.timestamp;
        turns.push(turn);
      }
      continue;
    }

    if (line.type !== "user" && line.type !== "assistant") continue;
    // A Channels message (Telegram, Discord, …) is the human speaking, though Claude Code marks it
    // meta like the lines it injects itself; skipping it dropped the whole inbound side (#6481).
    const channel = line.isMeta === true && line.origin?.kind === "channel";
    if (line.isMeta === true && !channel) continue;
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
    const content = channel ? channelText(line.message.content) : line.message.content;
    for (const rendered of renderLine(content, line.type, questionIds, failedIds)) {
      const turn: TransportTurn = { role: rendered.role, content: rendered.content };
      if (typeof line.timestamp === "string") turn.timestamp = line.timestamp;
      turns.push(turn);
    }
  }

  return turns;
}

/** The message inside a Channels line: Claude Code wraps it in `<channel source="…" …>…</channel>`. */
function channelText(
  content: string | ContentBlock[] | undefined
): string | ContentBlock[] | undefined {
  if (typeof content === "string") return unwrapChannel(content);
  // Stored as blocks (text plus an image, …): unwrap each text block, keep the rest as it is.
  return content?.map((b) =>
    b?.type === "text" && typeof b.text === "string" ? { ...b, text: unwrapChannel(b.text) } : b
  );
}

/** One `<channel …>…</channel>` text without its tag. The sender is kept in front (`alice: …`):
 *  in a group chat it is who said it. */
function unwrapChannel(t: string): string {
  return t
    .replace(/^\s*<channel\b([^>]*)>\s*/, (_, attrs: string) => {
      const user = /\buser="([^"]*)"/.exec(attrs)?.[1];
      return user ? `${user}: ` : "";
    })
    .replace(/\s*<\/channel>\s*$/, "");
}
