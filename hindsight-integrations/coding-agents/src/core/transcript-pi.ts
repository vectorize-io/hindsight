/**
 * pi live-transcript normalizer — shared by pi and its fork Prime Agent.
 *
 * These hosts hand an extension the completed exchange as an in-memory message list on the
 * `agent_end` event (not a JSONL file like Claude/Codex), so this is a pure function over that list,
 * mirroring transcript-opencode.ts. It produces the same rich `TransportTurn[]` shape (prose turns +
 * compact `role:"action"` tool turns) and reuses the shared `stripInjectedMemory`/`actionLine`
 * helpers so a retain never feeds injected memory back into recall and tool noise stays out of the
 * bank.
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/** Structural subset of a pi message content block (TextContent | ToolCall | dropped). */
export interface PiBlock {
  type?: string;
  text?: string; // text block
  name?: string; // toolCall block: the tool name
  arguments?: unknown; // toolCall block: the call input
}

/** Structural subset of a pi message ({ role, content }). */
export interface PiMessage {
  role?: string;
  content?: unknown; // string | PiBlock[]
}

/** Structural subset of one line of a stored pi session (`~/.pi/agent/sessions/**\/*.jsonl`).
 *  Conversation lives in `type:"message"` entries; the file's first line is the `type:"session"`
 *  header (id + cwd) and the rest are settings changes, which carry no conversation. */
interface PiEntry {
  type?: string;
  timestamp?: string;
  message?: PiMessage;
}

/**
 * Render one pi message into turns. Text (string content or text blocks) joins into one
 * prose turn (injected-memory stripped); each `toolCall` block becomes its own compact
 * `role:"action"` turn (tool name + primary target via `actionLine` — no args, no output). Other
 * block types and non-conversational roles are dropped.
 */
function renderMessage(m: PiMessage): TransportTurn[] {
  if (!m || typeof m !== "object") return [];
  const role = m.role;
  if (role !== "user" && role !== "assistant") return [];

  const texts: string[] = [];
  const actions: TransportTurn[] = [];

  if (typeof m.content === "string") {
    const t = stripInjectedMemory(m.content).trim();
    if (t) texts.push(t);
  } else if (Array.isArray(m.content)) {
    for (const part of m.content) {
      if (!part || typeof part !== "object") continue;
      const block = part as PiBlock;
      if (block.type === "text" && typeof block.text === "string") {
        const t = stripInjectedMemory(block.text).trim();
        if (t) texts.push(t);
      } else if (block.type === "toolCall" && typeof block.name === "string") {
        actions.push({ role: "action", content: actionLine(block.name, block.arguments) });
      }
    }
  }

  const out: TransportTurn[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role, content: joined });
  out.push(...actions);
  return out;
}

/**
 * Normalize a pi `agent_end` message list into transcript turns (user/assistant prose plus
 * compact action turns for tool calls). Never throws on malformed entries.
 */
export function readPiMessages(messages: readonly PiMessage[]): TransportTurn[] {
  return (messages || []).flatMap((m) => renderMessage(m));
}

/**
 * Read a STORED pi session file (the history-import path) into the same turns the live `agent_end`
 * path produces.
 *
 * pi and its fork Prime Agent both persist a session as JSONL whose conversation entries wrap
 * exactly the message objects the live event hands over, so this shares `renderMessage` with
 * readPiMessages rather than re-deriving the normalization — an imported session and a live one
 * must reach the bank identically. `toolResult` entries are dropped with every other
 * non-conversational role, matching how the Codex reader drops `function_call_output`.
 *
 * Bounded and fail-open like the other stored-transcript readers: a missing file, a torn line or a
 * transcript past the size cap yields fewer turns, never a throw (see core/jsonl.ts).
 */
export function readPiTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "pi" }).lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const entry = parsed as PiEntry;
    if (entry.type !== "message" || !entry.message) continue;
    // The entry's timestamp, not the inner message's: the outer one is the ISO string every other
    // reader's turns carry, while the inner is epoch milliseconds.
    for (const turn of renderMessage(entry.message)) {
      turns.push(entry.timestamp ? { ...turn, timestamp: entry.timestamp } : turn);
    }
  }
  return turns;
}
