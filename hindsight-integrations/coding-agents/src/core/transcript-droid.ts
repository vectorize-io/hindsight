/**
 * Factory Droid session transcript reader.
 *
 * Droid persists sessions as JSONL under `~/.factory/sessions/<project-slug>/<session>.jsonl`.
 * The Stop hook hands the hook the absolute path. Lines come in two shapes that matter here:
 *
 *   {"type":"message", "timestamp": "...", "message": {"role":"user"|"assistant", "content":[...]}}
 *   {"type":"session_start"|"agent_turn_outcome", ...}   (skipped)
 *
 * Keep the parser deliberately structural, like every other reader in this family: unknown line
 * types and unknown content blocks are skipped, a partially-written final line is skipped, and a
 * malformed file yields no turns. Text blocks join into one prose turn; each `tool_use` becomes a
 * compact `role:"action"` turn (name + primary target, via `actionLine`); `tool_result` blocks are
 * dropped as mechanical noise.
 *
 * Droid persists two kinds of one-sided messages: `user_only` is UI/hook state hidden from the
 * model, while `llm_only` is model context hidden from the user. Neither is user intent, so both
 * must be skipped. In particular, retaining `llm_only` would attribute system reminders,
 * compaction summaries and hook context to the user. `stripInjectedMemory` guards the same
 * retain-to-recall feedback loop for platforms that append injected context to ordinary turns.
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

interface ContentBlock {
  type?: unknown;
  text?: unknown;
  name?: unknown;
  input?: unknown;
}

interface MessagePayload {
  role?: unknown;
  visibility?: unknown;
  content?: unknown;
}

interface TranscriptLine {
  type?: unknown;
  timestamp?: unknown;
  message?: unknown;
}

/** Render one message's `content` into turns, mirroring the Claude reader's rules. */
function renderContent(content: unknown, role: string): TransportTurn[] {
  if (!Array.isArray(content)) return [];

  const texts: string[] = [];
  const actions: TransportTurn[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as ContentBlock;
    if (b.type === "text" && typeof b.text === "string") {
      const text = stripInjectedMemory(b.text).trim();
      if (text) texts.push(text);
    } else if (b.type === "tool_use" && typeof b.name === "string") {
      actions.push({ role: "action", content: actionLine(b.name, b.input) });
    }
    // tool_result: dropped - outputs are mechanical noise for extraction
  }

  const out: TransportTurn[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role, content: joined });
  out.push(...actions);
  return out;
}

/** Read a Factory Droid transcript JSONL into normalized turns. Never throws. */
export function readDroidTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "factory-droid" }).lines) {
    if (!rawLine.trim()) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawLine);
    } catch {
      continue;
    }
    if (!parsed || typeof parsed !== "object") continue;
    const line = parsed as TranscriptLine;
    if (line.type !== "message" || !line.message || typeof line.message !== "object") continue;

    const message = line.message as MessagePayload;
    const role = message.role;
    if (role !== "user" && role !== "assistant") continue;
    if (message.visibility === "user_only" || message.visibility === "llm_only") continue;

    const stamp =
      typeof line.timestamp === "string" && line.timestamp ? { timestamp: line.timestamp } : {};
    turns.push(...renderContent(message.content, role).map((turn) => ({ ...turn, ...stamp })));
  }
  return turns;
}
