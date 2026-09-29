/**
 * ZCode's EPHEMERAL Stop transcript reader.
 *
 * ZCode embeds the Claude Code agent runtime, so its records look like Claude's — one JSON object
 * per line wrapping a `message` — but the file the Stop hook is handed is not Claude's durable
 * session transcript. It is a temp file the agent writes for the hook and deletes as soon as the
 * hook returns, it holds the ASSISTANT side only, and its records omit the top-level `type` that
 * `readClaudeTranscript` drives role from. Hence a reader of its own, and hence the journal
 * (core/turn-journal.ts) for the conversation as a whole.
 *
 * Its only job here is a FALLBACK: the Stop payload normally carries the whole reply in
 * `responseText`, and this file answers the same question when it does not.
 *
 * Fail-open: a missing file, a malformed line, or a line that parses to a non-object yields no
 * turns rather than throwing — a Stop hook that throws loses the turn it was there to retain.
 */
import { readFileSync } from "node:fs";
import type { TransportTurn } from "./chat";
import { stripInjectedMemory } from "./transcript-util";

/** One `content` block of a Claude-shaped message. Only `text` blocks carry prose. */
interface ContentBlock {
  type?: string;
  text?: unknown;
}

/** Join the text blocks of a Claude-shaped `content` field, which is a string or a block list. */
function messageText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return (content as ContentBlock[])
    .filter((block) => block && typeof block === "object" && typeof block.text === "string")
    .map((block) => (block.text as string).trim())
    .filter(Boolean)
    .join("\n")
    .trim();
}

/**
 * Parse ZCode's Stop transcript into normalized turns.
 *
 * Role comes from `message.role`, falling back to the record's `type` — the reverse of the Claude
 * and Qwen readers, which prefer `type` because their assistant records carry a misleading nested
 * role. ZCode's ephemeral records have no `type` at all, so the nested role is the only evidence
 * present on the shape this reader exists for.
 */
export function readZcodeTranscript(path: string): TransportTurn[] {
  if (!path) return [];
  let body: string;
  try {
    body = readFileSync(path, "utf8");
  } catch {
    return [];
  }
  const turns: TransportTurn[] = [];
  for (const line of body.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const record = parsed as { type?: unknown; timestamp?: unknown; message?: unknown };
    const message =
      typeof record.message === "object" && record.message !== null
        ? (record.message as { role?: unknown; content?: unknown })
        : undefined;
    const role = typeof message?.role === "string" ? message.role : record.type;
    if (role !== "user" && role !== "assistant") continue;
    const content = stripInjectedMemory(messageText(message?.content)).trim();
    if (!content) continue;
    turns.push({
      role,
      content,
      ...(typeof record.timestamp === "string" ? { timestamp: record.timestamp } : {}),
    });
  }
  return turns;
}

/** The last assistant reply in a ZCode Stop transcript, or "" when it holds none. */
export function zcodeAssistantText(path: string | undefined): string {
  if (!path) return "";
  return (
    readZcodeTranscript(path)
      .filter((turn) => turn.role === "assistant")
      .at(-1)?.content ?? ""
  );
}
