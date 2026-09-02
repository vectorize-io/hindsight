/** Cursor CLI transcript reader for its `stop` write-back hook. */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

interface ContentBlock {
  type?: string;
  text?: string;
  name?: string;
  input?: unknown;
}

interface CursorLine {
  type?: string;
  role?: string;
  timestamp?: string;
  message?: { role?: string; content?: string | ContentBlock[] };
  content?: string | ContentBlock[];
  name?: string;
  args?: unknown;
}

function withTimestamp(turn: Omit<TransportTurn, "timestamp">, timestamp?: string): TransportTurn {
  return timestamp ? { ...turn, timestamp } : turn;
}

function textFrom(content: string | ContentBlock[] | undefined): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((block) => block?.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("\n");
}

/** Parse Cursor's JSONL message and tool-call events into durable text and compact action turns. */
export function readCursorTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "cursor-cli" }).lines) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawLine);
    } catch {
      continue;
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const line = parsed as CursorLine;
    const role =
      line.message?.role ??
      line.role ??
      (line.type === "user" || line.type === "assistant" ? line.type : undefined);
    const content = line.message?.content ?? line.content;

    if (role === "user" || role === "assistant") {
      const text = stripInjectedMemory(textFrom(content)).trim();
      if (text) turns.push(withTimestamp({ role, content: text }, line.timestamp));
      if (role === "assistant" && Array.isArray(content)) {
        for (const block of content) {
          if (block?.type === "tool_use" && typeof block.name === "string") {
            turns.push(
              withTimestamp(
                { role: "action", content: actionLine(block.name, block.input) },
                line.timestamp
              )
            );
          }
        }
      }
    } else if (line.type === "tool_call" && typeof line.name === "string") {
      turns.push(
        withTimestamp({ role: "action", content: actionLine(line.name, line.args) }, line.timestamp)
      );
    }
  }
  return turns;
}
