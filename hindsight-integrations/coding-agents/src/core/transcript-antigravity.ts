import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { stripInjectedMemory } from "./transcript-util";

/** Read Antigravity's transcript JSONL defensively. The documented hook contract guarantees only
 * the path, not the internal event schema, so support its user/assistant role and message variants. */
export function readAntigravityTranscript(path: string | undefined): TransportTurn[] {
  if (!path) return [];
  const turns: TransportTurn[] = [];
  for (const line of readJsonlTail(path, { scope: "antigravity-cli" }).lines) {
    try {
      const event = JSON.parse(line) as {
        role?: string;
        type?: string;
        content?: string;
        text?: string;
        message?: { role?: string; content?: string; text?: string };
        timestamp?: string;
      };
      const role = event.role ?? event.message?.role ?? event.type;
      const normalizedRole =
        role === "user" || role === "USER_INPUT"
          ? "user"
          : role === "assistant" || role === "model" || role === "PLANNER_RESPONSE"
            ? "assistant"
            : undefined;
      const content = event.content ?? event.text ?? event.message?.content ?? event.message?.text;
      const clean = typeof content === "string" ? stripInjectedMemory(content).trim() : "";
      if (normalizedRole && clean) {
        turns.push({
          role: normalizedRole,
          content: clean,
          ...(event.timestamp ? { timestamp: event.timestamp } : {}),
        });
      }
    } catch {
      // Transcript records may include agent/tool metadata; ignore unknown or malformed lines.
    }
  }
  return turns;
}
