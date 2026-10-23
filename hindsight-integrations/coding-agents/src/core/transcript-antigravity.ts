import type { TransportTurn } from "./chat";
import { readJsonl } from "./jsonl";
import { stripInjectedMemory } from "./transcript-util";

/** agy wraps each prompt as `<USER_REQUEST>…</USER_REQUEST>` plus an `<ADDITIONAL_METADATA>` block
 * (local time and the like); keep only the request so recall and retain see what the user typed (#5791). */
function unwrapUserRequest(text: string): string {
  const request = /<USER_REQUEST>([\s\S]*?)<\/USER_REQUEST>/.exec(text);
  return request
    ? request[1]
    : text.replace(/<ADDITIONAL_METADATA>[\s\S]*?<\/ADDITIONAL_METADATA>/g, "");
}

/** Read Antigravity's transcript JSONL defensively. The documented hook contract guarantees only
 * the path, not the internal event schema, so support its user/assistant role and message variants. */
export function readAntigravityTranscript(path: string | undefined): TransportTurn[] {
  if (!path) return [];
  const turns: TransportTurn[] = [];
  for (const line of readJsonl(path)) {
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
      const text = typeof content === "string" ? stripInjectedMemory(content) : "";
      const clean = (normalizedRole === "user" ? unwrapUserRequest(text) : text).trim();
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
