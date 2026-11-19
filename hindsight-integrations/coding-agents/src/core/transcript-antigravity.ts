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

/**
 * Recover the current user prompt for Antigravity's `PreInvocation` hook ONLY when the invocation
 * is the start of a new user turn.
 *
 * Antigravity fires `PreInvocation` before every planner step in the agent loop — not just on user
 * submission, but also after every tool result (`GENERIC`), tool-only planner step
 * (`PLANNER_RESPONSE` with empty `content`), and background task wakeup (`SYSTEM_MESSAGE`). Because
 * `readAntigravityTranscript` drops non-chat and empty-content records, taking `.filter(u).at(-1)`
 * from it re-emits the opening user prompt on every intra-turn tool step (e.g. 182 times across a
 * 9-turn session). Checking that the last non-housekeeping record in the JSONL is a `USER_INPUT` /
 * `user` record suppresses every intra-turn continuation via `runHook`'s `if (!prompt) return`.
 */
export function readAntigravityPromptTurn(path: string | undefined): string | undefined {
  if (!path) return undefined;
  let lastRole: string | undefined;
  let lastUserPrompt: string | undefined;
  for (const line of readJsonl(path)) {
    try {
      const event = JSON.parse(line) as {
        role?: string;
        type?: string;
        content?: string;
        text?: string;
        message?: { role?: string; content?: string; text?: string };
      };
      const role = event.role ?? event.message?.role ?? event.type;
      if (!role || role === "CHECKPOINT" || role === "EPHEMERAL_MESSAGE") continue;
      lastRole = role;
      if (role === "user" || role === "USER_INPUT") {
        const content =
          event.content ?? event.text ?? event.message?.content ?? event.message?.text;
        const text = typeof content === "string" ? stripInjectedMemory(content) : "";
        const clean = unwrapUserRequest(text).trim();
        lastUserPrompt = clean || undefined;
      }
    } catch {
      // Ignore malformed lines.
    }
  }
  return lastRole === "user" || lastRole === "USER_INPUT" ? lastUserPrompt : undefined;
}
