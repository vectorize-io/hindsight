/** Grok Build persists each session under ~/.grok/sessions/<url-encoded-cwd>/<session-id>.
 * Its Stop payload deliberately contains only the common camelCase session fields, so resolve the
 * local chat history ourselves instead of pretending it has Claude's transcript_path field. */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

const CHAT_HISTORY = "chat_history.jsonl";

/** Locate Grok's persisted chat history. Long workspace paths use a slug/hash directory with a
 * `.cwd` sidecar, so fall back to that documented layout when the direct encoded path is absent. */
export function grokTranscriptPath(
  cwd: string,
  sessionId: string,
  grokHome = process.env.GROK_HOME || join(homedir(), ".grok")
): string {
  const sessionsDir = join(grokHome, "sessions");
  const direct = join(sessionsDir, encodeURIComponent(cwd), sessionId, CHAT_HISTORY);
  if (existsSync(direct)) return direct;

  try {
    for (const candidate of readdirSync(sessionsDir)) {
      const directory = join(sessionsDir, candidate);
      if (readFileSync(join(directory, ".cwd"), "utf8").trim() === cwd) {
        return join(directory, sessionId, CHAT_HISTORY);
      }
    }
  } catch {
    // The direct path remains the best diagnostics target; the reader below fails open if absent.
  }
  return direct;
}

/** Normalize Grok's persisted chat-history records into user, assistant, and compact action turns.
 * Synthetic user records carry `synthetic_reason`; only prompt-indexed records are actual user work. */
export function readGrokTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "grok-build" }).lines) {
    try {
      const event = JSON.parse(rawLine) as {
        type?: string;
        content?: string | Array<{ type?: string; text?: string }>;
        prompt_index?: unknown;
        tool_calls?: Array<{ name?: string; arguments?: string | Record<string, unknown> }>;
      };
      if (event.type !== "user" && event.type !== "assistant") continue;
      if (event.type === "user" && typeof event.prompt_index !== "number") continue;

      const content =
        typeof event.content === "string"
          ? event.content
          : Array.isArray(event.content)
            ? event.content
                .filter((part) => part.type === "text")
                .map((part) => part.text ?? "")
                .join("\n")
            : "";
      const clean = stripInjectedMemory(content).trim();
      if (clean) turns.push({ role: event.type, content: clean });

      for (const toolCall of event.tool_calls ?? []) {
        if (!toolCall.name) continue;
        let input: unknown = toolCall.arguments;
        if (typeof input === "string") {
          try {
            input = JSON.parse(input) as unknown;
          } catch {
            /* preserve the raw string */
          }
        }
        turns.push({ role: "action", content: actionLine(toolCall.name, input) });
      }
    } catch {
      // Grok may write partial JSON while a session is stopping; ignore malformed records.
    }
  }
  return turns;
}
