/**
 * A harness transcript line as the messages of a Hindsight session append (POST
 * .../sessions/{id}/events): the OpenAI chat format most agents keep, plus an id per message. What
 * was said, the tools called, what they returned. Each id is derived from the transcript itself
 * (the line's uuid, or its byte offset), so a line sent twice — a retry, two hooks reading the same
 * tail — is stored once by the server.
 *
 * Text is kept whole: the session saves the conversation into memory from these messages, so a cut
 * message would be a lost one. Tool output is the exception — it is only read to spot failures, and
 * is never saved.
 */
import { stripInjectedMemory } from "./transcript-util";

/** One message of a session append. */
export interface SessionMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "system";
  content?: string;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
  is_error?: boolean;
}

/** How the session is set up (sent with the first append; see the server's SessionOptions). */
export interface SessionOptions {
  document_id?: string;
  live?: boolean;
  retain?: "turns" | "idle" | "off";
  tags?: string[];
  metadata?: Record<string, string>;
  context?: string;
  strategy?: string;
  observation_scopes?: unknown;
}

/** The body of one append. `state` is the working tree's latest snapshot, `ack` the last note
 *  this client handed over. */
export interface SessionAppend {
  messages: SessionMessage[];
  state?: { diff?: string };
  turn_end?: boolean;
  ack: number;
  session?: SessionOptions;
}

export interface SessionNote {
  seq: number;
  text: string;
  source?: { document_id: string; quote?: string; mentioned_at?: string | null } | null;
}

/** `checking`: the server is still judging the latest moment, so a note may follow. */
export type AppendResult =
  | { supported: true; notes: SessionNote[]; checking?: boolean }
  | { supported: false };

/** One transcript line, and where it starts in the file (the id when the line has none). */
export type EventsOf = (line: Record<string, unknown>, offset: number) => SessionMessage[];

const MESSAGE_CAP = 50_000;
const RESULT_CAP = 8_000;
/** Tool inputs keep every key; a long value (a file a Write creates) is cut, it is never saved. */
const INPUT_VALUE_CAP = 2_000;

const cap = (s: string, n: number): string => (s.length > n ? s.slice(0, n) : s);

function compactInput(input: unknown): Record<string, unknown> {
  if (!input || typeof input !== "object") return {};
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(input as Record<string, unknown>))
    out[k] = typeof v === "string" ? cap(v, INPUT_VALUE_CAP) : v;
  return out;
}

const textOf = (content: unknown): string =>
  typeof content === "string"
    ? content
    : Array.isArray(content)
      ? (content as Record<string, unknown>[])
          .map((c) => (typeof c?.text === "string" ? c.text : ""))
          .filter(Boolean)
          .join("\n")
      : "";

/** An assistant message calling one tool. The server ids the call `<message id>:<call id>`. */
const toolCall = (
  callId: string,
  name: string,
  input: Record<string, unknown>
): Omit<SessionMessage, "id"> => ({
  role: "assistant",
  tool_calls: [
    { id: callId, type: "function", function: { name, arguments: JSON.stringify(input) } },
  ],
});

/** Claude Code: user prompts, assistant text and tool calls, tool results. Meta lines, subagents'
 *  lines and compaction summaries are not the conversation. */
export const claudeEventsOf: EventsOf = (line, offset) => {
  if (line.isMeta === true || line.isSidechain === true || line.isCompactSummary === true)
    return [];
  const msg = line.message as { content?: unknown } | undefined;
  const content = msg && typeof msg === "object" ? msg.content : undefined;
  const base = typeof line.uuid === "string" && line.uuid ? line.uuid : `o${offset}`;
  const out: SessionMessage[] = [];
  const push = (m: Omit<SessionMessage, "id">) => out.push({ id: `${base}:${out.length}`, ...m });
  const said = (role: "user" | "assistant", text: string) =>
    push({ role, content: cap(stripInjectedMemory(text), MESSAGE_CAP) });
  if (line.type === "assistant" && Array.isArray(content)) {
    for (const b of content as Record<string, unknown>[]) {
      if (b.type === "text" && typeof b.text === "string" && b.text.trim())
        said("assistant", b.text);
      else if (b.type === "tool_use")
        push(toolCall(String(b.id ?? ""), String(b.name ?? ""), compactInput(b.input)));
    }
  } else if (line.type === "user") {
    if (typeof content === "string" && content.trim()) said("user", content);
    for (const b of Array.isArray(content) ? (content as Record<string, unknown>[]) : []) {
      if (b.type === "text" && typeof b.text === "string" && b.text.trim()) said("user", b.text);
      else if (b.type === "tool_result")
        push({
          role: "tool",
          tool_call_id: String(b.tool_use_id ?? ""),
          content: cap(textOf(b.content), RESULT_CAP),
          ...(b.is_error === true ? { is_error: true } : {}),
        });
    }
  }
  return out;
};

/** `["/bin/zsh", "-lc", script]` → script; a `cmd` string is already one. */
function codexCommand(item: Record<string, unknown>): string {
  const c = item.cmd ?? item.command;
  if (typeof c === "string") return c;
  if (!Array.isArray(c)) return "";
  return c.length === 3 && /^-[a-z]*c$/.test(String(c[1])) ? String(c[2]) : c.join(" ");
}

/** Codex: the finished items of its rollout (prompt, agent message, shell command, file change),
 *  read as Claude's Bash and Edit so the live session judges both with one set of rules. */
export const codexEventsOf: EventsOf = (line, offset) => {
  const p = (line.payload ?? {}) as Record<string, unknown>;
  if (line.type !== "event_msg" || p.type !== "item_completed") return [];
  const item = (p.item ?? {}) as Record<string, unknown>;
  const callId = String(item.id ?? `o${offset}`);
  const base = `${callId}@${offset}`;
  const msg = (i: number, m: Omit<SessionMessage, "id">): SessionMessage => ({
    id: `${base}:${i}`,
    ...m,
  });
  const result = (output: unknown, failed: boolean): SessionMessage =>
    msg(1, {
      role: "tool",
      tool_call_id: callId,
      content: cap(String(output ?? ""), RESULT_CAP),
      ...(failed ? { is_error: true } : {}),
    });
  switch (item.type) {
    case "UserMessage": {
      const said = textOf(item.content);
      return said.trim()
        ? [msg(0, { role: "user", content: cap(stripInjectedMemory(said), MESSAGE_CAP) })]
        : [];
    }
    case "AgentMessage": {
      const said = textOf(item.content);
      return said.trim() ? [msg(0, { role: "assistant", content: cap(said, MESSAGE_CAP) })] : [];
    }
    case "CommandExecution":
      return [
        msg(0, toolCall(callId, "Bash", { command: codexCommand(item) })),
        result(
          item.aggregated_output ?? item.stdout,
          item.exit_code !== undefined && Number(item.exit_code) !== 0
        ),
      ];
    case "FileChange": {
      const files = Object.keys((item.changes ?? {}) as Record<string, unknown>);
      return [
        msg(0, toolCall(callId, "Edit", { file_path: files.join(", ") })),
        result(item.stdout, item.status === "failed"),
      ];
    }
    default:
      return [];
  }
};

/** The harnesses whose transcript the plugin reads into a session, and the built entry file of
 *  each one's live session hook (core/live-entry.ts), which it also starts detached. */
export const SESSION_HARNESSES: Record<string, { entry: string; eventsOf: EventsOf }> = {
  "claude-code": { entry: "claude-live-hook.js", eventsOf: claudeEventsOf },
  codex: { entry: "codex-live-hook.js", eventsOf: codexEventsOf },
};
