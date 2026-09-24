/**
 * Codex CLI/Desktop rollout (JSONL) reader — the Codex counterpart to transcript.ts (Claude). Codex's
 * transcript is a different schema: each line is an event with a `type`; the conversation lives in
 * `type:"response_item"` lines whose `payload` is one of:
 *   - message (role user/assistant/developer; content is `input_text`/`output_text` blocks)
 *   - function_call        (tool call: name + arguments JSON string)
 *   - custom_tool_call     (tool call: name + free-form input; JSON for most tools, JavaScript for
 *                           the code-mode `exec` tool)
 *   - function_call_output (tool result: output string)
 *   - reasoning            (encrypted internal chain-of-thought — dropped, like Claude `thinking`)
 *
 * What we keep, normalized to the SAME `TransportTurn[]` shape as readClaudeTranscript so the live
 * write-back (retainLiveSession) renders both identically:
 *   - user text (real prompts only) — see below.
 *   - assistant text (all phases: commentary + final_answer).
 *   - function_call → a compact `role:"action"` turn (tool name + primary target, no args);
 *     function_call_output is dropped (outputs are mechanical noise for extraction).
 *
 * User turns: a `role:"user"` response_item is NOT necessarily the user — Codex also sends its
 * startup context (<recommended_plugins>, AGENTS.md, <environment_context>), <turn_aborted>
 * notices and compaction summaries that way. What the user actually typed is recorded separately as
 * an `event_msg` `item_completed` whose item is a `UserMessage`, right after the response_item and
 * before any reply. When a rollout has those events they are the only source of user turns, so no
 * text of the injected blocks has to be recognised. Rollouts without them (older Codex) fall back to
 * the response_item plus a prefix check for the startup message.
 *
 * Action turns: code-mode `exec` input is JavaScript, so its inner calls come from the
 * `McpToolCall` / `CommandExecution` items instead, and the `exec` wrapper is skipped.
 *
 * `developer`-role messages carry Codex's system prompt AND our hook-injected context
 * (<hindsight_knowledge>, <hindsight_memories>, <user_feedback>), so dropping them entirely is what
 * prevents a retain→recall feedback loop (plus stripInjectedMemory as a defensive second pass on the
 * text we do keep). Fail-open: never throws on a missing file or a malformed line.
 */
import type { TransportTurn } from "./chat";
import { readJsonl } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

interface ContentItem {
  type?: string;
  text?: string;
}
interface Payload {
  type?: string;
  role?: string;
  content?: ContentItem[];
  name?: string;
  arguments?: string;
  input?: string;
  output?: string;
  item?: {
    type?: string;
    content?: ContentItem[];
    server?: unknown;
    tool?: unknown;
    arguments?: unknown;
    command?: unknown;
  };
}
interface RolloutLine {
  type?: string;
  // Unvalidated: the line is a cast from JSON.parse, so the guard in stampOf is what makes
  // this a string by the time it reaches a turn.
  timestamp?: unknown;
  payload?: Payload;
}

/** The rollout envelope's event time, kept so a historical backfill retains when the event
 *  actually occurred rather than when it was ingested. Guarded like the Claude/qwen/droid
 *  readers (and factored out like transcript-dsh's stampOf): a rollout line is unvalidated
 *  JSON, so a non-string timestamp must never reach TransportTurn — it would be written
 *  verbatim into the retained JSONL by chat.ts. Absent/invalid → omitted, which is the
 *  pre-existing behaviour for older rollouts that carry no timestamp at all. */
function stampOf(line: RolloutLine): { timestamp?: string } {
  return typeof line.timestamp === "string" && line.timestamp ? { timestamp: line.timestamp } : {};
}

/** Fallback for rollouts without UserMessage events: Codex records its startup instructions
 *  (AGENTS.md + environment_context) as a normal user message — drop it. */
function isSyntheticUserText(text: string): boolean {
  const s = text.trimStart();
  return s.startsWith("# AGENTS.md instructions for ") || s.startsWith("<environment_context>");
}

/** Join the text blocks of a content list (input_text/output_text/text; images have no text). */
function contentText(content: ContentItem[] | undefined): string {
  return (content || [])
    .filter((c) => c && typeof c.text === "string")
    .map((c) => c.text as string)
    .join("\n");
}

function isUserMessageEvent(line: RolloutLine): boolean {
  return (
    line.type === "event_msg" &&
    line.payload?.type === "item_completed" &&
    line.payload.item?.type === "UserMessage"
  );
}

const TOOL_ITEM_TYPES = new Set(["McpToolCall", "CommandExecution"]);

function isToolItemEvent(line: RolloutLine): boolean {
  return (
    line.type === "event_msg" &&
    line.payload?.type === "item_completed" &&
    TOOL_ITEM_TYPES.has(line.payload.item?.type ?? "")
  );
}

function parseArgs(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

const SHELL_RE = /(?:^|\/)(?:sh|bash|zsh|dash|ksh|fish)$/;

/** The script of a `["/bin/zsh", "-lc", script]` style wrapper; other argv is joined as is. */
function commandText(command: unknown): string {
  if (typeof command === "string") return command;
  if (!Array.isArray(command) || !command.every((c) => typeof c === "string")) return "";
  const argv: string[] = command;
  if (argv.length === 3 && SHELL_RE.test(argv[0]) && /^-[a-z]*c$/.test(argv[1])) return argv[2];
  return argv.join(" ");
}

/** MCP tools are named `mcp__<server>__<tool>`, as in Claude transcripts. */
function toolItemAction(item: NonNullable<Payload["item"]>): string | undefined {
  if (item.type === "McpToolCall") {
    if (typeof item.tool !== "string" || !item.tool) return undefined;
    const name =
      typeof item.server === "string" && item.server
        ? `mcp__${item.server}__${item.tool}`
        : item.tool;
    return actionLine(name, parseArgs(item.arguments));
  }
  const command = commandText(item.command).trim();
  return command ? actionLine("exec_command", { command }) : undefined;
}

/** Parse a Codex rollout JSONL into normalized markdown turns (text + tool calls/results).
 *  Drops developer/system + synthetic-startup + reasoning + injected memory + empty turns.
 *  Never throws on bad lines. */
export function readCodexTranscript(path: string): TransportTurn[] {
  const lines: RolloutLine[] = [];
  for (const rawLine of readJsonl(path)) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (typeof parsed === "object" && parsed !== null) lines.push(parsed as RolloutLine);
    } catch {
      continue;
    }
  }
  const userFromEvents = lines.some(isUserMessageEvent);
  const actionsFromEvents = lines.some(isToolItemEvent);

  const turns: TransportTurn[] = [];
  const push = (role: string, raw: string, stamp: { timestamp?: string } = {}) => {
    const text = stripInjectedMemory(raw).trim();
    if (text) turns.push({ role, content: text, ...stamp });
  };
  for (const line of lines) {
    if (isUserMessageEvent(line)) {
      push("user", contentText(line.payload?.item?.content), stampOf(line));
      continue;
    }
    if (isToolItemEvent(line)) {
      const action = toolItemAction(line.payload?.item ?? {});
      if (action) turns.push({ role: "action", content: action, ...stampOf(line) });
      continue;
    }
    if (line.type !== "response_item") continue;
    const p = line.payload;
    if (!p || typeof p !== "object") continue;

    if (p.type === "message") {
      // `developer` messages are Codex's system prompt + OUR injected hook context → drop entirely.
      if (p.role === "assistant") push("assistant", contentText(p.content), stampOf(line));
      else if (p.role === "user" && !userFromEvents) {
        const text = contentText(p.content);
        if (!isSyntheticUserText(stripInjectedMemory(text))) push("user", text, stampOf(line));
      }
    } else if (
      (p.type === "function_call" || p.type === "custom_tool_call") &&
      typeof p.name === "string"
    ) {
      // Already covered by its tool items.
      if (actionsFromEvents && p.type === "custom_tool_call" && p.name === "exec") continue;
      // Codex CLI/Desktop emits both legacy function_call and current custom_tool_call records.
      // Keep one compact action representation and never retain raw arguments.
      const rawInput = p.type === "function_call" ? p.arguments : p.input;
      let input: unknown;
      try {
        input = JSON.parse(rawInput || "");
      } catch {
        input = undefined;
      }
      turns.push({ role: "action", content: actionLine(p.name, input), ...stampOf(line) });
    }
    // reasoning / other payloads: dropped.
  }

  return turns;
}
