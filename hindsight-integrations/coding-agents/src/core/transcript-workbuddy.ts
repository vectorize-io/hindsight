/**
 * WorkBuddy (and CodeBuddy Code — the same `@genie/agent-cli` engine) session transcript (JSONL)
 * reader.
 *
 * The file lives at `~/.workbuddy/projects/<cwd 的 "/" 换成 "-">/<uuid>.jsonl` and is NOT Claude's
 * schema: every record is `{ type, timestamp, cwd, … }`, and the conversation lives in
 * `type:"message"` records whose `role`/`content` sit at the TOP level. `content` is a BLOCK ARRAY
 * (`input_text` for the user, `output_text` for the assistant) — `readClaudeTranscript`'s
 * `type:"user"` gate would skip every line, which is why this host needs a reader of its own.
 *
 * What we keep, normalized to the SAME `TransportTurn[]` shape as every other reader:
 *   - user prompts (the host wraps them — see `unwrapUserQuery`)
 *   - assistant prose (`output_text` blocks)
 *   - each `function_call` → a compact `role:"action"` turn (tool name + primary target, no args)
 *
 * What we drop:
 *   - `function_call_result` — tool OUTPUT, the mechanical noise `actionLine` exists to keep out of
 *     the bank (the matching call is a separate `function_call` record, so nothing is lost).
 *   - `reasoning` — the model's own chain of thought, like Claude `thinking` (never retained).
 *   - `file-history-snapshot` / `ai-title` / `summary` / `resend-fork-notice` — UI metadata.
 *   - the host's OWN machine-written user records — context-compaction summaries and background-task
 *     notices (`isSyntheticUserText`). Retaining those files scaffolding as the user's words AND
 *     re-extracts turns already in the bank: the same failure Claude's `isCompactSummary` guard
 *     prevents (#3379).
 *
 * Fail-open: never throws on a missing file, a malformed line, or a line that parses to a non-object
 * JSON value (`null`, a number, …).
 */
import type { TransportTurn } from "./chat";
import { readJsonl } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/** One entry of a WorkBuddy message's `content` block array. Only text blocks carry prose. */
interface WorkbuddyBlock {
  type?: string;
  text?: string;
}

/** Structural subset of a WorkBuddy transcript record. */
interface WorkbuddyLine {
  type?: string;
  role?: string;
  /** Epoch MILLISECONDS — WorkBuddy stamps every record this way (not the ISO strings Claude uses). */
  timestamp?: number;
  content?: string | WorkbuddyBlock[] | null;
  name?: string;
  /** The model's raw JSON string of tool arguments, unparsed by design. */
  arguments?: string;
}

/** Join a message's text blocks (`input_text` for the user, `output_text` for the assistant). */
function messageText(content: WorkbuddyLine["content"]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((b) => b && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("\n");
}

/**
 * WorkBuddy wraps the human's prompt in `<user_query>…</user_query>`, after a `<system-reminder>`
 * block of host context that `stripInjectedMemory` already removes. It may also append an attached
 * image's local path in `<image_local_path>…</image_local_path>`. Both are host scaffolding, not the
 * user's words, so unwrap them — ANCHORED, so a prompt that merely QUOTES a tag is left alone (the
 * discipline transcript-qwen.ts documents for the same reason: a substring replace would delete a
 * genuine prompt that happens to mention the tag).
 *
 * Measured over 21 local transcripts / 334 user records: 284 are exactly a `<user_query>`, 3 carry
 * the image tail, and no other shape appears.
 *
 * Exported for the CodeBuddy IDE reader: this wrapper belongs to the ENGINE, so both of its hosts
 * spell it the same way (core/transcript-codebuddy-ide.ts).
 */
const USER_QUERY_RE =
  /^\s*<user_query>([\s\S]*?)<\/user_query>\s*(?:<image_local_path>[\s\S]*?<\/image_local_path>\s*)?$/;

export function unwrapUserQuery(text: string): string {
  const match = USER_QUERY_RE.exec(text);
  return (match ? match[1] : text).trim();
}

/**
 * Records WorkBuddy writes as `role:"user"` that the human did NOT say: the context-compaction
 * summaries (written when the window fills, and a summary of turns ALREADY retained) and
 * background-task notices. Prefix-based, like transcript-codex.ts's `isSyntheticUserText`.
 *
 * Exported for the CodeBuddy IDE reader, whose own store carries the same `<cb_summary>` marker.
 */
export function isSyntheticUserText(text: string): boolean {
  const s = text.trimStart();
  return (
    s.startsWith("<cb_summary>") ||
    s.startsWith("<conversation_history_summary>") ||
    s.startsWith("Use the TaskOutput tool ")
  );
}

/** `function_call.arguments` is the model's RAW JSON string, unparsed by design. */
function parseArgs(raw: string | undefined): unknown {
  if (!raw) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    return raw; // a malformed argument string still names what the model was reaching for
  }
}

/** WorkBuddy stamps epoch milliseconds; absent only on malformed input. */
function stampOf(line: WorkbuddyLine): { timestamp?: string } {
  return typeof line.timestamp === "number"
    ? { timestamp: new Date(line.timestamp).toISOString() }
    : {};
}

/**
 * Parse a WorkBuddy transcript JSONL into normalized turns (text + tool calls).
 * Drops tool results, reasoning, host metadata, machine-written user records, injected memory and
 * empty turns. Never throws on bad lines.
 */
export function readWorkbuddyTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonl(path)) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    // JSON.parse accepts non-object top-level values (`null`, numbers, …); guard so a corrupt line
    // cannot reach a property access below and throw.
    if (typeof parsed !== "object" || parsed === null) continue;
    const line = parsed as WorkbuddyLine;
    const stamp = stampOf(line);

    if (line.type === "message") {
      if (line.role !== "user" && line.role !== "assistant") continue;
      let text = stripInjectedMemory(messageText(line.content)).trim();
      if (!text) continue;
      if (line.role === "user") {
        if (isSyntheticUserText(text)) continue;
        text = unwrapUserQuery(text);
        if (!text) continue;
      }
      turns.push({ role: line.role, content: text, ...stamp });
    } else if (line.type === "function_call" && typeof line.name === "string") {
      turns.push({
        role: "action",
        content: actionLine(line.name, parseArgs(line.arguments)),
        ...stamp,
      });
    }
    // function_call_result / reasoning / file-history-snapshot / ai-title / summary /
    // resend-fork-notice: dropped.
  }
  return turns;
}
