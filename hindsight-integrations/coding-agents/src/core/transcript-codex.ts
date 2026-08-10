/**
 * Codex CLI rollout (JSONL) reader — the Codex counterpart to transcript.ts (Claude). Codex's
 * transcript is a different schema: each line is an event with a `type`; the conversation lives in
 * `type:"response_item"` lines whose `payload` is one of:
 *   - message (role user/assistant/developer; content is `input_text`/`output_text` blocks)
 *   - function_call        (tool call: name + arguments JSON string)
 *   - function_call_output (tool result: output string)
 *   - reasoning            (encrypted internal chain-of-thought — dropped, like Claude `thinking`)
 *
 * What we keep, normalized to the SAME `TransportTurn[]` shape as readClaudeTranscript so the live
 * write-back (retainLiveSession) renders both identically:
 *   - user text (real prompts) — synthetic Codex startup messages (AGENTS.md + <environment_context>)
 *     are dropped so we capture the user's work, not the agent rules.
 *   - assistant text (all phases: commentary + final_answer).
 *   - function_call → a compact `role:"action"` turn (tool name + primary target, no args);
 *     function_call_output is dropped (outputs are mechanical noise for extraction).
 *
 * `developer`-role messages carry Codex's system prompt AND our hook-injected context
 * (<hindsight_knowledge>, <hindsight_memories>, <user_feedback>), so dropping them entirely is what
 * prevents a retain→recall feedback loop (plus stripInjectedMemory as a defensive second pass on the
 * text we do keep). Fail-open: never throws on a missing file or a malformed line.
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
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
  output?: string;
}
interface RolloutLine {
  type?: string;
  payload?: Payload;
}

/** Codex records its startup instructions (AGENTS.md + environment_context) as a normal user
 *  message. Retaining that teaches the bank about agent rules, not the user's work — drop it. */
function isSyntheticUserText(text: string): boolean {
  const s = text.trimStart();
  return s.startsWith("# AGENTS.md instructions for ") || s.startsWith("<environment_context>");
}

/** Join a message payload's text blocks (input_text for user/developer, output_text for assistant). */
function messageText(payload: Payload): string {
  return (payload.content || [])
    .filter((c) => c && typeof c.text === "string")
    .map((c) => c.text as string)
    .join("\n");
}

/** Parse a Codex rollout JSONL into normalized markdown turns (text + tool calls/results).
 *  Drops developer/system + synthetic-startup + reasoning + injected memory + empty turns.
 *  Never throws on bad lines. */
export function readCodexTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "codex" }).lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const line = parsed as RolloutLine;
    if (line.type !== "response_item") continue;
    const p = line.payload;
    if (!p || typeof p !== "object") continue;

    if (p.type === "message") {
      // `developer` messages are Codex's system prompt + OUR injected hook context → drop entirely.
      if (p.role !== "user" && p.role !== "assistant") continue;
      const text = stripInjectedMemory(messageText(p)).trim();
      if (!text) continue;
      if (p.role === "user" && isSyntheticUserText(text)) continue;
      turns.push({ role: p.role, content: text });
    } else if (p.type === "function_call" && typeof p.name === "string") {
      let input: unknown;
      try {
        input = JSON.parse(p.arguments || "");
      } catch {
        input = undefined;
      }
      turns.push({ role: "action", content: actionLine(p.name, input) });
    }
    // reasoning / other payloads: dropped.
  }

  return turns;
}
