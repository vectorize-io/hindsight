/**
 * opencode live-transcript normalizer.
 *
 * opencode delivers the running conversation to a plugin as an in-memory message list (via
 * `experimental.chat.messages.transform`), NOT a JSONL file like Claude/Codex — so this is a pure
 * function over that list, not a file reader. It produces the SAME rich `TransportTurn[]` shape as
 * transcript.ts / transcript-codex.ts (prose turns + compact `role:"action"` tool turns), and
 * reuses the shared `stripInjectedMemory`/`actionLine` helpers so a retain never feeds injected
 * memory back into recall and tool noise stays out of the bank.
 */
import type { TransportTurn } from "./chat";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/** Structural subset of the opencode SDK's ToolState we render (avoids a hard dep on its types). */
interface OcToolState {
  status?: string;
  input?: unknown;
  output?: string; // present on status "completed"
  error?: string; // present on status "error"
}

/** Structural subset of an opencode message Part (TextPart | ToolPart | others we drop). */
export interface OcPart {
  type?: string;
  text?: string; // TextPart
  tool?: string; // ToolPart: the tool name
  state?: OcToolState; // ToolPart: call input + inline output/error
}

/** Structural subset of an opencode message ({ info, parts }). */
export interface OcMessage {
  info?: { role?: string; sessionID?: string; time?: { created?: number } };
  parts?: OcPart[];
}

/**
 * Render one opencode message's parts into turns. Text parts join into one prose turn
 * (injected-memory stripped); each tool part becomes its own compact `role:"action"` turn
 * (tool name + primary target via `actionLine` — no args, no output). reasoning/step/snapshot
 * parts and non-conversational roles are dropped.
 */
function renderMessage(m: OcMessage): TransportTurn[] {
  const role = m.info?.role;
  if (role !== "user" && role !== "assistant") return [];
  const created = m.info?.time?.created;
  const ts = created ? { timestamp: new Date(created).toISOString() } : {};

  const texts: string[] = [];
  const actions: TransportTurn[] = [];
  for (const p of m.parts || []) {
    if (!p || typeof p !== "object") continue;
    if (p.type === "text" && typeof p.text === "string") {
      const t = stripInjectedMemory(p.text).trim();
      if (t) texts.push(t);
    } else if (p.type === "tool" && typeof p.tool === "string") {
      actions.push({ role: "action", content: actionLine(p.tool, p.state?.input), ...ts });
    }
    // reasoning / step-start / step-finish / snapshot / patch / …: dropped
  }

  const out: TransportTurn[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role, content: joined, ...ts });
  out.push(...actions);
  return out;
}

/**
 * Normalize opencode's live message list into transcript turns (user/assistant prose plus compact
 * action turns for tool calls). Never throws on malformed entries.
 */
export function readOpencodeMessages(messages: OcMessage[]): TransportTurn[] {
  return (messages || []).flatMap((m) => renderMessage(m));
}

/** The session id carried on the messages (first message that has one), or undefined. */
export function opencodeSessionId(messages: OcMessage[]): string | undefined {
  return (messages || []).find((m) => m.info?.sessionID)?.info?.sessionID;
}
