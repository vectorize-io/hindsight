/**
 * opencode2 (opencode v2) live-transcript normalizer.
 *
 * opencode v2 rewrote its message model, so the v1 reader (transcript-opencode.ts) cannot be
 * reused: there is no `{info, parts}` envelope any more. A session's transcript comes back from
 * `ctx.session.context({sessionID})` as a flat `SessionMessageInfo[]` where
 *   - a user message carries its prose directly on `text`, and
 *   - an assistant message carries a `content` array whose entries are `{type:"text", text}` or
 *     `{type:"tool", name, state:{input, …}}`.
 * Non-conversational message types (`agent-selected`, `model-selected`, `system`, `compaction`, …)
 * ride the same list and are dropped here.
 *
 * The output is the SAME rich `TransportTurn[]` shape every other harness produces (prose turns
 * plus compact `role:"action"` tool turns) and reuses the shared `stripInjectedMemory`/`actionLine`
 * helpers, so a retain never feeds injected memory back into recall and tool noise stays out of the
 * bank.
 */
import type { TransportTurn } from "./chat";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/** Structural subset of an assistant message's content entry (text | tool | reasoning | …). */
interface Oc2Content {
  type?: string;
  text?: string; // type "text"
  name?: string; // type "tool": the tool name
  /** type "tool": the call's arguments plus its result. Only `input` is retained — a tool's output
   *  is deliberately dropped, so the bank records WHAT was touched, not mechanical noise. */
  state?: { input?: unknown; status?: string; content?: unknown };
}

/** Structural subset of one `SessionMessageInfo` (avoids a hard dep on the v2 SDK types). */
export interface Oc2Message {
  type?: string; // "user" | "assistant" | agent-selected | system | …
  text?: string; // user messages carry prose here
  content?: Oc2Content[]; // assistant messages carry parts here
  time?: { created?: number };
}

/** Render one v2 message into turns. Never throws on a malformed entry. */
function renderMessage(m: Oc2Message): TransportTurn[] {
  const role = m?.type;
  if (role !== "user" && role !== "assistant") return [];
  const created = m.time?.created;
  const ts = created ? { timestamp: new Date(created).toISOString() } : {};

  const texts: string[] = [];
  const actions: TransportTurn[] = [];
  // A user message's prose is a plain string; an assistant's is spread over content parts.
  if (typeof m.text === "string") texts.push(m.text);
  for (const c of m.content || []) {
    if (!c || typeof c !== "object") continue;
    if (c.type === "text" && typeof c.text === "string") texts.push(c.text);
    else if (c.type === "tool" && typeof c.name === "string")
      actions.push({ role: "action", content: actionLine(c.name, c.state?.input), ...ts });
    // reasoning / media / snapshot / …: dropped
  }

  const out: TransportTurn[] = [];
  const joined = stripInjectedMemory(texts.join("\n")).trim();
  if (joined) out.push({ role, content: joined, ...ts });
  out.push(...actions);
  return out;
}

/**
 * Normalize an opencode2 session's message list into transcript turns (user/assistant prose plus
 * compact action turns for tool calls). Never throws on malformed entries.
 */
export function readOpencode2Messages(messages: Oc2Message[]): TransportTurn[] {
  return (messages || []).flatMap((m) => renderMessage(m));
}
