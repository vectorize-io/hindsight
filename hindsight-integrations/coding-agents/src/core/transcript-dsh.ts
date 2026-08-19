/**
 * DeepSeek Harness (dsh) transcript normalizer.
 *
 * dsh keeps a session as an append-only log of typed `SessionEvent`s rather than a chat array, and
 * a plugin reads that log straight off `agent.session.events` — so this is a pure function over
 * those events, like the opencode normalizer, not a file reader. The SAME event vocabulary is what
 * dsh persists to disk, so the backfill reader (core/history.ts) feeds this exact function.
 *
 * Only three of the ~30 event types carry conversation:
 *   - `user/message`      surface, a UserMessage — the human prompt OR an injected plugin context
 *   - `assistant/message` surface, the assembled reply for one step
 *   - `tool/call`         log-only, the model's tool invocation (rendered as a compact action turn)
 * `tool/result` is deliberately skipped: its payload is raw tool output, which the shared
 * `actionLine` convention keeps out of the bank (see transcript-util.ts).
 *
 * A `user/message` is only kept when its source is an actual HUMAN (`source.kind === 'user'`).
 * dsh delivers plugin context — its own runtime-context snapshots, the skill catalog, file-change
 * notices, and OUR recalled memories — as user-role messages on the same surface, and retaining
 * those would file machine scaffolding as if the user had said it (and, for our own block, feed
 * recalled memory straight back into the next extraction). The tag-based `stripInjectedMemory`
 * still runs on what remains, for text that arrived inside a genuine prompt.
 */
import type { TransportTurn } from "./chat";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/**
 * The Cordis plugin name, which is also the `source.plugin` dsh records on every message this
 * integration injects — so an injected block is identifiable in a session log even though the
 * `kind: 'user'` filter below already keeps it (and every other plugin's context) out of a retain.
 */
export const HINDSIGHT_PLUGIN = "hindsight";

/** Structural subset of a dsh ContentBlock (text is the only kind that carries prose). */
interface DshContentBlock {
  type?: string;
  text?: string;
}

/** Structural subset of a dsh Message ({ id, role, content, source }). */
interface DshMessage {
  role?: string;
  content?: DshContentBlock[];
  source?: { kind?: string; plugin?: string };
}

/** Structural subset of a dsh SessionEvent. `data` is the per-type payload. */
export interface DshSessionEvent {
  type?: string;
  time?: number;
  data?: unknown;
}

/** Join a message's text blocks; reasoning/image/tool blocks carry no durable prose. */
function textOf(message: DshMessage): string {
  return (message.content || [])
    .filter((block) => block?.type === "text" && typeof block.text === "string")
    .map((block) => block.text!)
    .join("\n")
    .trim();
}

/** dsh stamps epoch milliseconds on every event; absent only on malformed input. */
function stampOf(event: DshSessionEvent): { timestamp?: string } {
  return typeof event.time === "number" ? { timestamp: new Date(event.time).toISOString() } : {};
}

/**
 * Normalize a dsh session event log into transcript turns: user/assistant prose plus compact
 * `role:"action"` turns for tool calls. Never throws on malformed entries.
 */
export function readDshEvents(events: readonly DshSessionEvent[]): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const event of events || []) {
    if (!event || typeof event !== "object") continue;
    const stamp = stampOf(event);
    if (event.type === "user/message") {
      const message = event.data as DshMessage | undefined;
      if (!message || message.source?.kind !== "user") continue;
      const text = stripInjectedMemory(textOf(message)).trim();
      if (text) turns.push({ role: "user", content: text, ...stamp });
    } else if (event.type === "assistant/message") {
      const message = (event.data as { message?: DshMessage } | undefined)?.message;
      if (!message) continue;
      const text = stripInjectedMemory(textOf(message)).trim();
      if (text) turns.push({ role: "assistant", content: text, ...stamp });
    } else if (event.type === "tool/call") {
      const call = event.data as { name?: string; arguments?: string } | undefined;
      if (!call?.name) continue;
      turns.push({
        role: "action",
        content: actionLine(call.name, parseArgs(call.arguments)),
        ...stamp,
      });
    }
  }
  return turns;
}

/** `tool/call.arguments` is the model's RAW JSON string, unparsed by design. */
function parseArgs(raw: string | undefined): unknown {
  if (!raw) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    return raw; // a malformed argument string still names what the model was reaching for
  }
}
