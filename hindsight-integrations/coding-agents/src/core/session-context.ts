/**
 * `sessionContext` (#5714): mental models and pages whose CONTENT goes into the agent's context
 * when a session starts, in the order listed. Same shape as the Hermes plugin's `session_context`.
 *
 * Everything else reaches the agent only when it searches, and an agent skips that search exactly
 * when a standing rule matters most — when it is about to act against it. Read once per session
 * (a stored-document read, no LLM call), so a model refreshed mid-session shows up next session.
 */
import { describeError, log } from "./log";

export interface SessionContextRef {
  kind: "mental_model" | "page";
  ref: string;
}

/** `mental_model:<id>` or `page:<id>`; a bare id is a mental model. Bad entries are dropped. */
export function parseSessionContext(value: unknown): SessionContextRef[] {
  if (!Array.isArray(value)) return [];
  const out: SessionContextRef[] = [];
  for (const item of value) {
    if (typeof item !== "string" || !item.trim()) continue;
    const i = item.indexOf(":");
    const kind = i < 0 ? "mental_model" : item.slice(0, i).trim();
    const ref = (i < 0 ? item : item.slice(i + 1)).trim();
    if ((kind !== "mental_model" && kind !== "page") || !ref) {
      log.warn("config", "ignoring sessionContext entry: expected mental_model:<id> or page:<id>", {
        entry: item,
      });
      continue;
    }
    out.push({ kind, ref });
  }
  return out;
}

export interface SessionContextClient {
  getPage(pageId: string, opts?: { timeoutMs?: number }): Promise<unknown>;
  getMentalModel(id: string, opts?: { timeoutMs?: number }): Promise<unknown>;
}

/** Each read is bounded on its own; they run together so N entries cost one wait, not N. */
const READ_TIMEOUT_MS = 5_000;

/** The configured entries rendered as one block, or "" when none could be read. Never throws:
 *  an entry that fails or is still empty is skipped with a warning. */
export async function loadSessionContext(
  client: SessionContextClient,
  refs: SessionContextRef[],
  harness = "session-context"
): Promise<string> {
  if (!refs.length) return "";
  const sections = await Promise.all(
    refs.map(async ({ kind, ref }) => {
      try {
        const doc = (
          kind === "page"
            ? await client.getPage(ref, { timeoutMs: READ_TIMEOUT_MS })
            : await client.getMentalModel(ref, { timeoutMs: READ_TIMEOUT_MS })
        ) as { name?: unknown; body?: unknown; content?: unknown } | null;
        const text = kind === "page" ? doc?.body : doc?.content;
        if (typeof text !== "string" || !text.trim()) {
          log.warn(harness, "sessionContext entry has no content yet", { kind, ref });
          return undefined;
        }
        const name = typeof doc?.name === "string" && doc.name ? doc.name : ref;
        return `## ${name}\n${text.trim()}`;
      } catch (e) {
        log.warn(harness, "sessionContext entry could not be read", {
          kind,
          ref,
          error: describeError(e),
        });
        return undefined;
      }
    })
  );
  const kept = sections.filter((s): s is string => s !== undefined);
  if (!kept.length) return "";
  return (
    "<hindsight_session_context>\n" +
    "Background from memory, read at session start. Treat it as standing context for this whole session.\n\n" +
    kept.join("\n\n") +
    "\n</hindsight_session_context>"
  );
}
