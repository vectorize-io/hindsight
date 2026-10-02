/**
 * CodeBuddy transcript reader — the IDE's own store, plus the dispatch onto the CLI's JSONL.
 *
 * CodeBuddy ships in TWO hosts and its Stop payload hands us whichever one is running, in two
 * entirely different on-disk shapes:
 *
 *   • CodeBuddy Code (the CLI) writes the WorkBuddy JSONL — `~/.codebuddy/projects/<cwd-encoded>/
 *     <uuid>.jsonl` — which core/transcript-workbuddy.ts already parses.
 *   • The CodeBuddy IDE hands the hook ITS OWN per-conversation DIRECTORY, built by its
 *     HistoryService: `<ext-data>/CodeBuddyExtension/Data/<uid>/CodeBuddyIDE/<uid>/history/
 *     <md5(workspace path)>/<conversationId>/`, whose `index.json` is what arrives as
 *     `transcript_path`:
 *
 *       index.json                 { messages: [{ id, type, role, isComplete }] }  ← order only, NO prose
 *       messages/<messageId>.json  { role, message, id, extra, createdAt }         ← `message` is an ESCAPED JSON string
 *
 * Why this reader has to exist: handed `index.json`, the JSONL reader finds not one `type:"message"`
 * record, so the retain path took its `turns.length === 0` exit and returned SILENTLY — no request,
 * no log line. A CodeBuddy IDE session was therefore never written back at all. The prose lives in
 * the sibling `messages/` files, which only this reader knows to open. (Host parity gotcha: the IDE
 * transcripts are NOT WorkBuddy's, so the "same engine ⇒ same schema" assumption does not hold here.)
 *
 * Mapping — the same `TransportTurn[]` shape as every other reader:
 *   text (user)      → user       (host scaffolding unwrapped, see `ideUserText`)
 *   text (assistant) → assistant
 *   tool-call        → action     (`actionLine(toolName, args)` — never the arguments themselves)
 * Dropped: `reasoning` (the model's chain of thought, never retained anywhere), `tool-result` and
 * the index's `role:"tool"` records (raw output is exactly the mechanical noise `actionLine` keeps
 * out), and `image` blocks — the same exclusions core/transcript-workbuddy.ts documents.
 *
 * Fail-open, like every reader here: an unreadable index, a missing message file, a malformed inner
 * record or a half-written directory contributes nothing and never throws. A hook must not fail the
 * host it is called by.
 */
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import type { TransportTurn } from "./chat";
import { actionLine, stripInjectedMemory } from "./transcript-util";
import {
  isSyntheticUserText,
  readWorkbuddyTranscript,
  unwrapUserQuery,
} from "./transcript-workbuddy";

/** One `index.json` entry: the message order the IDE recorded, carrying no prose of its own. */
interface IdeIndexEntry {
  id?: string;
}

/** One `messages/<messageId>.json`; `message` is the model-facing record as an escaped JSON string. */
interface IdeMessageFile {
  role?: string;
  message?: string;
  createdAt?: string;
}

/** A content block inside the decoded inner record. */
interface IdeBlock {
  type?: string;
  text?: string;
  toolName?: string;
  args?: unknown;
}

/** The decoded inner record (`JSON.parse(file.message)`). */
interface IdeInner {
  role?: string;
  content?: unknown;
}

/**
 * The context blocks the CodeBuddy IDE prefixes to a prompt — all of them OUTSIDE the
 * `<system-reminder>` wrapper WorkBuddy nests its own copy inside, which is why no reader had to
 * know them until now: the shared stripper removed WorkBuddy's as a side effect.
 *
 * Measured on disk, the IDE store carries the UNDERSCORE spelling of that wrapper
 * (`<system_reminder>`, 253 occurrences, absent from WorkBuddy's transcripts) plus three blocks of
 * its own: `<user_info>` (OS/shell/workspace), `<git_status>` (the repo state at session start) and
 * `<additional_data>` (ambient host context — today, the current time).
 *
 * Tag-structural, exactly like MEMORY_TAG_RE: the block is removed and any surrounding text KEPT,
 * so a prompt that merely QUOTES a marker still survives the anchored unwrap below.
 */
const IDE_CONTEXT_RE = /<(user_info|git_status|additional_data|system_reminder)\b[\s\S]*?<\/\1>/g;

/** Read + parse a file, or `undefined` for anything unreadable — this reader never throws. */
function readJson(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as unknown;
  } catch {
    return undefined;
  }
}

/**
 * A user record's text, or `""` when it is host scaffolding rather than the human's words: the
 * injected-memory blocks the shared stripper knows, then the IDE's own context blocks, then the
 * machine-written records `isSyntheticUserText` names (`<cb_summary>`), and only then the
 * `<user_query>` wrapper — in that order, because each step reveals the next one's prefix.
 */
function ideUserText(raw: string): string {
  let text = stripInjectedMemory(raw).replace(IDE_CONTEXT_RE, "").trim();
  if (!text || isSyntheticUserText(text)) return "";
  return unwrapUserQuery(text).trim();
}

/**
 * Parse a CodeBuddy IDE conversation directory, given the `index.json` path the Stop hook receives.
 * One text turn per record (WorkBuddy's rule), then that record's tool calls in block order.
 */
export function readCodebuddyIdeTranscript(indexPath: string): TransportTurn[] {
  const index = readJson(indexPath) as { messages?: IdeIndexEntry[] } | undefined;
  const entries = index?.messages;
  if (!Array.isArray(entries)) return [];

  const messagesDir = join(dirname(indexPath), "messages");
  const turns: TransportTurn[] = [];

  for (const entry of entries) {
    const id = entry?.id;
    if (typeof id !== "string") continue;
    const file = readJson(join(messagesDir, `${id}.json`)) as IdeMessageFile | undefined;
    const encoded = file?.message;
    if (typeof encoded !== "string") continue;

    let inner: IdeInner;
    try {
      inner = JSON.parse(encoded) as IdeInner;
    } catch {
      continue; // a truncated record contributes nothing rather than failing the hook
    }

    const role = typeof inner.role === "string" ? inner.role : file?.role;
    const isUser = role === "user";
    const isAssistant = role === "assistant";
    const stamp = typeof file?.createdAt === "string" ? { timestamp: file.createdAt } : {};

    const blocks = Array.isArray(inner.content) ? inner.content : [];
    const texts: string[] = [];
    const calls: { name: string; args: unknown }[] = [];
    for (const block of blocks) {
      if (!block || typeof block !== "object") continue;
      const { type, text, toolName, args } = block as IdeBlock;
      if (type === "text" && typeof text === "string") texts.push(text);
      else if (type === "tool-call" && typeof toolName === "string")
        calls.push({ name: toolName, args });
      // reasoning / tool-result / image: dropped.
    }

    if (texts.length) {
      const raw = texts.join("\n");
      if (isUser) {
        const text = ideUserText(raw);
        if (text) turns.push({ role: "user", content: text, ...stamp });
      } else if (isAssistant) {
        const text = stripInjectedMemory(raw).trim();
        if (text) turns.push({ role: "assistant", content: text, ...stamp });
      }
    }
    for (const call of calls) {
      turns.push({ role: "action", content: actionLine(call.name, call.args), ...stamp });
    }
  }

  return turns;
}

/**
 * The reader registered for the `codebuddy` harness, which covers BOTH of its hosts: dispatch on the
 * shape of the path we were handed. The IDE passes `<conversation>/index.json`; CodeBuddy Code
 * passes the WorkBuddy `.jsonl`. Anything else falls through to the JSONL reader, which fails open.
 */
export function readCodebuddyTranscript(path: string): TransportTurn[] {
  return basename(path) === "index.json"
    ? readCodebuddyIdeTranscript(path)
    : readWorkbuddyTranscript(path);
}
