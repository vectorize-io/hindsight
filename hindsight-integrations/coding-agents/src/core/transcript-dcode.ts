/**
 * DeepAgents Dcode Hooks V2 transcript reader.
 *
 * Dcode materializes a versioned JSONL projection rather than Codex's rollout events. Keep the
 * parser deliberately structural: the hook must continue to retain useful turns when Dcode adds
 * fields, but an unknown schema version must not be mistaken for a known transcript.
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";

interface TranscriptRecord {
  schema_version?: unknown;
  role?: unknown;
  content?: unknown;
  timestamp?: unknown;
  name?: unknown;
}

/** Join used everywhere text blocks are flattened. Shared so a value recovered from the Stop
 *  event compares equal to the same message read back from the transcript (see dcodeAssistantText). */
const BLOCK_JOIN = "\n";

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((block) => {
      if (typeof block === "string") return [block];
      if (!block || typeof block !== "object") return [];
      const text = (block as { text?: unknown }).text;
      return typeof text === "string" ? [text] : [];
    })
    .join(BLOCK_JOIN);
}

/**
 * Read one Python literal (the subset `repr()` emits for JSON-shaped data) starting at `i`.
 *
 * Dcode's Stop event carries `last_assistant_message` as `str(message.content)`
 * (`hooks/server_middleware.py:_last_assistant_text`), so whenever the provider returns content
 * BLOCKS rather than a plain string the field is a Python repr, not prose — see dcodeAssistantText.
 * Only the value forms `repr()` can produce for LangChain content are handled: str (with either
 * quote), int/float, True/False/None, list and dict. Anything else fails the parse, and the caller
 * falls back rather than guessing.
 *
 * Returns the parsed value and the index just past it, or null if the text is not that shape.
 */
function readPyValue(s: string, i: number): { value: unknown; end: number } | null {
  const skip = (j: number): number => {
    while (j < s.length && /\s/.test(s[j]!)) j++;
    return j;
  };
  i = skip(i);
  if (i >= s.length) return null;
  const c = s[i]!;

  if (c === "'" || c === '"') {
    let out = "";
    let j = i + 1;
    while (j < s.length) {
      const ch = s[j]!;
      if (ch === "\\") {
        const esc = s[j + 1];
        if (esc === undefined) return null;
        // Python's repr escapes are a subset of JS's; \x.. and \u.... carry through unchanged.
        if (esc === "n") out += "\n";
        else if (esc === "t") out += "\t";
        else if (esc === "r") out += "\r";
        else if (esc === "x") {
          const hex = s.slice(j + 2, j + 4);
          if (!/^[0-9a-fA-F]{2}$/.test(hex)) return null;
          out += String.fromCharCode(parseInt(hex, 16));
          j += 4;
          continue;
        } else if (esc === "u") {
          const hex = s.slice(j + 2, j + 6);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) return null;
          out += String.fromCharCode(parseInt(hex, 16));
          j += 6;
          continue;
        } else out += esc;
        j += 2;
        continue;
      }
      if (ch === c) return { value: out, end: j + 1 };
      out += ch;
      j++;
    }
    return null; // unterminated
  }

  if (s.startsWith("True", i)) return { value: true, end: i + 4 };
  if (s.startsWith("False", i)) return { value: false, end: i + 5 };
  if (s.startsWith("None", i)) return { value: null, end: i + 4 };

  if (c === "[" || c === "{") {
    const isList = c === "[";
    const close = isList ? "]" : "}";
    const list: unknown[] = [];
    const obj: Record<string, unknown> = {};
    let j = skip(i + 1);
    if (s[j] === close) return { value: isList ? list : obj, end: j + 1 };
    for (;;) {
      const key = readPyValue(s, j);
      if (!key) return null;
      j = skip(key.end);
      if (isList) {
        list.push(key.value);
      } else {
        if (s[j] !== ":" || typeof key.value !== "string") return null;
        const val = readPyValue(s, j + 1);
        if (!val) return null;
        obj[key.value] = val.value;
        j = skip(val.end);
      }
      if (s[j] === ",") {
        j = skip(j + 1);
        // Trailing comma before the closer is not something repr() emits, but tolerate it.
        if (s[j] === close) return { value: isList ? list : obj, end: j + 1 };
        continue;
      }
      if (s[j] === close) return { value: isList ? list : obj, end: j + 1 };
      return null;
    }
  }

  const num = /^-?\d+(\.\d+)?([eE][-+]?\d+)?/.exec(s.slice(i));
  if (num) return { value: Number(num[0]), end: i + num[0].length };
  return null;
}

/**
 * Recover the assistant's actual reply from Dcode's `last_assistant_message`.
 *
 * Dcode computes that field as `content if isinstance(content, str) else str(content)`. For any
 * provider that returns content BLOCKS — reasoning models in particular — the field is therefore a
 * Python repr of the block list, e.g.
 *
 *     [{'type': 'reasoning', 'encrypted_content': 'gAAAA…3KB…'}, {'type': 'text', 'text': 'done'}]
 *
 * Retaining that verbatim stores kilobytes of provider-internal reasoning payload as if the
 * assistant had said it, and — because the transcript reader yields the clean text — it can never
 * compare equal to the transcript's own copy, so the Stop hook would append a duplicate on every
 * turn as well. Parse it back down to the text blocks, joined exactly as contentText joins them.
 *
 * Returns "" when the value is a block list we cannot parse: the retain document is rebuilt from
 * the whole transcript on every Stop, so a turn dropped here is picked up by the next one — which
 * is strictly better than storing an unreadable blob.
 */
export function dcodeAssistantText(raw: string): string {
  const trimmed = raw.trim();
  // Plain prose is the common case (string content) and must pass through untouched.
  if (!trimmed.startsWith("[{") && !trimmed.startsWith("[ {")) return raw;
  const parsed = readPyValue(trimmed, 0);
  if (!parsed || readPyValue(trimmed, parsed.end) !== null) return "";
  if (!Array.isArray(parsed.value)) return "";
  return contentText(parsed.value);
}

/**
 * Read Dcode's materialized transcript into the normalized chat shape used by retention.
 * Tool results are mechanical noise; tool records become compact action turns when a name exists.
 * Invalid records are skipped so a partially-written Stop transcript remains fail-open.
 */
export function readDcodeTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  for (const rawLine of readJsonlTail(path, { scope: "dcode" }).lines) {
    if (!rawLine.trim()) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawLine);
    } catch {
      continue;
    }
    if (!parsed || typeof parsed !== "object") continue;
    const record = parsed as TranscriptRecord;
    if (record.schema_version !== 1) continue;

    const role = record.role;
    const stamp =
      typeof record.timestamp === "string" && record.timestamp
        ? { timestamp: record.timestamp }
        : {};
    if (role === "user" || role === "assistant") {
      const text = stripInjectedMemory(contentText(record.content)).trim();
      if (text) turns.push({ role, content: text, ...stamp });
    } else if (role === "tool" && typeof record.name === "string" && record.name.trim()) {
      turns.push({ role: "action", content: actionLine(record.name, record.content), ...stamp });
    }
  }
  return turns;
}
