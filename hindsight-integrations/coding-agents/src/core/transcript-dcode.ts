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
    .join("\n");
}

function timestampOf(value: unknown): { timestamp?: string } {
  return typeof value === "string" && value ? { timestamp: value } : {};
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
    const stamp = timestampOf(record.timestamp);
    if (role === "user" || role === "assistant") {
      const text = stripInjectedMemory(contentText(record.content)).trim();
      if (text) turns.push({ role, content: text, ...stamp });
    } else if (role === "tool" && typeof record.name === "string" && record.name.trim()) {
      turns.push({ role: "action", content: actionLine(record.name, record.content), ...stamp });
    }
  }
  return turns;
}
