// The OpenClaw transcript on disk, and the reader for it.
//
// Lifted out of `backfill-lib.ts` so the plugin can read a transcript without
// importing that module: `backfill-lib` imports `./index.js`, and having index
// import it back would make the cycle load-order dependent. The parsing is
// unchanged; `backfill-lib` re-exports it so its own surface is the same.
import { readFileSync } from "fs";

export interface SessionMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | Array<{ type?: string; text?: string }>;
}

export interface ParsedSessionFile {
  filePath: string;
  agentId: string;
  sessionId: string;
  sessionKey?: string;
  startedAt?: string;
  messages: SessionMessage[];
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter(
        (block): block is { type?: string; text?: string } => !!block && typeof block === "object"
      )
      .filter((block) => block.type === "text" && typeof block.text === "string")
      .map((block) => block.text || "")
      .join("\n");
  }
  return "";
}

function readJsonLines(filePath: string): unknown[] {
  const content = readFileSync(filePath, "utf8");
  return content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function parseSessionFile(filePath: string, agentId: string): ParsedSessionFile {
  const records = readJsonLines(filePath) as Array<Record<string, any>>;
  let sessionId =
    filePath
      .split("/")
      .pop()
      ?.replace(/\.jsonl$/, "") || "session";
  let sessionKey: string | undefined;
  let startedAt: string | undefined;
  const messages: SessionMessage[] = [];

  for (const record of records) {
    if (record.type === "session") {
      sessionId = typeof record.id === "string" ? record.id : sessionId;
      startedAt = typeof record.timestamp === "string" ? record.timestamp : startedAt;
      sessionKey = typeof record.sessionKey === "string" ? record.sessionKey : sessionKey;
      continue;
    }
    if (record.type !== "message" || !record.message || typeof record.message !== "object") {
      continue;
    }
    const message = record.message as Record<string, unknown>;
    const role = message.role;
    if (role !== "user" && role !== "assistant" && role !== "system" && role !== "tool") {
      continue;
    }
    const text = extractTextContent(message.content);
    if (!text.trim()) {
      continue;
    }
    messages.push({
      role,
      content: typeof message.content === "string" ? message.content : [{ type: "text", text }],
    });
    if (!sessionKey && typeof record.sessionKey === "string") {
      sessionKey = record.sessionKey;
    }
  }

  return {
    filePath,
    agentId,
    sessionId,
    sessionKey,
    startedAt,
    messages,
  };
}
