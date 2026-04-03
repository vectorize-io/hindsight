/**
 * Session retention logic for OpenCode.
 *
 * On session.idle, reads the completed conversation from OpenCode's SQLite
 * database and sends it to Hindsight via the retain API. Uses bun:sqlite
 * for zero-dependency database access.
 *
 * OpenCode stores session data in:
 *   ~/.local/share/opencode/opencode.db
 *
 * Schema:
 *   session: id, title, directory, time_created, time_updated
 *   message: id, session_id, data (JSON: {role, modelID, providerID, ...})
 *   part:    id, message_id, session_id, data (JSON: {type, text, tool, ...})
 */

import { Database } from "bun:sqlite";
import { existsSync } from "fs";
import { basename, join } from "path";
import { homedir } from "os";
import type { HindsightConfig } from "./config";
import type { HindsightClient } from "./client";
import { debugLog } from "./config";
import { deriveBankId, ensureBankMission } from "./bank";

const OPENCODE_DB_PATH = join(homedir(), ".local", "share", "opencode", "opencode.db");

interface SessionRow {
  id: string;
  title: string;
  directory: string;
  time_created: number;
}

interface ConversationRow {
  role: string;
  model: string | null;
  part_type: string;
  text: string | null;
  tool_name: string | null;
  msg_time: number;
}

function findSessionByDirectory(db: Database, directory: string): SessionRow | null {
  // Find the most recent non-subagent session matching the directory
  const row = db
    .query<SessionRow, [string]>(
      `SELECT id, title, directory, time_created
       FROM session
       WHERE directory = ?
       AND title NOT LIKE '%subagent%'
       ORDER BY time_created DESC
       LIMIT 1`,
    )
    .get(directory);

  return row ?? null;
}

function findSessionById(db: Database, sessionId: string): SessionRow | null {
  return db.query<SessionRow, [string]>(`SELECT id, title, directory, time_created FROM session WHERE id = ?`).get(sessionId) ?? null;
}

function getConversationParts(db: Database, sessionId: string): ConversationRow[] {
  return db
    .query<ConversationRow, [string]>(
      `SELECT
         json_extract(m.data, '$.role') as role,
         json_extract(m.data, '$.modelID') as model,
         json_extract(p.data, '$.type') as part_type,
         json_extract(p.data, '$.text') as text,
         json_extract(p.data, '$.tool') as tool_name,
         m.time_created as msg_time
       FROM message m
       JOIN part p ON p.message_id = m.id
       WHERE m.session_id = ?
       AND json_extract(p.data, '$.type') IN ('text', 'tool')
       ORDER BY m.time_created ASC, p.time_created ASC`,
    )
    .all(sessionId);
}

function reconstructTranscript(
  parts: ConversationRow[],
  includeToolCalls: boolean,
): { transcript: string; messageCount: number } {
  const lines: string[] = [];
  let messageCount = 0;

  for (const { role, model, part_type, text, tool_name, msg_time } of parts) {
    const ts = new Date(msg_time).toISOString().replace("T", " ").slice(0, 16);

    if (part_type === "text" && text) {
      const label = role === "user" ? "User" : `Assistant (${model || "unknown"})`;
      lines.push(`${label} (${ts}):\n${text}`);
      messageCount++;
    } else if (part_type === "tool" && tool_name && includeToolCalls) {
      lines.push(`[Tool: ${tool_name}]`);
    }
  }

  return { transcript: lines.join("\n\n"), messageCount };
}

export async function retainSession(
  event: Record<string, unknown>,
  config: HindsightConfig,
  client: HindsightClient,
  apiUrl: string,
): Promise<void> {
  if (!existsSync(OPENCODE_DB_PATH)) {
    debugLog(config, `OpenCode database not found at ${OPENCODE_DB_PATH}`);
    return;
  }

  const db = new Database(OPENCODE_DB_PATH, { readonly: true });

  try {
    // Determine which session just completed.
    // The session.idle event includes a session property from OpenCode.
    const sessionEvent = event as Record<string, unknown>;
    const directory = (sessionEvent.directory as string) || process.cwd();

    let session: SessionRow | null = null;

    // Try session ID from event first
    const sessionId = sessionEvent.sessionID as string | undefined;
    if (sessionId) {
      session = findSessionById(db, sessionId);
    }

    // Fall back to most recent session in the working directory
    if (!session) {
      session = findSessionByDirectory(db, directory);
    }

    if (!session) {
      debugLog(config, `No session found for directory: ${directory}`);
      return;
    }

    // Skip subagent sessions
    if (config.retainSkipSubagent && session.title.toLowerCase().includes("subagent")) {
      debugLog(config, `Skipping subagent session: ${session.title}`);
      return;
    }

    // Extract conversation
    const parts = getConversationParts(db, session.id);
    const { transcript, messageCount } = reconstructTranscript(parts, config.retainToolCalls);

    if (!transcript || transcript.length < config.retainMinChars) {
      debugLog(config, `Session too short (${transcript?.length || 0} chars), skipping`);
      return;
    }

    // Derive bank ID and ensure bank exists
    const bankId = deriveBankId(config, session.directory);
    await ensureBankMission(client, bankId, config);

    // Build metadata
    const project = basename(session.directory);
    const timestamp = new Date(session.time_created).toISOString();
    const metadata: Record<string, string> = {
      source: "opencode",
      session_id: session.id,
      project,
      directory: session.directory,
      title: session.title,
      retained_at: new Date().toISOString(),
      message_count: String(messageCount),
      ...config.retainMetadata,
    };

    // Build tags
    const tags = [...config.retainTags];
    tags.push(`project:${project}`);

    debugLog(
      config,
      `Retaining session "${session.title}" to bank "${bankId}" ` +
        `(${transcript.length} chars, ${messageCount} messages)`,
    );

    await client.retain({
      bankId,
      content: transcript,
      documentId: `opencode-session-${session.id}`,
      context: config.retainContext,
      timestamp,
      metadata,
      tags,
      async: true,
    });

    debugLog(config, `Session retained successfully`);
  } finally {
    db.close();
  }
}
