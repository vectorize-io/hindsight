import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { join } from "node:path";
import type { TransportTurn } from "./chat";
import { diag } from "./diag";
import { stripInjectedMemory } from "./transcript-util";

/** Where the Devin CLI persists its conversations. Resolved per call so tests can point elsewhere. */
export function devinSessionDb(home = homedir()): string {
  return join(home, ".local", "share", "devin", "cli", "sessions.db");
}

interface DevinMessageNode {
  node_id: number;
  chat_message: string;
}

/** Devin writes progressive assistant updates with a shared message id; retain their final value. */
export function parseDevinMessages(nodes: DevinMessageNode[]): TransportTurn[] {
  const messages = new Map<
    string,
    { index: number; role: "user" | "assistant"; content: string }
  >();
  for (const node of nodes) {
    try {
      const message = JSON.parse(node.chat_message) as {
        message_id?: string;
        role?: string;
        content?: string;
      };
      if ((message.role !== "user" && message.role !== "assistant") || !message.content?.trim())
        continue;
      const content = stripInjectedMemory(message.content).trim();
      if (!content) continue;
      const id = message.message_id || `node:${node.node_id}`;
      messages.set(id, {
        index: messages.get(id)?.index ?? node.node_id,
        role: message.role,
        content,
      });
    } catch {
      /* tool frames and incomplete rows are not conversational turns */
    }
  }
  return [...messages.values()]
    .sort((a, b) => a.index - b.index)
    .map(({ role, content }) => ({ role, content }));
}

interface SqliteStatement {
  all(...params: unknown[]): DevinMessageNode[];
}
interface SqliteDatabase {
  prepare(sql: string): SqliteStatement;
  close(): void;
}
type DatabaseCtor = new (path: string, options?: { readOnly?: boolean }) => SqliteDatabase;

const requireBuiltin = createRequire(import.meta.url);

/**
 * Resolve Node's built-in SQLite, lazily and on the Devin path only.
 *
 * `node:sqlite` ships with Node — nothing to install, nothing to bundle — but it does not exist
 * before 22.5 and is flag-gated in early 22.x. A static import would therefore throw at MODULE
 * LOAD, and this module is pulled in by hook-lifecycle, which every harness shares: an old Node
 * would take Claude Code and Codex down with it. Resolving inside the call keeps the blast radius
 * to Devin, where the failure is real.
 *
 * This previously shelled out to the `sqlite3` BINARY to avoid a native npm dependency
 * (better-sqlite3). The builtin removes both — and the binary was an undeclared prerequisite whose
 * absence silently produced an empty transcript (#3125).
 */
function loadSqlite(): DatabaseCtor | undefined {
  try {
    return (requireBuiltin("node:sqlite") as { DatabaseSync: DatabaseCtor }).DatabaseSync;
  } catch {
    return undefined;
  }
}

/**
 * Devin hooks provide no transcript path — only a session id — so the conversation has to come from
 * the CLI's own sessions.db.
 *
 * Every failure below is reported through `diag`. An empty array used to be returned for a missing
 * reader, an absent database and a genuinely idle session alike, which made a permanently
 * memory-less install indistinguishable from one that simply had nothing to retain yet (#3125).
 */
export function readDevinTranscript(
  sessionId: string | undefined,
  dbPath = devinSessionDb()
): TransportTurn[] {
  if (!sessionId) return [];
  const Database = loadSqlite();
  if (!Database) {
    diag("devin-cli", "sqlite_unavailable", { node: process.version, dbPath });
    return [];
  }
  if (!existsSync(dbPath)) {
    diag("devin-cli", "session_db_missing", { dbPath });
    return [];
  }
  let db: SqliteDatabase | undefined;
  try {
    db = new Database(dbPath, { readOnly: true });
    const rows = db
      .prepare(
        "SELECT node_id, chat_message FROM message_nodes WHERE session_id = ? ORDER BY node_id"
      )
      .all(sessionId);
    return parseDevinMessages(rows);
  } catch (error) {
    // A corrupt file or a Devin storage-schema change surfaces here instead of masquerading as an
    // empty conversation.
    diag("devin-cli", "session_db_read_failed", { dbPath, error: String(error) });
    return [];
  } finally {
    try {
      db?.close();
    } catch {
      /* the read already succeeded or was reported; a close failure adds nothing */
    }
  }
}
