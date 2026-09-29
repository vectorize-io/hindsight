import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { devinSessionDb, parseDevinMessages, readDevinTranscript } from "./transcript-devin";

describe("parseDevinMessages", () => {
  it("uses the final streamed assistant update", () => {
    expect(
      parseDevinMessages([
        {
          node_id: 1,
          chat_message: JSON.stringify({
            message_id: "u",
            role: "user",
            content: "Explain this repo",
          }),
        },
        {
          node_id: 2,
          chat_message: JSON.stringify({ message_id: "a", role: "assistant", content: "Partial" }),
        },
        {
          node_id: 3,
          chat_message: JSON.stringify({
            message_id: "a",
            role: "assistant",
            content: "Complete answer",
          }),
        },
      ])
    ).toEqual([
      { role: "user", content: "Explain this repo" },
      { role: "assistant", content: "Complete answer" },
    ]);
  });
});

/**
 * The SQLite read path itself, which used to shell out to an undeclared `sqlite3` binary and return
 * an empty transcript on every failure (#3125). Skipped rather than failed on a Node without
 * `node:sqlite` — that is precisely the environment the installer preflight now refuses.
 */
const sqlite = (() => {
  try {
    return createRequire(import.meta.url)("node:sqlite") as {
      DatabaseSync: new (path: string) => {
        exec(sql: string): void;
        prepare(sql: string): { run(...params: unknown[]): void };
        close(): void;
      };
    };
  } catch {
    return undefined;
  }
})();

describe.skipIf(!sqlite)("readDevinTranscript", () => {
  let dir: string;
  let dbPath: string;
  let diagFile: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hindsight-devin-"));
    dbPath = join(dir, "sessions.db");
    diagFile = join(dir, "diag.log");
    process.env.HINDSIGHT_DIAG_FILE = diagFile;
  });

  afterEach(() => {
    delete process.env.HINDSIGHT_DIAG_FILE;
    rmSync(dir, { recursive: true, force: true });
  });

  function seed(rows: Array<{ node_id: number; session_id: string; chat_message: string }>): void {
    const db = new sqlite!.DatabaseSync(dbPath);
    db.exec("CREATE TABLE message_nodes (node_id INTEGER, session_id TEXT, chat_message TEXT)");
    const insert = db.prepare("INSERT INTO message_nodes VALUES (?, ?, ?)");
    for (const row of rows) {
      insert.run(row.node_id, row.session_id, row.chat_message);
    }
    db.close();
  }

  it("reads the conversation for one session", () => {
    seed([
      {
        node_id: 1,
        session_id: "s1",
        chat_message: JSON.stringify({ message_id: "u", role: "user", content: "hi" }),
      },
      {
        node_id: 2,
        session_id: "s1",
        chat_message: JSON.stringify({ message_id: "a", role: "assistant", content: "hello" }),
      },
      {
        node_id: 3,
        session_id: "other",
        chat_message: JSON.stringify({ message_id: "x", role: "user", content: "not mine" }),
      },
    ]);
    expect(readDevinTranscript("s1", dbPath)).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  // The session id reaches SQLite as a bound parameter, so a quote in it can't terminate the
  // string literal the way the previous hand-escaped `sqlite3` command line could.
  it("binds the session id instead of interpolating it", () => {
    seed([
      {
        node_id: 1,
        session_id: "s'1",
        chat_message: JSON.stringify({ message_id: "u", role: "user", content: "quoted" }),
      },
    ]);
    expect(readDevinTranscript("s'1", dbPath)).toEqual([{ role: "user", content: "quoted" }]);
  });

  // Regression for #3125: an unreadable database must be DISTINGUISHABLE from an idle session,
  // both returning [] but only one leaving a trace.
  it("reports a missing database rather than looking idle", () => {
    expect(readDevinTranscript("s1", join(dir, "absent.db"))).toEqual([]);
    expect(readFileSync(diagFile, "utf8")).toContain("session_db_missing");
  });

  it("reports a schema change rather than looking idle", () => {
    const db = new sqlite!.DatabaseSync(dbPath);
    db.exec("CREATE TABLE renamed_in_a_future_release (node_id INTEGER)");
    db.close();
    expect(readDevinTranscript("s1", dbPath)).toEqual([]);
    expect(readFileSync(diagFile, "utf8")).toContain("session_db_read_failed");
  });
});

describe("devinSessionDb", () => {
  it("resolves under the given home", () => {
    expect(devinSessionDb("/home/x")).toBe("/home/x/.local/share/devin/cli/sessions.db");
  });
});
