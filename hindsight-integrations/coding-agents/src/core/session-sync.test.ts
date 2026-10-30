import { appendFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  claudeEventsOf,
  codexEventsOf,
  type AppendResult,
  type SessionAppend,
} from "./session-events";
import {
  emptySyncState,
  sessionOptions,
  sessionsUnsupported,
  syncSession,
  type SessionClient,
} from "./session-sync";
import { NOTE_CREDIT, NOTE_DISMISS, NOTE_LEAD, noteCredit, noteDismiss } from "./live";
import type { Config } from "./config";

const line = (o: unknown) => `${JSON.stringify(o)}\n`;
const prompt = (uuid: string, text: string) =>
  line({
    type: "user",
    uuid,
    timestamp: "2026-10-19T10:00:00Z",
    message: { role: "user", content: text },
  });
const edit = (uuid: string, id: string) =>
  line({
    type: "assistant",
    uuid: `${uuid}a`,
    message: {
      content: [
        {
          type: "tool_use",
          id,
          name: "Edit",
          input: { file_path: "units.py", new_string: "x".repeat(5000) },
        },
      ],
    },
  }) +
  line({
    type: "user",
    uuid: `${uuid}b`,
    message: { content: [{ type: "tool_result", tool_use_id: id, content: "ok" }] },
  });

const reply = (uuid: string, text: string) =>
  line({
    type: "assistant",
    uuid,
    message: { role: "assistant", content: [{ type: "text", text }] },
  });

function session() {
  const root = mkdtempSync(join(tmpdir(), "session-sync-"));
  const transcript = join(root, "t.jsonl");
  writeFileSync(transcript, "");
  return { transcript, dir: join(root, "state") };
}

/** A sessions server: records each append, answers with the notes it is given. */
function server(notes: AppendResult[] = []) {
  const appends: SessionAppend[] = [];
  const client: SessionClient = {
    appendSession: vi.fn(async (_id: string, body) => {
      appends.push(body);
      return notes.shift() ?? { supported: true, notes: [] };
    }),
  };
  return { client, appends };
}

const base = (s: ReturnType<typeof session>, client: SessionClient) => ({
  sessionId: "s1",
  transcriptPath: s.transcript,
  cwd: s.dir,
  dir: s.dir,
  client,
  eventsOf: claudeEventsOf,
  options: () => ({ live: true }),
  diff: () => "+++ b/units.py\n+K = 1024",
});

describe("claudeEventsOf", () => {
  it("keeps the conversation whole and ids it by the transcript line", () => {
    const long = "y".repeat(10_000);
    const events = claudeEventsOf(JSON.parse(prompt("u1", long)), 0);
    expect(events).toEqual([{ id: "u1:0", role: "user", content: long }]);
  });

  it("reads tool calls and results, cutting what is never saved", () => {
    const [call, result] = edit("u2", "t1")
      .trim()
      .split("\n")
      .map((l, i) => claudeEventsOf(JSON.parse(l), i)[0]);
    // The server ids the call `<message id>:<call id>`, as it always did.
    expect(call).toMatchObject({
      id: "u2a:0",
      role: "assistant",
      tool_calls: [{ id: "t1", type: "function", function: { name: "Edit" } }],
    });
    const input = JSON.parse(call.tool_calls?.[0].function.arguments ?? "{}");
    expect(input.new_string.length).toBe(2000);
    expect(result).toEqual({ id: "u2b:0", role: "tool", tool_call_id: "t1", content: "ok" });
  });

  it("skips meta lines and subagents", () => {
    expect(claudeEventsOf({ type: "user", isMeta: true, message: { content: "x" } }, 0)).toEqual(
      []
    );
    expect(
      claudeEventsOf({ type: "assistant", isSidechain: true, message: { content: [] } }, 0)
    ).toEqual([]);
  });
});

describe("codexEventsOf", () => {
  it("reads a command and a file change as Bash and Edit", () => {
    const cmd = codexEventsOf(
      {
        type: "event_msg",
        payload: {
          type: "item_completed",
          item: {
            type: "CommandExecution",
            id: "c1",
            command: ["/bin/zsh", "-lc", "make"],
            aggregated_output: "boom",
            exit_code: 2,
          },
        },
      },
      7
    );
    expect(cmd).toEqual([
      {
        id: "c1@7:0",
        role: "assistant",
        tool_calls: [
          {
            id: "c1",
            type: "function",
            function: { name: "Bash", arguments: '{"command":"make"}' },
          },
        ],
      },
      { id: "c1@7:1", role: "tool", tool_call_id: "c1", content: "boom", is_error: true },
    ]);
  });
});

describe("syncSession", () => {
  it("sends the new lines once, with the session's options on the first append only", async () => {
    const s = session();
    const { client, appends } = server();
    const options = vi.fn(() => ({ live: true }));
    const sync = () => syncSession({ ...base(s, client), options });
    appendFileSync(s.transcript, prompt("u1", "Fix parse_quantity."));
    await sync();
    await sync();
    appendFileSync(s.transcript, edit("u2", "t1"));
    await sync();
    expect(options).toHaveBeenCalledTimes(1);
    expect(appends.map((a) => a.messages.map((m) => m.id))).toEqual([
      ["u1:0"],
      [],
      ["u2a:0", "u2b:0"],
    ]);
    // A tool ran and the tree changed: the diff rides along.
    expect(appends.map((a) => a.state)).toEqual([
      undefined,
      undefined,
      { diff: "+++ b/units.py\n+K = 1024" },
    ]);
    expect(appends.map((a) => a.session)).toEqual([{ live: true }, undefined, undefined]);
  });

  it("opens the session again in a new bank when it moves repo, without resending the transcript", async () => {
    const s = session();
    const { client, appends } = server();
    const options = vi.fn(() => ({ live: true }));
    appendFileSync(s.transcript, prompt("u1", "Fix parse_quantity."));
    await syncSession({ ...base(s, client), options, bankId: "coding-agent::a" });
    appendFileSync(s.transcript, edit("u2", "t1"));
    await syncSession({ ...base(s, client), options, bankId: "coding-agent::b" });
    await syncSession({ ...base(s, client), options, bankId: "coding-agent::b" });
    expect(appends.map((a) => a.session)).toEqual([{ live: true }, { live: true }, undefined]);
    expect(appends.map((a) => a.messages.map((m) => m.id))).toEqual([
      ["u1:0"],
      ["u2a:0", "u2b:0"],
      [],
    ]);
  });

  it("returns new notes rendered for the agent, and acknowledges them on the next append", async () => {
    const s = session();
    const note = {
      seq: 3,
      text: "Before you change units.py: K is 1024.",
      source: {
        document_id: "conversation:old",
        quote: "K is 1024, never 1000.",
        mentioned_at: "2026-10-01T10:00:00Z",
      },
    };
    const { client, appends } = server([{ supported: true, notes: [note] }]);
    appendFileSync(s.transcript, prompt("u1", "Fix it."));
    const r = await syncSession(base(s, client));
    expect(r.notes).toEqual([
      `${NOTE_LEAD.replace("thought:", "thought #3:")} Before you change units.py: K is 1024. Source: a past session (2026-10-01) — "K is 1024, never 1000.". Share it with the user: "${noteCredit(3)} …" if you did something differently because of it, else "${noteDismiss(3)} …; no change: …"`,
    ]);
    await syncSession(base(s, client));
    expect(appends.map((a) => a.ack)).toEqual([0, 3]);
  });

  it("does not move past lines a failed append did not deliver", async () => {
    const s = session();
    const { client, appends } = server();
    const failing: SessionClient = {
      appendSession: vi.fn(async () => Promise.reject(new Error("down"))),
    };
    appendFileSync(s.transcript, prompt("u1", "Fix it."));
    await expect(syncSession(base(s, failing))).rejects.toThrow("down");
    await syncSession(base(s, client));
    expect(appends[0].messages.map((m) => m.id)).toEqual(["u1:0"]);
  });

  it("sends the prompt from the prompt hook at once, and skips its copy in the transcript", async () => {
    const s = session();
    const { client, appends } = server();
    // The prompt hook gets the prompt trimmed; the transcript keeps the newline it was typed with.
    await syncSession({ ...base(s, client), prompt: { text: "Fix it." } });
    appendFileSync(s.transcript, prompt("u1", "Fix it.\n"));
    await syncSession({ ...base(s, client), turnEnd: true });
    expect(appends[0].messages.map((m) => [m.role, m.content])).toEqual([["user", "Fix it."]]);
    expect(appends[1]).toMatchObject({ messages: [], turn_end: true });
  });

  it("marks a server without sessions, and stops asking it", async () => {
    const s = session();
    const { client } = server([{ supported: false }]);
    appendFileSync(s.transcript, prompt("u1", "Fix it."));
    expect((await syncSession(base(s, client))).unsupported).toBe(true);
    expect(sessionsUnsupported(s.dir)).toBe(true);
    expect((await syncSession(base(s, client))).unsupported).toBe(true);
    expect(client.appendSession).toHaveBeenCalledTimes(1);
  });

  it("starts over when the transcript was replaced by a shorter one", async () => {
    const s = session();
    const { client, appends } = server();
    appendFileSync(s.transcript, prompt("u1", "Fix it.") + prompt("u2", "And this."));
    await syncSession(base(s, client));
    writeFileSync(s.transcript, prompt("u3", "New file."));
    await syncSession(base(s, client));
    expect(appends[1].messages.map((m) => m.id)).toEqual(["u3:0"]);
  });
});

describe("sessionOptions", () => {
  it("saves into the conversation document the plugin always wrote, with its tags and scopes", () => {
    const cfg = {
      retainSessions: true,
      autoInject: "reflect",
      observationScopes: "shared",
      retainTags: ["team:a"],
    } as unknown as Config;
    const o = sessionOptions({
      cfg,
      harness: "claude-code",
      sessionId: "abc",
      bankId: "b",
      cwd: "/tmp/x",
    });
    expect(o).not.toHaveProperty("source");
    expect(o).toMatchObject({
      document_id: "conversation:abc",
      live: false,
      retain: "turns",
      tags: ["team:a", "source:chat", "harness:claude-code"],
      metadata: {
        source: "chat",
        session_id: "abc",
        ref_id: "conversation:abc",
        harness: "claude-code",
      },
      strategy: "conversation",
      observation_scopes: "shared",
    });
    expect(
      sessionOptions({
        cfg: { ...cfg, retainSessions: false },
        harness: "codex",
        sessionId: "x",
        bankId: "b",
        cwd: "/",
      }).retain
    ).toBe("off");
    // The live session runs only where it was asked for; the session saves the conversation either way.
    expect(
      sessionOptions({
        cfg: { ...cfg, autoInject: "live" },
        harness: "claude-code",
        sessionId: "x",
        bankId: "b",
        cwd: "/",
      }).live
    ).toBe(true);
  });

  it("starts from an empty state", () => {
    expect(emptySyncState()).toMatchObject({
      offset: 0,
      ack: 0,
      opened: false,
      unsupported: false,
    });
  });
  it("sends the final reply with the turn end, and not again when the transcript catches up", async () => {
    // Claude Code writes the final reply after its Stop hook runs: the hook hands it over itself.
    const s = session();
    const { client, appends } = server();
    appendFileSync(s.transcript, prompt("u1", "Fix it."));
    const credit = "> 🧠 **From Hindsight memory (live session)** — K is 1024.";
    await syncSession({ ...base(s, client), turnEnd: true, reply: credit });
    expect(appends[0].messages.map((m) => [m.role, m.content])).toEqual([
      ["user", "Fix it."],
      ["assistant", credit],
    ]);
    expect(appends[0].turn_end).toBe(true);
    appendFileSync(s.transcript, reply("a1", credit) + prompt("u2", "Thanks."));
    await syncSession(base(s, client));
    expect(appends[1].messages.map((m) => m.content)).toEqual(["Thanks."]);
  });
});
