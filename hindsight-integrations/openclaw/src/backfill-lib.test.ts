import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  buildBackfillPlan,
  readGatewaySessions,
  loadPluginConfigFromOpenClawRoot,
  stableDocumentId,
} from "./backfill-lib.js";

const tempDirs: string[] = [];

function makeTempRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), "hindsight-openclaw-backfill-"));
  tempDirs.push(dir);
  return dir;
}

function writeOpenClawConfig(root: string, config: Record<string, unknown>) {
  writeFileSync(join(root, "openclaw.json"), JSON.stringify(config, null, 2));
}

function writeSession(
  root: string,
  agentId: string,
  fileName: string,
  lines: unknown[],
  archive = false
) {
  const dir = archive
    ? join(root, "agents", agentId, "sessions-archive-from-migration_backup")
    : join(root, "agents", agentId, "sessions");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, fileName), lines.map((line) => JSON.stringify(line)).join("\n") + "\n");
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("backfill planning", () => {
  it("mirrors plugin bank routing from config by default", () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              dynamicBankId: true,
              dynamicBankGranularity: ["agent", "provider", "channel"],
            },
          },
        },
      },
    });
    writeSession(root, "proj-run", "one.jsonl", [
      { type: "session", id: "session-1", sessionKey: "agent:proj-run:discord:channel:123" },
      { type: "message", message: { role: "user", content: "hello" } },
      { type: "message", message: { role: "assistant", content: "world" } },
    ]);

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: true,
      bankStrategy: "mirror-config",
    });

    expect(result.discoveredSessions).toBe(1);
    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].bankId).toBe("proj-run::discord::channel%3A123");
    expect(result.entries[0].documentId).toBe(
      stableDocumentId(
        {
          filePath: result.entries[0].filePath,
          agentId: "proj-run",
          sessionId: "session-1",
          sessionKey: "agent:proj-run:discord:channel:123",
          messages: [],
        },
        result.entries[0].bankId
      )
    );
  });

  it("supports migration overrides for agent-only banks", () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, { plugins: { entries: { "hindsight-openclaw": { config: {} } } } });
    writeSession(root, "proj-debug", "two.jsonl", [
      { type: "session", id: "session-2", sessionKey: "agent:proj-debug:discord:group:abc" },
      { type: "message", message: { role: "user", content: "hello" } },
      { type: "message", message: { role: "assistant", content: "world" } },
    ]);

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: true,
      bankStrategy: "agent",
    });

    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].bankId).toBe("proj-debug");
  });

  it("can exclude archive sessions", () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, { plugins: { entries: { "hindsight-openclaw": { config: {} } } } });
    writeSession(root, "main", "live.jsonl", [
      { type: "session", id: "live" },
      { type: "message", message: { role: "user", content: "live" } },
      { type: "message", message: { role: "assistant", content: "reply" } },
    ]);
    writeSession(
      root,
      "main",
      "archive.jsonl",
      [
        { type: "session", id: "archive" },
        { type: "message", message: { role: "user", content: "archived" } },
        { type: "message", message: { role: "assistant", content: "reply" } },
      ],
      true
    );

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: false,
      bankStrategy: "mirror-config",
    });

    expect(result.discoveredSessions).toBe(1);
    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].sessionId).toBe("live");
  });

  it("reads sessions from the Gateway in order across pages (#5569)", async () => {
    // A stand-in Gateway: two sessions, the first one's history split over two
    // newest-first pages, as chat.history returns it.
    const calls: Array<[string, Record<string, unknown>]> = [];
    const gateway = async (method: string, params: Record<string, unknown>) => {
      calls.push([method, params]);
      if (method === "sessions.list") {
        return {
          sessions: [
            { key: "agent:main:telegram:direct:42", agentId: "main", sessionId: "s1" },
            { key: "agent:other:main", agentId: "other", sessionId: "s2" },
          ],
          hasMore: false,
        };
      }
      if (params.offset === 0) {
        return {
          messages: [{ role: "assistant", content: "Porto, noted.", __openclaw: { seq: 2 } }],
          hasMore: true,
          nextOffset: 1,
        };
      }
      return {
        messages: [{ role: "user", content: "My favourite city is Porto", __openclaw: { seq: 1 } }],
        hasMore: false,
      };
    };

    const sessions = await readGatewaySessions(gateway, new Set(["main"]));

    expect(sessions).toHaveLength(1);
    expect(sessions[0]).toMatchObject({ agentId: "main", sessionId: "s1" });
    expect(sessions[0].messages.map((m) => m.content)).toEqual([
      "My favourite city is Porto",
      "Porto, noted.",
    ]);
    // The unselected agent's history is never fetched.
    expect(calls.filter(([m]) => m === "chat.history")).toHaveLength(2);

    const plan = buildBackfillPlan(
      { dynamicBankId: true, retainRoles: ["user", "assistant"] },
      { openclawRoot: "/nowhere", includeArchive: true, bankStrategy: "agent" },
      sessions
    );
    expect(plan.entries).toHaveLength(1);
    expect(plan.entries[0].transcript).toContain("My favourite city is Porto");
  });

  it("skips the sessions live saving skips (#5569)", () => {
    const session = (key: string) => ({
      filePath: `gateway:${key}`,
      agentId: "main",
      sessionId: key,
      sessionKey: key,
      messages: [
        { role: "user" as const, content: "hello" },
        { role: "assistant" as const, content: "world" },
      ],
    });
    const sessions = [
      session("agent:main:cron:nightly"),
      session("agent:main:heartbeat:1"),
      session("agent:main:subagent:abc"),
      session("agent:main:secret:1"),
      session("agent:main:telegram:direct:42"),
    ];
    const opts = { openclawRoot: "/nowhere", includeArchive: true, bankStrategy: "agent" as const };
    const base = { retainRoles: ["user", "assistant"] as Array<"user" | "assistant"> };

    const plan = buildBackfillPlan(
      { ...base, ignoreSessionPatterns: ["agent:main:secret:*"] },
      opts,
      sessions
    );
    expect(plan.entries.map((e) => e.sessionId)).toEqual(["agent:main:telegram:direct:42"]);
    expect(plan.skippedFiltered).toBe(4);

    const optedIn = buildBackfillPlan({ ...base, skipOperationalSessions: false }, opts, sessions);
    expect(optedIn.entries).toHaveLength(5);
  });
});
