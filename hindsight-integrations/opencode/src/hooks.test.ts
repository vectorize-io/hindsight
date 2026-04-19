import { describe, it, expect, vi, beforeEach } from "vitest";
import { createHooks, type PluginState } from "./hooks.js";
import { makeConfig } from "./test-helpers.js";

function makeState(): PluginState {
  return {
    turnCount: 0,
    missionsSet: new Set(),
    recalledSessions: new Set(),
    lastRetainedTurn: new Map(),
  };
}

function makeClient() {
  return {
    retain: vi.fn().mockResolvedValue({}),
    recall: vi.fn().mockResolvedValue({ results: [] }),
    reflect: vi.fn().mockResolvedValue({ text: "" }),
    createBank: vi.fn().mockResolvedValue({}),
  } as any;
}

function makeOpencodeClient(
  messages: Array<{ info: { role: string }; parts: Array<{ type: string; text?: string }> }> = []
) {
  return {
    session: {
      messages: vi.fn().mockResolvedValue({ data: messages }),
    },
  };
}

describe("createHooks", () => {
  it("returns all required hooks", () => {
    const hooks = createHooks(
      makeClient(),
      "bank",
      makeConfig(),
      makeState(),
      makeOpencodeClient()
    );
    expect(hooks.event).toBeDefined();
    expect(hooks["experimental.session.compacting"]).toBeDefined();
    expect(hooks["experimental.chat.system.transform"]).toBeDefined();
  });
});

describe("event hook — session.idle", () => {
  it("auto-retains conversation on session.idle with document_id", async () => {
    const client = makeClient();
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi there" }] },
    ];
    const opencodeClient = makeOpencodeClient(messages);
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ retainEveryNTurns: 1 }),
      state,
      opencodeClient
    );

    await hooks.event({
      event: { type: "session.idle", properties: { sessionID: "sess-1" } },
    });

    expect(client.retain).toHaveBeenCalledTimes(1);
    expect(client.retain.mock.calls[0][0]).toBe("bank");
    // Full-session mode uses session ID as document_id
    const opts = client.retain.mock.calls[0][2];
    expect(opts.documentId).toBe("sess-1");
    expect(opts.metadata.session_id).toBe("sess-1");
  });

  it("skips retain when autoRetain is false", async () => {
    const client = makeClient();
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ autoRetain: false }),
      makeState(),
      makeOpencodeClient(messages)
    );

    await hooks.event({
      event: { type: "session.idle", properties: { sessionID: "sess-1" } },
    });

    expect(client.retain).not.toHaveBeenCalled();
  });

  it("uses chunked document_id with overlap in last-turn mode", async () => {
    const client = makeClient();
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Turn 1" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Reply 1" }] },
      { info: { role: "user" }, parts: [{ type: "text", text: "Turn 2" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Reply 2" }] },
    ];
    const config = makeConfig({
      retainMode: "last-turn",
      retainEveryNTurns: 1,
      retainOverlapTurns: 1,
    });
    const state = makeState();
    const hooks = createHooks(client, "bank", config, state, makeOpencodeClient(messages));

    await hooks.event({
      event: { type: "session.idle", properties: { sessionID: "sess-1" } },
    });

    expect(client.retain).toHaveBeenCalledTimes(1);
    const opts = client.retain.mock.calls[0][2];
    // Chunked mode uses session-timestamp format
    expect(opts.documentId).toMatch(/^sess-1-\d+$/);
  });

  it("respects retainEveryNTurns", async () => {
    const client = makeClient();
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const config = makeConfig({ retainEveryNTurns: 5 });
    const state = makeState();
    const hooks = createHooks(client, "bank", config, state, makeOpencodeClient(messages));

    await hooks.event({
      event: { type: "session.idle", properties: { sessionID: "sess-1" } },
    });

    // Only 1 user turn, needs 5 — should not retain
    expect(client.retain).not.toHaveBeenCalled();
  });

  it("does not throw on client error", async () => {
    const client = makeClient();
    client.retain.mockRejectedValue(new Error("Network error"));
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ retainEveryNTurns: 1 }),
      makeState(),
      makeOpencodeClient(messages)
    );

    await expect(
      hooks.event({
        event: { type: "session.idle", properties: { sessionID: "sess-1" } },
      })
    ).resolves.not.toThrow();
  });
});

describe("event hook — session.created", () => {
  it("tracks session for recall injection", async () => {
    const state = makeState();
    const hooks = createHooks(makeClient(), "bank", makeConfig(), state, makeOpencodeClient());

    await hooks.event({
      event: {
        type: "session.created",
        properties: { info: { id: "sess-1", title: "Test" } },
      },
    });

    expect(state.recalledSessions.has("sess-1")).toBe(true);
  });

  it("does not track when autoRecall is false", async () => {
    const state = makeState();
    const hooks = createHooks(
      makeClient(),
      "bank",
      makeConfig({ autoRecall: false }),
      state,
      makeOpencodeClient()
    );

    await hooks.event({
      event: {
        type: "session.created",
        properties: { info: { id: "sess-1" } },
      },
    });

    expect(state.recalledSessions.has("sess-1")).toBe(false);
  });
});

describe("compacting hook", () => {
  it("retains before compaction and recalls context", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "Important fact", type: "world" }],
    });
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Build the feature" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Working on it" }] },
    ];
    const output = { context: [] as string[], prompt: undefined };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      makeState(),
      makeOpencodeClient(messages)
    );

    await hooks["experimental.session.compacting"]({ sessionID: "sess-1" }, output);

    // Should have retained and recalled
    expect(client.retain).toHaveBeenCalled();
    expect(client.recall).toHaveBeenCalled();
    expect(output.context.length).toBeGreaterThan(0);
    expect(output.context[0]).toContain("hindsight_memories");
    expect(output.context[0]).toContain("Important fact");
  });

  it("pre-compaction retain includes documentId and session metadata", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const output = { context: [] as string[] };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      makeState(),
      makeOpencodeClient(messages)
    );

    await hooks["experimental.session.compacting"]({ sessionID: "sess-1" }, output);

    expect(client.retain).toHaveBeenCalledTimes(1);
    const opts = client.retain.mock.calls[0][2];
    expect(opts.documentId).toBe("sess-1");
    expect(opts.metadata.session_id).toBe("sess-1");
  });

  it("pre-compaction retain uses chunked documentId in last-turn mode", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const config = makeConfig({ retainMode: "last-turn", retainEveryNTurns: 1 });
    const output = { context: [] as string[] };
    const hooks = createHooks(client, "bank", config, makeState(), makeOpencodeClient(messages));

    await hooks["experimental.session.compacting"]({ sessionID: "sess-1" }, output);

    const opts = client.retain.mock.calls[0][2];
    expect(opts.documentId).toMatch(/^sess-1-\d+$/);
  });

  it("resets lastRetainedTurn so idle-retain resumes after compaction", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const messages = [
      { info: { role: "user" }, parts: [{ type: "text", text: "Hello" }] },
      { info: { role: "assistant" }, parts: [{ type: "text", text: "Hi" }] },
    ];
    const state = makeState();
    // Simulate prior retain at turn 10
    state.lastRetainedTurn.set("sess-1", 10);
    const output = { context: [] as string[] };
    const hooks = createHooks(client, "bank", makeConfig(), state, makeOpencodeClient(messages));

    await hooks["experimental.session.compacting"]({ sessionID: "sess-1" }, output);

    // After compaction, lastRetainedTurn should be cleared so idle-retain works again
    expect(state.lastRetainedTurn.has("sess-1")).toBe(false);
  });

  it("does not throw on error", async () => {
    const client = makeClient();
    client.recall.mockRejectedValue(new Error("Failed"));
    const messages = [{ info: { role: "user" }, parts: [{ type: "text", text: "Test" }] }];
    const output = { context: [] as string[] };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      makeState(),
      makeOpencodeClient(messages)
    );

    await expect(
      hooks["experimental.session.compacting"]({ sessionID: "s" }, output)
    ).resolves.not.toThrow();
  });
});

describe("system transform hook", () => {
  it("injects memories for tracked sessions", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "User is a developer", type: "world" }],
    });
    const state = makeState();
    state.recalledSessions.add("sess-1");
    const output = { system: [] as string[] };
    const hooks = createHooks(client, "bank", makeConfig(), state, makeOpencodeClient());

    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    expect(output.system.length).toBeGreaterThan(0);
    expect(output.system[0]).toContain("hindsight_memories");
    // Session should be removed after first injection
    expect(state.recalledSessions.has("sess-1")).toBe(false);
  });

  it("skips untracked sessions", async () => {
    const client = makeClient();
    const state = makeState();
    const output = { system: [] as string[] };
    const hooks = createHooks(client, "bank", makeConfig(), state, makeOpencodeClient());

    await hooks["experimental.chat.system.transform"](
      { sessionID: "sess-unknown", model: {} },
      output
    );

    expect(output.system.length).toBe(0);
    expect(client.recall).not.toHaveBeenCalled();
  });

  it("consumes session on empty recall (no repeated queries for empty banks)", async () => {
    const client = makeClient();
    // No results — empty bank
    client.recall.mockResolvedValue({ results: [] });
    const state = makeState();
    state.recalledSessions.add("sess-1");
    const output = { system: [] as string[] };
    const hooks = createHooks(client, "bank", makeConfig(), state, makeOpencodeClient());

    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    // No injection, but session consumed — won't re-query on next transform
    expect(output.system.length).toBe(0);
    expect(state.recalledSessions.has("sess-1")).toBe(false);
  });

  it("retries recall on next transform after transient API failure", async () => {
    const client = makeClient();
    // First call: API error (transient)
    client.recall.mockRejectedValueOnce(new Error("Connection refused"));
    // Second call: succeeds
    client.recall.mockResolvedValueOnce({
      results: [{ text: "Found it", type: "world" }],
    });
    const state = makeState();
    state.recalledSessions.add("sess-1");
    const hooks = createHooks(client, "bank", makeConfig(), state, makeOpencodeClient());

    // First attempt — API error, session preserved for retry
    const output1 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output1);
    expect(output1.system.length).toBe(0);
    expect(state.recalledSessions.has("sess-1")).toBe(true);

    // Second attempt — succeeds, session consumed
    const output2 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output2);
    expect(output2.system.length).toBeGreaterThan(0);
    expect(state.recalledSessions.has("sess-1")).toBe(false);
  });

  it("skips when autoRecall is false", async () => {
    const client = makeClient();
    const state = makeState();
    state.recalledSessions.add("sess-1");
    const output = { system: [] as string[] };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ autoRecall: false }),
      state,
      makeOpencodeClient()
    );

    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    expect(output.system.length).toBe(0);
  });
});
