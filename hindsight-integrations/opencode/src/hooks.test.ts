import { describe, it, expect, vi, beforeEach } from "vitest";
import { createHooks, type PluginState } from "./hooks.js";
import { makeConfig } from "./test-helpers.js";

function makeState(): PluginState {
  return {
    turnCount: 0,
    missionsSet: new Set(),
    lastRetainedTurn: new Map(),
    sessionTurnCount: new Map(),
    prefetchCache: new Map(),
    prefetchInProgress: new Map(),
    sessionTurns: new Map(),
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

  it("passes retainTags from config to retain call", async () => {
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
      makeConfig({ retainTags: ["user:alice", "shared"], retainEveryNTurns: 1 }),
      state,
      opencodeClient
    );

    await hooks.event({
      event: { type: "session.idle", properties: { sessionID: "sess-1" } },
    });

    expect(client.retain).toHaveBeenCalledTimes(1);
    const opts = client.retain.mock.calls[0][2];
    expect(opts.tags).toEqual(["user:alice", "shared"]);
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

// Helper messages that the system transform hook needs (it fetches session
// messages to build a contextual recall query).
const CONVO_MESSAGES = [
  { info: { role: "user" }, parts: [{ type: "text", text: "Help me with React" }] },
  { info: { role: "assistant" }, parts: [{ type: "text", text: "Sure, let me help" }] },
];

describe("system transform hook — recalls every turn", () => {
  it("injects memory instructions on the first turn", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const state = makeState();
    const output = { system: ["You are a helpful assistant."] as string[] };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    // Should inject memory instructions on turn 1 (hermes-agent memory tool style)
    expect(output.system[0]).toContain("Hindsight Memory");
    expect(output.system[0]).toContain("WHEN TO SAVE");
    expect(output.system[0]).toContain("proactively");
    expect(output.system[0]).toContain("PRIORITY");
    expect(output.system[0]).toContain("hindsight_recall");
    expect(output.system[0]).toContain("hindsight_reflect");
    expect(output.system[0]).toContain("hindsight_retain");
  });

  it("recalls on every turn (not just once per session)", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "User is a developer", type: "world" }],
    });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    // Turn 1 — live recall fires; background prefetch may also resolve synchronously
    const output1 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output1);
    // At least 1 call (live recall) — mock resolves synchronously so prefetch may also fire
    expect(client.recall.mock.calls.length).toBeGreaterThanOrEqual(1);
    const callsAfterTurn1 = client.recall.mock.calls.length;

    // Turn 2 — should recall again (no dedup by session)
    const output2 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output2);
    expect(client.recall.mock.calls.length).toBeGreaterThan(callsAfterTurn1);

    // Turn 3 — still recalls
    const output3 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output3);
    expect(client.recall.mock.calls.length).toBeGreaterThan(callsAfterTurn1 + 1);
  });

  it("appends recall into the existing first system section, not a new one", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "User is a developer", type: "world" }],
    });
    const state = makeState();
    const output = { system: ["You are a helpful coding assistant."] as string[] };
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    // Still a single system section — appended, not pushed.
    expect(output.system.length).toBe(1);
    expect(output.system[0]).toContain("You are a helpful coding assistant.");
    expect(output.system[0]).toContain("hindsight_memories");
    expect(output.system[0]).toContain("User is a developer");
    // Herms-agent style preamble
    expect(output.system[0]).toContain(
      "Do not call tools to look up information that is already present here"
    );
  });

  it("tracks turn count per session", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ autoRetain: false }),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    await hooks["experimental.chat.system.transform"](
      { sessionID: "sess-1", model: {} },
      { system: [] }
    );
    await hooks["experimental.chat.system.transform"](
      { sessionID: "sess-1", model: {} },
      { system: [] }
    );
    await hooks["experimental.chat.system.transform"](
      { sessionID: "sess-1", model: {} },
      { system: [] }
    );

    // Each system.transform increments once; retainTurn (disabled here) does not double-count
    expect(state.sessionTurnCount.get("sess-1")).toBe(3);
  });

  it("does not inject memory instructions on subsequent turns", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    // Turn 1 — instructions injected
    const output1 = { system: ["Base prompt."] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output1);
    expect(output1.system[0]).toContain("Hindsight Memory");

    // Turn 2 — instructions NOT re-injected
    const output2 = { system: ["Base prompt."] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output2);
    expect(output2.system[0]).not.toContain("Hindsight Memory");
  });

  it("skips when autoRecall is false", async () => {
    const client = makeClient();
    const state = makeState();
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
    expect(client.recall).not.toHaveBeenCalled();
  });

  it("retries recall after transient API failure", async () => {
    const client = makeClient();
    // First call: API error
    client.recall.mockRejectedValueOnce(new Error("Connection refused"));
    // Second call: succeeds
    client.recall.mockResolvedValueOnce({
      results: [{ text: "Found it", type: "world" }],
    });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    // First attempt — API error, recall content not injected (but memory instructions ARE on turn 1)
    const output1 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output1);
    // system[0] has memory instructions but no hindsight_memories (recall failed)
    expect(output1.system[0]).not.toContain("hindsight_memories");

    // Second attempt — succeeds (different turn since we no longer dedup)
    const output2 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output2);
    expect(output2.system[0]).toContain("Found it");
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

  it("pre-compaction retain passes retainTags from config", async () => {
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
      makeConfig({ retainTags: ["user:alice", "auto-tag"] }),
      makeState(),
      makeOpencodeClient(messages)
    );

    await hooks["experimental.session.compacting"]({ sessionID: "sess-1" }, output);

    expect(client.retain).toHaveBeenCalledTimes(1);
    const opts = client.retain.mock.calls[0][2];
    expect(opts.tags).toEqual(["user:alice", "auto-tag"]);
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

describe("background prefetch", () => {
  it("queues a background prefetch after each turn", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "Cached memory", type: "world" }],
    });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    const output = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    // Wait for background prefetch to complete
    await new Promise((r) => setTimeout(r, 100));

    // The prefetch should have been called (the live recall + the background prefetch)
    expect(client.recall).toHaveBeenCalled();
  });

  it("uses prefetch cache on the next turn", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({
      results: [{ text: "Prefetched data", type: "world" }],
    });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig(),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    // Turn 1 — triggers background prefetch
    const output1 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output1);

    // Wait for prefetch to complete
    await new Promise((r) => setTimeout(r, 100));

    // Turn 2 — should use cached prefetch
    const output2 = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output2);

    // The prefetch cache should have been used
    expect(output2.system.length).toBeGreaterThan(0);
  });
});

describe("retain every turn", () => {
  it("retains turns in the background during system.transform", async () => {
    const client = makeClient();
    client.recall.mockResolvedValue({ results: [] });
    const state = makeState();
    const hooks = createHooks(
      client,
      "bank",
      makeConfig({ retainEveryNTurns: 1 }),
      state,
      makeOpencodeClient(CONVO_MESSAGES)
    );

    const output = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ sessionID: "sess-1", model: {} }, output);

    // Wait for background retain to complete
    await new Promise((r) => setTimeout(r, 100));

    // retainTurn was called (may be in addition to other retains)
    // The key is that it was called at least once from the turn flow
    expect(client.retain).toHaveBeenCalled();
  });
});
