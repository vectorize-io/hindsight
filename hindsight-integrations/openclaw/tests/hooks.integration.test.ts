/**
 * Integration tests for the OpenClaw plugin hooks.
 *
 * Loads the plugin with a mock MoltbotPluginAPI in HTTP mode, then triggers
 * `before_prompt_build` and `agent_end` hooks with realistic event payloads.
 * Client methods (recall / retain) are spied on to verify the plugin
 * orchestrates them correctly without requiring a full LLM pipeline.
 *
 * Requirements:
 *   Running Hindsight API at HINDSIGHT_API_URL (default: http://localhost:8888)
 *
 * Run:
 *   npm run test:integration
 */

import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  afterEach,
  vi,
  type MockInstance,
} from "vitest";
import type { RecallResponse, RetainResponse } from "@vectorize-io/hindsight-client";
import type { MoltbotPluginAPI, PluginConfig } from "../src/types.js";

const HINDSIGHT_API_URL = process.env.HINDSIGHT_API_URL || "http://localhost:8888";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForApi(url: string, maxMs = 5000): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) return true;
    } catch {
      /* not ready yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

interface MockApiHandle {
  api: MoltbotPluginAPI;
  /** Trigger a registered hook and return the last handler's return value. */
  trigger(event: string, eventData: unknown, ctx?: unknown): Promise<unknown>;
  startServices(): Promise<void>;
  stopServices(): Promise<void>;
}

function createMockApi(pluginConfig: Partial<PluginConfig> = {}): MockApiHandle {
  const handlers = new Map<string, ((event: unknown, ctx?: unknown) => unknown)[]>();
  const services: { id: string; start(): Promise<void>; stop(): Promise<void> }[] = [];

  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": { enabled: true, config: pluginConfig as PluginConfig },
        },
      },
    },
    logger: {
      info: (msg: string) => console.log(msg),
      warn: (msg: string) => console.warn(msg),
      error: (msg: string) => console.error(msg),
    },
    registerService(svc: any) {
      services.push(svc);
    },
    on(event: string, handler: any) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
  };

  return {
    api,
    async trigger(event, eventData, ctx) {
      const list = handlers.get(event) ?? [];
      let result: unknown;
      for (const h of list) result = await h(eventData, ctx);
      return result;
    },
    async startServices() {
      for (const svc of services) await svc.start();
    },
    async stopServices() {
      for (const svc of services) await svc.stop();
    },
  };
}

const EMPTY_RECALL: RecallResponse = {
  results: [],
  entities: null,
  trace: null,
  chunks: null,
} as RecallResponse;
const OK_RETAIN = { operations: [], memory_units: [] } as unknown as RetainResponse;

interface MockMemoryResult {
  id: string;
  text: string;
  type: string;
  entities: string[];
  context: string;
  occurred_start: string | null;
  occurred_end: string | null;
  mentioned_at: string | null;
  document_id: string | null;
  metadata: Record<string, string> | null;
  chunk_id: string | null;
  tags: string[];
}

function makeMemoryResult(text: string): MockMemoryResult {
  return {
    id: `mem-${Math.random().toString(36).slice(2)}`,
    text,
    type: "fact",
    entities: [],
    context: "",
    occurred_start: null,
    occurred_end: null,
    mentioned_at: null,
    document_id: null,
    metadata: null,
    chunk_id: null,
    tags: [],
  };
}

// ---------------------------------------------------------------------------
// Module-level state shared across all hook describe blocks
// ---------------------------------------------------------------------------

let apiReachable = false;
let triggerHook: MockApiHandle["trigger"];
let stopServicesFn: () => Promise<void>;
// Typed loosely as MockInstance because vi.spyOn's generic form doesn't
// play nicely with the hindsight-client class shape (method overloads).
let recallSpy: MockInstance;
let retainSpy: MockInstance;

beforeAll(async () => {
  apiReachable = await waitForApi(HINDSIGHT_API_URL, 8000);
  if (!apiReachable) {
    console.warn(
      `[Hooks Integration] Hindsight API not reachable at ${HINDSIGHT_API_URL} – skipping hook tests.`
    );
    return;
  }

  // Reset module registry so we get a fresh module with clean state.
  vi.resetModules();

  const mod = await import("../src/index.js");
  const { HindsightClient } = await import("@vectorize-io/hindsight-client");
  const pluginFn = mod.default;
  const getClient = mod.getClient;

  // Plugin runs in external API mode (talks to the running test API), so no LLM
  // credentials are needed in the plugin config — the daemon handles them.
  const handle = createMockApi({
    hindsightApiUrl: HINDSIGHT_API_URL,
    dynamicBankId: true,
    excludeProviders: ["slack"],
    retainEveryNTurns: 1, // retain every turn so individual tests aren't affected by chunking
    recallContextTurns: 3,
    recallMaxQueryChars: 180,
    recallRoles: ["user"],
    // No bankMission — keeps init lean
  });
  triggerHook = handle.trigger;
  stopServicesFn = handle.stopServices;

  // Load the plugin — registers hooks and starts background init.
  pluginFn(handle.api);

  // service.start() awaits initPromise and health-checks the external API.
  await handle.startServices();

  // After startServices the client must be ready.
  if (!getClient())
    throw new Error("[Hooks Integration] Client not initialized after service start");

  // Spy on the HindsightClient prototype so all calls go through the spy.
  // The plugin's scopeClient() wrapper calls through these prototype methods.
  recallSpy = vi.spyOn(HindsightClient.prototype, "recall");
  retainSpy = vi.spyOn(HindsightClient.prototype, "retain");
}, 30_000);

afterAll(async () => {
  vi.restoreAllMocks();
  if (stopServicesFn) await stopServicesFn().catch(() => {});
}, 15_000);

afterEach(() => {
  // Reset spy call history between tests; don't remove the implementation.
  recallSpy?.mockReset();
  retainSpy?.mockReset();
});

// ---------------------------------------------------------------------------
// before_prompt_build hook
// ---------------------------------------------------------------------------

describe("before_prompt_build hook", () => {
  it("skips recall for excluded providers and returns undefined", async () => {
    if (!apiReachable) return;

    const result = await triggerHook(
      "before_prompt_build",
      { rawMessage: "What are my preferences?", prompt: "What are my preferences?", messages: [] },
      { messageProvider: "slack", senderId: "U001" }
    );

    expect(recallSpy).not.toHaveBeenCalled();
    expect(result).toBeUndefined();
  });

  it("skips recall when rawMessage is too short and returns undefined", async () => {
    if (!apiReachable) return;

    const result = await triggerHook(
      "before_prompt_build",
      { rawMessage: "Hi", prompt: "Hi", messages: [] },
      { messageProvider: "telegram", senderId: "U001" }
    );

    expect(recallSpy).not.toHaveBeenCalled();
    expect(result).toBeUndefined();
  });

  it("returns undefined when recall finds no results", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    const result = await triggerHook(
      "before_prompt_build",
      { rawMessage: "What programming language do I like?", prompt: "", messages: [] },
      { messageProvider: "telegram", senderId: "U002" }
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    expect(result).toBeUndefined();
  });

  it("returns { prependSystemContext } with <hindsight_memories> when recall returns results", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue({
      results: [makeMemoryResult("User likes Python")],
      entities: null,
      trace: null,
      chunks: null,
    } as RecallResponse);

    const result = (await triggerHook(
      "before_prompt_build",
      { rawMessage: "What programming language do I prefer?", prompt: "", messages: [] },
      { messageProvider: "telegram", senderId: "U003" }
    )) as { prependSystemContext: string; prependContext?: string };

    expect(result).toBeDefined();
    expect(result.prependContext).toBeUndefined();
    expect(result.prependSystemContext).toContain("<hindsight_memories>");
    expect(result.prependSystemContext).toContain("User likes Python");
    expect(result.prependSystemContext).toContain("</hindsight_memories>");
  });

  it("injects all memory result fields in the prependSystemContext", async () => {
    if (!apiReachable) return;
    const mem = makeMemoryResult("User prefers dark mode");
    mem.tags = ["preference"];
    mem.entities = ["dark_mode"];
    recallSpy.mockResolvedValue({
      results: [mem],
      entities: null,
      trace: null,
      chunks: null,
    } as RecallResponse);

    const result = (await triggerHook(
      "before_prompt_build",
      { rawMessage: "Do I prefer dark or light mode?", prompt: "", messages: [] },
      { messageProvider: "telegram", senderId: "U004" }
    )) as { prependSystemContext: string; prependContext?: string };

    // formatMemories returns a bullet list, not JSON
    expect(result.prependContext).toBeUndefined();
    expect(result.prependSystemContext).toContain("- User prefers dark mode");
    expect(result.prependSystemContext).toContain("<hindsight_memories>");
    expect(result.prependSystemContext).toContain("</hindsight_memories>");
  });

  it("extracts the inner query from an envelope-formatted prompt when rawMessage is absent", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    const envelopePrompt = "[Telegram Chat]\nWhat is my favorite food?\n[from: Alice]";
    await triggerHook(
      "before_prompt_build",
      { rawMessage: "", prompt: envelopePrompt, messages: [] },
      { messageProvider: "telegram", senderId: "U005" }
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    // HindsightClient.recall signature: (bankId, query, options?)
    const [, query] = recallSpy.mock.calls[0];
    expect(query).not.toContain("[Telegram");
    expect(query).not.toContain("[from: Alice]");
    expect(query).toContain("What is my favorite food?");
  });

  it("passes a latest-priority contextual recall query and respects max query chars", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    await triggerHook(
      "before_prompt_build",
      {
        rawMessage: "Do I still prefer dark mode?",
        prompt: "",
        messages: [
          { role: "user", content: "I prefer dark mode in IDEs." },
          { role: "assistant", content: "Noted: dark mode preference." },
          { role: "user", content: "Do I still prefer dark mode?" },
        ],
      },
      { messageProvider: "telegram", senderId: "U006A" }
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    const [, query] = recallSpy.mock.calls[0];
    expect(query).toContain("Do I still prefer dark mode?");
    expect(query).toContain("user: I prefer dark mode in IDEs.");
    expect(query).not.toContain("assistant: Noted: dark mode preference.");
    expect(query.length).toBeLessThanOrEqual(180);
  });

  it("passes maxTokens to recall", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    await triggerHook(
      "before_prompt_build",
      { rawMessage: "Tell me about my hobbies please.", prompt: "", messages: [] },
      { messageProvider: "telegram", senderId: "U006" }
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    const [, , options] = recallSpy.mock.calls[0];
    expect(options?.maxTokens).toBeGreaterThan(0);
  });

  it("includes recalled memories in the prependSystemContext block", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue({
      results: [makeMemoryResult("User loves hiking")],
      entities: null,
      trace: null,
      chunks: null,
    } as RecallResponse);

    const result = (await triggerHook(
      "before_prompt_build",
      { rawMessage: "What outdoor activities do I enjoy?", prompt: "", messages: [] },
      { messageProvider: "telegram", senderId: "U007" }
    )) as { prependSystemContext: string; prependContext?: string };

    expect(result.prependContext).toBeUndefined();
    expect(result.prependSystemContext).toContain("User loves hiking");
    expect(result.prependSystemContext).toContain("<hindsight_memories>");
  });

  it("uses identity cached in before_dispatch when later hooks lack sender metadata", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    await triggerHook(
      "before_dispatch",
      {
        sessionKey: "agent:main:telegram:direct:U020",
        channel: "telegram",
        senderId: "U020",
      },
      { sessionKey: "agent:main:telegram:direct:U020" }
    );

    await triggerHook(
      "before_prompt_build",
      { rawMessage: "What do I like?", prompt: "", messages: [] },
      { sessionKey: "agent:main:telegram:direct:U020" }
    );

    expect(recallSpy).toHaveBeenCalledOnce();
  });

  it("skips recall when before_dispatch detects a provider mismatch for the session", async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    await triggerHook(
      "before_dispatch",
      {
        sessionKey: "agent:main:telegram:direct:U021",
        channel: "discord",
        senderId: "U021",
      },
      { sessionKey: "agent:main:telegram:direct:U021" }
    );

    const result = await triggerHook(
      "before_prompt_build",
      { rawMessage: "What do I like?", prompt: "", messages: [] },
      { sessionKey: "agent:main:telegram:direct:U021" }
    );

    expect(recallSpy).not.toHaveBeenCalled();
    expect(result).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// agent_end hook
// ---------------------------------------------------------------------------

describe("agent_end hook", () => {
  it("skips retain when success is false", async () => {
    if (!apiReachable) return;

    await triggerHook(
      "agent_end",
      { success: false, messages: [{ role: "user", content: "Hello there world!" }] },
      { messageProvider: "telegram", senderId: "U010" }
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it("skips retain when messages array is empty", async () => {
    if (!apiReachable) return;

    await triggerHook(
      "agent_end",
      { success: true, messages: [] },
      { messageProvider: "telegram", senderId: "U011" }
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it("skips retain for excluded providers", async () => {
    if (!apiReachable) return;

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "I work as a software engineer." }],
      },
      { messageProvider: "slack", senderId: "U012" }
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it("calls retain with correctly formatted transcript for string content", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [
          { role: "user", content: "I love TypeScript." },
          { role: "assistant", content: "TypeScript is great!" },
        ],
      },
      { messageProvider: "telegram", senderId: "U013", sessionKey: "sess-ts-test" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    // HindsightClient.retain signature: (bankId, content, options?)
    // Default retainFormat is 'json' with Anthropic-shaped typed blocks.
    const [, content] = retainSpy.mock.calls[0];
    const parsed = JSON.parse(content);
    expect(parsed).toEqual([
      { role: "user", content: [{ type: "text", text: "I love TypeScript." }] },
      { role: "assistant", content: [{ type: "text", text: "TypeScript is great!" }] },
    ]);
  });

  it("includes session key in documentId", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "My favourite colour is blue." }],
      },
      { messageProvider: "telegram", senderId: "U014", sessionKey: "sess-colour" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, , options] = retainSpy.mock.calls[0];
    expect(options?.documentId).toContain("sess-colour");
  });

  it("populates metadata with channel_type, channel_id, and sender_id", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "My cat is named Whiskers." }],
      },
      {
        messageProvider: "telegram",
        channelId: "chat-999",
        senderId: "U015",
        sessionKey: "sess-cat",
      }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, , options] = retainSpy.mock.calls[0];
    expect(options?.metadata?.channel_type).toBe("telegram");
    expect(options?.metadata?.channel_id).toBe("chat-999");
    expect(options?.metadata?.sender_id).toBe("U015");
    expect(options?.metadata?.retained_at).toBeDefined();
    expect(options?.metadata?.message_count).toBe("1");
  });

  it("uses identity cached in before_dispatch for retain metadata when agent_end ctx is sparse", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "before_dispatch",
      {
        sessionKey: "agent:main:telegram:direct:U015B",
        channel: "telegram",
        senderId: "U015B",
      },
      { sessionKey: "agent:main:telegram:direct:U015B" }
    );

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "I like midnight blue." }],
      },
      { sessionKey: "agent:main:telegram:direct:U015B" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, , options] = retainSpy.mock.calls[0];
    expect(options?.metadata?.channel_type).toBe("telegram");
    expect(options?.metadata?.channel_id).toBe("direct:U015B");
    expect(options?.metadata?.sender_id).toBe("U015B");
  });

  it("keeps provider fallback without backfilling channel_type when only session parsing provides it", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "I prefer espresso." }],
      },
      { sessionKey: "agent:main:telegram:direct:U015C" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, , options] = retainSpy.mock.calls[0];
    expect(options?.metadata?.provider).toBe("telegram");
    expect(options?.metadata?.channel_type).toBeUndefined();
    expect(options?.metadata?.channel_id).toBe("direct:U015C");
    expect(options?.metadata?.sender_id).toBe("U015C");
  });

  it("strips <hindsight_memories> tags from content before retaining", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    const contentWithMemories =
      '<hindsight_memories>\nRelevant memories:\n[{"text":"old fact"}]\n</hindsight_memories>\nI enjoy reading science fiction.';

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: contentWithMemories }],
      },
      { messageProvider: "telegram", senderId: "U016", sessionKey: "sess-strip" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, content] = retainSpy.mock.calls[0];
    expect(content).not.toContain("<hindsight_memories>");
    expect(content).not.toContain("</hindsight_memories>");
    expect(content).not.toContain("old fact");
    expect(content).toContain("I enjoy reading science fiction.");
  });

  it("strips <relevant_memories> tags from content before retaining", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    const contentWithLegacyTag =
      "<relevant_memories>\nSome old memories\n</relevant_memories>\nI am learning Rust.";

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: contentWithLegacyTag }],
      },
      { messageProvider: "telegram", senderId: "U017", sessionKey: "sess-legacy" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, content] = retainSpy.mock.calls[0];
    expect(content).not.toContain("<relevant_memories>");
    expect(content).toContain("I am learning Rust.");
  });

  it("handles array content blocks (structured message format)", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "I prefer dark mode in all my editors." },
              { type: "image", source: "data:..." }, // non-text block — should be ignored
            ],
          },
        ],
      },
      { messageProvider: "telegram", senderId: "U018", sessionKey: "sess-array" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, content] = retainSpy.mock.calls[0];
    expect(content).toContain("I prefer dark mode in all my editors.");
    expect(content).not.toContain("data:");
  });

  it("retains a multi-turn conversation in the correct transcript format", async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      "agent_end",
      {
        success: true,
        messages: [
          { role: "user", content: "My name is Carol." },
          { role: "assistant", content: "Nice to meet you, Carol!" },
          { role: "user", content: "I work as a data scientist." },
          { role: "assistant", content: "That's a fascinating career!" },
        ],
      },
      { messageProvider: "telegram", senderId: "U019", sessionKey: "sess-multi" }
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [, content, options] = retainSpy.mock.calls[0];

    // Only the last turn (from last user message onwards) is retained.
    // Default retainFormat is 'json' with Anthropic-shaped typed blocks.
    const parsed = JSON.parse(content);
    expect(parsed).toEqual([
      { role: "user", content: [{ type: "text", text: "I work as a data scientist." }] },
      { role: "assistant", content: [{ type: "text", text: "That's a fascinating career!" }] },
    ]);
    expect(content).not.toContain("My name is Carol.");
    expect(options?.metadata?.message_count).toBe("2");
  });
});
