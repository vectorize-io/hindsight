import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  MemoryToolsContext,
  MemoryTurnCompletedContext,
  MemoryTurnStartedContext,
} from "eve/memory";
import { hindsightMemory } from "./memory-provider";
import { EMPTY_RECALL_CONTENT, RECALL_MESSAGE_ID } from "./provider-core";

const OPTS = { apiUrl: "http://test", apiKey: "k" };
const SCOPE = { key: "memscope1_abc", namespace: "ns", value: "user-1" };

/** Mock fetch, routing by URL; returns recall results, a reflect answer, or a retain ack. */
function mockFetch(
  recallResults: unknown[] = [],
  reflectText = "answer"
): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async (url: string) => ({
    ok: true,
    status: 200,
    json: async () =>
      url.endsWith("/recall")
        ? { results: recallResults }
        : url.endsWith("/reflect")
          ? { text: reflectText }
          : { success: true },
    text: async () => "",
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

function failingFetch(): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({
    ok: false,
    status: 500,
    json: async () => ({}),
    text: async () => "boom",
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

/** The parts of eve's contexts the provider reads; the rest is stubbed. */
function turnStarted(userText: string, overrides: Partial<MemoryTurnStartedContext> = {}) {
  return {
    abortSignal: new AbortController().signal,
    messages: [],
    operationId: "op-1",
    memory: { scope: SCOPE, slot: "hindsight" },
    session: { id: "s1" },
    turn: { id: "t1", input: [{ role: "user", content: userText }], sequence: 1 },
    ...overrides,
  } as unknown as MemoryTurnStartedContext;
}

function turnCompleted(userText: string, assistantText: string | null) {
  const messages: unknown[] = [{ role: "user", content: userText }];
  if (assistantText !== null) messages.push({ role: "assistant", content: assistantText });
  return {
    abortSignal: new AbortController().signal,
    messages,
    operationId: "op-2",
    memory: { scope: SCOPE, slot: "hindsight" },
    session: { id: "s1" },
    turn: { id: "t1", input: [{ role: "user", content: userText }], sequence: 1 },
  } as unknown as MemoryTurnCompletedContext;
}

const toolsCtx = {
  memory: { scope: SCOPE, slot: "hindsight" },
  turn: { id: "t1", input: [], sequence: 1 },
} as unknown as MemoryToolsContext;

afterEach(() => vi.unstubAllGlobals());

describe("hindsightMemory recall", () => {
  it("recalls on the live user message in the scope's bank and returns one keyed message", async () => {
    const fetchFn = mockFetch([{ id: "m1", text: "User prefers tabs" }]);
    const provider = hindsightMemory(OPTS);

    const result = await provider.recall["turn.started"](turnStarted("indent this file"));

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://test/v1/default/banks/memscope1_abc/memories/recall");
    expect(JSON.parse(init.body)).toEqual({
      query: "indent this file",
      budget: "mid",
      max_tokens: 1024,
    });
    expect(result).toEqual({
      messages: [
        { id: RECALL_MESSAGE_ID, content: expect.stringContaining("- User prefers tabs") },
      ],
    });
  });

  it("still returns the keyed message when nothing was recalled, so a stale block is superseded", async () => {
    mockFetch([]);
    const result = await hindsightMemory(OPTS).recall["turn.started"](turnStarted("hello"));
    expect(result).toEqual({
      messages: [{ id: RECALL_MESSAGE_ID, content: EMPTY_RECALL_CONTENT }],
    });
  });

  it("returns null and reports onError on failure instead of failing the turn", async () => {
    failingFetch();
    const onError = vi.fn();
    const result = await hindsightMemory({ ...OPTS, onError }).recall["turn.started"](
      turnStarted("hi")
    );
    expect(result).toBeNull();
    expect(onError).toHaveBeenCalledWith(expect.anything(), "recall");
  });

  it("stays quiet when the failure is the turn's own cancellation", async () => {
    failingFetch();
    const onError = vi.fn();
    const controller = new AbortController();
    controller.abort();
    const result = await hindsightMemory({ ...OPTS, onError }).recall["turn.started"](
      turnStarted("hi", { abortSignal: controller.signal })
    );
    expect(result).toBeNull();
    expect(onError).not.toHaveBeenCalled();
  });

  it("recalls after compaction with the fallback query when there is no turn", async () => {
    const fetchFn = mockFetch([]);
    const provider = hindsightMemory({ ...OPTS, recallQuery: "profile" });
    await provider.recall["compaction.completed"]!(turnStarted("", { turn: null }) as never);
    expect(JSON.parse(fetchFn.mock.calls[0][1].body).query).toBe("profile");
  });

  it("uses a custom bank resolver", async () => {
    const fetchFn = mockFetch([]);
    const provider = hindsightMemory({ ...OPTS, bankId: (scope) => `eve-${scope.value}` });
    await provider.recall["turn.started"](turnStarted("hi"));
    expect(fetchFn.mock.calls[0][0]).toBe(
      "http://test/v1/default/banks/eve-user-1/memories/recall"
    );
  });
});

describe("hindsightMemory capture", () => {
  it("retains the exchange keyed by operationId into the scope's bank", async () => {
    const fetchFn = mockFetch();
    const provider = hindsightMemory(OPTS);

    await provider.capture!["turn.completed"]!(turnCompleted("I prefer tabs", "Noted."));

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://test/v1/default/banks/memscope1_abc/memories");
    const body = JSON.parse(init.body);
    expect(body.async).toBe(true);
    expect(body.items).toHaveLength(1);
    expect(body.items[0]).toMatchObject({
      content: "User: I prefer tabs\n\nAssistant: Noted.",
      context: "eve",
      document_id: "op-2",
      metadata: { sessionId: "s1", turnId: "t1", slot: "hindsight" },
    });
    expect(typeof body.items[0].timestamp).toBe("string");
  });

  it("retains only the user message when includeAssistantReply is false", async () => {
    const fetchFn = mockFetch();
    const provider = hindsightMemory({ ...OPTS, includeAssistantReply: false });
    await provider.capture!["turn.completed"]!(turnCompleted("I prefer tabs", "Noted."));
    expect(JSON.parse(fetchFn.mock.calls[0][1].body).items[0].content).toBe("User: I prefer tabs");
  });

  it("skips a turn with no user text", async () => {
    const fetchFn = mockFetch();
    await hindsightMemory(OPTS).capture!["turn.completed"]!(turnCompleted("", "hi"));
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("reports onError on failure and never throws", async () => {
    failingFetch();
    const onError = vi.fn();
    await expect(
      hindsightMemory({ ...OPTS, onError }).capture!["turn.completed"]!(turnCompleted("a", "b"))
    ).resolves.toBeUndefined();
    expect(onError).toHaveBeenCalledWith(expect.anything(), "capture");
  });

  it("has no capture surface when capture is off", () => {
    expect(hindsightMemory({ ...OPTS, capture: false }).capture).toBeUndefined();
  });
});

describe("hindsightMemory tools", () => {
  it("exposes reflect bound to the scope's bank", async () => {
    const fetchFn = mockFetch([], "You said tabs.");
    const tools = await hindsightMemory(OPTS).tools!(toolsCtx);
    expect(tools).not.toBeNull();
    const reflect = tools!.reflect;
    expect(reflect.description).toMatch(/long-term memory/);

    const output = await reflect.execute(
      { question: "what indentation?" } as never,
      {
        abortSignal: new AbortController().signal,
      } as never
    );

    expect(output).toEqual({ answer: "You said tabs." });
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://test/v1/default/banks/memscope1_abc/reflect");
    expect(JSON.parse(init.body)).toEqual({ query: "what indentation?", budget: "low" });
  });

  it("returns no tools when tools are off", async () => {
    expect(await hindsightMemory({ ...OPTS, tools: false }).tools!(toolsCtx)).toBeNull();
  });
});
