import { afterEach, describe, expect, it, vi } from "vitest";
import { hindsightAutoRecall, hindsightRetainHook } from "./auto-memory";
import { SENTINEL_OPEN } from "./client";

const OPTS = { apiUrl: "http://test", apiKey: "k", bankId: "b" };
const CTX = { session: { id: "s1" }, channel: { kind: "web" } } as unknown;

/** Mock fetch, routing by URL; returns recall results or a retain ack. */
function mockFetch(recallResults: unknown[] = []): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async (url: string) => ({
    ok: true,
    status: 200,
    json: async () => (url.includes("/recall") ? { results: recallResults } : { success: true }),
    text: async () => "",
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const handlers = (def: { events: unknown }): any => def.events;

afterEach(() => vi.unstubAllGlobals());

describe("hindsightRetainHook", () => {
  it("retains both the user message and assistant reply on turn.completed (both by default)", async () => {
    const fetchFn = mockFetch();
    const ev = handlers(hindsightRetainHook(OPTS));

    ev["message.received"]({ data: { turnId: "t1", message: "I prefer tabs" } });
    ev["message.completed"]({ data: { turnId: "t1", message: "Got it.", finishReason: "stop" } });
    await ev["turn.completed"]({ data: { turnId: "t1" } }, CTX);

    expect(fetchFn).toHaveBeenCalledTimes(1);
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://test/v1/default/banks/b/memories");
    const body = JSON.parse(init.body);
    expect(body.async).toBe(true);
    expect(body.items[0].content).toBe("User: I prefer tabs\n\nAssistant: Got it.");
    expect(body.items[0].context).toBe("eve");
    expect(body.items[0].metadata).toMatchObject({ sessionId: "s1", turnId: "t1", channel: "web" });
  });

  it("stores only the user message when includeAssistantReply is false", async () => {
    const fetchFn = mockFetch();
    const ev = handlers(hindsightRetainHook({ ...OPTS, includeAssistantReply: false }));
    ev["message.received"]({ data: { turnId: "t1", message: "I prefer tabs" } });
    ev["message.completed"]({ data: { turnId: "t1", message: "Got it.", finishReason: "stop" } });
    await ev["turn.completed"]({ data: { turnId: "t1" } }, CTX);
    expect(JSON.parse(fetchFn.mock.calls[0][1].body).items[0].content).toBe("User: I prefer tabs");
  });

  it("ignores non-terminal assistant steps (finishReason !== 'stop')", async () => {
    const fetchFn = mockFetch();
    const ev = handlers(hindsightRetainHook(OPTS));
    ev["message.received"]({ data: { turnId: "t1", message: "hi" } });
    ev["message.completed"]({
      data: { turnId: "t1", message: "calling tool", finishReason: "tool-calls" },
    });
    await ev["turn.completed"]({ data: { turnId: "t1" } }, CTX);
    // still retains (user text present), but content has no assistant half
    expect(JSON.parse(fetchFn.mock.calls[0][1].body).items[0].content).toBe("User: hi");
  });

  it("does not retain a turn with no user message", async () => {
    const fetchFn = mockFetch();
    const ev = handlers(hindsightRetainHook(OPTS));
    ev["message.completed"]({ data: { turnId: "t1", message: "orphan", finishReason: "stop" } });
    await ev["turn.completed"]({ data: { turnId: "t1" } }, CTX);
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("never throws on a retain failure (degrades via onError)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        json: async () => ({}),
        text: async () => "boom",
      }))
    );
    const onError = vi.fn();
    const ev = handlers(hindsightRetainHook({ ...OPTS, onError }));
    ev["message.received"]({ data: { turnId: "t1", message: "x" } });
    await expect(ev["turn.completed"]({ data: { turnId: "t1" } }, CTX)).resolves.toBeUndefined();
    expect(onError).toHaveBeenCalledWith(expect.anything(), "retain");
  });
});

describe("hindsightAutoRecall", () => {
  it("recalls and returns injected instructions containing the memories", async () => {
    const fetchFn = mockFetch([{ id: "1", text: "prefers Python" }]);
    const ev = handlers(hindsightAutoRecall(OPTS));
    const result = await ev["turn.started"]({ data: { turnId: "t1" } }, CTX);

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://test/v1/default/banks/b/memories/recall");
    expect(JSON.parse(init.body).query).toBe("user preferences, identity, and working context");
    expect(result.markdown).toContain(SENTINEL_OPEN);
    expect(result.markdown).toContain("- prefers Python");
  });

  it("returns undefined when there is nothing to recall", async () => {
    mockFetch([]);
    const ev = handlers(hindsightAutoRecall(OPTS));
    expect(await ev["turn.started"]({ data: { turnId: "t1" } }, CTX)).toBeUndefined();
  });

  it("returns undefined and reports onError on a recall failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        json: async () => ({}),
        text: async () => "boom",
      }))
    );
    const onError = vi.fn();
    const ev = handlers(hindsightAutoRecall({ ...OPTS, onError }));
    expect(await ev["turn.started"]({ data: { turnId: "t1" } }, CTX)).toBeUndefined();
    expect(onError).toHaveBeenCalledWith(expect.anything(), "recall");
  });
});
