import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MoltbotPluginAPI, ServiceConfig } from "./types.js";

let directory: string;
let service: ServiceConfig;
let retainHook: Parameters<MoltbotPluginAPI["on"]>[1];
let sessionEndHook: Parameters<MoltbotPluginAPI["on"]>[1];
let hook: Parameters<MoltbotPluginAPI["on"]>[1];
let Client: typeof import("@vectorize-io/hindsight-client").HindsightClient;
const memory = { results: [{ id: "fixture", text: "A fixture observation", type: "observation" }] };
const ctx = {
  agentId: "main",
  sessionKey: "agent:main:telegram:direct:test-user",
  messageProvider: "telegram",
  channelId: "test-user",
  senderId: "test-user",
};
function recall() {
  return hook({ rawMessage: "What was the project decision?" }, ctx);
}

beforeEach(async () => {
  // Start with the actual never-started module state, not a stopped generation.
  vi.resetModules();
  Client = (await import("@vectorize-io/hindsight-client")).HindsightClient;
  const plugin = (await import("./index.js")).default;
  directory = mkdtempSync(join(tmpdir(), "hindsight-recall-lazy-"));
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            api_version: "0.10.0",
            features: { store_document_text: true },
          }),
          { headers: { "Content-Type": "application/json" } }
        )
    )
  );
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "http://localhost:8888",
              bankId: "test-bank",
              dynamicBankId: false,
              autoRecall: true,
              autoRetain: true,
              recallTimeoutMs: 1000,
              retainQueuePath: join(directory, "queue.jsonl"),
              logLevel: "error",
            },
          },
        },
      },
    },
    registerService: (registered) => {
      service = registered;
    },
    on: (name, handler) => {
      if (name === "before_prompt_build") hook = handler;
      if (name === "agent_end") retainHook = handler;
      if (name === "session_end") sessionEndHook = handler;
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  };
  plugin(api);
});
afterEach(async () => {
  await service.stop();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  rmSync(directory, { recursive: true, force: true });
});

it("preserves external-API lazy recall before service.start", async () => {
  vi.spyOn(Client.prototype, "recall").mockResolvedValue(memory as never);
  expect(await recall()).toEqual(
    expect.objectContaining({ prependContext: expect.stringContaining("A fixture observation") })
  );
});

it.each(["start", "stop"] as const)(
  "first service.%s cancels a lazy recall",
  async (transition) => {
    let signal: AbortSignal | undefined;
    let finish!: (value: unknown) => void;
    const response = new Promise((resolve) => {
      finish = resolve;
    });
    const send = vi
      .spyOn(Client.prototype, "recall")
      .mockImplementation((_bank, _query, options) => {
        signal = options?.signal;
        return response as ReturnType<InstanceType<typeof Client>["recall"]>;
      });
    const pending = recall();
    await vi.waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    await service[transition]();
    expect(signal?.aborted).toBe(true);
    finish(memory);
    expect(await pending).toBeUndefined();
  }
);

const transcript = {
  success: true,
  messages: [
    { role: "user", content: "The project release date is October 15." },
    { role: "assistant", content: "I will remember that release date." },
  ],
};

it("initializes retention when agent_end is the first hook and reuses the lifecycle", async () => {
  const retain = vi.spyOn(Client.prototype, "retain").mockResolvedValue({} as never);
  await retainHook(transcript, ctx);
  expect(retain).toHaveBeenCalledTimes(1);
  expect(JSON.stringify(retain.mock.calls)).toContain("The project release date is October 15.");
  const probes = vi.mocked(fetch).mock.calls.length;
  expect(probes).toBeGreaterThan(0);

  await retainHook(transcript, { ...ctx, sessionKey: `${ctx.sessionKey}-second` });
  expect(retain).toHaveBeenCalledTimes(2);
  expect(fetch).toHaveBeenCalledTimes(probes);
});

it("does not lazily initialize retention after an explicit stop", async () => {
  const retain = vi.spyOn(Client.prototype, "retain").mockResolvedValue({} as never);
  await service.stop();
  await retainHook(transcript, ctx);
  await sessionEndHook(transcript, ctx);
  expect(retain).not.toHaveBeenCalled();
  expect(fetch).not.toHaveBeenCalled();
});

it.each(["start", "stop"] as const)(
  "first service.%s cancels initialization triggered by agent_end",
  async (transition) => {
    let finish!: (response: Response) => void;
    const health = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    vi.mocked(fetch).mockReturnValueOnce(health);
    const retain = vi.spyOn(Client.prototype, "retain").mockResolvedValue({} as never);
    const pending = retainHook(transcript, ctx);
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    await service[transition]();
    finish(new Response("{}", { headers: { "Content-Type": "application/json" } }));
    await pending;
    expect(retain).not.toHaveBeenCalled();
  }
);

it.each([false, true])(
  "retains after lazy initialization (previously stopped: %s)",
  async (stopped) => {
    if (stopped) await service.stop();
    const globalClient = (
      globalThis as unknown as {
        __hindsightClient: { waitForReady(): Promise<void> };
      }
    ).__hindsightClient;
    await globalClient.waitForReady();
    const retain = vi.spyOn(Client.prototype, "retain").mockResolvedValue({} as never);
    await retainHook(transcript, ctx);
    expect(retain).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(retain.mock.calls)).toContain("The project release date is October 15.");
  }
);

it("does not publish a client or restart retention when stopped during lazy initialization", async () => {
  let finish!: (response: Response) => void;
  const health = new Promise<Response>((resolve) => {
    finish = resolve;
  });
  vi.mocked(fetch).mockReturnValueOnce(health);
  const globalClient = (
    globalThis as unknown as {
      __hindsightClient: { waitForReady(): Promise<void>; getClient(): unknown };
    }
  ).__hindsightClient;
  const pending = globalClient.waitForReady();
  await service.stop();
  finish(new Response("{}", { headers: { "Content-Type": "application/json" } }));
  await pending;
  expect(globalClient.getClient()).toBeNull();
  const retain = vi.spyOn(Client.prototype, "retain").mockResolvedValue({} as never);
  await retainHook(transcript, ctx);
  expect(retain).not.toHaveBeenCalled();
});

it.each(["recall", "retain"] as const)(
  "queues a failed retain after lazy %s and replays it on the flush timer",
  async (firstHook) => {
    vi.useFakeTimers();
    try {
      if (firstHook === "recall") {
        vi.spyOn(Client.prototype, "recall").mockResolvedValue(memory as never);
        expect(await recall()).toEqual(
          expect.objectContaining({
            prependContext: expect.stringContaining("A fixture observation"),
          })
        );
      }
      const retain = vi
        .spyOn(Client.prototype, "retain")
        .mockRejectedValueOnce(new Error("temporary transport failure"))
        .mockResolvedValue({} as never);
      await retainHook(transcript, ctx);
      expect(retain).toHaveBeenCalledTimes(1);
      expect(readFileSync(join(directory, "queue.jsonl"), "utf8")).toContain(
        "The project release date is October 15."
      );
      await vi.advanceTimersByTimeAsync(60_000);
      expect(retain).toHaveBeenCalledTimes(2);
      await service.stop();
      await vi.advanceTimersByTimeAsync(60_000);
      expect(retain).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  }
);
