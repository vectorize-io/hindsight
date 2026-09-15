import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import plugin, { scopeClient } from "./index.js";
import type { MoltbotPluginAPI, PluginHookAgentContext, ServiceConfig } from "./types.js";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
const memory = {
  results: [{ id: "fixture", text: "A fixture observation", type: "observation" as const }],
};

describe("bank-scoped recall cancellation", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  it("aborts the transport at the deadline even if it ignores cancellation", async () => {
    const recall = vi.fn(
      (_bank: string, _query: string, _options?: { signal?: AbortSignal }) =>
        new Promise<never>(() => {})
    );
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    const pending = scoped.recall({ query: "question" }, 1000).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(1000);
    expect(await pending).toMatchObject({ name: "TimeoutError" });
    expect(recall.mock.calls[0]?.[2]?.signal?.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("clears the deadline after success", async () => {
    const recall = vi.fn().mockResolvedValue(memory);
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    expect(await scoped.recall({ query: "question" }, 1000)).toEqual(memory);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("does not send after service cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    const recall = vi.fn().mockResolvedValue(memory);
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    await expect(scoped.recall({ query: "question" }, 1000, controller.signal)).rejects.toBe(
      controller.signal.reason
    );
    expect(recall).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("automatic recall service lifecycle", () => {
  let service: ServiceConfig;
  let hooks: Map<string, Parameters<MoltbotPluginAPI["on"]>[1]>;
  let directory: string;
  const ctx: PluginHookAgentContext = {
    agentId: "main",
    sessionKey: "agent:main:telegram:direct:test-user",
    messageProvider: "telegram",
    channelId: "test-user",
    senderId: "test-user",
  };
  const event = { rawMessage: "What was the project decision?", messages: [] };
  beforeEach(async () => {
    directory = mkdtempSync(join(tmpdir(), "hindsight-recall-cancellation-"));
    hooks = new Map();
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
                autoRetain: false,
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
        hooks.set(name, handler);
      },
      logger: { info: () => {}, warn: () => {}, error: () => {} },
    };
    plugin(api);
    await service.start();
  });
  afterEach(async () => {
    await service.stop();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    rmSync(directory, { recursive: true, force: true });
  });
  function run() {
    return hooks.get("before_prompt_build")!(event, ctx);
  }
  it("shares a recall within the running service", async () => {
    const response = deferred<typeof memory>();
    const started = deferred<void>();
    const recall = vi.spyOn(HindsightClient.prototype, "recall").mockImplementation(() => {
      started.resolve();
      return response.promise as ReturnType<HindsightClient["recall"]>;
    });
    const first = run();
    await started.promise;
    const second = run();
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(recall).toHaveBeenCalledTimes(1);
    response.resolve(memory);
    expect(await first).toEqual(
      expect.objectContaining({ prependContext: expect.stringContaining("A fixture observation") })
    );
    expect(await second).toEqual(await first);
  });
  it("suppresses a late transport success after stop", async () => {
    const response = deferred<typeof memory>();
    const started = deferred<void>();
    let signal: AbortSignal | undefined;
    vi.spyOn(HindsightClient.prototype, "recall").mockImplementation((_bank, _query, options) => {
      signal = options?.signal;
      started.resolve();
      return response.promise as ReturnType<HindsightClient["recall"]>;
    });
    const pending = run();
    await started.promise;
    await service.stop();
    response.resolve(memory);
    expect(await pending).toBeUndefined();
    expect(signal?.aborted).toBe(true);
  });
  it.each([false, true])(
    "does not reuse or evict a successor recall (stop first: %s)",
    async (stopFirst) => {
      const old = deferred<typeof memory>();
      const fresh = deferred<typeof memory>();
      const started = deferred<void>();
      const recall = vi
        .spyOn(HindsightClient.prototype, "recall")
        .mockImplementationOnce(() => {
          started.resolve();
          return old.promise as ReturnType<HindsightClient["recall"]>;
        })
        .mockImplementation(() => fresh.promise as ReturnType<HindsightClient["recall"]>);
      const previous = run();
      await started.promise;
      if (stopFirst) await service.stop();
      await service.start();
      const successor = run();
      await new Promise<void>((resolve) => setImmediate(resolve));
      old.resolve(memory);
      expect(await previous).toBeUndefined();
      const duplicate = run();
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(recall).toHaveBeenCalledTimes(2);
      fresh.resolve(memory);
      expect(await successor).toEqual(
        expect.objectContaining({ prependContext: expect.any(String) })
      );
      expect(await duplicate).toEqual(await successor);
    }
  );
  it("does not send if stopped while waiting for initialization", async () => {
    const ready = deferred<void>();
    const globalClient = (
      globalThis as unknown as { __hindsightClient: { waitForReady(): Promise<void> } }
    ).__hindsightClient;
    vi.spyOn(globalClient, "waitForReady").mockReturnValue(ready.promise);
    const recall = vi.spyOn(HindsightClient.prototype, "recall").mockResolvedValue(memory as never);
    const pending = run();
    await service.stop();
    ready.resolve();
    expect(await pending).toBeUndefined();
    expect(recall).not.toHaveBeenCalled();
  });
});
