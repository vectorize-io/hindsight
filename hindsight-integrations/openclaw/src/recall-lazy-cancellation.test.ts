import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MoltbotPluginAPI, ServiceConfig } from "./types.js";

let directory: string;
let service: ServiceConfig;
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
      if (name === "before_prompt_build") hook = handler;
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
