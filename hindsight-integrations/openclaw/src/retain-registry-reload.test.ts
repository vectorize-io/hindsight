import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MoltbotPluginAPI, ServiceConfig } from "./types.js";

// OpenClaw re-evaluates the plugin module for each plugin registry it loads, but
// only calls service.start() on the first. The hooks that run belong to the newest
// copy, which was never started. Retain must work there just like recall. (#4828)

let directory: string;
const ctx = {
  agentId: "main",
  sessionKey: "agent:main:telegram:direct:test-user",
  messageProvider: "telegram",
  channelId: "test-user",
  senderId: "test-user",
};
const turn = {
  messages: [
    { role: "user", content: "My favourite fruit is the persimmon." },
    { role: "assistant", content: "Noted, persimmons it is." },
  ],
};

interface PluginCopy {
  service: ServiceConfig;
  agentEnd: Parameters<MoltbotPluginAPI["on"]>[1];
  retain: ReturnType<typeof vi.fn>;
}

/** Evaluate a fresh copy of the plugin module, as a new OpenClaw registry does. */
async function loadCopy(): Promise<PluginCopy> {
  vi.resetModules();
  const Client = (await import("@vectorize-io/hindsight-client")).HindsightClient;
  const retain = vi.fn().mockResolvedValue({ success: true });
  vi.spyOn(Client.prototype, "retain").mockImplementation(retain);
  const plugin = (await import("./index.js")).default;
  const copy = {} as PluginCopy;
  copy.retain = retain;
  plugin({
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "http://localhost:8888",
              dynamicBankGranularity: ["agent"],
              autoRecall: false,
              autoRetain: true,
              retainQueuePath: join(directory, "queue.jsonl"),
              logLevel: "error",
            },
          },
        },
      },
    },
    registerService: (registered) => {
      copy.service = registered;
    },
    on: (name, handler) => {
      if (name === "agent_end") copy.agentEnd = handler;
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  } as MoltbotPluginAPI);
  return copy;
}

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), "hindsight-registry-reload-"));
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ api_version: "0.10.0", features: {} }), {
          headers: { "Content-Type": "application/json" },
        })
    )
  );
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  rmSync(directory, { recursive: true, force: true });
});

it("retains through a reloaded copy that was never started", async () => {
  const started = await loadCopy();
  await started.service.start();
  const reloaded = await loadCopy();

  await reloaded.agentEnd(turn, ctx);

  await vi.waitFor(() => expect(reloaded.retain).toHaveBeenCalledTimes(1));
  expect(reloaded.retain.mock.calls[0][1]).toContain("persimmon");
  await started.service.stop();
});

it("queues a failed retain from a never-started copy instead of dropping it", async () => {
  const reloaded = await loadCopy();
  reloaded.retain.mockRejectedValue(new Error("connect ECONNREFUSED"));

  await reloaded.agentEnd(turn, ctx);

  await vi.waitFor(() => expect(reloaded.retain).toHaveBeenCalledTimes(1));
  const queue = join(directory, "queue.jsonl");
  await vi.waitFor(() =>
    expect(existsSync(queue) && readFileSync(queue, "utf-8")).toContain("persimmon")
  );
});

it("does not retain after the copy is explicitly stopped", async () => {
  const copy = await loadCopy();
  await copy.service.start();
  await copy.service.stop();

  await copy.agentEnd(turn, ctx);

  expect(copy.retain).not.toHaveBeenCalled();
});
