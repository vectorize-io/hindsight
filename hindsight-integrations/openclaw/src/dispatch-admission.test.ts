import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import plugin from "./index.js";
import type {
  MoltbotPluginAPI,
  PluginConfig,
  PluginHookAgentContext,
  ServiceConfig,
} from "./types.js";

const services: ServiceConfig[] = [];
const directories: string[] = [];

afterEach(async () => {
  for (const service of services.splice(0)) await service.stop();
  vi.unstubAllGlobals();
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

async function register(config: Partial<PluginConfig> = {}) {
  const directory = mkdtempSync(join(tmpdir(), "hindsight-dispatch-admission-"));
  directories.push(directory);
  const hooks = new Map<string, Parameters<MoltbotPluginAPI["on"]>[1]>();
  const requests: Request[] = [];
  vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init);
    const path = new URL(request.url).pathname;
    let body: unknown;
    if (path === "/health") {
      body = { status: "ok" };
    } else if (path === "/version") {
      body = { api_version: "0.10.0", features: { store_document_text: true } };
    } else if (path.endsWith("/recall")) {
      requests.push(request);
      body = {
        results: [{ id: "fixture", text: "We chose the green design.", type: "observation" }],
      };
    } else if (request.method === "POST" && path.endsWith("/memories")) {
      requests.push(request);
      body = { success: true, items_count: 1, async: true };
    } else {
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    }
    return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  });
  let service!: ServiceConfig;
  plugin({
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "https://hindsight.test",
              retainQueuePath: join(directory, "queue.jsonl"),
              logLevel: "off",
              ...config,
            },
          },
        },
      },
    },
    registerService(registered) {
      service = registered;
      services.push(registered);
    },
    on(name, handler) {
      hooks.set(name, handler);
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  });
  await service.start();
  return { hooks, requests };
}

const recallEvent = { rawMessage: "What design did we choose?", messages: [] };
const retainEvent = {
  success: true,
  messages: [
    { role: "user", content: "We chose the green design." },
    { role: "assistant", content: "I will use the green design." },
  ],
};

describe("dispatch admission and identity resolution", () => {
  it.each([true, undefined])(
    "omits before_dispatch for agent-only dynamic banking (%s)",
    async (dynamicBankId) => {
      const { hooks, requests } = await register({
        dynamicBankId,
        dynamicBankGranularity: ["agent"],
      });
      expect(hooks.has("before_dispatch")).toBe(false);
      for (const agentId of ["main", "designer"]) {
        const ctx = { sessionKey: `agent:${agentId}:main` };
        expect(await hooks.get("before_prompt_build")!(recallEvent, ctx)).toEqual(
          expect.objectContaining({ prependContext: expect.stringContaining("green design") })
        );
        await hooks.get("agent_end")!(retainEvent, ctx);
        expect(requests.slice(-2).map((request) => new URL(request.url).pathname)).toEqual([
          `/v1/default/banks/${agentId}/memories/recall`,
          `/v1/default/banks/${agentId}/memories`,
        ]);
      }
    }
  );

  it.each([
    {},
    { dynamicBankGranularity: [] },
    { dynamicBankGranularity: ["agent", "user"] },
    { dynamicBankGranularity: ["agent", "channel"] },
    { dynamicBankGranularity: ["provider"] },
    { dynamicBankGranularity: ["user"] },
    { dynamicBankId: false, bankId: "static", dynamicBankGranularity: ["agent"] },
  ])("keeps dispatch identity capture for other routing configurations: %j", async (config) => {
    const { hooks } = await register(config);
    expect(hooks.has("before_dispatch")).toBe(true);
  });

  it("uses dispatch sender identity for user-scoped recall and retain", async () => {
    const { hooks, requests } = await register({ dynamicBankGranularity: ["user"] });
    const ctx = { sessionKey: "agent:main:discord:group:dispatch-sender" };
    await hooks.get("before_dispatch")!({ channel: "discord", senderId: "alice" }, ctx);
    await hooks.get("before_prompt_build")!(recallEvent, ctx);
    await hooks.get("agent_end")!(retainEvent, ctx);
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual([
      "/v1/default/banks/alice/memories/recall",
      "/v1/default/banks/alice/memories",
    ]);
  });

  it("keeps channel-scoped dispatch mismatch suppression", async () => {
    const { hooks, requests } = await register();
    const ctx = { sessionKey: "agent:main:discord:group:dispatch-mismatch", senderId: "alice" };
    await hooks.get("before_dispatch")!({ channel: "telegram" }, ctx);
    expect(await hooks.get("before_prompt_build")!(recallEvent, ctx)).toBeUndefined();
    await hooks.get("agent_end")!(retainEvent, ctx);
    expect(requests).toHaveLength(0);
  });

  it("does not fall back to an anonymous user bank without dispatch identity", async () => {
    const { hooks, requests } = await register({ dynamicBankGranularity: ["user"] });
    const ctx = { sessionKey: "agent:main:discord:group:missing-sender" };
    expect(await hooks.get("before_prompt_build")!(recallEvent, ctx)).toBeUndefined();
    await hooks.get("agent_end")!(retainEvent, ctx);
    expect(requests).toHaveLength(0);
  });

  it.each<PluginHookAgentContext>([
    { sessionKey: "agent:main:dashboard:missing-provider" },
    { sessionKey: "agent:main:cron:operational" },
    { sessionKey: "temp:ephemeral", agentId: "main", messageProvider: "webchat" },
    { sessionKey: "agent:main:telegram:direct:alice", senderId: "bob" },
  ])("keeps recall and retain identity checks without before_dispatch: %j", async (ctx) => {
    const { hooks, requests } = await register({ dynamicBankGranularity: ["agent"] });
    expect(await hooks.get("before_prompt_build")!(recallEvent, ctx)).toBeUndefined();
    await hooks.get("agent_end")!(retainEvent, ctx);
    expect(requests).toHaveLength(0);
  });
});
