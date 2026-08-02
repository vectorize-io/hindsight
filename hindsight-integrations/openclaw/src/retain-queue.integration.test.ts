import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import registerPlugin, { type AsyncRetainOperationIdCapability } from "./index.js";
import type { MoltbotPluginAPI, PluginHookAgentContext, ServiceConfig } from "./types.js";

const tempDirs: string[] = [];

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeApi(
  queuePath: string,
  flushIntervalMs: number
): {
  api: MoltbotPluginAPI;
  service: () => ServiceConfig;
  agentEnd: () => (event: unknown, ctx?: PluginHookAgentContext) => Promise<void>;
} {
  let registeredService: ServiceConfig | undefined;
  let agentEndHandler:
    ((event: unknown, ctx?: PluginHookAgentContext) => void | Promise<void>) | undefined;
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "https://hindsight.test",
              retainQueuePath: queuePath,
              retainQueueFlushIntervalMs: flushIntervalMs,
              dynamicBankId: false,
              bankId: "integration-bank",
              autoRecall: false,
              autoRetain: true,
              logLevel: "off",
            },
          },
        },
      },
    },
    registerService(config) {
      registeredService = config;
    },
    on(event, handler) {
      if (event === "agent_end") agentEndHandler = handler;
    },
    logger: {
      info: () => undefined,
      warn: () => undefined,
      error: () => undefined,
    },
  };
  registerPlugin(api);
  return {
    api,
    service: () => {
      if (!registeredService) throw new Error("service not registered");
      return registeredService;
    },
    agentEnd: () => {
      if (!agentEndHandler) throw new Error("agent_end not registered");
      return async (event, ctx) => {
        await agentEndHandler?.(event, ctx);
      };
    },
  };
}

describe("retain queue production capability transitions", () => {
  it("defers an unknown initial send, then reuses its persisted id through the timer", async () => {
    vi.useFakeTimers();
    const dir = mkdtempSync(join(tmpdir(), "hindsight-retain-integration-"));
    tempDirs.push(dir);
    const queuePath = join(dir, "retains.jsonl");
    let capability: AsyncRetainOperationIdCapability = "supported";
    let versionRequests = 0;
    let deferNextVersion = false;
    let resolveDeferredVersion: ((response: Response) => void) | undefined;
    let notifyDeferredVersionStarted: (() => void) | undefined;
    const retainBodies: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      if (request.url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (request.url.endsWith("/version")) {
        versionRequests++;
        if (deferNextVersion) {
          deferNextVersion = false;
          notifyDeferredVersionStarted?.();
          return await new Promise<Response>((resolve) => {
            resolveDeferredVersion = resolve;
          });
        }
        if (capability === "unknown") throw new Error("version probe unavailable");
        return new Response(
          JSON.stringify({
            api_version: capability === "supported" ? "0.8.6" : "0.8.5",
            features: { store_document_text: true },
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        );
      }
      if (request.method === "POST" && request.url.includes("/memories")) {
        const body = JSON.parse(await request.clone().text()) as Record<string, unknown>;
        retainBodies.push(body);
        return new Response(
          JSON.stringify({
            success: true,
            bank_id: "integration-bank",
            items_count: 1,
            async: true,
            operation_id: body.operation_id,
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          }
        );
      }
      throw new Error(`unexpected request: ${request.method} ${request.url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const first = makeApi(queuePath, 1_000);
    const firstService = first.service();
    await firstService.start();
    capability = "unknown";

    await first.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "Please remember that my favorite color is ultramarine." },
          { role: "assistant", content: "I will remember that preference." },
        ],
      },
      {
        agentId: "main",
        sessionKey: "agent:main:discord:direct:integration",
        messageProvider: "discord",
        channelId: "direct:integration",
        senderId: "user:integration",
      }
    );

    expect(retainBodies).toHaveLength(0);
    const queuedBeforeRestart = JSON.parse(readFileSync(queuePath, "utf8").trim()) as {
      operationId?: string;
    };
    expect(queuedBeforeRestart.operationId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    );
    await firstService.stop();

    capability = "supported";
    const second = makeApi(queuePath, 1_000);
    const secondService = second.service();
    await secondService.start();

    await vi.advanceTimersByTimeAsync(1_000);

    expect(retainBodies).toHaveLength(1);
    expect(retainBodies[0].operation_id).toBe(queuedBeforeRestart.operationId);

    const versionRequestsBeforeTriggeredRetain = versionRequests;
    await second.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "Please also remember that I prefer concise answers." },
          { role: "assistant", content: "I will remember that preference too." },
        ],
      },
      {
        agentId: "main",
        sessionKey: "agent:main:discord:direct:integration-2",
        messageProvider: "discord",
        channelId: "direct:integration-2",
        senderId: "user:integration",
      }
    );
    for (let i = 0; i < 10; i++) await Promise.resolve();

    expect(retainBodies).toHaveLength(2);
    expect(versionRequests - versionRequestsBeforeTriggeredRetain).toBe(2);

    capability = "unknown";
    await second.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "Remember this while the version endpoint is unavailable." },
          { role: "assistant", content: "I will defer that safely." },
        ],
      },
      {
        agentId: "main",
        sessionKey: "agent:main:discord:direct:integration-3",
        messageProvider: "discord",
        channelId: "direct:integration-3",
        senderId: "user:integration",
      }
    );
    expect(retainBodies).toHaveLength(2);

    capability = "supported";
    const deferredVersionStarted = new Promise<void>((resolve) => {
      notifyDeferredVersionStarted = resolve;
    });
    deferNextVersion = true;
    const timerAdvance = vi.advanceTimersByTimeAsync(1_000);
    await deferredVersionStarted;
    await secondService.stop();
    resolveDeferredVersion?.(
      new Response(
        JSON.stringify({
          api_version: "0.8.6",
          features: { store_document_text: true },
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );
    await timerAdvance;

    expect(retainBodies).toHaveLength(2);

    const third = makeApi(queuePath, 1_000);
    const thirdService = third.service();
    await thirdService.start();
    const globalClient = (global as any).__hindsightClient as {
      waitForReady(): Promise<void>;
    };
    const originalWaitForReady = globalClient.waitForReady;
    let releaseStaleHandler: (() => void) | undefined;
    let notifyStaleHandlerWaiting: (() => void) | undefined;
    const staleHandlerWaiting = new Promise<void>((resolve) => {
      notifyStaleHandlerWaiting = resolve;
    });
    const staleHandlerRelease = new Promise<void>((resolve) => {
      releaseStaleHandler = resolve;
    });
    globalClient.waitForReady = async () => {
      notifyStaleHandlerWaiting?.();
      await staleHandlerRelease;
    };
    const staleHandler = third.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "This stale hook must not use a restarted client." },
          { role: "assistant", content: "Acknowledged." },
        ],
      },
      {
        agentId: "main",
        sessionKey: "agent:main:discord:direct:stale",
        messageProvider: "discord",
        channelId: "direct:stale",
        senderId: "user:integration",
      }
    );
    await staleHandlerWaiting;
    await thirdService.stop();

    const fourth = makeApi(queuePath, 1_000);
    const fourthService = fourth.service();
    await fourthService.start();
    releaseStaleHandler?.();
    await staleHandler;

    expect(retainBodies).toHaveLength(2);
    globalClient.waitForReady = originalWaitForReady;
    await fourthService.stop();
  });
});
