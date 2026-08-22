import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
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
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

interface FakeServer {
  /** Bodies of every retain POST that reached the server. */
  retainBodies: Array<Record<string, unknown>>;
  /** How many `/version` probes have been answered (or refused). */
  versionRequests: () => number;
  /** "unknown" makes the probe throw, mimicking an unreachable /version. */
  setCapability: (next: AsyncRetainOperationIdCapability) => void;
  /** Make the next N retain POSTs fail after the server has seen the body. */
  failRetains: (count: number) => void;
  /** Hold the next probe open; resolves once the handler has been entered. */
  deferNextVersion: () => Promise<void>;
  releaseDeferredVersion: () => void;
}

function installFakeServer(initial: AsyncRetainOperationIdCapability): FakeServer {
  let capability = initial;
  let versionRequests = 0;
  let retainFailures = 0;
  let deferNext = false;
  let resolveDeferred: ((response: Response) => void) | undefined;
  let notifyDeferredStarted: (() => void) | undefined;
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
      if (deferNext) {
        deferNext = false;
        notifyDeferredStarted?.();
        return await new Promise<Response>((resolve) => {
          resolveDeferred = resolve;
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
      // Record the body first: a lost acknowledgement is a request the server
      // *did* process, which is the case operation_id has to cover.
      if (retainFailures > 0) {
        retainFailures--;
        throw new Error("connection reset before acknowledgement");
      }
      return new Response(
        JSON.stringify({
          success: true,
          bank_id: "integration-bank",
          items_count: 1,
          async: true,
          operation_id: body.operation_id,
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    throw new Error(`unexpected request: ${request.method} ${request.url}`);
  });
  vi.stubGlobal("fetch", fetchMock);

  return {
    retainBodies,
    versionRequests: () => versionRequests,
    setCapability: (next) => {
      capability = next;
    },
    failRetains: (count) => {
      retainFailures = count;
    },
    deferNextVersion: () => {
      deferNext = true;
      return new Promise<void>((resolve) => {
        notifyDeferredStarted = resolve;
      });
    },
    releaseDeferredVersion: () => {
      resolveDeferred?.(
        new Response(
          JSON.stringify({
            api_version: "0.8.6",
            features: { store_document_text: true },
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        )
      );
    },
  };
}

function conversation(text: string, session: string) {
  return {
    event: {
      success: true,
      messages: [
        { role: "user", content: text },
        { role: "assistant", content: "Noted." },
      ],
    },
    ctx: {
      agentId: "main",
      sessionKey: `agent:main:discord:direct:${session}`,
      messageProvider: "discord",
      channelId: `direct:${session}`,
      senderId: "user:integration",
    } as PluginHookAgentContext,
  };
}

function makeQueuePath(): string {
  const dir = mkdtempSync(join(tmpdir(), "hindsight-retain-integration-"));
  tempDirs.push(dir);
  return join(dir, "retains.jsonl");
}

function readQueue(queuePath: string): Array<{ operationId?: string }> {
  // A fully drained queue removes its file, so "missing" and "empty" are the
  // same observation here.
  if (!existsSync(queuePath)) return [];
  const raw = readFileSync(queuePath, "utf8").trim();
  if (!raw) return [];
  return raw.split("\n").map((line) => JSON.parse(line) as { operationId?: string });
}

describe("retain queue idempotent replay", () => {
  it("replays a lost acknowledgement under the id the first attempt already carried", async () => {
    vi.useFakeTimers();
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");

    const first = makeApi(queuePath, 1_000);
    const firstService = first.service();
    await firstService.start();

    server.failRetains(1);
    const { event, ctx } = conversation("My favourite colour is ultramarine.", "integration");
    await first.agentEnd()(event, ctx);

    // The server saw the request and processed it; only the acknowledgement was
    // lost. Without an operation id the replay below would store it a second time.
    expect(server.retainBodies).toHaveLength(1);
    const sentId = server.retainBodies[0].operation_id;
    expect(sentId).toMatch(UUID_RE);

    const queued = readQueue(queuePath);
    expect(queued).toHaveLength(1);
    expect(queued[0].operationId).toBe(sentId);
    await firstService.stop();

    // A restart must not mint a new id for work the server may already hold.
    const second = makeApi(queuePath, 1_000);
    const secondService = second.service();
    await secondService.start();
    await vi.advanceTimersByTimeAsync(1_000);

    expect(server.retainBodies).toHaveLength(2);
    expect(server.retainBodies[1].operation_id).toBe(sentId);
    expect(readQueue(queuePath)).toHaveLength(0);
    await secondService.stop();
  });

  it("still sends the first attempt when /version is unreachable, then holds the replay until it answers", async () => {
    vi.useFakeTimers();
    const queuePath = makeQueuePath();
    const server = installFakeServer("unknown");

    const api = makeApi(queuePath, 1_000);
    const service = api.service();
    await service.start();

    server.failRetains(1);
    const { event, ctx } = conversation("Remember this while /version is down.", "integration");
    await api.agentEnd()(event, ctx);

    // An unknown capability must not cost the user their turn: nothing is stored
    // server-side yet, so the first attempt goes out — just without the field.
    expect(server.retainBodies).toHaveLength(1);
    expect(server.retainBodies[0].operation_id).toBeUndefined();
    const queued = readQueue(queuePath);
    expect(queued).toHaveLength(1);
    expect(queued[0].operationId).toMatch(UUID_RE);

    // The replay is the half that can duplicate, so it waits for a real answer.
    await vi.advanceTimersByTimeAsync(1_000);
    expect(server.retainBodies).toHaveLength(1);
    expect(readQueue(queuePath)).toHaveLength(1);

    server.setCapability("supported");
    await vi.advanceTimersByTimeAsync(1_000);
    expect(server.retainBodies).toHaveLength(2);
    expect(server.retainBodies[1].operation_id).toBe(queued[0].operationId);
    await service.stop();
  });

  it("stops probing /version once the capability is known and the queue is empty", async () => {
    vi.useFakeTimers();
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");

    const api = makeApi(queuePath, 1_000);
    const service = api.service();
    await service.start();

    const first = conversation("First turn.", "probe-1");
    await api.agentEnd()(first.event, first.ctx);
    const afterFirstRetain = server.versionRequests();
    expect(afterFirstRetain).toBeGreaterThan(0); // resolved "unknown" once

    const second = conversation("Second turn.", "probe-2");
    await api.agentEnd()(second.event, second.ctx);
    // Cached capability, empty queue: a turn must not cost an extra round trip.
    expect(server.versionRequests()).toBe(afterFirstRetain);

    await vi.advanceTimersByTimeAsync(5_000);
    expect(server.versionRequests()).toBe(afterFirstRetain);
    expect(server.retainBodies).toHaveLength(2);
    await service.stop();
  });

  it("does not replay against a client that a restart has already replaced", async () => {
    vi.useFakeTimers();
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");

    const first = makeApi(queuePath, 1_000);
    const firstService = first.service();
    await firstService.start();
    server.failRetains(1);
    const { event, ctx } = conversation("Queued before the restart.", "stale");
    await first.agentEnd()(event, ctx);
    expect(readQueue(queuePath)).toHaveLength(1);

    // Stop the service while its flush is blocked inside the capability probe.
    const probeEntered = server.deferNextVersion();
    const timerAdvance = vi.advanceTimersByTimeAsync(1_000);
    await probeEntered;
    await firstService.stop();
    server.releaseDeferredVersion();
    await timerAdvance;

    // The stopped generation must not resume against a restarted client.
    expect(server.retainBodies).toHaveLength(1);
    expect(readQueue(queuePath)).toHaveLength(1);
  });
});
