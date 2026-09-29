import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
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
  flushIntervalMs: number,
  extraConfig: Record<string, unknown> = {}
): {
  api: MoltbotPluginAPI;
  service: () => ServiceConfig;
  agentEnd: () => (event: unknown, ctx?: PluginHookAgentContext) => Promise<void>;
  sessionEnd: () => (event: unknown, ctx?: PluginHookAgentContext) => Promise<void>;
} {
  let registeredService: ServiceConfig | undefined;
  let agentEndHandler:
    | ((event: unknown, ctx?: PluginHookAgentContext) => void | Promise<void>)
    | undefined;
  let sessionEndHandler:
    | ((event: unknown, ctx?: PluginHookAgentContext) => void | Promise<void>)
    | undefined;
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
              ...extraConfig,
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
      if (event === "session_end") sessionEndHandler = handler;
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
    sessionEnd: () => {
      if (!sessionEndHandler) throw new Error("session_end not registered");
      return async (event, ctx) => {
        await sessionEndHandler?.(event, ctx);
      };
    },
  };
}
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

interface FakeServer {
  /** Bodies of every retain POST that reached the server. */
  retainBodies: Array<Record<string, unknown>>;
  /** URLs of those same POSTs — the bank id is in the path, so this is the routing. */
  retainUrls: string[];
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
  const retainUrls: string[] = [];

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
      retainUrls.push(request.url);
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
    retainUrls,
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

describe("session_end flushes the un-retained tail (#4341)", () => {
  // The helper that reads the transcript has its own unit tests; this one pins the
  // wiring, which is where #1726's flush died: the hook fired, the guard above it
  // saw a payload with no `messages`, and the tail was dropped in silence. Only an
  // end-to-end retain proves the forced flush now reaches the server.
  function writeTranscript(sessionId: string, turns: Array<[string, string]>): string {
    const dir = mkdtempSync(join(tmpdir(), "hindsight-session-end-"));
    tempDirs.push(dir);
    const file = join(dir, `${sessionId}.jsonl`);
    const lines: unknown[] = [
      { type: "session", id: sessionId, timestamp: "2026-09-12T20:00:00Z" },
      ...turns.flatMap(([user, assistant]) => [
        { type: "message", message: { role: "user", content: user } },
        { type: "message", message: { role: "assistant", content: assistant } },
      ]),
    ];
    writeFileSync(file, lines.map((line) => JSON.stringify(line)).join("\n") + "\n", "utf8");
    return file;
  }

  it("retains the turns after the last cadence boundary when the session closes", async () => {
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");

    // retainEveryNTurns: 3 — two turns sit below the cadence boundary, so nothing
    // has been retained when the session ends.
    const api = makeApi(queuePath, 1_000, { retainEveryNTurns: 3, retainOverlapTurns: 1 });
    const service = api.service();
    await service.start();

    const sessionKey = "agent:main:discord:direct:session-end";
    const ctx = {
      agentId: "main",
      sessionKey,
      messageProvider: "discord",
      channelId: "direct:session-end",
      senderId: "user:integration",
    } as PluginHookAgentContext;

    await api.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "First turn." },
          { role: "assistant", content: "Noted." },
        ],
      },
      ctx
    );
    await api.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "The tail nobody retained." },
          { role: "assistant", content: "Understood." },
        ],
      },
      ctx
    );
    expect(server.retainBodies).toHaveLength(0);

    // The real payload: ids, counts and a sessionFile — no messages array, and a
    // context holding only ids (OpenClaw's buildSessionEndHookPayload()).
    const sessionFile = writeTranscript("sess-1", [
      ["First turn.", "Noted."],
      ["The tail nobody retained.", "Understood."],
    ]);
    await api.sessionEnd()(
      {
        sessionId: "sess-1",
        sessionKey,
        messageCount: 4,
        durationMs: 12_000,
        reason: "reset",
        sessionFile,
        context: { sessionId: "sess-1", sessionKey, agentId: "main" },
      },
      ctx
    );

    expect(server.retainBodies).toHaveLength(1);
    expect(JSON.stringify(server.retainBodies[0])).toContain("The tail nobody retained.");
    await service.stop();
  });

  it("skips the flush when the event points at no readable transcript", async () => {
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");

    const api = makeApi(queuePath, 1_000, { retainEveryNTurns: 3, retainOverlapTurns: 1 });
    const service = api.service();
    await service.start();

    const sessionKey = "agent:main:discord:direct:no-transcript";
    const ctx = {
      agentId: "main",
      sessionKey,
      messageProvider: "discord",
      channelId: "direct:no-transcript",
      senderId: "user:integration",
    } as PluginHookAgentContext;

    await api.agentEnd()(
      {
        success: true,
        messages: [
          { role: "user", content: "Only turn." },
          { role: "assistant", content: "Noted." },
        ],
      },
      ctx
    );

    await api.sessionEnd()(
      {
        sessionId: "sess-2",
        sessionKey,
        messageCount: 2,
        reason: "shutdown",
        sessionFile: join(tmpdir(), "hindsight-missing", "sess-2.jsonl"),
        context: { sessionId: "sess-2", sessionKey, agentId: "main" },
      },
      ctx
    );

    expect(server.retainBodies).toHaveLength(0);
    await service.stop();
  });
});

// The bank id lives in the retain URL, so these pin the routing itself rather than
// deriveBankId's return value — what would break is a call site reaching the client
// without consulting the map. (#3890)
describe("agentBankMap routes retains to the mapped bank", () => {
  const ctxFor = (agentId: string, session: string) =>
    ({
      agentId,
      sessionKey: `agent:${agentId}:discord:direct:${session}`,
      messageProvider: "discord",
      channelId: `direct:${session}`,
      senderId: `user-${session}`,
    }) as PluginHookAgentContext;

  const oneTurn = (text: string) => ({
    success: true,
    messages: [
      { role: "user", content: text },
      { role: "assistant", content: "Noted." },
    ],
  });

  it("sends mapped agents to one shared bank and leaves an unmapped one derived", async () => {
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");
    const api = makeApi(queuePath, 1_000, {
      dynamicBankId: true,
      bankId: undefined,
      dynamicBankGranularity: ["agent", "channel", "user"],
      agentBankMap: { inbound: "ps-technology", outbound: "ps-technology" },
    });
    const service = api.service();
    await service.start();

    await api.agentEnd()(oneTurn("Postgres 16 in production."), ctxFor("inbound", "s1"));
    await api.agentEnd()(oneTurn("Campaigns ship on Tuesday."), ctxFor("outbound", "s2"));
    await api.agentEnd()(oneTurn("This agent is not in the map."), ctxFor("stranger", "s3"));

    expect(server.retainUrls).toHaveLength(3);
    // Two different agents, one named bank.
    expect(server.retainUrls[0]).toContain("/banks/ps-technology/");
    expect(server.retainUrls[1]).toContain("/banks/ps-technology/");
    // The unmapped agent keeps the bank it would have derived anyway.
    expect(server.retainUrls[2]).toContain("stranger");
    expect(server.retainUrls[2]).not.toContain("ps-technology");

    await service.stop();
  });

  it("overrides a static bank for mapped agents only", async () => {
    const queuePath = makeQueuePath();
    const server = installFakeServer("supported");
    // The harness config is dynamicBankId:false — without the map every agent
    // would share `integration-bank`.
    const api = makeApi(queuePath, 1_000, {
      agentBankMap: { limpieza: "ps-limpieza" },
    });
    const service = api.service();
    await service.start();

    await api.agentEnd()(oneTurn("The crew starts at 06:00."), ctxFor("limpieza", "s1"));
    await api.agentEnd()(oneTurn("Anything else."), ctxFor("other", "s2"));

    expect(server.retainUrls).toHaveLength(2);
    expect(server.retainUrls[0]).toContain("/banks/ps-limpieza/");
    expect(server.retainUrls[1]).toContain("/banks/integration-bank/");

    await service.stop();
  });
});
