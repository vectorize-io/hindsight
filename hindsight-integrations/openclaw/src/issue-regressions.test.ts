import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import registerPlugin, { isActiveMemoryChildSession, resolveAndCacheIdentity } from "./index.js";
import type {
  MoltbotPluginAPI,
  PluginHookAgentContext,
  PluginTool,
  PluginToolContext,
  ServiceConfig,
} from "./types.js";

// Regressions for the OpenClaw issue sweep: #6073, #6196, #6201, #6303, #6515, #6572.

const dirs: string[] = [];
let service: ServiceConfig | undefined;

afterEach(async () => {
  await service?.stop();
  service = undefined;
  vi.unstubAllGlobals();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

type Handler = (event: unknown, ctx?: PluginHookAgentContext) => unknown;
interface Seen {
  method: string;
  url: string;
  auth: string | null;
  body: any;
}

async function loadPlugin(extraConfig: Record<string, unknown> = {}) {
  const dir = mkdtempSync(join(tmpdir(), "hindsight-issues-"));
  dirs.push(dir);
  const seen: Seen[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const text = request.method === "GET" ? "" : await request.text();
      seen.push({
        method: request.method,
        url: request.url,
        auth: request.headers.get("authorization"),
        body: text ? JSON.parse(text) : undefined,
      });
      if (request.url.endsWith("/health")) return Response.json({ status: "healthy" });
      if (request.url.endsWith("/version")) {
        return Response.json({ api_version: "1.3.0", features: { store_document_text: true } });
      }
      if (request.url.includes("/memories/recall")) return Response.json({ results: [] });
      if (request.method === "POST" && request.url.includes("/memories")) {
        return Response.json({ success: true, bank_id: "b", items_count: 1, async: true });
      }
      return Response.json({});
    })
  );
  const hooks = new Map<string, Handler>();
  const tools: Array<(ctx: PluginToolContext) => PluginTool[] | null> = [];
  const capabilities: Array<{ recallToolNames?: readonly string[] }> = [];
  const queuePath = join(dir, "queue.jsonl");
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "https://hindsight.test",
              retainQueuePath: queuePath,
              dynamicBankId: false,
              bankId: "b",
              autoRecall: false,
              logLevel: "off",
              ...extraConfig,
            },
          },
        },
      },
    },
    registerService: (registered) => {
      service = registered;
    },
    on: (name, handler) => {
      hooks.set(name, handler as Handler);
    },
    registerTool: (factory) => {
      tools.push(factory);
    },
    registerMemoryCapability: (capability) => {
      capabilities.push(capability);
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  };
  registerPlugin(api);
  // Rejects when the API can't be reached at start; the hooks recover lazily.
  await service!.start().catch(() => {});
  return { hooks, tools, capabilities, seen, queuePath };
}

// Run state is per process, keyed by session: each test gets its own session.
let sessionSeq = 0;
function newCtx(): PluginHookAgentContext {
  sessionSeq += 1;
  return {
    agentId: "main",
    sessionKey: `agent:main:discord:direct:u${sessionSeq}`,
    messageProvider: "discord",
    channelId: `direct:u${sessionSeq}`,
    senderId: `u${sessionSeq}`,
  } as PluginHookAgentContext;
}

describe("OpenClaw issue regressions", () => {
  it("names its own recall tools for Active Memory deep recall (#6303)", async () => {
    const { capabilities } = await loadPlugin();
    expect(capabilities[0]?.recallToolNames).toEqual(["hindsight_recall"]);
  });

  it("recalls on currentUserMessage, not a prompt that leads with long context (#6515)", async () => {
    const { hooks, seen } = await loadPlugin({ autoRecall: true });
    const ctx = newCtx();
    const question = "Which cuisines did I enjoy in Lisbon?";
    await hooks.get("before_prompt_build")!(
      {
        prompt: `${"Background context line. ".repeat(200)}\n${question}`,
        currentUserMessage: question,
        messages: [],
      },
      ctx
    );
    const recall = seen.find((r) => r.url.includes("/memories/recall"));
    expect(recall?.body.query).toContain(question);
  });

  it("resolves the tools' bank in a Control UI session from the run's channel (#6572)", async () => {
    const { tools, seen } = await loadPlugin({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent"],
      agentBankMap: { main: "main-bank" },
    });
    const recallTool = tools
      .flatMap(
        (factory) =>
          factory({
            agentId: "main",
            sessionKey: "agent:main:dashboard:3f9a1c2e-0000-0000-0000-000000000000",
            messageChannel: "webchat",
          }) ?? []
      )
      .find((tool) => tool.name === "hindsight_recall")!;
    const result = await recallTool.execute("t1", { query: "what do I like?" });
    expect(JSON.stringify(result)).not.toContain("missing stable message provider");
    expect(seen.some((r) => r.url.includes("/banks/main-bank/memories/recall"))).toBe(true);
  });

  it("saves a run whose user message the host left out of agent_end (#6073)", async () => {
    const { hooks, seen } = await loadPlugin();
    const ctx = newCtx();
    await hooks.get("before_prompt_build")!(
      { prompt: "Plan my trip", currentUserMessage: "Plan my trip to Porto", messages: [] },
      ctx
    );
    // Codex's suppressNextUserMessagePersistence: the snapshot has only the answer.
    await hooks.get("agent_end")!(
      { success: true, messages: [{ role: "assistant", content: "Day one: Ribeira." }] },
      ctx
    );
    const retain = seen.find((r) => r.method === "POST" && r.url.endsWith("/memories"));
    const content = retain?.body.items[0].content as string;
    expect(content).toContain("Plan my trip to Porto");
    expect(content).toContain("Day one: Ribeira.");
  });

  it("never sends an unresolved SecretRef token, and queues the turn instead (#6201, #6196)", async () => {
    const { hooks, seen, queuePath } = await loadPlugin({
      hindsightApiToken: { source: "env", provider: "default", id: "HINDSIGHT_TOKEN" },
    });
    const ctx = newCtx();
    await hooks.get("agent_end")!(
      {
        success: true,
        messages: [
          { role: "user", content: "Remember that I prefer window seats" },
          { role: "assistant", content: "Noted: window seats." },
        ],
      },
      ctx
    );
    expect(seen.some((r) => r.auth?.includes("[object Object]"))).toBe(false);
    expect(seen.some((r) => r.method === "POST" && r.url.endsWith("/memories"))).toBe(false);
    expect(readFileSync(queuePath, "utf8")).toContain("I prefer window seats");
  });

  it("needs no message provider when the provider does not pick the bank", () => {
    // `openclaw agent --session-key agent:main:x` and Control UI runs carry no provider.
    const run = (pluginConfig: Record<string, unknown>) =>
      resolveAndCacheIdentity({
        sessionKey: `agent:main:cli-${(sessionSeq += 1)}`,
        ctx: { agentId: "main" } as PluginHookAgentContext,
        pluginConfig,
      }).skipReason?.detail;
    expect(run({ dynamicBankId: true, dynamicBankGranularity: ["agent"] })).toBeUndefined();
    expect(run({ dynamicBankId: false, bankId: "b" })).toBeUndefined();
    // Banks keyed by user or channel still need it.
    expect(run({ dynamicBankId: true })).toBe("missing stable message provider");
  });

  it("neither recalls into nor saves Active Memory's child turn (#6303)", async () => {
    const { hooks, seen } = await loadPlugin({ autoRecall: true });
    const parent = newCtx();
    const child = { ...parent, sessionKey: `${parent.sessionKey}:active-memory:9b3e087b087e` };
    expect(isActiveMemoryChildSession(child.sessionKey!)).toBe(true);
    expect(isActiveMemoryChildSession(parent.sessionKey!)).toBe(false);
    await hooks.get("before_prompt_build")!(
      { prompt: "You are a memory search agent.", currentUserMessage: "sister?", messages: [] },
      child
    );
    await hooks.get("agent_end")!(
      {
        success: true,
        messages: [
          { role: "user", content: "You are a memory search agent." },
          { role: "assistant", content: "Sister is Giulia." },
        ],
      },
      child
    );
    expect(seen.some((r) => r.url.includes("/memories"))).toBe(false);
  });

  it("returns from start() at once while the API is down (#6196)", async () => {
    // OpenClaw fails, stops and drops the turns of a service whose start() runs past ~5s;
    // the health check alone retries for longer than that.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      })
    );
    let started: ServiceConfig | undefined;
    registerPlugin({
      config: {
        plugins: {
          entries: {
            "hindsight-openclaw": {
              config: {
                hindsightApiUrl: "https://hindsight.test",
                // Never the real ~/.openclaw queue.
                retainQueuePath: join(mkdtempSync(join(tmpdir(), "hindsight-down-")), "q.jsonl"),
                logLevel: "off",
              },
            },
          },
        },
      },
      registerService: (registered) => {
        started = registered;
      },
      on: () => {},
      logger: { info: () => {}, warn: () => {}, error: () => {} },
    });
    service = started;
    const begin = Date.now();
    await started!.start();
    expect(Date.now() - begin).toBeLessThan(1000);
  });
});
