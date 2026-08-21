import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MoltbotPluginAPI, PluginHookAgentContext, ServiceConfig } from "./types.js";

const daemonMocks = vi.hoisted(() => ({
  start: vi.fn(async () => undefined),
  stop: vi.fn(async () => undefined),
  checkHealth: vi.fn(async () => true),
  getBaseUrl: vi.fn(() => "http://127.0.0.1:19077"),
}));

const clientMocks = vi.hoisted(() => ({
  retain: vi.fn(async () => undefined),
}));

vi.mock("@vectorize-io/hindsight-all", () => ({
  HindsightServer: vi.fn(
    class {
      start = daemonMocks.start;
      stop = daemonMocks.stop;
      checkHealth = daemonMocks.checkHealth;
      getBaseUrl = daemonMocks.getBaseUrl;
    }
  ),
}));

vi.mock("@vectorize-io/hindsight-client", () => ({
  HindsightClient: vi.fn(
    class {
      retain = clientMocks.retain;
    }
  ),
}));

import registerPlugin from "./index.js";

interface PluginHarness {
  service: ServiceConfig;
  agentEnd: (event: unknown, ctx?: PluginHookAgentContext) => Promise<void>;
}

let activeService: ServiceConfig | undefined;
let fetchMock: ReturnType<typeof vi.fn>;

function makeHarness(): PluginHarness {
  let registeredService: ServiceConfig | undefined;
  let agentEndHandler:
    | ((event: unknown, ctx?: PluginHookAgentContext) => void | Promise<void>)
    | undefined;
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              llmProvider: "ollama",
              dynamicBankId: false,
              bankId: "local-bank",
              autoRecall: false,
              autoRetain: true,
              logLevel: "off",
            },
          },
        },
      },
    },
    registerService(service) {
      registeredService = service;
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
  if (!registeredService) throw new Error("service not registered");
  if (!agentEndHandler) throw new Error("agent_end not registered");

  return {
    service: registeredService,
    agentEnd: async (event, ctx) => {
      await agentEndHandler(event, ctx);
    },
  };
}

async function retainOneTurn(harness: PluginHarness, session: string): Promise<void> {
  await harness.agentEnd(
    {
      success: true,
      messages: [
        { role: "user", content: "Remember that my preferred editor theme is solarized." },
        { role: "assistant", content: "I will remember that preference." },
      ],
    },
    {
      agentId: "main",
      sessionKey: `agent:main:discord:direct:${session}`,
      messageProvider: "discord",
      channelId: `direct:${session}`,
      senderId: "user:local-test",
    }
  );
}

function expectAppendRetain(session: string): void {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock.mock.calls[0]?.[0]).toBe("http://127.0.0.1:19077/version");
  expect(clientMocks.retain).toHaveBeenCalledWith(
    "local-bank",
    expect.any(String),
    expect.objectContaining({
      documentId: `openclaw:agent:main:discord:direct:${session}`,
      updateMode: "append",
      async: true,
    })
  );
}

beforeEach(() => {
  daemonMocks.start.mockReset().mockResolvedValue(undefined);
  daemonMocks.stop.mockReset().mockResolvedValue(undefined);
  daemonMocks.checkHealth.mockReset().mockResolvedValue(true);
  daemonMocks.getBaseUrl.mockReset().mockReturnValue("http://127.0.0.1:19077");
  clientMocks.retain.mockReset().mockResolvedValue(undefined);
  fetchMock = vi.fn(async (input: string | URL | Request) => {
    const url = input instanceof Request ? input.url : input.toString();
    if (url !== "http://127.0.0.1:19077/version") {
      throw new Error(`unexpected request: ${url}`);
    }
    return new Response(
      JSON.stringify({
        api_version: "0.9.0",
        features: { store_document_text: true },
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(async () => {
  if (activeService) {
    await activeService.stop();
    activeService = undefined;
  }
  vi.unstubAllGlobals();
});

describe("local daemon capability detection", () => {
  it("uses append retention after the initial daemon start", async () => {
    const harness = makeHarness();
    activeService = harness.service;

    await harness.service.start();
    await retainOneTurn(harness, "initial");

    expect(daemonMocks.start).toHaveBeenCalledTimes(1);
    expectAppendRetain("initial");
  });

  it("uses append retention after recovering from an initial daemon start failure", async () => {
    daemonMocks.start.mockRejectedValueOnce(new Error("initial daemon failed to start"));
    const harness = makeHarness();
    activeService = harness.service;

    await harness.service.start();
    await retainOneTurn(harness, "recovered");

    expect(daemonMocks.start).toHaveBeenCalledTimes(2);
    expectAppendRetain("recovered");
  });
});
