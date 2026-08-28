/**
 * Local-daemon mode must behave exactly like external-API mode. (#3686)
 *
 * The append-capability probe used to run only on the external-API code paths,
 * so a plugin that spawned its own daemon kept `supportsUpdateModeAppend` at
 * `false` forever, fell back to per-turn document ids sequenced from an
 * in-memory counter, and silently overwrote the previous run's documents after
 * every host restart.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const daemonStart = vi.fn(async () => undefined);
const daemonStop = vi.fn(async () => undefined);
const daemonCheckHealth = vi.fn(async () => true);
const DAEMON_BASE_URL = "http://127.0.0.1:9077";

vi.mock("@vectorize-io/hindsight-all", () => ({
  HindsightServer: class {
    start = daemonStart;
    stop = daemonStop;
    checkHealth = daemonCheckHealth;
    getBaseUrl = () => DAEMON_BASE_URL;
  },
}));

const plugin = (await import("./index.js")).default;
const { isAppendModeSupported, getDocumentIdBootToken } = await import("./index.js");

import type { MoltbotPluginAPI } from "./types.js";

interface StartedService {
  start(): Promise<void>;
  stop(): Promise<void>;
}

function makeApi(config: Record<string, unknown>): {
  api: MoltbotPluginAPI;
  service: () => StartedService;
  infoLines: string[];
} {
  let service: StartedService | undefined;
  const infoLines: string[] = [];
  const api = {
    config: { plugins: { entries: { "hindsight-openclaw": { config } } } },
    registerService: (svc: StartedService) => {
      service = svc;
    },
    on: () => undefined,
    logger: {
      info: (msg: string) => infoLines.push(String(msg)),
      warn: () => undefined,
      error: () => undefined,
    },
  } as unknown as MoltbotPluginAPI;
  return {
    api,
    service: () => {
      if (!service) throw new Error("service was never registered");
      return service;
    },
    infoLines,
  };
}

let originalFetch: typeof globalThis.fetch;
let tempDir: string;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  tempDir = mkdtempSync(join(tmpdir(), "hindsight-openclaw-"));
  daemonStart.mockClear();
  daemonStop.mockClear();
  daemonCheckHealth.mockClear();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  rmSync(tempDir, { recursive: true, force: true });
});

function mockVersionEndpoint(payload: unknown): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () => ({ ok: true, json: async () => payload }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("local-daemon mode parity (#3686)", () => {
  it("probes the daemon's /version so retains use append instead of per-turn ids", async () => {
    const fetchMock = mockVersionEndpoint({
      api_version: "0.9.1",
      features: { store_document_text: true },
    });
    const { api, service } = makeApi({
      llmProvider: "openai",
      llmApiKey: "sk-test",
      retainQueuePath: join(tempDir, "queue.jsonl"),
    });

    plugin(api);
    await service().start();
    try {
      expect(daemonStart).toHaveBeenCalled();
      const probed = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(probed).toContain(`${DAEMON_BASE_URL}/version`);
      expect(isAppendModeSupported()).toBe(true);
    } finally {
      await service().stop();
    }
  });

  it("keeps the per-turn fallback when the daemon cannot store document text", async () => {
    const fetchMock = mockVersionEndpoint({
      api_version: "0.9.1",
      features: { store_document_text: false },
    });
    const { api, service } = makeApi({
      llmProvider: "openai",
      llmApiKey: "sk-test",
      retainQueuePath: join(tempDir, "queue.jsonl"),
    });

    plugin(api);
    await service().start();
    try {
      const probed = fetchMock.mock.calls.map((call) => String(call[0]));
      expect(probed).toContain(`${DAEMON_BASE_URL}/version`);
      expect(isAppendModeSupported()).toBe(false);
    } finally {
      await service().stop();
    }
  });

  it("mints a fresh document-id boot token on every host process", async () => {
    // Reloading the module is the closest in-process stand-in for the restart
    // that used to replay `…:turn:000001` onto the previous run's document.
    vi.resetModules();
    const restarted = await import("./index.js");

    expect(restarted.getDocumentIdBootToken()).not.toBe(getDocumentIdBootToken());
  });

  it("opens the retain queue so a daemon that is down cannot drop retains", async () => {
    mockVersionEndpoint({ api_version: "0.9.1", features: { store_document_text: true } });
    const queuePath = join(tempDir, "queue.jsonl");
    writeFileSync(
      queuePath,
      JSON.stringify({
        id: "1-abcd",
        bankId: "bank",
        content: "hello",
        documentId: "openclaw:session",
        metadata: {},
        createdAt: new Date().toISOString(),
      }) + "\n",
      "utf8"
    );
    const { api, service, infoLines } = makeApi({
      llmProvider: "openai",
      llmApiKey: "sk-test",
      retainQueuePath: queuePath,
    });

    plugin(api);
    await service().start();
    try {
      expect(infoLines.some((line) => line.includes("retain queue: 1 items pending"))).toBe(true);
    } finally {
      await service().stop();
    }
  });
});
