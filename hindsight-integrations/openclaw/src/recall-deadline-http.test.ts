/** Exercise the published client transport against a synthetic loopback server. */
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createServer, type Server, type ServerResponse } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MoltbotPluginAPI, ServiceConfig } from "./types.js";

const plugin = (await import("./index.js")).default;

let server: Server;
let service: ServiceConfig;
let hook: Parameters<MoltbotPluginAPI["on"]>[1];
let directory: string;
let requests: number;
let respond: (response: ServerResponse) => void;
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
function json(response: ServerResponse, body: unknown, status = 200) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

beforeEach(async () => {
  directory = mkdtempSync(join(tmpdir(), "hindsight-recall-http-"));
  requests = 0;
  respond = (response) => json(response, memory);
  server = createServer(async (request, response) => {
    // Consume the whole request before responding, as the real API does.
    for await (const _chunk of request) {
      /* drain */
    }
    if (request.url?.endsWith("/recall")) {
      requests++;
      respond(response);
    } else {
      json(response, { api_version: "0.10.0", features: { store_document_text: true } });
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No test listener");
  const baseUrl = `http://127.0.0.1:${address.port}`;
  const nativeFetch = globalThis.fetch;
  vi.stubGlobal("fetch", ((input, options) => {
    const url = input instanceof Request ? input.url : String(input);
    if (!url.startsWith(`${baseUrl}/`)) throw new Error("Unexpected external request in fixture");
    return nativeFetch(input, options);
  }) as typeof fetch);
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: baseUrl,
              bankId: "test-bank",
              dynamicBankId: false,
              autoRecall: true,
              autoRetain: false,
              // 1000 is the floor: getPluginConfig drops anything lower, and the
              // hook then falls back to the 10s default — which blows this test's
              // budget rather than exercising the deadline.
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
  await service.start();
});
afterEach(async () => {
  await service.stop();
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve()))
  );
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  rmSync(directory, { recursive: true, force: true });
});

it("disconnects an unanswered HTTP recall when its deadline expires", async () => {
  let disconnected = false;
  respond = (response) => {
    response.on("close", () => {
      disconnected = true;
    });
  };
  expect(await recall()).toBeUndefined();
  await vi.waitFor(() => expect(disconnected).toBe(true));
  expect(requests).toBe(1);
});
