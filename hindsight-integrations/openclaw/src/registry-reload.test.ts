import { expect, it, vi } from "vitest";
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MoltbotPluginAPI, ServiceConfig } from "./types.js";

it("retains across registry owners until the active client owner stops", async () => {
  const directory = mkdtempSync(join(tmpdir(), "hindsight-registry-"));
  const retains: string[] = [];
  const services: ServiceConfig[] = [];
  const server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    response.setHeader("Content-Type", "application/json");
    if (request.method === "POST" && request.url?.endsWith("/memories")) retains.push(body);
    response.end(
      JSON.stringify(
        request.url?.endsWith("/version")
          ? { api_version: "0.10.1", features: { store_document_text: true } }
          : { success: true }
      )
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Expected a TCP listener");

  async function load() {
    // Registry reloads evaluate index independently while sharing its global client facade.
    vi.resetModules();
    const plugin = (await import("./index.js")).default;
    let service!: ServiceConfig;
    let retain!: Parameters<MoltbotPluginAPI["on"]>[1];
    const api: MoltbotPluginAPI = {
      config: {
        plugins: {
          entries: {
            "hindsight-openclaw": {
              config: {
                hindsightApiUrl: `http://127.0.0.1:${address.port}`,
                bankId: "fixture-bank",
                dynamicBankId: false,
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
        service = registered;
        services.push(service);
      },
      on: (name, hook) => {
        if (name === "agent_end") retain = hook;
      },
      logger: { info: () => {}, warn: () => {}, error: () => {} },
    };
    plugin(api);
    return { service, retain };
  }

  const transcript = {
    success: true,
    messages: [
      { role: "user", content: "The project release date is October 15." },
      { role: "assistant", content: "I will remember that release date." },
    ],
  };
  const context = {
    agentId: "main",
    sessionKey: "agent:main:telegram:direct:fixture",
    messageProvider: "telegram",
    channelId: "fixture",
    senderId: "fixture",
  };
  try {
    const a = await load();
    await a.service.start();
    const b = await load();
    await b.retain(transcript, context);
    expect(retains).toHaveLength(1);
    expect(retains[0]).toContain("The project release date is October 15.");

    const c = await load();
    await c.service.start();
    await b.retain(transcript, { ...context, sessionKey: `${context.sessionKey}-replacement` });
    expect(retains).toHaveLength(2);
    await a.service.stop();
    await b.retain(transcript, { ...context, sessionKey: `${context.sessionKey}-stale-stop` });
    expect(retains).toHaveLength(3);
    await c.service.stop();
    await b.retain(transcript, { ...context, sessionKey: `${context.sessionKey}-active-stop` });
    expect(retains).toHaveLength(3);
  } finally {
    for (const service of services.splice(0)) await service.stop();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    rmSync(directory, { recursive: true, force: true });
    vi.resetModules();
  }
});
