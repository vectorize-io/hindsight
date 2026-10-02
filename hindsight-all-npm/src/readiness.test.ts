import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect, vi, afterEach } from "vitest";
import { HindsightServer } from "./server.js";

// CLI success is deterministic here; readiness itself exercises a real local
// HTTP connection, including cancellation when the server sends no headers.
vi.mock("child_process", () => ({
  spawn: () => {
    const child = {
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      on: (event: string, handler: (code?: number) => void) => {
        if (event === "exit") handler(0);
        return child;
      },
    };
    return child;
  },
}));

describe("daemon readiness deadline", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each(["pending", "rejected"])(
    "accepts healthy headers when body cleanup is %s",
    async (mode) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          body: {
            cancel: () =>
              mode === "pending"
                ? new Promise<void>(() => {})
                : Promise.reject(new Error("cleanup failed")),
          },
        })
      );
      const server = new HindsightServer({ readyTimeoutMs: 50 });
      await expect(server.start()).resolves.toBeUndefined();
    },
    200
  );

  it("bounds an unhealthy probe even if body cleanup never completes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, body: { cancel: () => new Promise<void>(() => {}) } })
    );
    const server = new HindsightServer({ readyTimeoutMs: 50, readyPollIntervalMs: 1000 });
    await expect(server.start()).rejects.toThrow("within 50ms");
  }, 200);

  it("does not accept healthy headers returned after the deadline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async () => {
        await new Promise((resolve) => setTimeout(resolve, 70));
        return { ok: true, body: null };
      })
    );
    const server = new HindsightServer({ readyTimeoutMs: 50 });
    await expect(server.start()).rejects.toThrow("within 50ms");
  });

  it("accepts a healthy status without waiting for its response body", async () => {
    const http = createServer((_request, response) => {
      response.writeHead(200);
      response.write("ready");
      // Leave the body open: a health probe only needs the successful status.
    });
    http.listen(0, "127.0.0.1");
    await once(http, "listening");
    const address = http.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const server = new HindsightServer({ port: address.port, readyTimeoutMs: 500 });
      await expect(server.start()).resolves.toBeUndefined();
    } finally {
      http.closeAllConnections();
      await new Promise<void>((resolve, reject) =>
        http.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });

  it.each(["unavailable", "hanging"])("bounds startup when health is %s", async (mode) => {
    const http = createServer((_request, response) => {
      if (mode === "unavailable") {
        response.writeHead(503);
        response.end("not ready");
      }
    });
    http.listen(0, "127.0.0.1");
    await once(http, "listening");
    const address = http.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    const server = new HindsightServer({
      port: address.port,
      readyTimeoutMs: 50,
      readyPollIntervalMs: 1000,
    });
    const start = Date.now();
    try {
      await expect(server.start()).rejects.toThrow("within 50ms");
      expect(Date.now() - start).toBeLessThan(500);
    } finally {
      http.closeAllConnections();
      await new Promise<void>((resolve, reject) =>
        http.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
