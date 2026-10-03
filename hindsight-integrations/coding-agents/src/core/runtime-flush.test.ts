import { createServer } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import { diag } from "./diag";
import { HindsightClient } from "./hindsight";
import { log } from "./log";
import { RuntimeCore } from "./runtime";

const daemon = vi.hoisted(() => vi.fn(async () => {}));
vi.mock("./daemon", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./daemon")>()),
  ensureDaemon: daemon,
}));
vi.mock("./diag", () => ({ diag: vi.fn() }));

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

afterEach(() => {
  daemon.mockReset().mockResolvedValue(undefined);
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("RuntimeCore write-back flush", () => {
  it.each([200, 500])(
    "awaits a delayed HTTP retain response (%i) without resending",
    async (status) => {
      const received = deferred();
      const respond = deferred();
      const bodies: { items: { document_id: string; content: string }[] }[] = [];
      const server = createServer(async (req, res) => {
        res.setHeader("content-type", "application/json");
        if (req.method !== "POST") {
          res.end(JSON.stringify({ version: "0.8.0" }));
          return;
        }
        let body = "";
        for await (const chunk of req) body += chunk;
        bodies.push(JSON.parse(body));
        received.resolve();
        await respond.promise;
        res.statusCode = status;
        res.end(JSON.stringify({ operation_id: "op-1" }));
      });
      await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("missing server address");
      const client = new HindsightClient({
        apiUrl: `http://127.0.0.1:${address.port}`,
        bank: "bank",
      });
      const runtime = new RuntimeCore(client, "bank", resolveConfig({}), "pi");
      const warn = vi.spyOn(log, "warn").mockImplementation(() => {});
      try {
        // agent_end returns while the submission is in flight; the completed answer must survive
        // the host's subsequent awaited shutdown event, without re-submitting that transcript.
        await runtime.onTranscript(
          "session",
          [
            { role: "user", content: "final preference" },
            { role: "assistant", content: "final summary" },
          ],
          true
        );
        await received.promise;
        let finished = false;
        const flushed = runtime.flushRetains().then((ok) => {
          finished = true;
          return ok;
        });
        await Promise.resolve();
        expect(finished).toBe(false);
        respond.resolve();
        expect(await flushed).toBe(status === 200);
        expect(await runtime.flushRetains()).toBe(true);
        expect(bodies).toHaveLength(1);
        expect(bodies[0].items[0].document_id).toBe("conversation:session");
        expect(bodies[0].items[0].content).toContain("final summary");
        if (status === 500) {
          expect(warn).toHaveBeenCalledWith(
            "pi",
            "session write-back failed",
            expect.objectContaining({ session: "session" })
          );
          expect(diag).toHaveBeenCalledWith(
            "pi",
            "retain_failed",
            expect.objectContaining({ session: "session" })
          );
        }
      } finally {
        respond.resolve();
        server.closeAllConnections();
        await new Promise<void>((r) => server.close(() => r()));
      }
    }
  );

  it.each(["daemon", "retain"])(
    "bounds stalled %s work and keeps it available to a later flush",
    async (phase) => {
      vi.useFakeTimers();
      const ready = deferred();
      const accepted = deferred();
      if (phase === "daemon") daemon.mockReturnValue(ready.promise);
      const client = { retain: vi.fn(() => accepted.promise) } as unknown as HindsightClient;
      const runtime = new RuntimeCore(client, "bank", resolveConfig({}), "pi");
      const warn = vi.spyOn(log, "warn").mockImplementation(() => {});
      await runtime.onTranscript("session", [{ role: "user", content: "hi" }], true);
      const flushed = runtime.flushRetains();
      await vi.advanceTimersByTimeAsync(10_000);
      expect(await flushed).toBe(false);
      expect(client.retain).toHaveBeenCalledTimes(phase === "daemon" ? 0 : 1);
      expect(diag).toHaveBeenCalledWith("pi", "retain_flush_timeout", {
        timeoutMs: 10_000,
        pending: 1,
      });
      expect(warn).toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);

      const again = runtime.flushRetains();
      ready.resolve();
      accepted.resolve();
      expect(await again).toBe(true);
      expect(client.retain).toHaveBeenCalledOnce();
      expect(vi.getTimerCount()).toBe(0);
    }
  );

  it("drains overlapping per-turn writes in order and does not duplicate an unchanged transcript", async () => {
    const accepted = deferred();
    const retain = vi
      .fn()
      .mockImplementationOnce(() => accepted.promise)
      .mockResolvedValue(undefined);
    const runtime = new RuntimeCore(
      { retain, supportsAppendRetain: async () => true } as unknown as HindsightClient,
      "bank",
      resolveConfig({}),
      "pi"
    );
    const first = [{ role: "user" as const, content: "first" }];
    const second = [...first, { role: "assistant" as const, content: "last" }];
    await runtime.onTranscript("session", first, true);
    await runtime.onTranscript("session", second, true);
    await runtime.onTranscript("session", second, true);
    const flushed = runtime.flushRetains();
    accepted.resolve();
    expect(await flushed).toBe(true);
    expect(retain).toHaveBeenCalledTimes(2);
    expect(retain.mock.calls[0][0]).not.toContain("last");
    expect(retain.mock.calls[1][0]).toContain("last");
  });

  it("is a no-op when session write-back is disabled", async () => {
    const retain = vi.fn();
    const runtime = new RuntimeCore(
      { retain } as unknown as HindsightClient,
      "bank",
      resolveConfig({ retainSessions: false }),
      "pi"
    );
    await runtime.onTranscript("session", [{ role: "user", content: "hi" }], true);
    expect(await runtime.flushRetains()).toBe(true);
    expect(retain).not.toHaveBeenCalled();
    expect(daemon).not.toHaveBeenCalled();
  });
});
