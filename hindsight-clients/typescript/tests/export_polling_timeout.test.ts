/** Completion deadlines through the real generated SDK and loopback HTTP. */
import { createServer, Server, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";
import { HindsightClient, HindsightError } from "../src";

const OPERATION_ID = "123e4567-e89b-12d3-a456-426614174000";
const DOWNLOAD_URL = "/v1/default/files/download/banks/test-bank/export.zip";
const ARCHIVE = new Uint8Array([0x50, 0x4b, 0x03, 0x04]);
const METHODS = ["exportDocuments", "exportBank"] as const;
type Stage = "submission" | "poll" | "download";
type Outcome = { kind: "success"; value: Uint8Array } | { kind: "error"; error: unknown };

class Gate {
  readonly promise: Promise<void>;
  open!: () => void;
  constructor() {
    this.promise = new Promise<void>((resolve) => {
      this.open = resolve;
    });
  }
}
function settle(call: Promise<Uint8Array>): Promise<Outcome> {
  return call.then(
    (value) => ({ kind: "success", value }),
    (error: unknown) => ({ kind: "error", error })
  );
}
async function within<T>(call: Promise<T>, milliseconds = 1000): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      call,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("operation exceeded test safety guard")),
          milliseconds
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
class ExportServer {
  server: Server;
  baseUrl = "";
  status = "completed";
  hold: Stage | undefined;
  submissions = 0;
  polls = 0;
  downloads = 0;
  submitted = new Gate();
  polled = new Gate();
  downloading = new Gate();
  release = new Gate();
  responded = new Gate();
  constructor() {
    this.server = createServer((request, response) => {
      const path = new URL(request.url ?? "/", "http://localhost").pathname;
      void this.respond(request.method ?? "", path, response);
    });
  }
  async start(): Promise<void> {
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    this.baseUrl = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }
  async stop(): Promise<void> {
    this.release.open();
    this.server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      this.server.close((error) => (error ? reject(error) : resolve()))
    );
  }
  private async respond(method: string, path: string, response: ServerResponse): Promise<void> {
    if (
      method === "POST" &&
      [
        "/v1/default/banks/test-bank/document-transfer/export",
        "/v1/default/banks/test-bank/transfer/export",
      ].includes(path)
    ) {
      this.submissions++;
      this.submitted.open();
      if (this.hold === "submission") await this.release.promise;
      response.writeHead(202, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ operation_id: OPERATION_ID, status: "pending" }));
    } else if (
      method === "GET" &&
      path === `/v1/default/banks/test-bank/operations/${OPERATION_ID}`
    ) {
      this.polls++;
      this.polled.open();
      if (this.hold === "poll") await this.release.promise;
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify({
          operation_id: OPERATION_ID,
          status: this.status,
          result_metadata: this.status === "completed" ? { download_url: DOWNLOAD_URL } : null,
          error_message: "synthetic export failure",
        })
      );
      this.responded.open();
    } else if (method === "GET" && path === DOWNLOAD_URL) {
      this.downloads++;
      this.downloading.open();
      if (this.hold === "download") await this.release.promise;
      response.writeHead(200, { "Content-Type": "application/zip" });
      response.end(ARCHIVE);
    } else {
      response.writeHead(404);
      response.end();
    }
  }
}

describe.each(METHODS)("%s completion timeout", (method) => {
  let server: ExportServer;
  let client: HindsightClient;
  let controller: AbortController;
  let outcome: Promise<Outcome> | undefined;
  beforeEach(async () => {
    server = new ExportServer();
    await server.start();
    client = new HindsightClient({ baseUrl: server.baseUrl });
    controller = new AbortController();
    outcome = undefined;
  });
  afterEach(async () => {
    controller.abort();
    server.release.open();
    // Baseline polling sleeps are not cancellable. Let that pending sleep finish
    // in cleanup without confusing it with the deadline assertion itself.
    if (outcome) await within(outcome, 3000);
    await server.stop();
  });
  it.each(["processing", "completed"])(
    "bounds a held status response ending %s",
    async (status) => {
      server.hold = "poll";
      server.status = status;
      outcome = settle(
        client[method]("test-bank", { timeoutMs: 50, pollIntervalMs: 0, signal: controller.signal })
      );
      await within(server.polled.promise);
      // The peer cannot respond until released; the 50ms deadline gets 1s slack.
      const result = await within(outcome);
      expect(result.kind).toBe("error");
      if (result.kind === "error") {
        expect(result.error).toBeInstanceOf(HindsightError);
        expect((result.error as Error).message).toMatch(/did not complete within/);
      }
      server.release.open();
      await within(server.responded.promise);
      await expect(within(server.downloading.promise, 100)).rejects.toThrow("test safety guard");
      expect(server.downloads).toBe(0);
    }
  );
  it("bounds a polling interval longer than the remaining timeout", async () => {
    server.status = "processing";
    outcome = settle(
      client[method]("test-bank", {
        timeoutMs: 50,
        pollIntervalMs: 2000,
        signal: controller.signal,
      })
    );
    await within(server.responded.promise);
    const result = await within(outcome);
    expect(result.kind).toBe("error");
    if (result.kind === "error")
      expect((result.error as Error).message).toMatch(/did not complete within/);
    expect(server.polls).toBe(1);
    expect(server.downloads).toBe(0);
  });
  it("preserves caller cancellation and never downloads a late archive", async () => {
    server.hold = "poll";
    outcome = settle(client[method]("test-bank", { timeoutMs: 60000, signal: controller.signal }));
    await within(server.polled.promise);
    controller.abort(new Error("caller stopped export"));
    const result = await within(outcome);
    expect(result.kind).toBe("error");
    if (result.kind === "error")
      expect((result.error as Error).message).toContain("caller stopped export");
    server.release.open();
    await within(server.responded.promise);
    await expect(within(server.downloading.promise, 100)).rejects.toThrow("test safety guard");
    expect(server.downloads).toBe(0);
  });
  it.each(["completed", "failed", "cancelled"])("preserves %s status", async (status) => {
    server.status = status;
    outcome = settle(client[method]("test-bank", { timeoutMs: 5000, signal: controller.signal }));
    const result = await within(outcome);
    if (status === "completed") {
      expect(result).toEqual({ kind: "success", value: ARCHIVE });
      expect(server.downloads).toBe(1);
    } else {
      expect(result.kind).toBe("error");
      if (result.kind === "error")
        expect((result.error as Error).message).toContain(`${status}: synthetic export failure`);
      expect(server.downloads).toBe(0);
    }
    expect(server.submissions).toBe(1);
    expect(server.polls).toBe(1);
  });
  it.each<Stage>(["submission", "download"])(
    "does not extend polling deadline scope to %s",
    async (stage) => {
      server.hold = stage;
      outcome = settle(
        client[method]("test-bank", { timeoutMs: 50, pollIntervalMs: 0, signal: controller.signal })
      );
      await within(stage === "submission" ? server.submitted.promise : server.downloading.promise);
      let settled = false;
      void outcome.then(() => {
        settled = true;
      });
      await new Promise<void>((resolve) => setTimeout(resolve, 150));
      expect(settled).toBe(false);
      server.release.open();
      expect(await within(outcome)).toEqual({ kind: "success", value: ARCHIVE });
    }
  );

  it.each([0, -1])("does not poll for nonpositive timeout %s", async (timeoutMs) => {
    outcome = settle(client[method]("test-bank", { timeoutMs, signal: controller.signal }));
    const result = await within(outcome);
    expect(result.kind).toBe("error");
    if (result.kind === "error")
      expect((result.error as Error).message).toMatch(/did not complete within/);
    expect(server.submissions).toBe(1);
    expect(server.polls).toBe(0);
    expect(server.downloads).toBe(0);
  });
  it("preserves caller cancellation between polls", async () => {
    server.status = "processing";
    outcome = settle(
      client[method]("test-bank", {
        timeoutMs: 60000,
        pollIntervalMs: 2000,
        signal: controller.signal,
      })
    );
    await within(server.responded.promise);
    await new Promise<void>((resolve) => setTimeout(resolve, 100));
    controller.abort(new Error("caller stopped between polls"));
    const result = await within(outcome);
    expect(result.kind).toBe("error");
    if (result.kind === "error")
      expect((result.error as Error).message).toContain("caller stopped between polls");
    expect(server.polls).toBe(1);
    expect(server.downloads).toBe(0);
  });
  it("can reuse the client after a timed-out export", async () => {
    server.hold = "poll";
    outcome = settle(client[method]("test-bank", { timeoutMs: 50, signal: controller.signal }));
    await within(server.polled.promise);
    const result = await within(outcome);
    expect(result.kind).toBe("error");
    server.release.open();
    await within(server.responded.promise);
    server.hold = undefined;
    expect(await client[method]("test-bank", { timeoutMs: 5000 })).toEqual(ARCHIVE);
    expect(server.downloads).toBe(1);
  });
});
