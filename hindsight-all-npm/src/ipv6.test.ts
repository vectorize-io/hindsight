import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect } from "vitest";
import { HindsightServer } from "./server.js";

describe("IPv6 daemon hosts", () => {
  it.each(["::1", "[::1]", "2001:db8::1"])("formats %s as a valid HTTP authority", (host) => {
    const server = new HindsightServer({ host, port: 9077 });
    const url = new URL(server.getBaseUrl());
    expect(url.port).toBe("9077");
    expect(url.hostname).toBe(host.startsWith("[") ? host : `[${host}]`);
  });

  it("can check an actual IPv6 loopback daemon", async (context) => {
    const http = createServer((_request, response) => response.end("ready"));
    http.listen(0, "::1");
    try {
      await once(http, "listening");
    } catch (error) {
      // Some CI hosts disable IPv6; only that platform limitation is skipped.
      if (["EAFNOSUPPORT", "EADDRNOTAVAIL"].includes((error as NodeJS.ErrnoException).code ?? "")) {
        context.skip();
        return;
      }
      throw error;
    }
    const address = http.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const server = new HindsightServer({ host: "::1", port: address.port });
      await expect(server.checkHealth()).resolves.toBe(true);
    } finally {
      http.closeAllConnections();
      await new Promise<void>((resolve, reject) =>
        http.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
