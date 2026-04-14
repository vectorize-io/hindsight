import { describe, expect, it, vi } from "vitest";
import { proxyRequest } from "./proxy";

const baseEnv = {
  HINDSIGHT_ORIGIN: "https://hindsight-origin.example.com",
  PROXY_SECRET: "proxy-secret-value",
  HINDSIGHT_API_TOKEN: "server-api-token",
};

describe("proxyRequest", () => {
  it("rewrites the target URL to the Hindsight origin", async () => {
    let calledUrl: string | undefined;
    const fetchImpl = vi.fn<typeof fetch>(async (url) => {
      calledUrl = url.toString();
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "POST",
      body: JSON.stringify({ jsonrpc: "2.0", method: "ping" }),
      headers: { "Content-Type": "application/json" },
    });

    await proxyRequest(request, baseEnv, { fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(calledUrl).toBe("https://hindsight-origin.example.com/mcp");
  });

  it("strips client Authorization and injects the server bearer token", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      const headers = new Headers(init?.headers);
      expect(headers.get("Authorization")).toBe("Bearer server-api-token");
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "POST",
      headers: { Authorization: "Bearer client-leaked-token" },
      body: "{}",
    });

    await proxyRequest(request, baseEnv, { fetchImpl });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("injects the X-Proxy-Secret header even if the client tries to set one", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      const headers = new Headers(init?.headers);
      expect(headers.get("X-Proxy-Secret")).toBe("proxy-secret-value");
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "POST",
      headers: { "X-Proxy-Secret": "client-forged-value" },
      body: "{}",
    });

    await proxyRequest(request, baseEnv, { fetchImpl });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("does not forward a body on GET requests", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      expect(init?.body).toBeNull();
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "GET",
    });

    await proxyRequest(request, baseEnv, { fetchImpl });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("forwards the request body as an ArrayBuffer for POST", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      const body = init?.body;
      expect(body).toBeInstanceOf(ArrayBuffer);
      expect(new TextDecoder().decode(body as ArrayBuffer)).toBe('{"hello":"world"}');
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "POST",
      body: '{"hello":"world"}',
      headers: { "Content-Type": "application/json" },
    });

    await proxyRequest(request, baseEnv, { fetchImpl });
  });

  it("drops upstream CORS and Set-Cookie headers from the response", async () => {
    const fetchImpl = vi.fn<typeof fetch>(
      async () =>
        new Response("payload", {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Set-Cookie": "session=leak",
            "X-Custom-Leak": "yes",
            "Mcp-Session-Id": "session-123",
          },
        }),
    );
    const request = new Request("https://hindsight.mydomain.com/mcp", { method: "GET" });

    const response = await proxyRequest(request, baseEnv, { fetchImpl });
    expect(response.headers.get("Content-Type")).toBe("application/json");
    expect(response.headers.get("Mcp-Session-Id")).toBe("session-123");
    expect(response.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(response.headers.get("Set-Cookie")).toBeNull();
    expect(response.headers.get("X-Custom-Leak")).toBeNull();
  });

  it("forwards the upstream status code", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("nope", { status: 418 }));
    const request = new Request("https://hindsight.mydomain.com/mcp", { method: "GET" });

    const response = await proxyRequest(request, baseEnv, { fetchImpl });
    expect(response.status).toBe(418);
  });

  it("returns a 502 JSON error when the upstream fetch throws", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchImpl = vi.fn<typeof fetch>(async () => {
      throw new Error("ECONNREFUSED");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", { method: "GET" });

    const response = await proxyRequest(request, baseEnv, { fetchImpl });
    expect(response.status).toBe(502);
    expect(response.headers.get("Content-Type")).toBe("application/json");
    const body = await response.json();
    expect(body).toEqual({ error: "Backend unavailable" });

    errorSpy.mockRestore();
  });

  it("does not forward the Host header to the upstream", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      const headers = new Headers(init?.headers);
      // `fetch` controls Host from the URL; forwarding the client's would
      // point the upstream at the wrong vhost.
      expect(headers.get("host")).toBeNull();
      return new Response("ok");
    });
    const request = new Request("https://hindsight.mydomain.com/mcp", {
      method: "GET",
      headers: { Host: "hindsight.mydomain.com" },
    });

    await proxyRequest(request, baseEnv, { fetchImpl });
  });
});
