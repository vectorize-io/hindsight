import { describe, expect, it, vi } from "vitest";
import { createWorker, type ProviderLike } from "./router";
import type { Env } from "./env";

function fakeEnv(): Env {
  return {
    OAUTH_KV: {} as KVNamespace,
    OAUTH_PROVIDER: {
      parseAuthRequest: vi.fn(),
      completeAuthorization: vi.fn(),
    },
    HINDSIGHT_ORIGIN: "https://origin.example.com",
    ALLOWED_EMAIL: "test@example.com",
    SESSION_SECRET: "secret",
    PROXY_SECRET: "proxy-secret",
    HINDSIGHT_API_TOKEN: "token",
  };
}

const fakeCtx = {} as ExecutionContext;

describe("createWorker", () => {
  describe("OPTIONS preflight", () => {
    it("returns 204 + CORS headers for an allowlisted origin", async () => {
      const provider: ProviderLike = { fetch: vi.fn() };
      const worker = createWorker(provider);
      const response = await worker.fetch(
        new Request("https://proxy.example.com/mcp", {
          method: "OPTIONS",
          headers: { Origin: "https://claude.ai" },
        }),
        fakeEnv(),
        fakeCtx
      );

      expect(response.status).toBe(204);
      expect(response.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
      expect(provider.fetch).not.toHaveBeenCalled();
    });

    it("returns 403 for a disallowed origin without touching the provider", async () => {
      const provider: ProviderLike = { fetch: vi.fn() };
      const worker = createWorker(provider);
      const response = await worker.fetch(
        new Request("https://proxy.example.com/mcp", {
          method: "OPTIONS",
          headers: { Origin: "https://evil.example.com" },
        }),
        fakeEnv(),
        fakeCtx
      );

      expect(response.status).toBe(403);
      expect(provider.fetch).not.toHaveBeenCalled();
    });
  });

  describe("/.well-known/oauth-authorization-server", () => {
    it("forces code_challenge_methods_supported to ['S256']", async () => {
      const provider: ProviderLike = {
        fetch: vi.fn(
          async () =>
            new Response(
              JSON.stringify({
                issuer: "https://proxy.example.com",
                code_challenge_methods_supported: ["S256", "plain"],
                grant_types_supported: ["authorization_code"],
              }),
              { headers: { "Content-Type": "application/json" } }
            )
        ),
      };
      const worker = createWorker(provider);

      const response = await worker.fetch(
        new Request("https://proxy.example.com/.well-known/oauth-authorization-server", {
          headers: { Origin: "https://claude.ai" },
        }),
        fakeEnv(),
        fakeCtx
      );

      const metadata = (await response.json()) as Record<string, unknown>;
      expect(metadata.code_challenge_methods_supported).toEqual(["S256"]);
      expect(metadata.grant_types_supported).toEqual(["authorization_code"]);
      expect(response.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
    });

    it("strips CORS headers when the request origin is not allowlisted", async () => {
      const provider: ProviderLike = {
        fetch: vi.fn(
          async () =>
            new Response(JSON.stringify({ code_challenge_methods_supported: ["plain"] }), {
              headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
              },
            })
        ),
      };
      const worker = createWorker(provider);

      const response = await worker.fetch(
        new Request("https://proxy.example.com/.well-known/oauth-authorization-server"),
        fakeEnv(),
        fakeCtx
      );

      expect(response.headers.get("Access-Control-Allow-Origin")).toBeNull();
    });
  });

  describe("delegation to the OAuth provider", () => {
    it("passes non-preflight, non-metadata requests through and applies CORS", async () => {
      const providerFetch = vi.fn(async () => new Response("provider-body"));
      const provider: ProviderLike = { fetch: providerFetch };
      const worker = createWorker(provider);

      const response = await worker.fetch(
        new Request("https://proxy.example.com/authorize", {
          headers: { Origin: "https://claude.ai" },
        }),
        fakeEnv(),
        fakeCtx
      );

      expect(providerFetch).toHaveBeenCalledTimes(1);
      expect(response.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
      expect(await response.text()).toBe("provider-body");
    });

    it("strips provider-set CORS headers for non-allowlisted origins", async () => {
      const provider: ProviderLike = {
        fetch: vi.fn(
          async () =>
            new Response("ok", {
              headers: { "Access-Control-Allow-Origin": "*" },
            })
        ),
      };
      const worker = createWorker(provider);

      const response = await worker.fetch(
        new Request("https://proxy.example.com/authorize", {
          headers: { Origin: "https://evil.example.com" },
        }),
        fakeEnv(),
        fakeCtx
      );

      expect(response.headers.get("Access-Control-Allow-Origin")).toBeNull();
    });
  });
});
