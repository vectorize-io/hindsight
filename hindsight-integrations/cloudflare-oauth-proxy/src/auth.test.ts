import { beforeEach, describe, expect, it, vi } from "vitest";
import { handleDefaultRequest, verifyPassword } from "./auth";
import type { Env, OAuthHelpers, OAuthReqInfo } from "./env";

function createMockKv(): KVNamespace {
  const store = new Map<string, string>();
  const kv = {
    get: async (key: string): Promise<string | null> => store.get(key) ?? null,
    put: async (key: string, value: string): Promise<void> => {
      store.set(key, value);
    },
    delete: async (key: string): Promise<void> => {
      store.delete(key);
    },
  };
  return kv as unknown as KVNamespace;
}

function createMockProvider(): OAuthHelpers & {
  parseAuthRequest: ReturnType<typeof vi.fn>;
  completeAuthorization: ReturnType<typeof vi.fn>;
} {
  return {
    parseAuthRequest: vi.fn(
      async (): Promise<OAuthReqInfo> => ({
        clientId: "test-client",
        scope: "mcp:full",
      }),
    ),
    completeAuthorization: vi.fn(
      async () => ({
        redirectTo: "https://claude.ai/oauth/callback?code=abc",
      }),
    ),
  };
}

function createEnv(overrides: Partial<Env> = {}): Env {
  return {
    OAUTH_KV: createMockKv(),
    OAUTH_PROVIDER: createMockProvider(),
    HINDSIGHT_ORIGIN: "https://hindsight-origin.example.com",
    ALLOWED_EMAIL: "test@example.com",
    SESSION_SECRET: "correct-horse-battery-staple",
    PROXY_SECRET: "proxy-secret",
    HINDSIGHT_API_TOKEN: "server-token",
    ...overrides,
  };
}

describe("verifyPassword", () => {
  it("returns true for matching passwords", async () => {
    expect(await verifyPassword("hunter2", "hunter2")).toBe(true);
  });

  it("returns false for mismatched passwords", async () => {
    expect(await verifyPassword("hunter2", "hunter3")).toBe(false);
  });

  it("returns false for an empty provided password", async () => {
    expect(await verifyPassword("", "hunter2")).toBe(false);
  });

  it("is insensitive to length differences (no timing leak)", async () => {
    // Just exercising both short+long inputs; the assertion is that both
    // return deterministically and don't throw.
    expect(await verifyPassword("a", "a-very-long-password-indeed")).toBe(false);
    expect(await verifyPassword("a-very-long-password-indeed", "a")).toBe(false);
  });
});

describe("handleDefaultRequest", () => {
  let env: Env;
  beforeEach(() => {
    env = createEnv();
  });

  describe("/health", () => {
    it("returns { status: 'ok' }", async () => {
      const response = await handleDefaultRequest(
        new Request("https://hindsight.mydomain.com/health"),
        env,
      );
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual({ status: "ok" });
    });
  });

  describe("/ (service root)", () => {
    it("returns the service identifier", async () => {
      const response = await handleDefaultRequest(
        new Request("https://hindsight.mydomain.com/"),
        env,
      );
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual({ service: "Hindsight MCP OAuth Proxy" });
    });
  });

  describe("unknown paths", () => {
    it("returns 404", async () => {
      const response = await handleDefaultRequest(
        new Request("https://hindsight.mydomain.com/nope"),
        env,
      );
      expect(response.status).toBe(404);
    });
  });

  describe("GET /authorize", () => {
    it("stores OAuth state in KV and renders the login page", async () => {
      const response = await handleDefaultRequest(
        new Request("https://hindsight.mydomain.com/authorize?client_id=x"),
        env,
      );

      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toBe("text/html");
      const html = await response.text();
      expect(html).toContain("Hindsight MCP");
      expect(html).toContain('name="stateKey"');

      // Extract stateKey and verify it was stored in KV
      const match = html.match(/name="stateKey" value="([^"]+)"/);
      expect(match).not.toBeNull();
      const stateKey = match![1];
      const stored = await env.OAUTH_KV.get("auth_state:" + stateKey);
      expect(stored).not.toBeNull();
      const parsed = JSON.parse(stored!) as OAuthReqInfo;
      expect(parsed.clientId).toBe("test-client");

      // Provider was called with the request
      expect(env.OAUTH_PROVIDER.parseAuthRequest).toHaveBeenCalledTimes(1);
    });
  });

  describe("POST /authorize", () => {
    async function postForm(form: Record<string, string>, e: Env = env): Promise<Response> {
      const body = new URLSearchParams(form);
      return handleDefaultRequest(
        new Request("https://hindsight.mydomain.com/authorize", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body,
        }),
        e,
      );
    }

    it("returns 401 and re-renders the login page on wrong password", async () => {
      await env.OAUTH_KV.put(
        "auth_state:state-1",
        JSON.stringify({ clientId: "c", scope: "mcp:full" }),
      );
      const response = await postForm({ password: "wrong", stateKey: "state-1" });
      expect(response.status).toBe(401);
      const html = await response.text();
      expect(html).toContain("Incorrect password.");

      // State must not be consumed on a failed attempt
      expect(await env.OAUTH_KV.get("auth_state:state-1")).not.toBeNull();
    });

    it("returns 401 when password field is missing entirely", async () => {
      const response = await postForm({ stateKey: "state-1" });
      expect(response.status).toBe(401);
      expect(env.OAUTH_PROVIDER.completeAuthorization).not.toHaveBeenCalled();
    });

    it("returns 400 when state is missing", async () => {
      const response = await postForm({ password: "correct-horse-battery-staple" });
      expect(response.status).toBe(400);
      expect(await response.text()).toContain("Missing state");
    });

    it("returns 400 when state is expired/unknown", async () => {
      const response = await postForm({
        password: "correct-horse-battery-staple",
        stateKey: "does-not-exist",
      });
      expect(response.status).toBe(400);
      expect(await response.text()).toContain("Authorization expired");
    });

    it("happy path: calls completeAuthorization and redirects", async () => {
      await env.OAUTH_KV.put(
        "auth_state:state-2",
        JSON.stringify({ clientId: "test-client", scope: "mcp:full" }),
      );
      const response = await postForm({
        password: "correct-horse-battery-staple",
        stateKey: "state-2",
      });
      expect(response.status).toBe(302);
      expect(response.headers.get("Location")).toBe(
        "https://claude.ai/oauth/callback?code=abc",
      );

      expect(env.OAUTH_PROVIDER.completeAuthorization).toHaveBeenCalledTimes(1);
      const call = (env.OAUTH_PROVIDER.completeAuthorization as ReturnType<typeof vi.fn>)
        .mock.calls[0][0];
      expect(call.userId).toBe("test@example.com");
      expect(call.scope).toBe("mcp:full");
      expect(call.props.email).toBe("test@example.com");
      expect(typeof call.props.authenticatedAt).toBe("number");

      // State must be deleted after successful consumption
      expect(await env.OAUTH_KV.get("auth_state:state-2")).toBeNull();
    });

    it("defaults scope to mcp:full when the OAuth request carries none", async () => {
      await env.OAUTH_KV.put(
        "auth_state:state-3",
        JSON.stringify({ clientId: "test-client" }),
      );
      await postForm({
        password: "correct-horse-battery-staple",
        stateKey: "state-3",
      });

      const call = (env.OAUTH_PROVIDER.completeAuthorization as ReturnType<typeof vi.fn>)
        .mock.calls[0][0];
      expect(call.scope).toBe("mcp:full");
    });

    it("returns 500 when completeAuthorization throws", async () => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      const broken = createEnv();
      (broken.OAUTH_PROVIDER.completeAuthorization as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("library error"),
      );
      await broken.OAUTH_KV.put(
        "auth_state:state-4",
        JSON.stringify({ clientId: "test-client", scope: "mcp:full" }),
      );

      const response = await postForm(
        { password: "correct-horse-battery-staple", stateKey: "state-4" },
        broken,
      );
      expect(response.status).toBe(500);
      errorSpy.mockRestore();
    });
  });
});
