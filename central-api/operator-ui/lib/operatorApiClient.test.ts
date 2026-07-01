import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAuthHeaders,
  createOperatorApiClient,
  maskSecret,
  normalizeApiBaseUrl,
  type ConnectionConfig,
} from "./operatorApiClient";

const baseConfig = (overrides: Partial<ConnectionConfig> = {}): ConnectionConfig => ({
  apiBaseUrl: "http://api.example.test///",
  mode: "local",
  connected: false,
  ...overrides,
});

const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });

test("normalizes base URLs", () => {
  assert.equal(normalizeApiBaseUrl(" https://api.example.test/// "), "https://api.example.test");
  assert.equal(normalizeApiBaseUrl("http://localhost:3050/api/"), "http://localhost:3050/api");
  assert.equal(normalizeApiBaseUrl("api.example.test///"), "http://api.example.test");
});

test("injects API key auth headers", async () => {
  const seenHeaders: HeadersInit[] = [];
  const client = createOperatorApiClient(baseConfig({ apiKey: "mem11_sk_secret" }), {
    fetchImpl: async (_input, init) => {
      seenHeaders.push(init?.headers ?? {});
      return jsonResponse({ ok: true });
    },
  });

  await client.validateAuth();
  const headers = new Headers(seenHeaders[0]);
  assert.equal(headers.get("authorization"), "Bearer mem11_sk_secret");
  assert.equal(headers.get("x-api-key"), "mem11_sk_secret");
});

test("injects JWT auth header before API key", async () => {
  const seenHeaders: HeadersInit[] = [];
  const client = createOperatorApiClient(
    baseConfig({ apiKey: "mem11_sk_secret", jwtToken: "ey.jwt.token" }),
    {
      fetchImpl: async (_input, init) => {
        seenHeaders.push(init?.headers ?? {});
        return jsonResponse({ ok: true });
      },
    },
  );

  await client.validateAuth();
  const headers = new Headers(seenHeaders[0]);
  assert.equal(headers.get("authorization"), "Bearer ey.jwt.token");
  assert.equal(headers.get("x-api-key"), null);
  assert.deepEqual(buildAuthHeaders(baseConfig({ jwtToken: "ey.jwt.token" })), {
    Authorization: "Bearer ey.jwt.token",
  });
});

test("masks configured secrets and common token patterns in errors", async () => {
  const client = createOperatorApiClient(baseConfig({ apiKey: "mem11_sk_secret" }), {
    fetchImpl: async () =>
      jsonResponse(
        { detail: "bad Authorization: Bearer mem11_sk_secret" },
        { status: 401, statusText: "Unauthorized" },
      ),
  });

  const result = await client.validateAuth();
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.match(result.error.message, /\[REDACTED\]/);
    assert.doesNotMatch(JSON.stringify(result.error), /mem11_sk_secret/);
  }

  assert.equal(maskSecret("Authorization: Bearer abc.def.ghi"), "Authorization: Bearer [REDACTED]");
});

test("timeout failures are normalized and do not leak secrets", async () => {
  const client = createOperatorApiClient(baseConfig({ apiKey: "mem11_sk_timeout_secret" }), {
    timeoutMs: 1,
    fetchImpl: async (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const error = new Error("timed out with mem11_sk_timeout_secret");
          error.name = "AbortError";
          reject(error);
        });
      }),
  });

  const result = await client.validateAuth();
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.equal(result.status, 0);
    assert.equal(result.error.code, "network_error");
    assert.equal(result.error.message, "Request timed out");
    assert.doesNotMatch(JSON.stringify(result.error), /mem11_sk_timeout_secret/);
  }
});

test("health succeeds on the first available endpoint", async () => {
  const paths: string[] = [];
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async (input) => {
      paths.push(String(input));
      return jsonResponse({ status: "ok", service: "central-api" });
    },
  });

  const result = await client.health();
  assert.equal(result.ok, true);
  assert.deepEqual(paths, ["http://api.example.test/health"]);
  if (result.ok) assert.equal(result.data.service, "central-api");
});

test("health falls back after missing endpoints and returns success", async () => {
  const paths: string[] = [];
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async (input) => {
      paths.push(new URL(String(input)).pathname);
      if (paths.length < 3) return jsonResponse({ detail: "missing" }, { status: 404 });
      return jsonResponse([{ backend: "internal", status: "ok" }]);
    },
  });

  const result = await client.health();
  assert.equal(result.ok, true);
  assert.deepEqual(paths, ["/health", "/api/health", "/api/health/engines"]);
  if (result.ok) {
    assert.equal(result.data.engines?.[0]?.backend, "internal");
  }
});

test("healthDependencies returns richer readiness data", async () => {
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async () =>
      jsonResponse({
        status: "degraded",
        timestamp_ms: 123,
        services: { memory: { name: "memory", port: 3020, status: "healthy", latency_ms: 12, last_checked_at: 123 } },
        providers: [],
        governance_healthy: true,
        audit_events_count: 4,
        quarantine_items_count: 0,
        is_quarantined: false,
      }),
  });

  const result = await client.healthDependencies();
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.equal(result.data.services.memory.status, "healthy");
    assert.equal(result.data.audit_events_count, 4);
  }
});

test("health failure is normalized without fake success", async () => {
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async () => jsonResponse({ detail: "database down" }, { status: 503 }),
  });

  const result = await client.health();
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.equal(result.status, 503);
    assert.equal(result.error.code, "server_error");
    assert.equal(result.error.message, "database down");
  }
});

test("missing endpoint returns a clean error", async () => {
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async () => jsonResponse({ detail: "not found" }, { status: 404 }),
  });

  const result = await client.request("/api/executions/does-not-exist", { method: "GET" });
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.equal(result.status, 404);
    assert.equal(result.error.code, "not_found");
    assert.equal(result.error.path?.startsWith("/api/executions/does-not-exist"), true);
  }
});

test("memory search normalizes an empty state", async () => {
  let seenPath = "";
  let seenBody = "";
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async (input, init) => {
      seenPath = String(input);
      seenBody = String(init?.body ?? "");
      return jsonResponse({ query: "nothing", mode: "hybrid", results: [] });
    },
  });

  const result = await client.memorySearch({ query: "nothing" });
  assert.equal(result.ok, true);
  assert.equal(seenPath.endsWith("/api/memory/search/unified"), true);
  assert.match(seenBody, /nothing/);
  if (result.ok) {
    assert.equal(result.data.count, 0);
    assert.deepEqual(result.data.results, []);
  }
});

test("audit events response is normalized", async () => {
  const client = createOperatorApiClient(baseConfig(), {
    fetchImpl: async () => jsonResponse({ tenant_id: "tenant-1", events: [{ operation: "search" }] }),
  });

  const result = await client.auditEvents();
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.equal(result.data.count, 1);
    assert.equal(result.data.events[0]?.operation, "search");
  }
});
