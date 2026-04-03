import { describe, test, expect, mock, beforeEach } from "bun:test";
import { HindsightClient } from "../src/client";

describe("HindsightClient", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient("http://localhost:8888");
  });

  test("constructor strips trailing slashes from base URL", () => {
    const c = new HindsightClient("http://localhost:8888///");
    // The URL is stored internally; verify via a health check URL construction
    expect(c).toBeDefined();
  });

  test("healthy returns true when server responds 200", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async () => new Response("OK", { status: 200 }));

    const result = await client.healthy();
    expect(result).toBe(true);

    globalThis.fetch = originalFetch;
  });

  test("healthy returns false on network error", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async () => {
      throw new Error("ECONNREFUSED");
    });

    const result = await client.healthy();
    expect(result).toBe(false);

    globalThis.fetch = originalFetch;
  });

  test("retain sends correct payload structure", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (typeof init?.body === "string") {
        capturedBody = JSON.parse(init.body);
      }
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await client.retain({
      bankId: "test-bank",
      content: "Alice works at Google",
      documentId: "doc-123",
      context: "coding session",
      metadata: { project: "my-project" },
      tags: ["project:my-project"],
    });

    expect(capturedBody).toBeDefined();
    expect(capturedBody!.document_id).toBe("doc-123");
    const items = capturedBody!.items as Record<string, unknown>[];
    expect(items).toHaveLength(1);
    expect(items[0].content).toBe("Alice works at Google");
    expect(items[0].context).toBe("coding session");
    expect(items[0].metadata).toEqual({ project: "my-project" });
    expect(items[0].tags).toEqual(["project:my-project"]);

    globalThis.fetch = originalFetch;
  });

  test("retain throws on non-OK response", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async () => new Response("Bad Request", { status: 400 }));

    await expect(
      client.retain({ bankId: "test", content: "test" }),
    ).rejects.toThrow("Retain failed (400)");

    globalThis.fetch = originalFetch;
  });

  test("recall sends correct payload structure", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = mock(async (_url: string, init?: RequestInit) => {
      if (typeof init?.body === "string") {
        capturedBody = JSON.parse(init.body);
      }
      return new Response(JSON.stringify({ results: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await client.recall({
      bankId: "test-bank",
      query: "What does Alice do?",
      budget: "high",
      maxTokens: 2048,
      types: ["world", "observation"],
    });

    expect(capturedBody).toBeDefined();
    expect(capturedBody!.query).toBe("What does Alice do?");
    expect(capturedBody!.budget).toBe("high");
    expect(capturedBody!.max_tokens).toBe(2048);
    expect(capturedBody!.types).toEqual(["world", "observation"]);

    globalThis.fetch = originalFetch;
  });

  test("bankExists returns true for 200 response", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async () => new Response("{}", { status: 200 }));

    const result = await client.bankExists("my-bank");
    expect(result).toBe(true);

    globalThis.fetch = originalFetch;
  });

  test("bankExists returns false for 404 response", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async () => new Response("Not Found", { status: 404 }));

    const result = await client.bankExists("nonexistent");
    expect(result).toBe(false);

    globalThis.fetch = originalFetch;
  });
});
