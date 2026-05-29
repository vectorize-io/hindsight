import { afterEach, describe, expect, it, vi } from "vitest";

import { normalizeBasePath, sanitizeReturnTo, stripBasePath, withBasePath } from "@/lib/base-path";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("base path helpers", () => {
  it("normalizes missing, root, and slash variants", () => {
    expect(normalizeBasePath(undefined)).toBe("");
    expect(normalizeBasePath("")).toBe("");
    expect(normalizeBasePath("/")).toBe("");
    expect(normalizeBasePath("ai-memory/")).toBe("/ai-memory");
    expect(normalizeBasePath("/ai-memory///")).toBe("/ai-memory");
  });

  it("prefixes app-relative paths when NEXT_PUBLIC_BASE_PATH is set", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    expect(withBasePath("/login")).toBe("/ai-memory/login");
    expect(withBasePath("api/auth/login")).toBe("/ai-memory/api/auth/login");
  });

  it("does not double-prefix paths that already include the base path", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    expect(withBasePath("/ai-memory/login")).toBe("/ai-memory/login");
    expect(withBasePath("/ai-memory/login?returnTo=%2Fdashboard")).toBe(
      "/ai-memory/login?returnTo=%2Fdashboard"
    );
  });

  it("strips the base path from returnTo-style paths while preserving query strings", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    expect(stripBasePath("/ai-memory/dashboard")).toBe("/dashboard");
    expect(stripBasePath("/ai-memory/dashboard?view=data")).toBe("/dashboard?view=data");
    expect(stripBasePath("/dashboard")).toBe("/dashboard");
  });

  it("builds a base-prefixed middleware login redirect with an app-relative returnTo", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    const loginUrl = new URL(withBasePath("/login"), "https://example.com/ai-memory/dashboard");
    loginUrl.searchParams.set("returnTo", stripBasePath("/ai-memory/dashboard"));

    expect(loginUrl.toString()).toBe("https://example.com/ai-memory/login?returnTo=%2Fdashboard");
  });

  describe("sanitizeReturnTo", () => {
    it("falls back when value is missing or empty", () => {
      expect(sanitizeReturnTo(null)).toBe("/dashboard");
      expect(sanitizeReturnTo(undefined)).toBe("/dashboard");
      expect(sanitizeReturnTo("")).toBe("/dashboard");
    });

    it("accepts safe app-relative paths and strips the base path", () => {
      vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

      expect(sanitizeReturnTo("/dashboard")).toBe("/dashboard");
      expect(sanitizeReturnTo("/dashboard?view=data")).toBe("/dashboard?view=data");
      expect(sanitizeReturnTo("/ai-memory/dashboard")).toBe("/dashboard");
    });

    it("rejects open-redirect payloads", () => {
      expect(sanitizeReturnTo("//evil.com/phish")).toBe("/dashboard");
      expect(sanitizeReturnTo("https://evil.com")).toBe("/dashboard");
      expect(sanitizeReturnTo("javascript:alert(1)")).toBe("/dashboard");
      expect(sanitizeReturnTo("data:text/html,<script>1</script>")).toBe("/dashboard");
      expect(sanitizeReturnTo("/\\evil.com")).toBe("/dashboard");
      expect(sanitizeReturnTo("dashboard")).toBe("/dashboard");
    });

    it("strips browser-ignored leading control chars before validating", () => {
      expect(sanitizeReturnTo(" \t\n//evil.com")).toBe("/dashboard");
      expect(sanitizeReturnTo("  /dashboard")).toBe("/dashboard");
    });

    it("honors a custom fallback", () => {
      expect(sanitizeReturnTo("//evil.com", "/home")).toBe("/home");
    });
  });

  it("leaves absolute URLs unchanged", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    expect(withBasePath("https://example.com/login")).toBe("https://example.com/login");
    expect(stripBasePath("https://example.com/ai-memory/login")).toBe(
      "https://example.com/ai-memory/login"
    );
  });
});
