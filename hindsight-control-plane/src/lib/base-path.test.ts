import { afterEach, describe, expect, it, vi } from "vitest";

import { normalizeBasePath, stripBasePath, withBasePath } from "./base-path";

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

  it("leaves absolute URLs unchanged", () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/ai-memory");

    expect(withBasePath("https://example.com/login")).toBe("https://example.com/login");
    expect(stripBasePath("https://example.com/ai-memory/login")).toBe(
      "https://example.com/ai-memory/login"
    );
  });
});
