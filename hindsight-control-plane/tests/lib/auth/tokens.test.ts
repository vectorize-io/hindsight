import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { bankAllowed, bankIdHasTraversal, resolveToken } from "@/lib/auth/tokens";

const ORIGINAL_ENV = { ...process.env };

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
  vi.restoreAllMocks();
});

describe("bankAllowed", () => {
  it("admin (empty prefix) reaches any well-formed bank", () => {
    expect(bankAllowed("", "u2")).toBe(true);
    expect(bankAllowed("", "u2--notes")).toBe(true);
    expect(bankAllowed("", "anything")).toBe(true);
  });

  it("matches the prefix exactly or as a namespaced child", () => {
    expect(bankAllowed("u2", "u2")).toBe(true);
    expect(bankAllowed("u2", "u2--notes")).toBe(true);
    expect(bankAllowed("u2", "u2--a--b")).toBe(true);
  });

  it("does not treat a longer numeric prefix as a child (u2 vs u20 boundary)", () => {
    expect(bankAllowed("u2", "u20")).toBe(false);
    expect(bankAllowed("u2", "u20--notes")).toBe(false);
    expect(bankAllowed("u2", "u2x")).toBe(false);
  });

  it("rejects a foreign bank", () => {
    expect(bankAllowed("u2", "u5")).toBe(false);
    expect(bankAllowed("u2", "victim")).toBe(false);
  });

  it("rejects traversal-shaped ids even when the prefix would otherwise match", () => {
    // The exact exploit shape from the review: passes startsWith('u2--') but
    // normalizes to a foreign bank once the dot segment collapses.
    expect(bankAllowed("u2", "u2--x/../victim")).toBe(false);
    expect(bankAllowed("u2", "u2--x/../../victim")).toBe(false);
    expect(bankAllowed("u2", "u2--x\\..\\victim")).toBe(false);
    expect(bankAllowed("u2", "u2--sub/child")).toBe(false);
  });

  it("rejects traversal-shaped ids even under the admin scope", () => {
    expect(bankAllowed("", "u2--x/../victim")).toBe(false);
    expect(bankAllowed("", "../victim")).toBe(false);
    expect(bankAllowed("", "a/b")).toBe(false);
  });
});

describe("bankIdHasTraversal", () => {
  it("flags separators and dot-dot segments", () => {
    expect(bankIdHasTraversal("a/b")).toBe(true);
    expect(bankIdHasTraversal("a\\b")).toBe(true);
    expect(bankIdHasTraversal("..")).toBe(true);
    expect(bankIdHasTraversal("../x")).toBe(true);
    expect(bankIdHasTraversal("x/..")).toBe(true);
    expect(bankIdHasTraversal("x/../y")).toBe(true);
  });

  it("does not flag legitimate ids, including a literal double dot mid-token", () => {
    expect(bankIdHasTraversal("u2")).toBe(false);
    expect(bankIdHasTraversal("u2--notes")).toBe(false);
    expect(bankIdHasTraversal("SX.Products.GovComply.Build")).toBe(false);
    expect(bankIdHasTraversal("a..b")).toBe(false); // not a bounded ".." segment
  });
});

describe("resolveToken scoped-token parsing", () => {
  it("resolves the admin access key to the empty prefix", () => {
    process.env.HINDSIGHT_CP_ACCESS_KEY = "admin-key";
    delete process.env.HINDSIGHT_CP_TOKENS;
    expect(resolveToken("admin-key")).toEqual({ prefix: "", label: "admin" });
    expect(resolveToken("nope")).toBeNull();
  });

  it("resolves a configured scoped token to its prefix", () => {
    process.env.HINDSIGHT_CP_ACCESS_KEY = "admin-key";
    process.env.HINDSIGHT_CP_TOKENS = JSON.stringify([
      { token: "u2-token", prefix: "u2", label: "user 2" },
    ]);
    expect(resolveToken("u2-token")).toEqual({ prefix: "u2", label: "user 2" });
  });

  it("drops an entry with an empty prefix rather than granting admin", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    process.env.HINDSIGHT_CP_ACCESS_KEY = "admin-key";
    process.env.HINDSIGHT_CP_TOKENS = JSON.stringify([{ token: "typo-token", prefix: "" }]);
    // The empty-prefix entry must NOT resolve (it would otherwise be admin).
    expect(resolveToken("typo-token")).toBeNull();
    expect(warn).toHaveBeenCalled();
  });

  it("drops a non-string prefix entry", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    process.env.HINDSIGHT_CP_ACCESS_KEY = "admin-key";
    process.env.HINDSIGHT_CP_TOKENS = JSON.stringify([{ token: "t", prefix: 123 }]);
    expect(resolveToken("t")).toBeNull();
  });
});
