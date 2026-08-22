import { describe, expect, it, vi } from "vitest";

vi.mock("@vectorize-io/hindsight-client", () => ({
  HindsightClient: class {},
  HindsightError: class extends Error {},
  createClient: () => ({}),
  createConfig: (c: unknown) => c,
  sdk: {},
}));

import { bankIdHasTraversal, dataplaneBankUrl } from "@/lib/hindsight-client";

describe("bankIdHasTraversal", () => {
  it("flags path separators and bounded .. segments", () => {
    expect(bankIdHasTraversal("a/b")).toBe(true);
    expect(bankIdHasTraversal("a\\b")).toBe(true);
    expect(bankIdHasTraversal("..")).toBe(true);
    expect(bankIdHasTraversal("../x")).toBe(true);
    expect(bankIdHasTraversal("x/..")).toBe(true);
    expect(bankIdHasTraversal("u2--x/../victim")).toBe(true);
  });

  it("does not flag legitimate bank ids", () => {
    expect(bankIdHasTraversal("u2")).toBe(false);
    expect(bankIdHasTraversal("u2--notes")).toBe(false);
    expect(bankIdHasTraversal("agent-1::channel-2::user-3")).toBe(false);
    expect(bankIdHasTraversal("SX.Products.GovComply.Build")).toBe(false);
    expect(bankIdHasTraversal("a..b")).toBe(false); // not a bounded ".." segment
  });
});

describe("dataplaneBankUrl", () => {
  it("percent-encodes the bank id into a single path segment", () => {
    expect(dataplaneBankUrl("my bank")).toContain("/v1/default/banks/my%20bank");
    expect(dataplaneBankUrl("agent-1::channel-2")).toContain(
      "/v1/default/banks/agent-1%3A%3Achannel-2"
    );
  });

  it("appends the suffix after the encoded id", () => {
    expect(dataplaneBankUrl("u2", "/graph?limit=5")).toContain(
      "/v1/default/banks/u2/graph?limit=5"
    );
  });

  it("throws on a traversal-shaped bank id instead of building a URL", () => {
    // Without the guard this would normalize to .../banks/victim/graph.
    expect(() => dataplaneBankUrl("u2--x/../victim", "/graph")).toThrow(/Invalid bank_id/);
    expect(() => dataplaneBankUrl("../victim")).toThrow(/Invalid bank_id/);
    expect(() => dataplaneBankUrl("a\\b")).toThrow(/Invalid bank_id/);
  });

  it("keeps a normalized URL free of dot segments", () => {
    const url = dataplaneBankUrl("u2--notes", "/documents/d1/chunks");
    expect(new URL(url).pathname).toBe("/v1/default/banks/u2--notes/documents/d1/chunks");
  });
});
