import { describe, it, expect } from "vitest";
import { hasHindsightPolicy, mergeHindsightPolicy, serializePolicy } from "./policy-writer.js";
import { parseSandboxPolicy } from "./policy-reader.js";
import type { SandboxPolicy } from "./types.js";
import { HINDSIGHT_HOST, OPENCLAW_BINARY } from "./types.js";

const BASE_POLICY: SandboxPolicy = {
  version: 1,
  filesystem_policy: {
    include_workdir: true,
    read_only: ["/usr", "/lib"],
    read_write: ["/sandbox", "/tmp"],
  },
  network_policies: {
    claude_code: {
      name: "claude_code",
      endpoints: [
        {
          host: "api.anthropic.com",
          port: 443,
          rules: [{ allow: { method: "*", path: "/**" } }],
        },
      ],
      binaries: [{ path: "/usr/local/bin/claude" }],
    },
  },
};

describe("hasHindsightPolicy", () => {
  it("returns false when no hindsight policy exists", () => {
    expect(hasHindsightPolicy(BASE_POLICY)).toBe(false);
  });

  it("returns false when network_policies is undefined", () => {
    expect(hasHindsightPolicy({ version: 1 })).toBe(false);
  });

  it("returns true when hindsight policy is present", () => {
    const withHindsight = mergeHindsightPolicy(BASE_POLICY);
    expect(hasHindsightPolicy(withHindsight)).toBe(true);
  });
});

describe("mergeHindsightPolicy", () => {
  it("adds the hindsight network policy block", () => {
    const result = mergeHindsightPolicy(BASE_POLICY);
    expect(result.network_policies?.hindsight).toBeDefined();
    expect(result.network_policies?.hindsight?.endpoints[0].host).toBe(HINDSIGHT_HOST);
  });

  it("preserves all existing network policies", () => {
    const result = mergeHindsightPolicy(BASE_POLICY);
    expect(result.network_policies?.claude_code).toBeDefined();
    expect(result.network_policies?.claude_code?.name).toBe("claude_code");
  });

  it("sets the correct binary path", () => {
    const result = mergeHindsightPolicy(BASE_POLICY);
    const binaries = result.network_policies?.hindsight?.binaries ?? [];
    expect(binaries.some((b) => b.path === OPENCLAW_BINARY)).toBe(true);
  });

  it("includes GET, POST, and PUT rules", () => {
    const result = mergeHindsightPolicy(BASE_POLICY);
    const rules = result.network_policies?.hindsight?.endpoints[0].rules ?? [];
    const methods = rules.map((r) => r.allow.method);
    expect(methods).toContain("GET");
    expect(methods).toContain("POST");
    expect(methods).toContain("PUT");
  });

  it("is idempotent — merging twice yields the same result", () => {
    const once = mergeHindsightPolicy(BASE_POLICY);
    const twice = mergeHindsightPolicy(once);
    expect(JSON.stringify(twice.network_policies?.hindsight)).toBe(
      JSON.stringify(once.network_policies?.hindsight)
    );
  });

  it("does not mutate the original policy", () => {
    const original = JSON.parse(JSON.stringify(BASE_POLICY)) as SandboxPolicy;
    mergeHindsightPolicy(BASE_POLICY);
    expect(BASE_POLICY.network_policies?.hindsight).toBeUndefined();
    expect(JSON.stringify(BASE_POLICY)).toBe(JSON.stringify(original));
  });
});

describe("serializePolicy", () => {
  it("produces valid YAML that round-trips through parseSandboxPolicy", () => {
    const merged = mergeHindsightPolicy(BASE_POLICY);
    const yamlStr = serializePolicy(merged);
    // Wrap in Policy: header as parseSandboxPolicy expects
    const wrapped =
      "Policy:\n" +
      yamlStr
        .split("\n")
        .map((l) => `  ${l}`)
        .join("\n");
    const reparsed = parseSandboxPolicy(wrapped);
    expect(reparsed.version).toBe(merged.version);
    expect(reparsed.network_policies?.hindsight?.endpoints[0].host).toBe(HINDSIGHT_HOST);
    expect(reparsed.network_policies?.claude_code).toBeDefined();
  });

  it("includes all network policies in output", () => {
    const merged = mergeHindsightPolicy(BASE_POLICY);
    const yaml = serializePolicy(merged);
    expect(yaml).toContain("claude_code:");
    expect(yaml).toContain("hindsight:");
    expect(yaml).toContain(HINDSIGHT_HOST);
  });
});
