import { describe, it, expect } from "vitest";
import { stripAnsi, extractPolicyYaml, parseSandboxPolicy } from "./policy-reader.js";
import { serializePolicy } from "./policy-writer.js";

// Fixture: actual output of `openshell sandbox get my-assistant`
// (ANSI codes represented as escape sequences)
const FIXTURE_RAW = `\x1b[1m\x1b[36mSandbox:\x1b[39m\x1b[0m

  \x1b[2mId:\x1b[0m 61c993f1-010f-4eca-a1ac-d6ddec9d604a
  \x1b[2mName:\x1b[0m my-assistant
  \x1b[2mNamespace:\x1b[0m openshell
  \x1b[2mPhase:\x1b[0m Ready

\x1b[1m\x1b[36mPolicy:\x1b[39m\x1b[0m

  \x1b[2mversion\x1b[0m\x1b[2m:\x1b[0m 1
  \x1b[2mfilesystem_policy\x1b[0m\x1b[2m:\x1b[0m
    \x1b[2minclude_workdir\x1b[0m\x1b[2m:\x1b[0m true
    \x1b[2mread_only\x1b[0m\x1b[2m:\x1b[0m
    \x1b[2m- \x1b[0m/usr
    \x1b[2m- \x1b[0m/lib
    \x1b[2mread_write\x1b[0m\x1b[2m:\x1b[0m
    \x1b[2m- \x1b[0m/sandbox
    \x1b[2m- \x1b[0m/tmp
  \x1b[2mnetwork_policies\x1b[0m\x1b[2m:\x1b[0m
    \x1b[2mclaude_code\x1b[0m\x1b[2m:\x1b[0m
      \x1b[2mname\x1b[0m\x1b[2m:\x1b[0m claude_code
      \x1b[2mendpoints\x1b[0m\x1b[2m:\x1b[0m
      \x1b[2m- \x1b[0mhost: api.anthropic.com
        \x1b[2mport\x1b[0m\x1b[2m:\x1b[0m 443
        \x1b[2mrules\x1b[0m\x1b[2m:\x1b[0m
        \x1b[2m- \x1b[0mallow:
            \x1b[2mmethod\x1b[0m\x1b[2m:\x1b[0m '*'
            \x1b[2mpath\x1b[0m\x1b[2m:\x1b[0m /**
      \x1b[2mbinaries\x1b[0m\x1b[2m:\x1b[0m
      \x1b[2m- \x1b[0mpath: /usr/local/bin/claude
`;

describe("stripAnsi", () => {
  it("removes ANSI escape codes", () => {
    expect(stripAnsi("\x1b[1m\x1b[36mHello\x1b[39m\x1b[0m")).toBe("Hello");
  });

  it("leaves plain strings unchanged", () => {
    expect(stripAnsi("version: 1")).toBe("version: 1");
  });

  it("handles strings with no ANSI codes", () => {
    expect(stripAnsi("  - /usr")).toBe("  - /usr");
  });
});

describe("extractPolicyYaml", () => {
  it("extracts the Policy: section and dedents by 2 spaces", () => {
    const result = extractPolicyYaml(FIXTURE_RAW);
    expect(result).toContain("version: 1");
    expect(result).toContain("filesystem_policy:");
    expect(result).toContain("network_policies:");
  });

  it("does not include the Sandbox: section", () => {
    const result = extractPolicyYaml(FIXTURE_RAW);
    expect(result).not.toContain("Sandbox:");
    expect(result).not.toContain("my-assistant");
  });

  it("throws if Policy: section is missing", () => {
    expect(() => extractPolicyYaml("no policy here")).toThrow('Could not find "Policy:"');
  });
});

describe("parseSandboxPolicy", () => {
  it("parses version field", () => {
    const policy = parseSandboxPolicy(FIXTURE_RAW);
    expect(policy.version).toBe(1);
  });

  it("parses filesystem_policy", () => {
    const policy = parseSandboxPolicy(FIXTURE_RAW);
    expect(policy.filesystem_policy?.include_workdir).toBe(true);
    expect(policy.filesystem_policy?.read_only).toContain("/usr");
  });

  it("parses network_policies", () => {
    const policy = parseSandboxPolicy(FIXTURE_RAW);
    expect(policy.network_policies).toBeDefined();
    expect(policy.network_policies?.claude_code).toBeDefined();
    expect(policy.network_policies?.claude_code?.name).toBe("claude_code");
  });

  it("is idempotent — parse → serialize → parse yields same structure", () => {
    const policy1 = parseSandboxPolicy(FIXTURE_RAW);
    const yamlStr = serializePolicy(policy1);
    // Re-wrap in a Policy: header to match the expected format
    const wrapped =
      "Policy:\n" +
      yamlStr
        .split("\n")
        .map((l: string) => `  ${l}`)
        .join("\n");
    const policy2 = parseSandboxPolicy(wrapped);
    expect(policy2.version).toBe(policy1.version);
    expect(Object.keys(policy2.network_policies ?? {})).toEqual(
      Object.keys(policy1.network_policies ?? {})
    );
  });
});
