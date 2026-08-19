import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { HARNESS_LOGO_REGISTRY, documentHarness, resolveHarnessLogo } from "@/lib/harness-logo";

const PUBLIC_DIR = resolve(__dirname, "..", "..", "public");

describe("documentHarness", () => {
  it("reads the harness metadata field written by the coding-agent integrations", () => {
    expect(documentHarness({ source: "chat", harness: "claude-code" }, ["source:chat"])).toBe(
      "claude-code"
    );
  });

  it("falls back to the harness:<id> tag when metadata is absent", () => {
    // Documents retained before the metadata field existed, and hand-tagged ones,
    // only carry the tag.
    expect(documentHarness(null, ["source:chat", "harness:codex"])).toBe("codex");
    expect(documentHarness({ source: "chat" }, ["harness:cursor-cli"])).toBe("cursor-cli");
  });

  it("prefers metadata over the tag", () => {
    expect(documentHarness({ harness: "opencode" }, ["harness:codex"])).toBe("opencode");
  });

  it("returns null when nothing identifies a harness", () => {
    expect(documentHarness(null, null)).toBeNull();
    expect(documentHarness({ source: "git" }, ["source:git"])).toBeNull();
    // A bare prefix carries no value and must not resolve to the empty string.
    expect(documentHarness({ harness: "  " }, ["harness:"])).toBeNull();
  });
});

describe("resolveHarnessLogo", () => {
  // The exact set hindsight-coding-agents emits: one per HookSpec in
  // src/harness/hook-lifecycle.ts, plus the persistent-plugin harnesses whose id
  // is their entrypoint's createPluginEntry(...) argument (opencode, kilo,
  // cline-cli). The registry must cover these and nothing speculative.
  const EMITTED_HARNESSES = [
    "antigravity-cli",
    "claude-code",
    "cline-cli",
    "codex",
    "copilot-cli",
    "cursor-cli",
    "devin-cli",
    "dsh",
    "grok-build",
    "kilo",
    "opencode",
    "prime-agent",
  ];
  // Ids the integration used to emit. Kept so documents already retained under
  // them keep their logo; a new id never belongs here.
  const RETIRED_HARNESSES = ["gemini"];

  it("resolves every id the coding-agent integration emits", () => {
    for (const id of EMITTED_HARNESSES) {
      expect(resolveHarnessLogo(id)?.id).toBe(id);
    }
  });

  it("registers no harness that nothing writes", () => {
    expect(Object.keys(HARNESS_LOGO_REGISTRY).sort()).toEqual(
      [...EMITTED_HARNESSES, ...RETIRED_HARNESSES].sort()
    );
  });

  it("normalizes case, spacing and underscores", () => {
    // The harness can come from the user's own config key, not only a HookSpec.
    expect(resolveHarnessLogo("Claude_Code")?.id).toBe("claude-code");
    expect(resolveHarnessLogo(" CLAUDE CODE ")?.id).toBe("claude-code");
  });

  it("returns null for an unknown or empty harness instead of a placeholder", () => {
    expect(resolveHarnessLogo("some-new-agent")).toBeNull();
    expect(resolveHarnessLogo("")).toBeNull();
    expect(resolveHarnessLogo(null)).toBeNull();
  });

  it("points every entry at an asset that is actually shipped", () => {
    // The registry and public/img/harness are edited in separate steps; a typo
    // or a forgotten copy would only show as a broken image in the browser.
    for (const logo of Object.values(HARNESS_LOGO_REGISTRY)) {
      expect(existsSync(join(PUBLIC_DIR, logo.src)), `${logo.id} -> ${logo.src}`).toBe(true);
    }
  });
});
