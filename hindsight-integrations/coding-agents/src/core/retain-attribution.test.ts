import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveRetainAttribution } from "./retain-attribution";

const roots: string[] = [];

afterEach(() => {
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("resolveRetainAttribution", () => {
  it("resolves project, harness, cwd, and bank placeholders", () => {
    const root = mkdtempSync(join(tmpdir(), "hs-retain-tags-"));
    roots.push(root);

    expect(
      resolveRetainAttribution(
        {
          retainTags: ["project:{gitProject}", "harness:{harness}", "bank:{bankId}"],
          retainMetadata: { cwd: "{cwd}", project: "{project}" },
        },
        root,
        "codex",
        "default"
      )
    ).toEqual({
      tags: [`project:${root.split("/").at(-1)}`, "harness:codex", "bank:default"],
      metadata: { cwd: root, project: root.split("/").at(-1) },
    });
  });

  it("deduplicates tags and drops templates with an empty namespace value", () => {
    expect(
      resolveRetainAttribution(
        { retainTags: ["source:chat", "source:chat", "optional:"] },
        "/tmp/project",
        "codex",
        "default"
      ).tags
    ).toEqual(["source:chat"]);
  });

  it("omits values containing unknown placeholders instead of storing ambiguous attribution", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(
      resolveRetainAttribution(
        {
          retainTags: ["project:{missing}", "harness:{harness}", "other:{missing}"],
          retainMetadata: { project: "{missing}", harness: "{harness}" },
        },
        "/tmp/project",
        "codex",
        "default"
      )
    ).toEqual({ tags: ["harness:codex"], metadata: { harness: "codex" } });
    expect(error).toHaveBeenCalledTimes(1);
  });
});
