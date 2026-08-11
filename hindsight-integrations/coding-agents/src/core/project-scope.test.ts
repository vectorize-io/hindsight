import { describe, expect, it } from "vitest";
import { resolveConfig } from "./config";
import { buildScopedRetainStamp, resolveProjectScope, scopeTagGroups } from "./project-scope";

describe("project tag scope", () => {
  it("is disabled by default", () => {
    expect(resolveProjectScope(resolveConfig({}), "/tmp/p", "codex", "default")).toBeUndefined();
  });

  it("resolves strict project and optional global tag groups", () => {
    const scope = resolveProjectScope(
      resolveConfig({ projectScope: "tags", globalTags: ["scope:global"] }),
      "/tmp/potcodev",
      "codex",
      "default"
    );
    expect(scope).toEqual({ projectTag: "project:potcodev", globalTags: ["scope:global"] });
    expect(scopeTagGroups(scope!)).toEqual([
      {
        or: [
          { tags: ["project:potcodev"], match: "any_strict" },
          { tags: ["scope:global"], match: "any_strict" },
        ],
      },
    ]);
  });

  it("adds and deduplicates the project tag only while tag scope is active", () => {
    const ctx = { directory: "/tmp/potcodev", harness: "codex", bankId: "default" };
    expect(
      buildScopedRetainStamp(
        resolveConfig({ projectScope: "tags", retainTags: ["project:{gitProject}", "env:work"] }),
        ctx
      ).tags
    ).toEqual(["project:potcodev", "env:work"]);
    expect(buildScopedRetainStamp(resolveConfig({ projectScope: "bank" }), ctx).tags).toEqual([]);
  });
});
